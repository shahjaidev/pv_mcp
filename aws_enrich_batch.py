#!/usr/bin/env python3
"""Batch AWS Bedrock enrichment utility for large CSV jobs.

Processes an input CSV (for example ~7k rows) by calling an Amazon Titan
model via the Bedrock Conversation API with bounded concurrency, retries,
and rate limiting. Results are streamed to JSON Lines locally and can be
uploaded to S3 when complete.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from aws_enrich_simple import (
    build_prompt,
    call_bedrock,
    create_default_key,
    ensure_genres,
    upload_to_s3,
)


class RateLimiter:
    """Simple rate limiter for worker threads."""

    def __init__(self, rate_per_second: float) -> None:
        self._rate = max(rate_per_second, 0.0)
        self._lock = threading.Lock()
        self._next_allowed = time.time()
        self._interval = (1.0 / self._rate) if self._rate > 0 else 0.0

    def acquire(self) -> None:
        if self._rate <= 0:
            return
        while True:
            with self._lock:
                now = time.time()
                if now >= self._next_allowed:
                    self._next_allowed = now + self._interval
                    return
                sleep_for = self._next_allowed - now
            time.sleep(max(sleep_for, 0.0))


def iter_rows(path: str, limit: Optional[int]) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            yield idx, row
            if limit is not None and idx >= limit:
                break


def enrich_single(
    index: int,
    row: Dict[str, Any],
    *,
    client: Any,
    model_id: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    rate_limiter: RateLimiter,
    max_retries: int,
    retry_backoff: float,
    debug: bool,
) -> Tuple[int, Dict[str, Any]]:
    prompt = build_prompt(row)
    attempt = 0
    while True:
        attempt += 1
        rate_limiter.acquire()
        payload, raw_text, finish_reason, error = call_bedrock(
            client,
            model_id,
            prompt,
            temperature,
            max_tokens,
            top_p,
            debug,
        )
        summary = str(payload.get("summary", "")).strip()
        genres = ensure_genres(payload.get("genres"))
        enriched_row: Dict[str, Any] = {**row, "summary": summary, "genres": genres}

        final_error = error or ""
        if (not summary or not genres) and not final_error:
            final_error = "Missing summary or genres"
        if final_error:
            if raw_text:
                enriched_row["raw_response"] = raw_text
            if finish_reason:
                enriched_row["finish_reason"] = finish_reason
            enriched_row["error"] = final_error

        if "error" not in enriched_row or attempt > max_retries:
            return index, enriched_row

        sleep_for = retry_backoff * (2 ** (attempt - 1))
        time.sleep(min(sleep_for, 60))


def write_jsonl_line(handle: Any, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False))
    handle.write("\n")
    handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich CSV rows using Amazon Titan via Bedrock with bounded concurrency. "
            "Designed for large jobs (e.g. ~7000 rows)."
        )
    )
    parser.add_argument("--input", default="pv_all_US.csv", help="Input CSV path")
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows to process")
    parser.add_argument("--model", default="amazon.titan-text-premier-v1:0", help="Bedrock model ID")
    parser.add_argument("--region", default="us-east-1", help="AWS region for Bedrock and S3 clients")
    parser.add_argument("--temperature", type=float, default=0.5, help="Generation temperature")
    parser.add_argument("--max-output-tokens", type=int, default=1024, help="Maximum output tokens")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling value")
    parser.add_argument("--max-workers", type=int, default=4, help="Concurrent Bedrock requests")
    parser.add_argument(
        "--rate-per-second",
        type=float,
        default=2.0,
        help="Approximate maximum Bedrock requests per second",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retry attempts for errors")
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=2.0,
        help="Base seconds for exponential backoff between retries",
    )
    parser.add_argument(
        "--output",
        help="Local JSONL output path (defaults to input stem + '_enriched.jsonl')",
    )
    parser.add_argument("--status-every", type=int, default=100, help="Progress print frequency")
    parser.add_argument("--s3-bucket", help="Optional destination S3 bucket name")
    parser.add_argument("--s3-key", help="Destination S3 object key (JSONL)")
    parser.add_argument("--s3-prefix", help="Optional prefix for generated S3 key if --s3-key not set")
    parser.add_argument("--debug", action="store_true", help="Print responses to stderr for debugging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output) if args.output else Path(args.input).with_suffix("").with_name(Path(args.input).stem + "_enriched.jsonl")

    bedrock_client = boto3.client("bedrock-runtime", region_name=args.region)

    rate_limiter = RateLimiter(args.rate_per_second)
    success_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor, open(
        output_path, "w", encoding="utf-8"
    ) as output_handle:
        futures: List[Future[Tuple[int, Dict[str, Any]]]] = []
        for index, row in iter_rows(args.input, args.limit):
            future = executor.submit(
                enrich_single,
                index,
                row,
                client=bedrock_client,
                model_id=args.model,
                temperature=args.temperature,
                max_tokens=args.max_output_tokens,
                top_p=args.top_p,
                rate_limiter=rate_limiter,
                max_retries=args.max_retries,
                retry_backoff=args.retry_backoff,
                debug=args.debug,
            )
            futures.append(future)

        pending: Dict[int, Dict[str, Any]] = {}
        next_to_write = 1
        for future in as_completed(futures):
            idx, enriched_row = future.result()
            pending[idx] = enriched_row
            if "error" in enriched_row:
                error_count += 1
            else:
                success_count += 1
            while next_to_write in pending:
                write_jsonl_line(output_handle, pending.pop(next_to_write))
                next_to_write += 1
            if args.status_every and (success_count + error_count) % args.status_every == 0:
                print(
                    f"Processed {success_count + error_count} rows (success={success_count}, errors={error_count})",
                    file=sys.stderr,
                )

    print(
        f"Completed enrichment. Success={success_count}, Errors={error_count}. Output written to {output_path}",
        file=sys.stderr,
    )

    if args.s3_bucket:
        with open(output_path, "r", encoding="utf-8") as result_handle:
            body = result_handle.read()
        key = args.s3_key or create_default_key(args.s3_prefix)
        try:
            upload_to_s3(args.s3_bucket, key, body, args.region)
        except (ClientError, BotoCoreError) as exc:  # pragma: no cover - network path
            print(f"Failed to upload to s3://{args.s3_bucket}/{key}: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"Uploaded output to s3://{args.s3_bucket}/{key}")


if __name__ == "__main__":
    main()
