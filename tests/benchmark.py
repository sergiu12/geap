#!/usr/bin/env python3
"""
High-Performance Benchmarking Suite for LLM Endpoints on Agent Platform (fka Vertex AI).
Measures:
  - Total Request Throughput (requests/sec)
  - Output Token Generation Throughput (tokens/sec)
  - End-to-End Latency percentiles (P50, P90, P95, P99)
  - Time per Output Token (TPOT / ms per token)
  - Error rate and concurrency saturation
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import httpx


@dataclass
class RequestBenchmarkResult:
    request_id: int
    success: bool
    latency_sec: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error_msg: Optional[str] = None


@dataclass
class BenchmarkSummary:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_time_sec: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    concurrency: int = 1
    latencies: List[float] = field(default_factory=list)


def get_gcloud_auth_token() -> str:
    """Retrieve OAuth 2.0 access token via gcloud CLI."""
    token = os.getenv("GOOGLE_OAUTH_TOKEN")
    if token:
        return token.strip()
    try:
        result = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"Warning: Could not retrieve gcloud access token: {e}", file=sys.stderr)
        return ""


def calculate_percentiles(values: List[float], percentiles: List[float]) -> Dict[str, float]:
    """Calculate percentiles for a list of values."""
    if not values:
        return {f"P{int(p*100)}": 0.0 for p in percentiles}
    sorted_vals = sorted(values)
    results = {}
    for p in percentiles:
        k = (len(sorted_vals) - 1) * p
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_vals) else f
        d = k - f
        val = sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * d
        results[f"P{int(p*100)}"] = val
    return results


async def send_single_benchmark_request(
    client: httpx.AsyncClient,
    request_id: int,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    is_vertex_predict: bool,
    semaphore: asyncio.Semaphore,
) -> RequestBenchmarkResult:
    """Sends a single benchmark request with accurate latency and token accounting."""
    async with semaphore:
        start_time = time.perf_counter()
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=300.0)
            latency = time.perf_counter() - start_time

            if response.status_code != 200:
                return RequestBenchmarkResult(
                    request_id=request_id,
                    success=False,
                    latency_sec=latency,
                    error_msg=f"HTTP {response.status_code}: {response.text[:200]}",
                )

            data = response.json()
            input_tokens = 0
            output_tokens = 0

            if is_vertex_predict:
                predictions = data.get("predictions", [])
                if predictions:
                    usage = predictions[0].get("usage", {})
                    input_tokens = usage.get("prompt_tokens", 0)
                    output_tokens = usage.get("completion_tokens", 0)
                    if output_tokens == 0:
                        content = predictions[0].get("choices", [{}])[0].get("message", {}).get("content", "")
                        output_tokens = max(1, len(content.split()))
            else:
                usage = data.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                if output_tokens == 0:
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    output_tokens = max(1, len(content.split()))

            return RequestBenchmarkResult(
                request_id=request_id,
                success=True,
                latency_sec=latency,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except Exception as e:
            latency = time.perf_counter() - start_time
            return RequestBenchmarkResult(
                request_id=request_id,
                success=False,
                latency_sec=latency,
                error_msg=str(e),
            )


def generate_benchmark_prompt(approx_tokens: int) -> str:
    """Generates a synthetic prompt of specified approximate token size."""
    base_text = "The quick brown fox jumps over the lazy dog and explores modern artificial intelligence architectures. "
    repeats = max(1, approx_tokens // 15)
    return f"Write a comprehensive explanation on distributed AI inference and hardware acceleration. Background context: {base_text * repeats}"


async def run_benchmark(
    endpoint_url: str,
    headers: Dict[str, str],
    is_vertex_predict: bool,
    num_prompts: int,
    concurrency: int,
    prompt_tokens: int,
    max_tokens: int,
    temperature: float,
) -> BenchmarkSummary:
    """Executes the asynchronous benchmark against the target endpoint."""
    print(f"\n==================================================================")
    print(f" LLM Endpoint Benchmarking Suite")
    print(f"==================================================================")
    print(f" Target Endpoint:        {endpoint_url}")
    print(f" Total Requests:         {num_prompts}")
    print(f" Concurrency Level:      {concurrency}")
    print(f" Target Output Tokens:   {max_tokens}")
    print(f" Target Prompt Tokens:   ~{prompt_tokens}")
    print(f" Protocol:               {'Agent Platform (/predict)' if is_vertex_predict else 'OpenAI (/v1/chat/completions)'}")
    print(f"==================================================================\n")

    prompt_text = generate_benchmark_prompt(prompt_tokens)

    if is_vertex_predict:
        payload = {
            "instances": [
                {
                    "messages": [
                        {"role": "system", "content": "You are an expert AI performance benchmarking assistant."},
                        {"role": "user", "content": prompt_text},
                    ],
                    "temperature": temperature,
                }
            ],
            "parameters": {
                "max_tokens": max_tokens,
            },
        }
    else:
        payload = {
            "model": "default",
            "messages": [
                {"role": "system", "content": "You are an expert AI performance benchmarking assistant."},
                {"role": "user", "content": prompt_text},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    semaphore = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_keepalive_connections=concurrency * 2, max_connections=concurrency * 4)

    print(f"Warming up connection pool & starting benchmark...")
    start_bench_time = time.perf_counter()

    async with httpx.AsyncClient(limits=limits, timeout=300.0) as client:
        tasks = [
            send_single_benchmark_request(
                client=client,
                request_id=i + 1,
                url=endpoint_url,
                headers=headers,
                payload=payload,
                is_vertex_predict=is_vertex_predict,
                semaphore=semaphore,
            )
            for i in range(num_prompts)
        ]
        results = await asyncio.gather(*tasks)

    total_bench_time = time.perf_counter() - start_bench_time

    summary = BenchmarkSummary(
        total_requests=num_prompts,
        concurrency=concurrency,
        total_time_sec=total_bench_time,
    )

    for r in results:
        if r.success:
            summary.successful_requests += 1
            summary.total_input_tokens += r.input_tokens
            summary.total_output_tokens += r.output_tokens
            summary.latencies.append(r.latency_sec)
        else:
            summary.failed_requests += 1
            print(f" [!] Request {r.request_id} failed: {r.error_msg}", file=sys.stderr)

    return summary


def display_results_table(summary: BenchmarkSummary):
    """Formats and prints comprehensive benchmark performance metrics."""
    total_tokens = summary.total_input_tokens + summary.total_output_tokens
    req_per_sec = summary.successful_requests / summary.total_time_sec if summary.total_time_sec > 0 else 0.0
    out_tokens_per_sec = summary.total_output_tokens / summary.total_time_sec if summary.total_time_sec > 0 else 0.0
    total_tokens_per_sec = total_tokens / summary.total_time_sec if summary.total_time_sec > 0 else 0.0

    percentiles = calculate_percentiles(summary.latencies, [0.50, 0.90, 0.95, 0.99])
    avg_latency = sum(summary.latencies) / len(summary.latencies) if summary.latencies else 0.0
    min_latency = min(summary.latencies) if summary.latencies else 0.0
    max_latency = max(summary.latencies) if summary.latencies else 0.0

    avg_output_tokens_per_req = (
        summary.total_output_tokens / summary.successful_requests if summary.successful_requests > 0 else 0.0
    )
    tpot_ms = (avg_latency / avg_output_tokens_per_req * 1000.0) if avg_output_tokens_per_req > 0 else 0.0

    print("\n==================================================================")
    print("                      BENCHMARK RESULTS                           ")
    print("==================================================================")
    print(f" Benchmark Duration:      {summary.total_time_sec:.2f} s")
    print(f" Concurrency:             {summary.concurrency}")
    print(f" Total Requests Sent:     {summary.total_requests}")
    print(f" Successful Requests:     {summary.successful_requests} ({summary.successful_requests/summary.total_requests*100:.1f}%)")
    print(f" Failed Requests:         {summary.failed_requests}")
    print("------------------------------------------------------------------")
    print(" THROUGHPUT METRICS:")
    print(f"  • Request Throughput:    {req_per_sec:.2f} requests/sec")
    print(f"  • Output Token Rate:     {out_tokens_per_sec:.2f} tokens/sec")
    print(f"  • Total Token Rate:      {total_tokens_per_sec:.2f} tokens/sec")
    print(f"  • Total Output Generated: {summary.total_output_tokens} tokens")
    print("------------------------------------------------------------------")
    print(" LATENCY METRICS (End-to-End):")
    print(f"  • Mean Latency:          {avg_latency:.3f} s")
    print(f"  • Min Latency:           {min_latency:.3f} s")
    print(f"  • P50 (Median):          {percentiles.get('P50', 0):.3f} s")
    print(f"  • P90:                   {percentiles.get('P90', 0):.3f} s")
    print(f"  • P95:                   {percentiles.get('P95', 0):.3f} s")
    print(f"  • P99:                   {percentiles.get('P99', 0):.3f} s")
    print(f"  • Max Latency:           {max_latency:.3f} s")
    print(f"  • Avg Time/Token (TPOT): {tpot_ms:.2f} ms/token")
    print("==================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="vLLM Benchmarking Suite for Agent Platform Endpoints")
    parser.add_argument("--endpoint-id", help="Agent Platform numeric Endpoint ID")
    parser.add_argument("--project", help="Google Cloud Project ID")
    parser.add_argument("--region", default="us-central1", help="Google Cloud Region")
    parser.add_argument("--local-url", help="Local / direct server URL (e.g. http://localhost:8080)")
    parser.add_argument("--num-prompts", type=int, default=10, help="Total number of benchmark prompts to send")
    parser.add_argument("--concurrency", type=int, default=2, help="Number of concurrent requests")
    parser.add_argument("--prompt-tokens", type=int, default=128, help="Approximate prompt token length")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum generated tokens per request")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")

    args = parser.parse_args()

    headers = {"Content-Type": "application/json"}
    is_vertex_predict = False

    if args.endpoint_id and args.project:
        endpoint_url = (
            f"https://{args.region}-aiplatform.googleapis.com/v1/"
            f"projects/{args.project}/locations/{args.region}/endpoints/{args.endpoint_id}:predict"
        )
        token = get_gcloud_auth_token()
        if not token:
            print("Error: Authentication token required for Agent Platform endpoint.", file=sys.stderr)
            sys.exit(1)
        headers["Authorization"] = f"Bearer {token}"
        is_vertex_predict = True
    elif args.local_url:
        endpoint_url = args.local_url.rstrip("/")
        if not endpoint_url.endswith("/v1/chat/completions") and not endpoint_url.endswith("/predict"):
            endpoint_url = f"{endpoint_url}/predict"
        if endpoint_url.endswith("/predict"):
            is_vertex_predict = True
    else:
        # Fallback default
        endpoint_url = "http://localhost:8080/predict"
        is_vertex_predict = True

    summary = asyncio.run(
        run_benchmark(
            endpoint_url=endpoint_url,
            headers=headers,
            is_vertex_predict=is_vertex_predict,
            num_prompts=args.num_prompts,
            concurrency=args.concurrency,
            prompt_tokens=args.prompt_tokens,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    )

    display_results_table(summary)


if __name__ == "__main__":
    main()
