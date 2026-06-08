"""
Benchmark: openai.gpt-5.4 on Bedrock Mantle (Responses API)
Canonical prompts: 1K, 10K, 20K input tokens
Output configs: varies per input size
Runs N times per config, reports p50/p95/p99/mean/stddev.
"""

import os
import sys
import time
import json
import statistics
from datetime import datetime, timezone
from aws_bedrock_token_generator import provide_token as _provide_token
from openai import OpenAI

def provide_token():
    region = os.environ.get("AWS_REGION", "us-west-2")
    return _provide_token(region=region)

BASE_URL = os.environ.get("MANTLE_BASE_URL", "https://bedrock-mantle.us-west-2.api.aws/openai/v1")
MODEL    = os.environ.get("MANTLE_MODEL",    "openai.gpt-5.4")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.dirname(__file__)

PROMPTS = {
    "1k":  {"file": "prompt_1k.txt",  "tokens": 1002},
    "5k":  {"file": "prompt_5k.txt",  "tokens": 4750},
    "10k": {"file": "prompt_10k.txt", "tokens": 9984},
    "20k": {"file": "prompt_20k.txt", "tokens": 20136},
}

OUTPUT_CONFIGS = {
    "1k":  [100, 500, 1000],
    "5k":  [500, 1000, 5000],
    "10k": [500, 1000, 5000],
    "20k": [1000, 5000, 10000],
}

INSTRUCTION = "\n\nContinue writing about this topic in detail."


def load_prompt(label):
    path = os.path.join(DATA_DIR, PROMPTS[label]["file"])
    return open(path).read() + INSTRUCTION


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    idx = (p / 100) * (len(sorted_vals) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def summarize(values):
    s = sorted(values)
    return {
        "mean":   round(statistics.mean(s), 1),
        "stddev": round(statistics.stdev(s), 1) if len(s) > 1 else 0.0,
        "p50":    round(percentile(s, 50), 1),
        "p95":    round(percentile(s, 95), 1),
        "p99":    round(percentile(s, 99), 1),
        "min":    round(min(s), 1),
        "max":    round(max(s), 1),
    }


def run_single(client, prompt, max_output_tokens, max_retries=3):
    for attempt in range(max_retries):
        try:
            return _run_single_attempt(client, prompt, max_output_tokens)
        except Exception as e:
            if attempt < max_retries - 1 and any(x in str(e) for x in ["Connection reset", "ReadError", "RemoteProtocolError"]):
                time.sleep(2 ** attempt * 2)
                client._custom_auth = None  # force reconnect
                continue
            raise
    raise RuntimeError("max retries exceeded")


def _run_single_attempt(client, prompt, max_output_tokens):
    first_token_time = None
    output_tokens = 0
    input_tokens = 0
    full_text = []

    start = time.perf_counter()

    stream = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": prompt}],
        max_output_tokens=max_output_tokens,
        stream=True,
    )

    for event in stream:
        now = time.perf_counter()
        event_type = getattr(event, "type", None)

        if event_type == "response.output_text.delta":
            text = getattr(event, "delta", None)
            if text:
                if first_token_time is None:
                    first_token_time = now
                full_text.append(text)

        elif event_type in ("response.completed", "response.incomplete"):
            resp = getattr(event, "response", None)
            usage = getattr(resp, "usage", None) if resp else None
            if usage:
                output_tokens = getattr(usage, "output_tokens", output_tokens)
                input_tokens = getattr(usage, "input_tokens", input_tokens)

    end = time.perf_counter()

    e2e = end - start
    ttft = (first_token_time - start) if first_token_time else None
    gen_time = (end - first_token_time) if first_token_time else e2e
    otps = output_tokens / gen_time if gen_time > 0 and output_tokens > 0 else 0

    return {
        "max_output_tokens": max_output_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "ttft_ms": round(ttft * 1000, 1) if ttft else None,
        "otps": round(otps, 1),
        "e2e_ms": round(e2e * 1000, 1),
    }


def run_benchmark(input_label, n_runs=25):
    prompt = load_prompt(input_label)
    output_configs = OUTPUT_CONFIGS[input_label]
    nominal_input_tokens = PROMPTS[input_label]["tokens"]
    total_calls = len(output_configs) * n_runs

    token = provide_token()
    client = OpenAI(api_key=token, base_url=BASE_URL)

    started_at = datetime.now(timezone.utc)
    print(f"\nBenchmark: {input_label} input (~{nominal_input_tokens} tokens) | {MODEL}")
    print(f"Runs:      {n_runs} per config ({total_calls} total calls)")
    print(f"Started:   {started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # raw[max_out] = list of result dicts
    raw = {max_out: [] for max_out in output_configs}
    call_num = 0

    for max_out in output_configs:
        for run_idx in range(n_runs):
            call_num += 1
            print(f"  [{call_num:>3}/{total_calls}] input={input_label} max_out={max_out:>5} run={run_idx+1:>2}/{n_runs} ... ", end="", flush=True)
            r = run_single(client, prompt, max_out)
            raw[max_out].append(r)
            print(f"ttft={r['ttft_ms']}ms  otps={r['otps']}  e2e={r['e2e_ms']}ms")

    ended_at = datetime.now(timezone.utc)

    # Build summary stats per config
    summary = []
    print(f"\n{'='*80}")
    print(f"SUMMARY: {input_label} input | {n_runs} runs per config")
    print(f"{'='*80}")
    print(f"{'Max Out':>8}  {'Metric':>10}  {'Mean':>8}  {'Stddev':>8}  {'p50':>8}  {'p95':>8}  {'p99':>8}  {'Min':>8}  {'Max':>8}")
    print(f"{'-'*80}")

    for max_out in output_configs:
        runs = raw[max_out]
        ttft_vals = [r["ttft_ms"] for r in runs if r["ttft_ms"] is not None]
        otps_vals = [r["otps"] for r in runs if r["otps"] > 0]
        e2e_vals  = [r["e2e_ms"] for r in runs]

        ttft_stats = summarize(ttft_vals)
        otps_stats = summarize(otps_vals)
        e2e_stats  = summarize(e2e_vals)

        for metric, stats in [("TTFT(ms)", ttft_stats), ("Tok/s", otps_stats), ("E2E(ms)", e2e_stats)]:
            print(f"{max_out:>8}  {metric:>10}  {stats['mean']:>8}  {stats['stddev']:>8}  "
                  f"{stats['p50']:>8}  {stats['p95']:>8}  {stats['p99']:>8}  "
                  f"{stats['min']:>8}  {stats['max']:>8}")

        summary.append({
            "max_output_tokens": max_out,
            "n_runs": n_runs,
            "ttft_ms": ttft_stats,
            "otps": otps_stats,
            "e2e_ms": e2e_stats,
        })

    print(f"{'='*80}")
    print(f"Ended:     {ended_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    duration = (ended_at - started_at).total_seconds()
    print(f"Duration:  {duration/60:.1f} min")

    ts = started_at.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(RESULTS_DIR, f"results_{input_label}input_{n_runs}runs_{ts}.json")
    payload = {
        "model": MODEL,
        "base_url": BASE_URL,
        "input_label": input_label,
        "nominal_input_tokens": nominal_input_tokens,
        "n_runs": n_runs,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "duration_seconds": round(duration, 1),
        "summary": summary,
        "raw": {str(k): v for k, v in raw.items()},
    }
    with open(filename, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved:     {os.path.basename(filename)}")

    return payload


if __name__ == "__main__":
    args = sys.argv[1:]
    n_runs = 25
    labels = []

    for a in args:
        if a.startswith("--runs="):
            n_runs = int(a.split("=")[1])
        elif a in PROMPTS:
            labels.append(a)

    if not labels:
        labels = ["1k", "10k", "20k"]

    for label in labels:
        run_benchmark(label, n_runs=n_runs)
