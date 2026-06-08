"""
Benchmark: gpt-5.x on OpenAI SaaS API (direct).
Full matrix: 1K/5K/10K/20K input × multiple output configs, 25 runs each.
Model set via MODEL env var (default: gpt-5.5).
"""

import os, sys, time, json, statistics
from datetime import datetime, timezone
from openai import OpenAI

BASE_URL  = "https://api.openai.com/v1"
MODEL     = os.environ.get("SAAS_MODEL", "gpt-5.5")
DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.dirname(__file__)
N_RUNS    = 25

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
    return open(os.path.join(DATA_DIR, PROMPTS[label]["file"])).read() + INSTRUCTION


def percentile(s, p):
    if not s: return None
    idx = (p / 100) * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


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


def run_single(client, prompt, max_output_tokens, max_retries=5):
    for attempt in range(max_retries):
        try:
            first_token_time = None
            output_tokens = 0
            input_tokens = 0
            full_text = []

            start = time.perf_counter()
            stream = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_output_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )
            for chunk in stream:
                now = time.perf_counter()
                if chunk.usage:
                    output_tokens = chunk.usage.completion_tokens
                    input_tokens  = chunk.usage.prompt_tokens
                if not chunk.choices: continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    if first_token_time is None:
                        first_token_time = now
                    full_text.append(delta.content)
            end = time.perf_counter()

            e2e = end - start
            ttft = (first_token_time - start) if first_token_time else None
            gen_time = (end - first_token_time) if first_token_time else e2e
            otps = output_tokens / gen_time if gen_time > 0 and output_tokens > 0 else 0

            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "ttft_ms": round(ttft * 1000, 1) if ttft else None,
                "otps": round(otps, 1),
                "e2e_ms": round(e2e * 1000, 1),
                "error": None,
            }
        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err:
                wait = 2 ** attempt * 5
                print(f" [rate limit, wait {wait}s]", end="", flush=True)
                time.sleep(wait)
            else:
                return {"input_tokens":0,"output_tokens":0,"ttft_ms":None,
                        "otps":0,"e2e_ms":0,"error":err[:200]}
    return {"input_tokens":0,"output_tokens":0,"ttft_ms":None,
            "otps":0,"e2e_ms":0,"error":"max retries exceeded"}


def run_benchmark(input_label):
    prompt = load_prompt(input_label)
    output_configs = OUTPUT_CONFIGS[input_label]
    nominal_input_tokens = PROMPTS[input_label]["tokens"]

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=BASE_URL)
    started_at = datetime.now(timezone.utc)
    total_calls = len(output_configs) * N_RUNS

    print(f"\nBenchmark: {input_label} input (~{nominal_input_tokens} tokens) | {MODEL} | OAI SaaS")
    print(f"Runs: {N_RUNS} per config ({total_calls} total) | Started: {started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 68)
    print(f"{'Max Out':>8}  {'In Tok':>7}  {'Out Tok':>7}  {'TTFT(ms)':>9}  {'Tok/s':>6}  {'E2E(ms)':>9}")
    print("-" * 68)

    raw = {max_out: [] for max_out in output_configs}
    call_num = 0

    for max_out in output_configs:
        for run_idx in range(N_RUNS):
            call_num += 1
            print(f"  [{call_num:>3}/{total_calls}] max_out={max_out:>5} run={run_idx+1:>2}/{N_RUNS} ... ", end="", flush=True)
            r = run_single(client, prompt, max_out)
            raw[max_out].append(r)
            print(f"ttft={r['ttft_ms']}ms  otps={r['otps']}  e2e={r['e2e_ms']}ms")

    ended_at = datetime.now(timezone.utc)
    summary = []
    for max_out in output_configs:
        runs = raw[max_out]
        ttft_vals = [r["ttft_ms"] for r in runs if r["ttft_ms"]]
        otps_vals = [r["otps"] for r in runs if r["otps"] > 0]
        e2e_vals  = [r["e2e_ms"] for r in runs]
        ttft_s = summarize(ttft_vals)
        otps_s = summarize(otps_vals)
        e2e_s  = summarize(e2e_vals)
        print(f"{max_out:>8}  {'~'+str(nominal_input_tokens):>7}  {runs[0]['output_tokens']:>7}  "
              f"{ttft_s['p50']:>9.1f}  {otps_s['p50']:>6.1f}  {e2e_s['p50']:>9.1f}")
        summary.append({"max_output_tokens": max_out, "n_runs": N_RUNS,
                        "ttft_ms": ttft_s, "otps": otps_s, "e2e_ms": e2e_s})

    duration = (ended_at - started_at).total_seconds()
    print(f"\nEnded: {ended_at.strftime('%Y-%m-%d %H:%M:%S UTC')} | Duration: {duration/60:.1f} min")

    ts = started_at.strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(RESULTS_DIR, f"results_saas_{MODEL}_{input_label}input_{N_RUNS}runs_{ts}.json")
    json.dump({
        "model": MODEL, "backend": "saas", "base_url": BASE_URL,
        "input_label": input_label, "nominal_input_tokens": nominal_input_tokens,
        "n_runs": N_RUNS, "started_at": started_at.isoformat(), "ended_at": ended_at.isoformat(),
        "duration_seconds": round(duration, 1),
        "summary": summary, "raw": {str(k): v for k, v in raw.items()},
    }, open(fname, "w"), indent=2)
    print(f"Saved: {os.path.basename(fname)}")


if __name__ == "__main__":
    labels = sys.argv[1:] if len(sys.argv) > 1 else ["1k", "5k", "10k", "20k"]
    for label in labels:
        run_benchmark(label)
