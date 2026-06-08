"""
GPQA Diamond eval on openai.gpt-5.4 (Bedrock Mantle).
Methodology follows Artificial Analysis:
  - 5 repeats, pass@1 aggregated across all repeats
  - temperature=0.6 (reasoning model)
  - Zero-shot instruction prompted, no examples
  - 4-option multiple choice, answer options randomized per question
  - Regex extraction: looks for "Answer: X" on last line, fallbacks
  - Published GPT-5.4 score: 92.8%
"""

import os
import sys
import re
import json
import random
import time
import argparse
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from openai import OpenAI
from eval_utils import capture_error
from datasets import load_dataset

N_REPEATS   = 5
TEMPERATURE = 0.6
RESULTS_DIR = os.path.dirname(__file__)


def make_client(backend):
    if backend == "mantle":
        if os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
            token = os.environ["AWS_BEARER_TOKEN_BEDROCK"]
        else:
            from aws_bedrock_token_generator import provide_token
            region = os.environ.get("AWS_REGION", "us-west-2")
            token = provide_token(region=region)
        base_url = os.environ.get("MANTLE_BASE_URL", "https://bedrock-mantle.us-west-2.api.aws/openai/v1")
        model = os.environ.get("MANTLE_MODEL", "openai.gpt-5.4")
        return OpenAI(api_key=token, base_url=base_url), model
    else:
        return OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY_SAAS", os.environ.get("OPENAI_API_KEY", "")),
            base_url="https://api.openai.com/v1"
        ), os.environ.get("SAAS_MODEL", "gpt-5.4")

SYSTEM_PROMPT = (
    "You are an expert scientist. Answer the following multiple choice question. "
    "Think through the problem carefully, then on the very last line of your response, "
    "write your answer in the format: Answer: A (or B, C, or D)."
)



def shuffle_choices(question_data):
    """Randomize answer order, return (choices_dict, correct_letter)."""
    correct = question_data["Correct Answer"]
    incorrect = [
        question_data["Incorrect Answer 1"],
        question_data["Incorrect Answer 2"],
        question_data["Incorrect Answer 3"],
    ]
    all_answers = [correct] + incorrect
    random.shuffle(all_answers)
    letters = ["A", "B", "C", "D"]
    choices = {letters[i]: all_answers[i] for i in range(4)}
    correct_letter = next(l for l, v in choices.items() if v == correct)
    return choices, correct_letter


def format_question(question_text, choices):
    lines = [question_text.strip(), ""]
    for letter, text in choices.items():
        lines.append(f"{letter}) {text.strip()}")
    return "\n".join(lines)


def extract_answer(response_text):
    """Multi-stage regex extraction, takes last match."""
    text = response_text.strip()

    # Primary: "Answer: X" or "Answer: (X)"
    matches = re.findall(r"[Aa]nswer:\s*\(?([A-D])\)?", text)
    if matches:
        return matches[-1].upper()

    # Fallback: \boxed{A}
    matches = re.findall(r"\\boxed\{([A-D])\}", text)
    if matches:
        return matches[-1].upper()

    # Fallback: "the answer is X"
    matches = re.findall(r"(?:the\s+)?answer\s+is\s+\(?([A-D])\)?", text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    # Fallback: standalone letter at end
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        m = re.match(r"^([A-D])[.\s)]?$", lines[-1])
        if m:
            return m.group(1).upper()

    return None


def run_question(client, model, question_data, repeat_idx):
    choices, correct_letter = shuffle_choices(question_data)
    prompt = format_question(question_data["Question"], choices)

    for attempt in range(5):
        try:
            kwargs = dict(
                model=model,
                instructions=SYSTEM_PROMPT,
                input=[{"role": "user", "content": prompt}],
                max_output_tokens=2048,
            )
            # gpt-5.5 reasoning models don't support temperature
            if "5.5" not in model:
                kwargs["temperature"] = TEMPERATURE
            r = client.responses.create(**kwargs)
            response_text = r.output_text
            break
        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err:
                wait = 2 ** attempt * 5
                time.sleep(wait)
                continue
            return {
                "correct": False, "predicted": None,
                "correct_letter": correct_letter, "choices": choices,
                "response_preview": None, "error": capture_error(e),
            }
    else:
        return {
            "correct": False, "predicted": None,
            "correct_letter": correct_letter, "choices": choices,
            "response_preview": None, "error": {"error_message": "max retries exceeded"},
        }
    try:
        predicted = extract_answer(response_text)
        correct = (predicted == correct_letter) if predicted else False
        return {
            "correct": correct,
            "predicted": predicted,
            "correct_letter": correct_letter,
            "choices": choices,
            "response_preview": response_text[-200:],
            "error": None,
        }
    except Exception as e:
        return {
            "correct": False,
            "predicted": None,
            "correct_letter": correct_letter,
            "choices": choices,
            "response_preview": None,
            "error": capture_error(e),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["mantle", "saas"], default="mantle")
    args = parser.parse_args()

    random.seed(42)
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    questions = list(dataset)
    n_questions = len(questions)

    client, model = make_client(args.backend)

    started_at = datetime.now(timezone.utc)
    print(f"\nGPQA Diamond Eval — {model} ({args.backend})")
    print(f"Questions: {n_questions}  |  Repeats: {N_REPEATS}  |  Total calls: {n_questions * N_REPEATS}")
    print(f"Temperature: {TEMPERATURE}  |  Started: {started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Published GPT-5.4 score: 92.8%")
    print("=" * 65)

    all_results = []
    total_calls = n_questions * N_REPEATS
    call_num = 0
    correct_per_repeat = [0] * N_REPEATS

    for qi, q in enumerate(questions):
        q_results = []
        for ri in range(N_REPEATS):
            call_num += 1
            result = run_question(client, model, q, ri)
            result["question_idx"] = qi
            result["repeat"] = ri
            result["subdomain"] = q.get("Subdomain", "")
            result["domain"] = q.get("High-level domain", "")
            q_results.append(result)
            if result["correct"]:
                correct_per_repeat[ri] += 1

            status = "✅" if result["correct"] else ("⚠️" if result["predicted"] is None else "❌")
            print(f"  [{call_num:>4}/{total_calls}] Q{qi+1:>3} R{ri+1} {status} "
                  f"pred={result['predicted'] or '?'} correct={result['correct_letter']} "
                  f"[{result['domain'][:20]}]", flush=True)

            # Refresh mantle token every 50 calls
            if args.backend == "mantle" and call_num % 50 == 0:
                client, model = make_client(args.backend)

        all_results.extend(q_results)

    ended_at = datetime.now(timezone.utc)

    # Score: pass@1 per question (correct in any repeat), then average
    per_question_pass = []
    for qi in range(n_questions):
        q_reps = [r for r in all_results if r["question_idx"] == qi]
        # pass@1 = fraction of repeats correct
        frac = sum(r["correct"] for r in q_reps) / N_REPEATS
        per_question_pass.append(frac)

    # Overall accuracy = mean of per-question pass@1
    accuracy = sum(per_question_pass) / n_questions * 100

    # Per-repeat accuracy
    rep_accs = [correct_per_repeat[ri] / n_questions * 100 for ri in range(N_REPEATS)]

    # Per-domain breakdown
    domains = {}
    for r in all_results:
        d = r["domain"]
        if d not in domains:
            domains[d] = {"correct": 0, "total": 0}
        domains[d]["total"] += 1
        domains[d]["correct"] += r["correct"]

    print("\n" + "=" * 65)
    print(f"RESULTS — GPQA Diamond | {model} ({args.backend})")
    print(f"Overall accuracy (mean pass@1): {accuracy:.1f}%")
    print(f"Published GPT-5.4 score:        92.8%")
    print(f"\nPer-repeat accuracy:")
    for ri, acc in enumerate(rep_accs):
        print(f"  Repeat {ri+1}: {acc:.1f}%")
    print(f"\nPer-domain breakdown:")
    for d, v in sorted(domains.items()):
        domain_acc = v["correct"] / v["total"] * 100
        print(f"  {d:<20} {domain_acc:.1f}%  ({v['correct']}/{v['total']})")
    print(f"\nEnded:    {ended_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    duration = (ended_at - started_at).total_seconds()
    print(f"Duration: {duration/60:.1f} min")

    ts = started_at.strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(RESULTS_DIR, f"gpqa_diamond_{args.backend}_{ts}.json")
    payload = {
        "eval": "gpqa_diamond",
        "model": model,
        "backend": args.backend,
        "n_questions": n_questions,
        "n_repeats": N_REPEATS,
        "temperature": TEMPERATURE,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "duration_seconds": round(duration, 1),
        "accuracy_pct": round(accuracy, 2),
        "per_repeat_accuracy_pct": [round(a, 2) for a in rep_accs],
        "published_gpt54_score_pct": 92.8,
        "domain_breakdown": {
            d: {"accuracy_pct": round(v["correct"]/v["total"]*100, 1), "correct": v["correct"], "total": v["total"]}
            for d, v in domains.items()
        },
        "results": all_results,
    }
    with open(fname, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved:    {os.path.basename(fname)}")


if __name__ == "__main__":
    main()
