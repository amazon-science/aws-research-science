"""
Re-score HLE results using Claude as an LLM judge (via Bedrock).
Exact string match is too strict for HLE — answers can be mathematically
equivalent but expressed differently (LaTeX, decimals, fractions, etc.).
Judge: anthropic.claude-haiku-4-5 on Bedrock (fast, cheap)
"""

import os, sys, json, time, boto3
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

RESULTS_DIR = os.path.dirname(__file__)
JUDGE_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
REGION = "us-west-2"

JUDGE_PROMPT = """You are a strict answer grader.

Given a question, the correct answer, and a predicted answer, determine if the predicted answer is correct.

Be generous with formatting differences:
- Mathematical expressions that are equivalent should be marked correct (e.g. "1/2" == "0.5" == "$\\frac{1}{2}$")
- Case differences should be ignored
- Minor wording differences that preserve meaning should be marked correct
- Extra explanation around a correct answer should be marked correct if the core answer is right

Respond with ONLY one word: "correct" or "incorrect".

Question: {question}
Correct answer: {correct}
Predicted answer: {predicted}

Is the predicted answer correct?"""


def judge_answer(bedrock, question, correct, predicted):
    if predicted is None:
        return False, "no_prediction"

    prompt = (JUDGE_PROMPT
              .replace("{question}", question[:500])
              .replace("{correct}", correct[:200])
              .replace("{predicted}", str(predicted)[:200]))

    for attempt in range(3):
        try:
            response = bedrock.invoke_model(
                modelId=JUDGE_MODEL,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": prompt}]
                })
            )
            result = json.loads(response['body'].read())
            verdict = result['content'][0]['text'].strip().lower()
            is_correct = verdict.startswith("correct")
            return is_correct, verdict
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return False, f"error: {str(e)[:100]}"


def main():
    # Find latest HLE mantle results file
    import glob
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "hle_mantle_*.json")))
    if not files:
        print("No HLE mantle results found")
        return

    fname = files[-1]
    print(f"Loading: {fname}")
    data = json.load(open(fname))
    results = data["results"]

    bedrock = boto3.client("bedrock-runtime", region_name=REGION)

    started_at = datetime.now(timezone.utc)
    print(f"\nRe-scoring {len(results)} HLE answers with {JUDGE_MODEL}")
    print(f"Started: {started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 65)

    # Load original questions for context
    from datasets import load_dataset
    ds = load_dataset("cais/hle", split="test")
    text_only = [q for q in ds if not q.get("image")]
    q_map = {q["id"]: q["question"] for q in text_only}

    correct_count = 0
    exact_count = 0
    errors = 0
    rescored = []

    for i, r in enumerate(results):
        correct_answer = r["correct_answer"]
        predicted = r["predicted"]
        question_id = r.get("question_id", "")
        question_text = q_map.get(question_id, "")

        # Track original exact match
        exact = (predicted or "").strip().lower() == (correct_answer or "").strip().lower()
        if exact:
            exact_count += 1

        # LLM judge
        if predicted is None or r.get("error"):
            is_correct = False
            verdict = "skipped"
            errors += 1
        else:
            is_correct, verdict = judge_answer(bedrock, question_text, correct_answer, predicted)

        if is_correct:
            correct_count += 1

        rescored.append({**r, "llm_correct": is_correct, "llm_verdict": verdict, "exact_match": exact})

        if (i + 1) % 50 == 0 or (i + 1) == len(results):
            acc = correct_count / (i + 1) * 100
            print(f"  [{i+1:>4}/{len(results)}] running accuracy: {acc:.1f}%  (exact so far: {exact_count/(i+1)*100:.1f}%)", flush=True)

    ended_at = datetime.now(timezone.utc)
    duration = (ended_at - started_at).total_seconds()

    llm_accuracy = correct_count / len(results) * 100
    exact_accuracy = exact_count / len(results) * 100
    answerable = len(results) - errors
    llm_answerable = correct_count / answerable * 100 if answerable > 0 else 0

    print("\n" + "=" * 65)
    print(f"RESCORED HLE — {data['model']} (mantle)")
    print(f"Exact match accuracy:          {exact_accuracy:.1f}%  ({exact_count}/{len(results)})")
    print(f"LLM judge accuracy (all):      {llm_accuracy:.1f}%  ({correct_count}/{len(results)})")
    print(f"LLM judge accuracy (answerable): {llm_answerable:.1f}%  ({correct_count}/{answerable})")
    print(f"Unanswered/errors:             {errors}/{len(results)}")
    print(f"Published GPT-5.4 (no tools):  39.8%")
    print(f"Duration: {duration/60:.1f} min")

    out_fname = fname.replace(".json", "_rescored.json")
    json.dump({
        **data,
        "rescoring": {
            "judge_model": JUDGE_MODEL,
            "exact_accuracy_pct": round(exact_accuracy, 2),
            "llm_accuracy_pct": round(llm_accuracy, 2),
            "llm_answerable_accuracy_pct": round(llm_answerable, 2),
            "correct_count": correct_count,
            "exact_count": exact_count,
            "errors": errors,
            "rescored_at": started_at.isoformat(),
        },
        "results": rescored,
    }, open(out_fname, "w"), indent=2)
    print(f"Saved: {os.path.basename(out_fname)}")


if __name__ == "__main__":
    main()
