# Quality Benchmarks — openai.gpt-5.4 on Bedrock Mantle

**Model:** openai.gpt-5.4  
**Endpoint:** https://bedrock-mantle.us-west-2.api.aws/openai/v1  
**Auth:** IAM credentials via aws_bedrock_token_generator  
**Date:** 2026-05-28  

---

## Methodology

Following Artificial Analysis methodology where applicable:
- Temperature: 0.6 (reasoning model)
- Zero-shot instruction prompted, no examples
- Pass@1 aggregated across all repeats
- Answer extraction: multi-stage regex, last match wins
- Failed/blocked calls recorded separately, not counted as correct

---

## 1. GPQA Diamond

### Setup
- **Dataset:** `Idavidrein/gpqa` (gpqa_diamond split), HuggingFace
- **Questions:** 198
- **Repeats:** 5 (990 total API calls)
- **Scoring:** Mean pass@1 across repeats
- **Answer format:** 4-option MCQ, options randomized per question per repeat
- **Random seed:** 42

### Results — Mantle vs SaaS vs Published

| Metric | Mantle | OAI SaaS | Published (AA) |
|--------|--------|----------|----------------|
| **Overall accuracy** | **72.5%** | **76.8%** | **92.8%** |
| Physics | 88.1% | 90.9% | — |
| Biology | 69.5% | 72.6% | — |
| Chemistry | 58.7% | 64.5% | — |
| Duration | 125.5 min | ~90 min | — |

**Mantle per-repeat accuracy:** 72.2% / 69.2% / 75.8% / 73.7% / 71.7% (range: 69–76%)

### Observations

1. **Mantle 4.3 points below SaaS (72.5% vs 76.8%)** — small but consistent gap across all domains.
2. **Both 16–20 points below published AA score (92.8%).** AA tests from GCP us-central1-a with 8 runs/day rolling p50 — geographic and methodology differences likely account for some gap.
3. **Physics is strong on both** (88–91%) — chemistry is the weak domain (58–64%).
4. **Per-repeat variance is low** — score is stable, not a fluke.
5. **No content filter issues** during GPQA runs.

---

## 2. AIME 2024

### Setup
- **Dataset:** `qq8933/AIME_1983_2024`, Year=2024 (AIME II only — AIME I absent from dataset)
- **Questions:** 14 of 30
- **Repeats:** 5 (70 total API calls)
- **Scoring:** Exact integer match
- **OAI SaaS run:** Failed — personal API key quota exhausted (429 on all calls)

### Results — Mantle

| Metric | Value |
|--------|-------|
| **Raw accuracy** | **71.4%** (50/70 calls) |
| **Answerable accuracy** (ex-filter) | **83.3%** (50/60) |
| Content filter blocks | 10/70 (14.3%) |
| Genuine misses | Q5 (0/5), Q14 (0/5) |

**Per-question:** Q1,3,4,6,8,11,12,13 perfect (5/5) | Q2,9,10,12 partial filter | Q7 blocked 5/5 | Q5,Q14 genuine misses

### Content Filter False Positives — Critical Issue

10/70 calls blocked with `"Your request was flagged as potentially violating our usage policy"`. All blocked problems are standard competition math with no harmful content.

**Q7 (blocked 5/5 — deterministic on Mantle, passes on OAI SaaS):**
> "Let N be the greatest four-digit integer with the property that whenever one of its digits is changed to 1, the resulting number is divisible by 7..."

**Confirmed:** OAI SaaS handles Q7 and Q12 without issue. This is a Mantle-specific false positive.  
**Timestamp:** 2026-05-28 02:19–02:36 UTC, account <redacted>, us-west-2.  
**Note:** Could not reproduce in later testing — filter may have been fixed.

---

## 3. HLE (Humanity's Last Exam — text-only)

### Setup
- **Dataset:** `cais/hle`, test split, text-only questions (images excluded)
- **Questions:** 2,158 of 2,500 (342 image questions excluded)
- **Repeats:** 1
- **Scoring:** Exact string match (primary) + LLM judge re-score (Claude Haiku 4.5 on Bedrock)
- **OAI SaaS run:** Not completed — personal API key quota exhausted

### Results — Mantle

| Metric | Score |
|--------|-------|
| **Exact match accuracy** | **5.0%** (107/2158) |
| **LLM judge accuracy (all)** | **10.2%** (220/2158) |
| LLM judge (answerable only) | 10.6% (220/2081) |
| Unanswered / errors | 77/2158 (3.6%) |
| Published GPT-5.4 (no tools) | 39.8% |
| Gap vs published | -29.6 pts (LLM-judged) |

### Observations

1. **Exact match severely underestimates accuracy** — LLM judge doubled score from 5% to 10.2% by catching mathematically equivalent answers in different notation (LaTeX, fractions, decimals).

2. **Still 30 points below published (10.2% vs 39.8%).** Possible causes:
   - AA uses GPT-4o as judge — may be more lenient than Haiku
   - AA's published score is on full 2,500 questions (includes image questions with their own difficulty distribution); our text-only subset may be harder
   - Genuine performance gap on Mantle vs direct SaaS

3. **OAI SaaS comparison needed** — running the same test against SaaS direct is the only way to isolate Mantle-specific effects. Pending API quota refresh.

4. **High error/unanswered rate (3.6%)** — some questions returned None predictions (model didn't follow Answer: format or response was truncated).

---

## 4. Summary Table

| Eval | Mantle | OAI SaaS | Published (AA) | Notes |
|------|--------|----------|----------------|-------|
| GPQA Diamond | 72.5% | 76.8% | 92.8% | 5 repeats, 198 Qs |
| AIME 2024 | 71.4% raw / 83.3% ex-filter | ❌ quota | N/A | 14 Qs, content filter issue |
| HLE (text-only) | 10.2% (LLM judge) | ❌ quota | 39.8% | 2158 Qs, 1 repeat |

---

## 5. Open Questions / Follow-up

1. **GPQA/HLE on SaaS** — run when API quota refreshes for direct Mantle vs SaaS comparison.
2. **Content filter** — Q7 blocked deterministically during 02:19–02:36 UTC window. Could not reproduce later. Reported to Mantle team with timestamp and account ID.
3. **HLE scoring** — consider using Claude Opus or Sonnet as judge instead of Haiku for better accuracy on complex mathematical equivalences.
4. **AIME I 2024 missing** — need full 30-problem dataset.
5. **HLE image subset** — 342 image questions excluded; vision not functional on Mantle.
