# Data

This folder contains all datasets used in the evaluation.

## `labels.json` (root level)

The primary evaluation dataset: 32 human-labeled job descriptions (Track A).

| Field | Description |
|---|---|
| `jd_id` | Unique identifier |
| `human_relevance` | Human label: 0 (no fit) to 3 (strong fit) |
| `ai_score` | Career Pilot score (0–100) |
| `ai_verdict` | Apply / Consider / Skip |
| `masked` | Whether company name was masked |

32 labeled JDs: 16 positive (H ≥ 2), 16 negative (H ≤ 1).
Labels were assigned blind — before any AI analysis was run.

## `batch-results/`

Pre-computed results from all evaluation experiments. Used directly by the notebook.

| File | Contents |
|---|---|
| `prompt_p0.json` | Scores under minimal prompt (P0) |
| `prompt_p1.json` | Scores under rubric prompt P1 — deployed version |
| `prompt_p2.json` | Scores under chain-of-thought prompt (P2) |
| `prompt_comparison_v1_v2.json` | P0 vs P1 comparison |
| `prompt_comparison_v1_v2_v3.json` | P0 vs P1 vs P2 comparison |
| `consistency_icc.json` | Test-retest reliability (3 runs, n=24 JDs) |
| `robustness_cv_paraphrase.json` | Paraphrased CV scores (n=23) |
| `synthetic_cv_scores.json` | UX Researcher synthetic CV scores (n=24) |
| `company_name_bias.json` | Original vs masked JD scores (n=32) |
| `multi_model_comparison.json` | Gemini vs GPT-4o vs deepseek-r1 |
| `llm_annotation_audit.json` | LLM-as-judge annotation audit (Track D) |
| `structured_v1_baseline.json` | Structured matching layer scores |
| `structured_llm_sensitivity.json` | Structured + LLM combined |
| `track_b_prepared.json` | Track B dataset (historical applications) |
| `track_b_scores.json` | Track B AI scores |
| `llm_judge_scores.json` | Track E explanation quality scores (removed from scope) |

## `jds/`

Two additional job descriptions collected separately (not part of the 32 labeled set).
Kept for reference.

## `batch-jobs/`

Raw JD text files (140+ public job postings) collected via Jobicy and The Muse APIs.
Not included in this repository — used only to generate the `batch-results/` files.
To re-collect: run `scripts/fetch_real_jds_free.py`.
