# Evaluating an LLM-Based CV–Job Fit Scoring System: A Psychometric Analysis

LLM systems are increasingly used to evaluate people — screening CVs, ranking candidates, and making recommendations.

But some of these systems are deployed without the measurement standards required to ensure their outputs are reliable, valid, or even meaningful. This project applies those standards to an LLM-based job fit analyser.

## What This Is

This repository presents a structured evaluation of an LLM-based CV–job matching system.

- Not a product
- Not a benchmark experiment  
- A controlled analysis of how such systems behave when used to evaluate human data

---

## Key Findings

The system produces stable scores but unreliable decisions — a critical failure mode for AI systems used in evaluation.

- **The system is a strong ranker, not a reliable classifier.** Ranking quality is high (NDCG@5 = 0.956, MRR = 1.0). Binary classification is limited: false positive rate reaches 50%, with zero false negatives. The asymmetry is structural, not correctable by threshold adjustment alone.

- **High consistency under controlled conditions.** Test-retest reliability is ICC = 0.994 (95% CI [0.988, 0.997]) across three independent runs at temperature = 0.4 — near-perfect stability for an LLM-based system.

- **Systematic mid-range scoring pattern.** All 8 false positive cases receive a score of exactly 65 — a pattern that persists across three prompt versions (P0/P1/P2) and three models (Gemini, GPT-4o, deepseek-r1). Confirmed at scale: 14.8% of n=200 scored job descriptions land in the 60–70 band. This indicates where the system concentrates uncertainty, rather than random error.

- **Prompt engineering does not resolve the core failure.** McNemar's test across prompt versions: p = 1.0. The same job descriptions fail regardless of whether the prompt includes a rubric, chain-of-thought instructions, or neither.

---

## Why This Matters

If an LLM system is used to evaluate people:

- fixed thresholds can produce incorrect decisions when scores concentrate at the boundary  
- high consistency can mask systematic bias  
- poor calibration makes scores hard to interpret  

Without proper evaluation, these systems introduce risk into hiring and decision-making processes.

---

## What I Do

I design and evaluate LLM-based systems that operate on human data.

This includes:
- auditing AI scoring and ranking systems  
- designing evaluation frameworks (validity, reliability, calibration)  
- identifying structural failure modes in model outputs

---
## Scope

System under evaluation: Career Pilot  
- Input: CV + Job Description  
- Output: Score (0–100) + verdict (Apply / Consider / Skip)  

The system is intentionally simplified to make its measurement properties observable.

---

## Evaluation Dimensions

The analysis covers two perspectives:

- Psychometric: validity, reliability, fairness, invariance  
- AI/ML: ranking quality, calibration, prompt sensitivity, cross-model consistency  

---

## Methodology

**Dataset:** 32 job descriptions, human-labeled blind (before any AI analysis). 16 positive (H ≥ 2), 16 negative (H ≤ 1). Labels assigned by a single domain expert on a 0–3 behavioural scale. An additional 168 unlabeled JDs (Jobicy + The Muse APIs) were used for distributional analysis.

**Evaluation tracks completed:**

| Track | Method | Status |
|---|---|---|
| A — Construct validity | Agreement, ranking, calibration, reliability, robustness, fairness | ✅ Complete |
| C — Multi-model | Gemini vs GPT-4o vs deepseek-r1:8b | ✅ Complete |
| D — Annotation audit | LLM-as-judge agreement with human labels (κ = 0.56–0.63) | ✅ Complete |
| B — Predictive validity | Outcome-based validation (interview / offer) | ⏳ Data accumulating |

**Statistical methods:** ICC(2,1) (Shrout & Fleiss, 1979), Cohen's κ with Wilson CIs, NDCG@k with bootstrap CIs, McNemar's test, paired t-tests, OLS regression, Cohen's d.

---

## Structure

```
├── career_pilot_evaluation.ipynb   ← full analysis with outputs
├── case-study/                     ← written evaluation report (Markdown)
├── framework/                      ← psychometric evaluation framework (publishing separately)
├── scripts/                        ← all experiment scripts
├── scripts-future/                 ← Track B and multi-model extensions (in progress)
├── data/
│   ├── labels.json                 ← 32 human-labeled job descriptions
│   └── batch-results/              ← pre-computed results for all experiments
├── cv/                             ← CV variants used in experiments
└── results/figures/                ← output plots
```

---

## Detailed Report & Framework

> 📄 **Case study (evaluation report):**  
[career_pilot_case_study.md](case-study/career_pilot_case_study.md)  
*Core analysis complete; additional sections are being refined*

> 📄 **Companion framework (in progress)**  
A general framework for evaluating LLM-based decision systems operating on human data is being developed and will be added to this repository.

---

## Get Involved

This project is ongoing. If you'd like to contribute or collaborate:

- **Use the app and share outcomes** — Track B (predictive validity) is accumulating data. If you use Career Pilot and track whether applications led to interviews or offers, that data is directly useful. [Try the app →](https://cv-matcher-azure.vercel.app)
- **SME review** — The current dataset uses a single annotator. A second independent rater for ICC calculation would strengthen the annotation study. Psychometrics or HR background helpful.
- **Apply this methodology to your own tool or product** — if you're building or using an LLM-based system that evaluates people and want to understand its measurement properties, get in touch.

Reach out: [olga.maslenkova@gmail.com](mailto:olga.maslenkova@gmail.com)

---

## Use

Free to use for personal development, job searching, or research. If you find this useful, [buy me a coffee ☕](https://ko-fi.com/V7V11WRGZX).

---

## Related Work

The system being evaluated is described at [github.com/Holly-olly/career-pilot](https://github.com/Holly-olly/career-pilot).
This evaluation is independent and was designed after the system was built, using a dataset constructed blind to AI scores.
