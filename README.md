# Evaluating an LLM-Based CV–Job Fit Scoring System: A Psychometric Analysis

**How do you know if an AI scoring system is actually measuring what it claims to measure?**

LLMs are increasingly used to evaluate people — screening CVs, ranking candidates, scoring applications. Yet most deployments lack the measurement standards that psychometricians and assessment scientists apply to any scoring system: validity evidence, reliability estimates, bias checks, and calibration analysis. This project applies that methodology to an LLM-based job fit analyser.

---

## What This Analysis Covers

The system under evaluation — Career Pilot — scores the fit between a candidate's CV and a job description (0–100) and assigns a verdict (Apply / Consider / Skip). It is intentionally simple: one model, one candidate profile, one domain. That simplicity is the point. It makes the measurement properties visible and analysable in ways that production systems — with their black-box pipelines and proprietary data — do not permit.

The evaluation is structured around ten research questions spanning two analytical lenses:

| Lens | Questions examined |
|---|---|
| **Psychometric** | Criterion validity, reliability (ICC), measurement invariance, fairness (DIF) |
| **AI / ML evaluation** | Ranking quality (NDCG, MRR), calibration, prompt sensitivity, cross-model consistency |

---

## Key Findings

- **The system is a strong ranker, not a reliable classifier.** Ranking quality is high (NDCG@5 = 0.956, MRR = 1.0). Binary classification is limited: false positive rate reaches 50%, with zero false negatives. The asymmetry is structural, not correctable by threshold adjustment alone.

- **High consistency under controlled conditions.** Test-retest reliability is ICC = 0.994 (95% CI [0.988, 0.997]) across three independent runs at temperature = 0.4 — near-perfect stability for an LLM-based system.

- **Systematic mid-range scoring pattern.** All 8 false positive cases receive a score of exactly 65 — a pattern that persists across three prompt versions (P0/P1/P2) and three models (Gemini, GPT-4o, deepseek-r1). Confirmed at scale: 14.8% of n=200 scored job descriptions land in the 60–70 band. This is a signal about where the system's uncertainty is concentrated, not random error.

- **Prompt engineering does not fix the core failure.** McNemar's test across prompt versions: p = 1.0. The same job descriptions fail regardless of whether the prompt includes a rubric, chain-of-thought instructions, or neither.

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

## Methodology Summary

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

## Companion Framework

The methodology is documented in a companion framework — *"Evaluating LLM-Based Decision Support Systems for Human Data: A Psychometric and Regulatory Framework"* — which generalises the approach beyond this specific system. It is being published separately and will be linked here.

---

## Related Work

The system being evaluated is described at [github.com/Holly-olly/career-pilot](https://github.com/Holly-olly/career-pilot).
This evaluation is independent and was designed after the system was built, using a dataset constructed blind to AI scores.
