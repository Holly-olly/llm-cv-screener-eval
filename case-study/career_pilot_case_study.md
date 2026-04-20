# Career Pilot: Psychometric Evaluation of an LLM-Based CV–Job Fit Analyser

**April 2026 · Olga Maslenkova**
Track A (n=200: 32 labeled + 168 unlabeled) · Tracks C–E complete · Track B planned

---

> **This is Document 2 of 2 — a case study applying the framework described in:**
> `framework.md` — *Evaluating LLM-Based Decision Support Systems for Human Data: A Psychometric and Regulatory Framework*
>
> Each section opens with a framework reference `[→ FW §X.X]`.
> Results are linked to the notebook where reproduced: `[→ NB: Part X]`.

---

## Table of Contents

- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. Scope, Use Case, and System Definition](#2-scope-use-case-and-system-definition)
- [3. Legal and Ethical Considerations](#3-legal-and-ethical-considerations)
- [4. Construct Definition: What Is Being Measured](#4-construct-definition-what-is-being-measured)
- [5. Measurement Design and System Architecture](#5-measurement-design-and-system-architecture)
- [6. Data and Annotation Protocol](#6-data-and-annotation-protocol)
- [7. Validity Evidence](#7-validity-evidence)
- [8. Reliability and Stability](#8-reliability-and-stability)
- [9. Robustness and Measurement Invariance](#9-robustness-and-measurement-invariance)
- [10. Fairness and Bias Analysis](#10-fairness-and-bias-analysis)
- [11. Performance as a Decision Support System](#11-performance-as-a-decision-support-system)
- [12. Output Quality and Explainability](#12-output-quality-and-explainability)
- [13. Human Factors and Practical Use](#13-human-factors-and-practical-use)
- [14. Limitations and Risk Analysis](#14-limitations-and-risk-analysis)
- [15. Synthesis of Findings](#15-synthesis-of-findings)
- [16. Predictive Validity: Future Work — Track B](#16-predictive-validity-future-work--track-b)
- [17. Reproducibility and Implementation](#17-reproducibility-and-implementation)
- [18. Conclusion](#18-conclusion)
- [References](#references)

---

## Abbreviations

| Abbreviation | Full form |
|---|---|
| **JD** | Job description |
| **CV** | Curriculum vitae (resume) |
| **LLM** | Large language model |
| **FPR** | False positive rate — proportion of negative cases incorrectly classified as positive |
| **FNR** | False negative rate — proportion of positive cases incorrectly classified as negative |
| **ICC** | Intraclass correlation coefficient — measure of reliability/consistency |
| **CI** | Confidence interval |
| **κ** | Cohen's kappa — chance-adjusted agreement between two raters |
| **R²** | Coefficient of determination — proportion of variance explained in a regression |
| **NDCG** | Normalised Discounted Cumulative Gain — ranking quality metric |
| **MRR** | Mean Reciprocal Rank — position of the first relevant item in a ranked list |
| **DIF** | Differential Item Functioning — psychometric measure of bias |
| **IRT** | Item Response Theory |
| **CFA** | Confirmatory Factor Analysis |
| **ATS** | Applicant Tracking System |

---

## Abstract

Career Pilot is an LLM-powered tool that scores the fit between a candidate's CV and a job description (JD) (0–100), assigns a verdict (Apply / Consider / Skip), and generates a structured explanation. This report documents a systematic evaluation of the scoring system across a set of evaluation questions: human–AI agreement, ranking quality, score calibration, test-retest reliability, input robustness, company name bias, candidate discrimination, prompt design sensitivity, multi-model consistency, and human label trustworthiness. The core evaluation uses a purpose-built dataset of 32 masked job descriptions (n=16 positive, n=16 negative; human labels 0–3, blind to AI), three CV variants, three prompt versions, and three LLM models.

**Key findings:** The system functions as a reliable ranking instrument rather than a calibrated classifier. Ranking quality is high (NDCG@5 = 0.956, MRR = 1.0) and test-retest reliability is near-perfect (ICC = 0.994), but binary classification is limited by a structural scoring pattern: all borderline-negative cases receive a score of exactly 65, inflating the false positive rate to 50% while keeping false negatives at zero. This pattern is invariant to prompt design and model choice, indicating task-level ambiguity at the H=1/H=2 boundary rather than a correctable model failure. A subsequent analysis on n=200 unrated job descriptions confirms the 65-cluster as a systematic feature of the score distribution (14.8% of JDs), not a small-sample artefact.

---

## 1. Introduction

> **→ Framework:** [→ FW §1] — Introduction and framework rationale

Career Pilot was designed from scratch as a fully transparent scoring system — every component exposed for evaluation. That design choice, described in the framework, is what makes this case study possible: the scoring rubric, the hidden context injection, the prompt versions, the verdict thresholds, and the temperature are all accessible and modifiable. This enabled the ablation studies (P0→P1→P2), model substitutions (Gemini→GPT-4o→deepseek-r1), and structural diagnosis (the 65-cluster artefact) reported below.

The eleven evaluation questions are mapped to the framework's dual lens (psychometric + AI/engineering) in [→ FW §1.2]. This document reports the results for each.

**System overview:** Career Pilot is a web-based job fit analyser for senior professionals conducting targeted job searches. It accepts a candidate's CV and a job description and returns a score (0–100), a verdict (Apply / Consider / Skip), and a structured report. Full system description in Section 5.

### Cross-References

> **→ Framework:** [→ FW §1] — Rationale, dual-lens framework, evaluation question mapping

---

## 2. Scope, Use Case, and System Definition

> **→ Framework:** [→ FW §2] — Scope and system definition principles

### 2.1 Intended Use: Decision Support vs Automated Decision-Making

Career Pilot is a **decision-support instrument**, not an automated decision-making system. The score and verdict are presented to a candidate who retains full agency over application decisions. No hiring authority uses the system output; the tool operates entirely on the candidate side of the hiring process. This distinction is important: the system is not making a hiring decision — it is informing the candidate's decision about whether to invest time in an application.

Most published LLM-based CV screening systems operate under the same candidate-facing, decision-support framing — placing the human user in the final decision role rather than automating the outcome. > ⏳ *[Proof placeholder — citations to similar systems to be added.]*

This framing determines the appropriate error tolerance: asymmetric errors (missing a relevant job is worse than flagging a borderline irrelevant one) are the correct design target for a candidate-facing screening tool.

### 2.2 Target Population and Application Context

Career Pilot is designed for senior professionals conducting targeted job searches. The current evaluation applies this tool to one candidate profile (psychometrician / people data scientist, 10+ years experience) and one domain (people analytics, HR data science, assessment science, and adjacent roles). Generalisation to other profiles or domains is untested.

### 2.3 Definition of System Outputs and Their Interpretation

| Output | Format | Interpretation |
|--------|--------|----------------|
| **Score** | Integer 0–100 | Degree of fit; higher = stronger match |
| **Verdict** | Apply / Consider / Skip | Categorical decision proxy based on score threshold |
| **Report** | Free text | SUMMARY · CV TWEAKS · ATS KEYWORDS · FINAL TIP |

Verdict thresholds: **Apply** ≥ 70 · **Consider** 45–69 · **Skip** < 45.

These thresholds were set a priori. The sensitivity of classification performance to threshold choice is examined in Section 11.3.

### 2.4 Boundaries of Use and Non-Intended Applications

> ⏳ *[Planned — to be added in a future update.]*

### Cross-References

> **→ Framework:** [→ FW §2] — General scope definition principles
> **→ Notebook:** Part 2 — "Baseline Overview"

---

## 3. Legal and Ethical Considerations

> **→ Framework:** [→ FW §3] — Legal and ethical considerations

### 3.1 Risk Framing Under European AI Governance Principles

> ⏳ *[Planned — to be added in a future update.]*

### 3.2 Data Sources, Processing, and Transparency

The system processes two inputs per analysis: the candidate's own CV (provided by the user) and a job description (pasted or scraped from public sources). No data is stored server-side; all analysis history is stored in the user's local browser (localStorage only). The Gemini API receives CV text and JD text; no persistent candidate database is created.

In addition to the CV text, the scoring prompt injects a *hidden context block* — unlisted talents, current career focus, and dealbreakers not present in the CV text — to account for factors the candidate has not disclosed publicly. This context is defined by the user and stored locally. It is included in every API call but is not visible in the JD-facing output.

The evaluation dataset (Track A, n=32) was constructed from public job postings. All company names were replaced with tokens (`Company_XX`) before publication. No personal data of hiring companies or their employees is included.

### 3.3 Human Oversight and Responsibility

> ⏳ *[Planned — to be added in a future update.]*

### 3.4 Ethical Constraints in Evaluating Human Profiles

> ⏳ *[Planned — to be added in a future update.]*

### Cross-References

> **→ Framework:** [→ FW §3] — Legal and ethical framework

---

## 4. Construct Definition: What Is Being Measured

> **→ Framework:** [→ FW §4] — Construct definition principles

### 4.1 Defining "Fit" as a Latent Construct

"Fit" is operationalised as a latent variable representing the degree to which a candidate's profile matches a role's requirements across five dimensions: domain alignment, technical skill overlap, seniority match, educational background, and strategic direction. From the human annotator's perspective, fit is defined behaviourally — as the likelihood that the candidate would submit an application — on a 0–3 ordinal scale anchored to specific decision rules (Section 6.2).

The weighting of dimensions (Section 4.2) follows the pattern common in LLM-based screening systems, where role-domain fit and skill overlap receive the highest weights [[10]](#references). The specific values (35/25/20/15/5%) reflect the researcher's domain judgment about relative importance for the target role family (psychometrician / people data scientist) and were set a priori before any analysis was run. They are not empirically optimised; future work should validate whether these weights maximise predictive validity for this profile type.

### 4.2 Dimensional Structure of the Construct

The LLM scoring rubric embeds the following dimensional guidance into the prompt text:

| Dimension | Prompt weight | Description |
|-----------|-------:|-------------|
| Domain fit | 35% | Core field match — psychometrics, people data science, assessment science |
| Technical skills | 25% | Tool and methodology overlap (R, Python, IRT, CFA, NLP, etc.) |
| Seniority alignment | 20% | Level match between candidate experience and role requirements |
| Education | 15% | Degree field and level alignment |
| Strategic fit | 5% | Alignment with candidate's stated career focus and dealbreakers |

**Important:** these weights are instructions embedded in the prompt — they tell the LLM how to weight each dimension when reasoning toward a score. They are **not separately computed subscores**: the system produces a single composite score (0–100), and dimension-level contributions cannot be decomposed after the fact. The output should therefore be interpreted as a holistic fit estimate informed by the rubric, not as a mathematically weighted sum of five independent components.

> ⏳ *[Future work: implement separate dimension scoring so the composite can be verified and the weights validated empirically. See CLAUDE.md for tracking.]*

Domain fit receives the highest weight because mismatches at the domain level are rarely compensable by technical skills — a strong statistician without psychometric training is not a viable psychometrician candidate.

### 4.3 Construct Boundaries and Excluded Factors

> ⏳ *[Planned — to be added in a future update.]*

### 4.4 Mapping Construct to Observable Inputs

Observable inputs are: (a) the CV text, encoding domain experience, technical skills, seniority, and education, and (b) the JD text, encoding role requirements and context. These are mapped through the LLM to a 0–100 score via the rubric. The hidden context block extends the observable CV with factors the candidate does not publicly disclose — particularly dealbreakers and career focus constraints that would produce an immediate Skip if violated.

The technical skills dimension is supported by a deterministic pre-processing layer that uses a **custom 21-item skill vocabulary** implemented in `skillExtractor.js` (not an external standard such as ESCO or O*NET). When a JD specifies no skills that appear in this vocabulary — as observed for psychometrician roles, where domain-specific terminology (IRT, DIF, CFA) falls outside the 21-item set — the skill-match component returns 100% by default (no required skills extracted, no penalty applied). For this reason, the structured skill-match score was included only in the hard NO constraint gate and excluded from the scoring composite (see Section 5.2 and Section 7.4).

The mapping is not fully transparent: the LLM is instructed to apply the rubric but does not expose intermediate dimension scores. The output is a single composite score, not a dimension-level profile. This limits diagnostic interpretation but simplifies the output for candidate use.

### Cross-References

> **→ Framework:** [→ FW §4] — Construct definition and dimensional structure principles

---

## 5. Measurement Design and System Architecture

> **→ Framework:** [→ FW §5] — Measurement design principles

### 5.1 Overview of the Scoring Pipeline

Career Pilot operates through two sequential stages: a deterministic pre-filter and an LLM-based scoring call.

```
Input: cv_text, jd_text, settings

Step 1 — Hard NO gate (deterministic)
  if any(dealbreaker in jd for dealbreaker in settings.dealbreakers):
      return score=0, verdict='Skip'

Step 2 — LLM scoring (P1 prompt)
  prompt = build_prompt(cv, jd, hidden_context, rubric_weights)
  response = gemini(prompt, temperature=0.4, max_tokens=2048)
  score, verdict, summary = parse_response(response)

Step 3 — Output
  score: int [0–100]
  verdict: 'Apply' (≥70) | 'Consider' (45–69) | 'Skip' (<45)
  report: SUMMARY + CV TWEAKS + ATS KEYWORDS + FINAL TIP
```

### 5.2 Deterministic Pre-Processing and Constraint Handling

A rule-based pre-filter extracts explicit features from both documents before any LLM call. This layer checks for hard constraint violations (dealbreakers defined by the user — e.g., "no fully on-site roles", "no military tech") and computes a partial structured score across dimensions including domain classification, seniority extraction, location preference, and skill overlap (using the custom 21-item vocabulary described in Section 4.4). If a hard constraint is violated, the LLM stage is bypassed entirely and the score is set to zero.

> **Finding — Structured Layer:** The structured matching layer was evaluated independently (Section 7.4) and found to be decorative rather than additive: correlation with human labels r = −0.33, mean score shift +2.2 pts. It was retained only for the hard NO constraint gate and excluded from the scoring composite.

> **→ Notebook:** Part 13 — "Phase 1: Structured Matching Layer"

### 5.3 LLM-Based Scoring Mechanism

The primary scoring mechanism is a single call to **Google Gemini** (`gemini-3.1-flash-lite-preview`, temperature = 0.4, max_tokens = 2048). The model is instructed to act as a Senior Hiring Manager and evaluate the CV–JD pair on the weighted rubric defined in Section 4.2.

**Gemini was selected as the primary model** because it offers a free API tier, significantly lowering the barrier to entry for end users who supply their own API key. This is an end-user design choice, not a claim about Gemini's superiority for this task; the multi-model benchmark (Section 7.4) shows Gemini is the best-calibrated of the three models tested, but all models share the same structural limitation (the 65-cluster artefact).

Temperature = 0.4 was selected as a balance between consistency (lower temperature) and coverage (higher temperature). The ICC analysis (Section 8.1) confirms this produces near-perfect test-retest reliability under the evaluated conditions.

### 5.4 Prompt Design and Context Injection

Three prompt versions were designed and tested:

| Version | Design | Key feature |
|---------|--------|-------------|
| **P0** | Minimal | No rubric, no weights, no hidden context |
| **P1** | Rubric + context | Weighted 5-dimension rubric + candidate hidden context. **Deployed.** |
| **P2** | Chain-of-Thought | P1 rubric + explicit CoT instruction ("reason step by step before scoring") |

The deployed prompt (P1) injects the hidden context block (unlisted talents, career focus, dealbreakers) in addition to the CV text, and structures the rubric with explicit dimension weights. The ablation results are reported in Section 7.4.

> **→ Notebook:** Part 12 — "Prompt Engineering: Can We Fix the FPR?"

### 5.5 Output Structure: Score, Verdict, and Explanation

The deployed output (P1) consists of: a SUMMARY (2–3 sentences on main fit signal and key gap), CV TWEAKS (role-specific edits), ATS KEYWORDS (keywords the candidate has demonstrably used), and a FINAL TIP. Scores are parsed via regex; the verdict threshold is fixed at Apply ≥ 70, Consider 45–69, Skip < 45.

### 5.6 Separation of Measurement and Decision Layers

> ⏳ *[Planned — to be added in a future update. The core argument is implicit in Section 15.2: measurement quality (ICC, NDCG, R²) and decision policy (threshold setting, verdict assignment) are separable. The system's measurement quality is high; the FPR problem is a decision policy calibration issue that can be addressed at the threshold layer without changing the underlying instrument.]*

### Cross-References

> **→ Framework:** [→ FW §5] — Pipeline design, prompt design principles, deterministic layer evaluation
> **→ Notebook:** Part 12 — "Prompt Engineering" · Part 13 — "Structured Matching Layer"

---

## 6. Data and Annotation Protocol

> **→ Framework:** [→ FW §6] — Dataset construction and annotation methodology

### 6.1 Dataset Construction and Sampling Strategy

**Core evaluation dataset (Track A, n=32)**

32 real job descriptions collected from public job boards (LinkedIn, company career pages). All JDs were selected from roles the researcher had *not applied to*, eliminating hindsight bias. Labels were assigned blind — before any AI analysis was run.

| Property | Value |
|----------|-------|
| Total JDs | 32 (24 original + 8 extended) |
| Positive (H ≥ 2) | 16 (50%) |
| Negative (H ≤ 1) | 16 (50%) |
| Label distribution | H=0: 10 · H=1: 6 · H=2: 10 · H=3: 6 |
| Annotator | Single domain expert (psychometrician, 10+ years) |
| Blinding | AI scores not available at labelling time |
| Masking | Company names replaced with `Company_XX` for publication |
| Status | Frozen — no additions after analysis began |

**Dataset expansion — n=200 unlabeled set**

Following the core Track A study, 168 additional job descriptions were collected from public job APIs (Jobicy, The Muse) and scored using the deployed P1 prompt. These JDs carry no human relevance labels and are used exclusively for: score distribution characterisation, 65-cluster replication at scale, and cross-model consistency checks on a larger sample.

| Property | Value |
|----------|-------|
| Expansion JDs | 168 |
| Total scored | 196 of 200 (4 file errors) |
| Sources | Jobicy API (data-science, hr, engineering, marketing tags) · The Muse API (Data and Analytics, Science and Engineering categories) |
| Domain mix | Strong-fit adjacent (~20%) · Moderate (~40%) · Mismatch (~40%) |

**Score distribution — full n=200 dataset:**

| Score band | Count | % | Interpretation |
|------------|------:|--:|----------------|
| 0–24 | 51 | 26% | Clear mismatch |
| 25–49 | 77 | 39% | Weak fit |
| 50–74 | 59 | 30% | Partial fit |
| 75–100 | 9 | 5% | Strong fit |
| **Mean** | **40.6** | — | Well below Apply threshold (70) |
| **Median** | **40.0** | — | |
| **SD** | **22.0** | — | |

**65-cluster at scale:** 29 of 196 scored JDs (14.8%) received a score in the 60–70 band.

> **→ Notebook:** Part 2b — "Scale Extension: n=200 Dataset"

### 6.2 Human Annotation Procedure and Scale Definition

| Score | Label | Behavioural anchor | Decision rule |
|-------|-------|--------------------|---------------|
| **0** | Not relevant | Would not apply | Different domain; no meaningful overlap; or hard dealbreaker present |
| **1** | Weak overlap | Would not apply | Shared surface keywords but core role is misaligned |
| **2** | Partial fit | Would apply on a chance | Adjacent role; some skill match but clear gaps; low outcome expectation |
| **3** | Strong fit | Would apply with confidence | Core domain match (psychometrics, assessment science, people data science) |

**Key boundary rules:** 0 vs 1 — any data/research element at all? If none → 0. 1 vs 2 — would you actually submit an application? If yes → 2. 2 vs 3 — is psychometrics/measurement science a core requirement? If yes → 3.

Binary threshold: **H ≥ 2 = relevant**, **H ≤ 1 = not relevant**. AI threshold: **score ≥ 50 = predicted relevant**.

### 6.3 Blinding and Bias Control Measures

**Temporal blinding:** human relevance labels were assigned before any AI analysis was run. The annotator had no knowledge of AI scores at labelling time. The AI system had no access to human labels during analysis.

**Company name masking:** all 32 JDs were programmatically masked using a regex replacement dictionary. Each company name and its variants were replaced with `Company_XX`. Masking was verified manually on a random 20% subset. The bias test (Section 10.2) confirms masking does not materially affect AI scores.

**Selection bias control:** JDs were selected from roles the researcher had not applied to, removing hindsight contamination. Track A and Track B datasets do not overlap by design (Section 16.2).

### 6.4 Input Variants and Experimental Conditions

| File | Purpose | Used in |
|------|---------|---------|
| `cv_primary.txt` | Main CV — all primary analyses | Sections 7.1, 7.2, 7.3, 7.4, 8.1, 10.2, 11, 12 |
| `cv_paraphrased.txt` | Same content, reworded — tests surface sensitivity | Section 9.1 |
| `cv_synthetic_control.txt` | Fictional UX Researcher profile — tests discrimination | Section 7.3 |

### 6.5 Limitations of the Annotation Process

The annotation was conducted by a single domain expert. While the annotation audit (Section 11 / Q10) confirms substantial agreement with two independent LLM judges (κ ≥ 0.562), single-annotator datasets are susceptible to systematic perspective bias. The H=1/H=2 boundary — exactly where the 65-cluster cases sit — is the most ambiguous region: two of the eight borderline cases were rated H=2 by at least one LLM judge, suggesting the human label may be at or near the annotation error boundary for those cases.

**Additional limitation:** the sole annotator is also the candidate whose CV is being evaluated against the JDs. This creates a potential self-interest bias — the annotator may unconsciously rate as relevant those JDs where they feel confident applying, inflating the positive class in alignment with their own application decisions. This conflict of interest is not controllable in a single-candidate design. Future work should use at least two independent annotators.

The small sample (n=32) limits statistical power. Wilson CIs on FPR are wide ([28%, 72%]), and calibration estimates (R² = 0.674) carry uncertainty appropriate to the sample size.

### Cross-References

> **→ Framework:** [→ FW §6] — Dataset construction requirements, annotation scale design, blinding requirements
> **→ Notebook:** Part 2 — "Baseline Overview" · Part 2b — "Scale Extension: n=200"

---

## 7. Validity Evidence

> **→ Framework:** [→ FW §7] — Validity evidence: metrics, interpretation, study design

### 7.1 Criterion-Related Evidence: Human–AI Agreement

*Q1 · Does the AI agree with expert judgment?*

**Methods:** Binary accuracy (threshold AI ≥ 50 = positive, AI < 50 = negative) across n = 32 JDs. Cohen's κ [[1]](#references) for chance-adjusted agreement. Wilson score confidence intervals [[2]](#references) for proportions. Computed with `scipy.stats` and manual Wilson CI implementation.

**Results:**

| Metric | Value | 95% Wilson CI | Interpretation |
|--------|------:|--------------|----------------|
| Accuracy | **75%** | [58%, 87%] | 24/32 correct classifications |
| False Positive Rate (FPR) | **50%** | [28%, 72%] | 8/16 negatives incorrectly scored ≥ 50 |
| False Negative Rate (FNR) | **0%** | [0%, 21%] | 0/16 positives missed — every relevant job found |
| True Positive Rate (Recall) | **100%** | [79%, 100%] | All 16 relevant JDs scored ≥ 50 |
| Precision | **67%** | — | 16 true positives / 24 positive predictions |
| Cohen's κ | **0.500** | — | Moderate agreement beyond chance (Landis & Koch scale) |

**The 65-cluster artefact:** All 8 false positive JDs received an AI score of **exactly 65**. This pattern was confirmed across two independent analysis batches. The score 65 appears to function as a model default for uncertain cases — cases where the model cannot confidently place the JD above or below the relevance threshold. This is a structural property of the model, not a prompt wording issue, as it persists across prompt versions P0, P1, and P2 (Section 7.4) and across all three models tested (Section 7.4).

> **Key Finding:** FPR = 50% is driven entirely by 8 JDs that all receive score = 65. At a binary threshold of 50, all 8 fall on the positive side, inflating FPR. At a threshold of 66, all 8 flip to negative (FNR rises from 0% to non-zero). There is no threshold that simultaneously eliminates false positives without introducing false negatives.

> **Asymmetric error design:** The system has asymmetric error characteristics by design: it never misses a genuinely relevant job (FNR = 0%) but flags half of the borderline-irrelevant cases. For a job search context, this is arguably the correct failure mode — missing a relevant opportunity is more costly than spending five minutes reading a borderline posting.

### 7.2 Calibration and Score Meaningfulness

*Q3 · Do the scores mean anything?*

**DV:** Human relevance label (0–3). **IV:** AI score (0–100). **N:** 32 JDs. **Test:** OLS linear regression; Pearson r for association.

**Methods:** Linear regression of AI score onto human relevance label (0–3). R² as a calibration index; Pearson r for association. Computed with `numpy`.

| Measure | Value | Interpretation |
|---------|------:|----------------|
| Pearson r (AI score, human label, n=32) | **0.821** | Strong positive association |
| R² (OLS, AI score → human label) | **0.674** | 67% of variance in human ratings explained by AI score |
| Deployed threshold | **50** | Zero FNR — no relevant job missed |
| Optimal binary threshold | **66** | Maximises accuracy but introduces FNs |

R² = 0.674 indicates good linear calibration across the full score range (n=32). In plain terms: the model's scores track the human ratings closely (r = 0.821), explaining about two-thirds of the variation in expert judgment. The threshold sensitivity analysis (Section 11.3) reveals a *hard discontinuity* at the 65-cluster: raising the threshold from 65 to 66 simultaneously eliminates all 8 false positives and introduces several false negatives.

> **→ Notebook:** Part 5 — "Calibration" · Part 6 — "Error Analysis: The 65 Problem"

### 7.3 Discriminant Validity: Candidate Differentiation

*Q7 · Can the system differentiate between candidate profiles?*

**DV:** AI score (0–100). **IV:** CV type (primary vs synthetic). **N:** 32 JDs × 2 CV versions = 64 scored pairs. **Test:** paired t-test on score differences.

**Design:** A synthetic control CV was constructed representing a UX Researcher profile — adjacent to people data science in surface features but with no psychometrics, IRT, CFA, or quantitative assessment background. Both CVs were scored against all 32 JDs (n=32 pairs).

| CV Profile | Mean Score | Score Range |
|------------|-----------:|-------------|
| Primary CV (psychometrician) | **64.9** | 25–85 |
| Synthetic CV (UX Researcher) | **38.2** | 15–65 |
| Mean difference | **+26.7 pts** | 95% CI [19.3, 34.1] |

**Statistical test:** paired t-test, t(23) = 7.04, p < 0.001, Cohen's d = 1.44 (large effect). N = 24 JDs with non-zero variance across both CV versions.

> **Finding — Correct Discrimination:** The 26.7-point mean difference (95% CI [19.3, 34.1]) confirms meaningful differentiation between candidates. The effect is large (d = 1.44). The synthetic UX profile scores substantially lower on psychometrics and assessment roles (largest gap) but comparably on generic data analyst roles — which is the correct pattern.

> **→ Notebook:** Part 10 — "Discrimination: Synthetic CV Test"

### 7.4 Structural Validity: Prompt and Model Invariance

*Q8 & Q9 · Does prompt design matter? Is this a model problem or a task problem?*

**Prompt ablation — P0/P1/P2:**

| Prompt | Accuracy | FPR | FNR | 65-cluster | Notes |
|--------|---------:|----:|----:|------------|-------|
| P0 — minimal | 72% | 50% | 6% | Partial | Misses 1 relevant JD |
| **P1 — rubric** | **75%** | **50%** | **0%** | Yes (8×65) | **Deployed** |
| P2 — CoT | 75% | 44% | 6% | Absent | Fewer FPs but misses 1 relevant JD |

McNemar's test [[5]](#references) P0 vs P1 vs P2: p = 1.0 (all pairwise). Despite adequate power (n_neg = 16 > 11 required for 80% power at this effect size), the same JDs are misclassified by each prompt. The error is in the JDs (genuine ambiguity), not the prompt.

**Structured matching layer:** Pearson r (structured score vs human label, n=32) = −0.33, 95% CI [−0.61, 0.02]; mean score shift when combined with LLM = +2.2 pts. Removed from scoring composite; retained only for hard NO gate. The negative correlation indicates the structured layer does not add predictive information beyond the LLM score for this role family — likely due to the 21-item skill vocabulary failing to match psychometrics-specific terminology (see Section 4.4).

**Cross-model benchmark (Track C):**

| Model | Accuracy | FPR | FNR | FPs (of 16 neg) | Mean Score |
|-------|---------:|----:|----:|----------------:|-----------:|
| **Gemini V1** (baseline) | 75% | **50%** | 0% | 8/16 | 64.9 |
| GPT-4o | 69% | 63% | 6% | 10/16 | 65.8 |
| deepseek-r1:8b | 56% | 81% | 6% | 13/16 | 58.4 |

| Agreement Measure | Value |
|-------------------|------:|
| ICC (GPT-4o + deepseek) | 0.721 |
| ICC (all models incl. Gemini) | 0.698 |
| Pearson r: Gemini vs GPT-4o | 0.81 |
| Pearson r: Gemini vs deepseek | 0.74 |

> **Key Finding — Model-Dependent but Shared Failure Mode:** All three models struggle with the same subset of borderline JDs. The FPR range is 50%–81% (31pp across models), which suggests both a shared task-level difficulty and model-specific calibration differences — Gemini achieves the lowest FPR. The repeated score of 65 on these cases may reflect the model treating them as borderline rather than confident misclassifications, though alternative explanations (e.g., a fallback score in the model's distribution) cannot be ruled out from this evidence alone.

> **→ Notebook:** Part 12 — "Prompt Engineering" · Part 15 — "Multi-Model Comparison (Track C)"

### 7.5 Interpretation of Validity Evidence Across Metrics

The validity evidence is best interpreted by separating two distinct performance profiles. See Section 15.2 for the explicit synthesis. Briefly: the system demonstrates strong construct validity (R² = 0.674, r = 0.821, n=32; discriminant validity d=1.44 confirmed) and structural validity (prompt-invariant error pattern, cross-model consistency ICC = 0.698–0.721), but exhibits a calibration limitation that concentrates errors in a single cluster — the 65-score cases in the genuinely ambiguous region of the rating scale.

### Cross-References

> **→ Framework:** [→ FW §7] — Validity metrics, interpretation guidance, discriminant validity design
> **→ Notebook:** Part 3 — "Agreement Accuracy" · Part 4 — "Ranking Quality" · Part 5 — "Calibration" · Part 6 — "Error Analysis: The 65 Problem" · Part 10 — "Synthetic CV Test" · Part 12 — "Prompt Engineering" · Part 15 — "Multi-Model Comparison"

---

## 8. Reliability and Stability

> **→ Framework:** [→ FW §8] — Reliability metrics, ICC interpretation, non-deterministic systems

### 8.1 Test–Retest Consistency Under Controlled Conditions

*Q4 · Can you trust it to give the same answer twice?*

**Methods:** ICC(2,1) [[4]](#references) — two-way mixed effects model, single measures, absolute agreement. Applied to n = 24 JDs × 3 runs (masked version). Computed with a custom `numpy` implementation of the Shrout & Fleiss (1979) formula. Interpretation: ICC > 0.90 = excellent (Koo & Mae, 2016).

| Condition | ICC(2,1) | 95% CI | Interpretation |
|-----------|---------:|--------|----------------|
| Runs 1 & 2 (masked JDs only, n=24) | **0.994** | [0.988, 0.997] | Excellent — near-perfect consistency |
| All 3 runs (full matrix, n=24) | **0.984** | — | Excellent |

CI computed using the Shrout & Fleiss (1979) formula-based approach [[4]](#references), n=24, k=3.

> **Key Finding — Reliability (n=24):** ICC(2,1) = 0.994, 95% CI [0.988, 0.997], is exceptionally high for an LLM-based system at temperature = 0.4. Score variation across runs is negligible relative to variation across JDs. This demonstrates high stability under controlled conditions, with the understanding that ICC stability at this level is contingent on this candidate profile, this model version, and this temperature setting.

> **→ Notebook:** Part 8 — "Reliability: Test-Retest Consistency"

### 8.2 Sensitivity to Prompt and Model Variation

Sensitivity to prompt version was assessed by McNemar's test across P0/P1/P2 (result: p = 1.0 — same JDs misclassified regardless of prompt). Sensitivity to model was assessed by ICC across three models (ICC = 0.698–0.721 — moderate cross-model consistency). Full results are in Section 7.4.

The key contrast: **within-model reliability** (ICC = 0.994) is dramatically higher than **cross-model consistency** (ICC = 0.698–0.721). This is expected — different model architectures encode different inductive biases — but the convergence on the same failure cases (the same 8 borderline JDs) is more meaningful than the ICC difference suggests.

### 8.3 Interpretation of Reliability in Non-Deterministic Systems

The correct interpretation of ICC = 0.994 is: *under identical prompt, model, temperature, and candidate profile, the system is near-perfectly repeatable*. It is not evidence that the system would produce stable scores under different prompt conditions (addressed in 8.2) or when the candidate profile changes (untested).

### Cross-References

> **→ Framework:** [→ FW §8] — ICC design, interpretation thresholds, reliability in LLM systems
> **→ Notebook:** Part 8 — "Reliability: Test-Retest Consistency"

---

## 9. Robustness and Measurement Invariance

> **→ Framework:** [→ FW §9] — Robustness testing design and interpretation

### 9.1 Sensitivity to Surface Variations in Input (CV Paraphrase)

*Q5 · Is it sensitive to irrelevant CV surface changes?*

**DV:** Score difference (paraphrased − original). **IV:** CV version (original vs paraphrased). **N:** 23 JDs (9 JDs excluded: score = 0 in both versions, no meaningful comparison). **Test:** paired t-test.

**Methods:** A paraphrased version of the primary CV was constructed by manually rewriting sentences while preserving all factual content (roles, years of experience, tools, methods). Example: *"Led psychometric quality control of 350+ assessments"* (original) vs *"Maintained psychometric standards across a portfolio of over 350 live tests"* (paraphrased). The paraphrased CV was run through the system on the same n=32 JDs. Score differences (score_paraphrased − score_original) were computed.

**Note on sample size:** n=23 usable pairs; this is a small sample that limits statistical power. The paired t-test is borderline (see below). Results should be treated as exploratory.

| Metric | Value | 95% CI / Statistic | Interpretation |
|--------|------:|--------------------|----------------|
| Scores stable (\|Δ\| = 0) | **65%** (15/23) | Wilson CI [0.45, 0.81] | Majority unchanged |
| Mean score shift (paraphrased − original) | **+2.7 pts** | 95% CI [−0.2, 5.5] | Paraphrased CV slightly higher on average |
| Paired t-test | t(22) = 1.83 | p = 0.08 | Not statistically significant |
| Effect size | Cohen's d = 0.38 | — | Small-medium |

> **→ Notebook:** Part 9 — "Robustness: CV Paraphrase Test"

### 9.2 Stability Under Input Transformations

65% of JDs received identical scores across CV versions (Wilson CI [0.45, 0.81]) — the majority of the signal comes from content, not surface phrasing. The mean shift of +2.7 points (95% CI [−0.2, 5.5]; p = 0.08) is not statistically significant at the conventional α = 0.05 level, though the effect is in a consistent direction. The paraphrased CV's slight advantage may reflect word choice closer to common JD language, suggesting a minor lexical overlap effect. Given the small sample (n=23), this finding should be treated as exploratory.

### 9.3 Effects of Contextual Augmentation (Hidden Context)

> ⏳ *[Planned — to be added in a future update. The hidden context block is described in Section 5.4; its independent effect on scores has not been formally evaluated. A planned ablation: run n=32 JDs with and without the hidden context block and measure score shift and verdict change rate.]*

### 9.4 Implications for Measurement Invariance

A mean surface shift of +2.7 points is generally acceptable for screening: it would not change most verdicts, given that verdict thresholds are spaced 25 points apart (Apply/Consider boundary at 70, Consider/Skip at 45). However, individual shifts of > 10 points could change an Apply to Consider or vice versa for scores near threshold. Combined with the McNemar finding (prompt version does not change which JDs are misclassified), the overall picture is: the system is *ranking-invariant* (same JDs fail regardless of prompt or minor CV phrasing) but *score-level sensitive* to surface changes, especially near threshold boundaries.

### Cross-References

> **→ Framework:** [→ FW §9] — Robustness design, invariance interpretation
> **→ Notebook:** Part 9 — "Robustness: CV Paraphrase Test"

---

## 10. Fairness and Bias Analysis

> **→ Framework:** [→ FW §10] — Fairness definitions, DIF, bias testing design

### 10.1 Definition of Fairness in LLM-Based Assessment

In the psychometric tradition, fairness refers to the absence of **differential item functioning** (DIF): the score should be determined by the construct being measured (fit), not by irrelevant features of the input (employer identity, candidate name, protected characteristics). This section evaluates one such factor: employer name (Section 10.2).

**Note on company masking:** The company name masking (Section 6.3) was introduced originally as a privacy measure for dataset publication — not as a planned fairness analysis. The bias analysis in Section 10.2 was added post-hoc after noting the masking data were already available. This limits the interpretability of the test: the masking was not designed as a randomised experiment.

> ⏳ *[A broader fairness analysis — including name-coded bias in CVs and language-based effects — is planned for a future update.]*

### 10.2 Influence of Irrelevant Features (Company Name Test)

*Q6 · Does including the company name inflate or deflate scores?*

**Design:** All 32 JDs scored twice — once with company names visible (original), once with names replaced by `Company_XX` (masked). Paired t-test for mean score difference; Cohen's *d* for effect size; ICC between masked and original. 72 Gemini API calls total.

| Metric | Value | Interpretation |
|--------|------:|----------------|
| Mean bias (original − masked) | **−0.96 pts** | Negligible directional bias |
| Paired t-test p-value | **0.67** | No statistically significant difference |
| Cohen's *d* | **−0.18** | Small effect size |
| ICC (original vs masked) | **0.984** | Excellent agreement |

> **Finding — No Meaningful Company Name Bias (limited scope):** Company name masking does not meaningfully change AI scores on this dataset. **This test covers one irrelevant feature on one candidate profile.** It does not constitute a fairness audit — group fairness, protected-attribute bias (gender, nationality, age), and candidate-facing bias are not tested here. The result supports the validity of using masked JDs for publication, and nothing more.

### 10.3 Candidate-Level Discrimination

The discrimination analysis is reported in Section 7.3. From a fairness perspective, the observed 26.7-point gap represents *valid differentiation* (different qualifications), not bias. The critical fairness test — equivalent qualifications, irrelevant feature varies — is planned for a future update.

### 10.4 Limitations of Fairness Evaluation in Current Design

The current design tests only one axis of irrelevant-feature bias (company name). Candidate-facing bias — name-based discrimination, gendered language effects, cultural signalling in CV formatting — has not been tested. The single-candidate design means the discrimination finding characterises one specific profile comparison, not the general case.

### Cross-References

> **→ Framework:** [→ FW §10] — Fairness definitions, bias test design, limitations of pilot fairness evaluation
> **→ Notebook:** [→ NB: Part ??] — Company Name Bias Test *(notebook part TBC)*

---

## 11. Performance as a Decision Support System

> **→ Framework:** [→ FW §11] — Ranking and classification metrics, threshold sensitivity

### 11.1 Ranking Performance (NDCG, MRR, Recall@k)

*Q2 · Does it rank jobs correctly?*

**Methods:** NDCG@k [[3]](#references) — normalised discounted cumulative gain, computed with `sklearn.metrics.ndcg_score`. MRR — mean reciprocal rank of the first truly relevant item (H ≥ 2). Recall@k — proportion of relevant items in the top-k ranked results. Bootstrap confidence intervals (B = 5000 resamples, stratified).

| Metric | Value | Bootstrap 95% CI | Pre-registered target |
|--------|------:|-----------------|----------------------|
| NDCG@5 | **0.956** ✓ | [0.92, 0.98] | > 0.80 |
| NDCG@10 | **0.931** | — | — |
| MRR | **1.0** ✓ | — | > 0.80 |
| Recall@10 | **0.667** ✗ | — | > 0.75 |

> **Key Finding — Ranking Quality (interpret cautiously, n=32):** NDCG@5 = 0.956 means the top 5 results are nearly perfectly ordered by relevance. Recall@10 = 0.667 is below the 0.75 target, reflecting that some H=2 JDs are ranked below the 65-cluster cases. Note: MRR = 1.0 is a near-mathematical consequence of FNR = 0% — when no relevant job is missed and at least one receives the top score, MRR = 1.0 is almost guaranteed by construction. It is reported for completeness but is not an independent quality signal. With n=32, all ranking metrics are sensitive to single-item fluctuations; no bootstrap CIs are reported for MRR or NDCG@10. These results should be treated as indicative pending replication on a larger sample.

> **→ Notebook:** Part 4 — "Ranking Quality"

### 11.2 Binary Classification Performance (Accuracy, FPR, FNR)

Summary table for binary classification at AI threshold ≥ 50 (full analysis in Section 7.1):

| Metric | Value | 95% Wilson CI |
|--------|------:|--------------|
| Accuracy | **75%** | [58%, 87%] |
| FPR | **50%** | [28%, 72%] |
| FNR | **0%** | [0%, 21%] |
| Cohen's κ | **0.500** | — |

All 8 false positives are concentrated at score = 65 (the 65-cluster artefact).

> **→ Notebook:** Part 3 — "Agreement Accuracy" · Part 7 — "Threshold Sensitivity"

### 11.3 Threshold Sensitivity and Decision Trade-offs

The threshold sensitivity analysis sweeps the binary decision threshold from 45 to 80 in steps of 1 point. The key result: the score distribution is non-continuous at the 65-cluster. There is a *hard discontinuity* at threshold = 66:

- At threshold ≤ 65: all 8 FP cases classified as positive (FPR = 50%, FNR = 0%)
- At threshold = 66: all 8 FP cases flip to negative simultaneously (FPR = 0%, but FNR rises)

| Threshold | Accuracy | FPR | FNR | Notes |
|----------:|---------:|----:|----:|-------|
| 50 (deployed) | 75% | 50% | 0% | Zero FNR; all relevant jobs found |
| 66 (optimal accuracy) | 100% | 0% | 0%* | *Depends on exact cluster membership |

### 11.4 Distinguishing Measurement Quality from Decision Policy

> ⏳ *[Planned — to be added in a future update. The core argument is implicit in Section 15.2: measurement quality (ICC, NDCG, R²) and decision policy (threshold setting, verdict assignment) are separable.]*

### Cross-References

> **→ Framework:** [→ FW §11] — Ranking metrics, classification metrics, threshold sensitivity
> **→ Notebook:** Part 4 — "Ranking Quality" · Part 7 — "Threshold Sensitivity"

---

## 12. Output Quality and Explainability

> **→ Framework:** [→ FW §12] — LLM-as-judge methodology, verbosity bias, explanation evaluation

> ⏳ *[Section removed from current scope. The LLM-as-judge explanation quality analysis (Track E) has been saved to `docs/saved_for_later.md` for potential future inclusion. The corresponding notebook analysis (Part 17) was also removed. Reason: methodological concerns about the circular evaluation design (LLM judging LLM outputs from the same model family) and limited contribution to the core construct validity argument.]*

---

## 13. Human Factors and Practical Use

> **→ Framework:** [→ FW §13] — Human factors framework, over-reliance risks

### 13.1 Interaction Between User and System Output

In a typical use scenario, the candidate receives a score and verdict and decides whether to apply. The system's asymmetric error design (FNR = 0%, FPR = 50%) means all relevant opportunities appear in the candidate's review queue, at the cost of also surfacing some borderline-irrelevant roles. The 65-cluster cases — where the model cannot confidently classify — are presented with a *Consider* verdict (score 65 > threshold 45), inviting the candidate to make their own judgement on ambiguous cases rather than filtering them out automatically.

The critical user-interaction question: does the candidate follow the verdict, or do they override it? Track B (Section 16) will test this: if candidates consistently apply despite *Skip* verdicts or skip despite *Apply* verdicts, that is evidence the tool's utility is limited relative to candidate's own judgment.

### 13.2 Appropriate Use of Scores in Decision-Making

> ⏳ *[Planned — to be added in a future update.]*

### 13.3 Risks of Over-Reliance and Misinterpretation

> ⏳ *[Planned — to be added in a future update.]*

### Cross-References

> **→ Framework:** [→ FW §13] — Human factors principles, over-reliance design

---

## 14. Limitations and Risk Analysis

> **→ Framework:** [→ FW §14] — General limitations of pilot evaluation design

### 14.1 Data and Sampling Limitations

- **Small dataset.** n = 32 JDs (labeled) provides limited statistical power. Wilson CIs on FPR ([28%, 72%]) are wide. Findings should be treated as pilot evidence until replicated on a larger labeled dataset.
- **Single domain, single candidate.** All JDs were selected for one candidate profile (psychometrician / people data scientist). Generalisation to other domains, seniority levels, or candidate profiles is untested.
- **Selection of job APIs.** The n=168 unlabeled expansion used Jobicy and The Muse, which skew toward remote-first, US/UK roles. The domain mix does not include the rarest strong-fit categories (psychometrician, occupational psychologist) at meaningful frequency.

### 14.2 Annotation and Construct Limitations

- **Single annotator.** Human labels were assigned by a single domain expert. While the annotation audit (Track D) confirms substantial inter-rater reliability with LLM judges (κ ≥ 0.562), single-annotator datasets are susceptible to systematic perspective bias. The H=1/H=2 boundary received divergent labels from LLM judges in 2 of 8 cases, suggesting annotation error cannot be ruled out for those cases.
- **Threshold arbitrariness.** The binary threshold (≥ 50 = relevant) was set a priori. Alternative thresholds produce different FPR/FNR trade-offs.

### 14.3 Model and Prompt Dependence

- **Prompt as instrument.** The evaluation assesses a specific prompt (P1), not the LLM capability in general.
- **Model deprecation.** `gemini-3.1-flash-lite-preview` is a preview model. Results may not replicate on stable model versions or future releases.
- **Track C infrastructure.** Hardware variance, response caching, and version pinning are not controlled across models.

### 14.4 Interpretation Constraints and Uncertainty

- **No predictive validity yet.** Track A demonstrates construct validity. Track B has not been run. Construct validity is necessary but not sufficient for practical utility.
- **Researcher conflict of interest.** This study was designed, executed, and interpreted by the researcher who built the tool being evaluated. The design controls for hindsight bias in labelling. It does not control for motivated reasoning in how results are framed — particularly for the 65-cluster, where the "model expressing uncertainty" interpretation is plausible but also convenient.

### Cross-References

> **→ Framework:** [→ FW §14] — Pilot evaluation limitations, single annotator, researcher conflict of interest

---

## 15. Synthesis of Findings

> **→ Framework:** [→ FW §15] — Synthesis approach, ranking vs classification distinction

### 15.1 Summary Across Evaluation Dimensions

| Evaluation Question | Metric | Result | Status |
|---------------------|--------|-------:|--------|
| Human–AI agreement | Accuracy / κ | 75% / 0.500 | Moderate |
| Ranking quality | NDCG@5 / MRR | 0.956 / 1.0 | Excellent |
| Score calibration | r / R² (n=32) | 0.821 / 0.674 | Good |
| Test-retest reliability | ICC(2,1) | 0.994 [0.988, 0.997] | Excellent |
| CV robustness | % stable / mean shift | 65% / +2.7 pts (p=0.08) | Acceptable |
| Company name bias | p-value / d | 0.67 / −0.18 | None |
| Candidate discrimination | Mean gap / d | +26.7 pts / 1.44 | Correct |
| Prompt sensitivity | McNemar p | 1.0 | Invariant |
| Multi-model consistency | ICC (3 models) | 0.698 | Moderate |
| Annotation audit | κ (human vs LLM) | 0.562–0.625 | Substantial |
| 65-cluster prevalence | % of JDs in 60–70 band | 14.8% (n=200) | Present at scale |

> **→ Notebook:** Part 11 — "Full Evaluation Summary (Track A)"

### 15.2 Key Insight: Ranking vs Classification Behaviour

The system has two distinct performance profiles:

- **Ranking performance:** Near-perfect (NDCG@5 = 0.956, MRR = 1.0). The model correctly orders JDs from most to least relevant.
- **Binary classification performance:** Moderate (75% accuracy, FPR = 50%). At any fixed threshold, roughly half of the borderline-negative cases are over-scored.

> **The system behaves as a ranking instrument rather than a calibrated classifier.** The FPR issue is fundamentally a threshold calibration problem, not a ranking failure — the model orders opportunities correctly but does not place a well-calibrated decision boundary between "apply" and "skip".

This distinction has direct implications for how the system should be used (ranking and prioritisation, not automated filtering) and how its performance should be reported (ranking metrics are the primary validity evidence; classification accuracy is a secondary, threshold-dependent metric).

> **→ Notebook:** Part 11 — "Full Evaluation Summary (Track A)"

### 15.3 Interpretation of System Strengths and Weaknesses

The 65-cluster score pattern (14.8% of all scored JDs landing in the 60–70 band) is a structural feature of the model's output distribution. One plausible interpretation is that the model uses this region to express uncertainty on genuinely ambiguous cases; however, this interpretation is based on a small labeled sample (n=8 false positives) and should not be overstated. The Track D annotation audit finds that 6 of 8 such cases are borderline even by independent LLM judgment — but LLM judges have their own limitations as ground truth.

**External methodology review:** An independent GPT-4o critique rated the methodology as *Adequate* and the conclusions as *Weak*, flagging three valid points: single annotator significance is understated; the "task-fundamental" interpretation for the 65-cluster is stated with more confidence than three models warrant; Track D uses AI judges to partially validate AI-produced patterns, creating a circularity. These are acknowledged in Section 14.

### Cross-References

> **→ Framework:** [→ FW §15] — Synthesis framework, ranking vs classification interpretation
> **→ Notebook:** Part 11 — "Full Evaluation Summary (Track A)"

---

## 16. Predictive Validity: Future Work — Track B

> **→ Framework:** [→ FW §16] — Predictive validity study design principles

*Psychometric: Criterion (Predictive) Validity*

### 16.1 From Construct Validity to Real-World Outcomes

Track A answered the first question: the AI score correlates with expert relevance judgment (R² = 0.762) and ranks roles correctly (NDCG@5 = 0.956). That is **construct validity** — the score measures what it claims to measure.

But construct validity is not the same as **predictive validity**. Track B asks that harder question: *What does a score of 75 actually mean? Does it predict whether the candidate applies, whether the company responds, whether there is a match?*

### 16.2 Study Design for Outcome-Based Evaluation

| Property | Value |
|----------|-------|
| Dataset | 68–85+ historical application records |
| Criterion variables | Applied (binary) · Interview received (binary) · Offer received (binary) |
| Primary metrics | AUC-ROC, point-biserial correlation, Lift@k |
| Secondary metrics | Calibration curve (Brier score), precision at score ≥ 70 (Apply verdict) |
| Status | ⏳ Waiting for ≥ 10–15 positive hiring outcomes (~7 currently) |

Scripts are ready: `evaluation/track_b_prepare.py`, `evaluation/track_b_analyze.py`.

**Track A vs Track B comparison:**

| Property | Track A (completed) | Track B (planned) |
|----------|--------------------|--------------------|
| Dataset | 32 new JDs (no outcome) | 68–85+ historical applications |
| Criterion | Expert relevance label (0–3) | Real outcome (interview / rejection) |
| Primary metric | Accuracy, NDCG, ICC | AUC-ROC, point-biserial r |
| Labelling required | Yes (human labels) | No (outcome is objective) |
| Status | ✅ Complete | ⏳ Waiting for outcomes |

### 16.3 Planned Metrics and Data Collection Strategy

**Chapter 1 — Did the candidate apply?** If the AI advises *Apply* and the candidate skips the role — or if the AI says *Skip* and the candidate applies anyway — that is information about the tool's utility.

**Chapter 2 — Did the company respond?** A first-round interview is the earliest objective signal that the company also perceives a fit.

**Chapter 3 — Was there a match?** Beyond the first screening — did the process advance to an offer?

### 16.4 Challenges in Predictive Validation

Track B avoids retrospective label contamination by design — the criterion is an objective administrative record, not a retrospective label, and the Track A and Track B datasets do not overlap. A deeper challenge: Track B operates on a single candidate's historical data, limiting statistical power and precluding general conclusions.

> **→ Notebook:** Part 14 — "What's Next: Track B (Predictive Validity)"

### Cross-References

> **→ Framework:** [→ FW §16] — Predictive validity study design, outcome-based evaluation metrics
> **→ Notebook:** Part 14 — "Track B"

---

## 17. Reproducibility and Implementation

> **→ Framework:** [→ FW §17] — Reproducibility requirements, implementation documentation

### 17.1 Tools, Libraries, and Infrastructure

| Library / Tool | Version | Used for |
|----------------|---------|----------|
| `numpy` | ≥1.24 | Array operations, bootstrap resampling, ICC computation |
| `scipy.stats` | ≥1.10 | Cohen's κ, paired t-test, Pearson r, Wilson CI |
| `sklearn.metrics` | ≥1.3 | NDCG, accuracy, confusion matrix |
| `statsmodels` | ≥0.14 | McNemar's test, power analysis |
| `matplotlib` / `seaborn` | — | Calibration plots, ICC matrix, score distributions |
| `pandas` | ≥2.0 | Data loading, tabular analysis |
| `openai` | ≥1.0 | GPT-4o API; Ollama local models via `base_url='http://localhost:11434/v1'` |
| `google.generativeai` / HTTP | — | Gemini API calls (primary model, batch analysis) |
| Jupyter Notebook | — | Primary analysis environment |
| Ollama | — | Local model serving (deepseek-r1:8b, qwen3:8b) |

### 17.2 Experimental Pipeline and Scripts

All evaluation scripts are in the `evaluation/` directory:

| Script | Purpose |
|--------|---------|
| `label_jds.py` | Human annotation workflow |
| `batch_analyze.py` | Batch AI scoring (supports `--include-unlabeled` flag for n=200) |
| `fetch_real_jds_free.py` | Collects JDs from Jobicy and The Muse APIs (no auth required) |
| `consistency_test.py` | Test-retest reliability (ICC) |
| `robustness_test.py` | CV paraphrase robustness |
| `company_name_bias_test.py` | Company name masking experiment |
| `synthetic_cv_test.py` | Candidate discrimination test |
| `prompt_p0_simple.py` / `prompt_p1_revised_rubric.py` / `prompt_p2_cot.py` | Prompt ablation |
| `multi_model_analyze.py` | Cross-model benchmark (Track C) |
| `llm_annotation_audit.py` | LLM-as-judge annotation audit (Track D) |
| `openai_critique.py` | LLM-as-judge output quality (Track E) + methodology review |
| `track_b_prepare.py` / `track_b_analyze.py` | Predictive validity pipeline (Track B, planned) |

**Implementation notes:** ICC(2,1) was implemented from the Shrout & Fleiss (1979) formula directly in NumPy. Bootstrap CIs used stratified resampling (preserving positive/negative ratio) with B = 5000 iterations. The `parse_score()` function for deepseek-r1:8b includes handling for both closed and truncated `<think>` reasoning blocks.

### 17.3 Guidelines for Replication

> ⏳ *[Planned — to be added in a future update. Will include: data format specification, environment setup, step-by-step script execution order, and notes on model version pinning.]*

### Cross-References

> **→ Framework:** [→ FW §17] — Reproducibility documentation requirements

---

## 18. Conclusion

> **→ Framework:** [→ FW §18] — Framework conclusion and future directions

### 18.1 Toward Standardised Evaluation of LLM-Based Assessment Systems

This case study demonstrates the framework applied end-to-end to a real system. The psychometric framework surfaces findings that a standard AI benchmark would miss: the 65-cluster artefact is not a benchmark failure but a structural property of the system's uncertainty representation; the ICC = 0.994 is precisely characterised and appropriately caveated; the FPR = 50% is not a simple failure but a threshold calibration problem concentrated in genuinely ambiguous cases.

The framework is replicable: any LLM-based system that accepts structured inputs and produces scored outputs can be evaluated using the eleven-question structure defined in the companion framework document.

### 18.2 Implications for Research and Regulation

> ⏳ *[Planned — to be added in a future update.]*

### 18.3 Future Directions

1. **Cross-domain and cross-candidate generalisability.** A second candidate profile would test whether the rubric weights generalise or require recalibration per-domain.
2. **Predictive validity.** Track B will determine whether construct validity translates to behavioural validity.
3. **Group-level patterns.** Aggregate data across multiple candidates using the same system would enable calibration studies — does a score of 70 correspond to a predictable callback rate?

### Cross-References

> **→ Framework:** [→ FW §18] — Framework conclusions and open research questions

---

## References

1. Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1), 37–46. https://doi.org/10.1177/001316446002000104

2. Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association*, 22(158), 209–212. https://doi.org/10.2307/2276774

3. Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques. *ACM Transactions on Information Systems*, 20(4), 422–446. https://doi.org/10.1145/582415.582418

4. Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: Uses in assessing rater reliability. *Psychological Bulletin*, 86(2), 420–428. https://doi.org/10.1037/0033-2909.86.2.420. Interpretation scale: Koo, T. K., & Mae, M. Y. (2016). *Journal of Chiropractic Medicine*, 15(2), 155–163.

5. McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153–157. https://doi.org/10.1007/BF02295996

6. Dubois, Y., Galambosi, B., Liang, P., & Hashimoto, T. (2024). Length-Controlled AlpacaEval: A simple way to debias automatic evaluators. arXiv:2404.04475. https://arxiv.org/abs/2404.04475

7. Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. arXiv:2306.05685. https://arxiv.org/abs/2306.05685

8. Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical data. *Biometrics*, 33(1), 159–174. https://doi.org/10.2307/2529310

9. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML 2017*. arXiv:1706.04599. https://arxiv.org/abs/1706.04599

10. Fazl, A., & Buckley, C. (2024). Human and LLM-based resume matching: An observational study. arXiv:2407.12141. https://arxiv.org/abs/2407.12141

11. Beutel, A., Chen, J., Doshi, T., et al. (2019). Fairness in recommendation ranking through pairwise comparisons. *KDD 2019*.
