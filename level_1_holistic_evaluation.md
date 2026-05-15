# Level 1 — Holistic Evaluation

---

## 1. Purpose

Level 1 treats the LLM as a black box. The CV and JD enter the same prompt and a single fit score comes out. This is the simplest scoring design and the one a typical "LLM-as-a-judge" pipeline implements.

It establishes the **baseline measurement properties** (reliability, validity, discrimination) that Levels 2 and 3 must match or exceed to justify their added complexity.

---

## 2. Method

```
CV + JD
    ↓
  Prompt (one of P0 / P1 / P2)
    ↓
  LLM → one number + one verdict
```

Three prompt variants run with the same model (Gemini 3.1 Flash Lite preview, temperature = 1.0, 3 runs per pair):

| Prompt | What it asks | LLM output |
|---|---|---|
| **P0** | Minimal — score the fit | `score` ∈ {0, 1, 2, 3} + verdict ∈ {Skip, Consider, Apply} |
| **P1** | Rubric — skills 60 %, experience 30 %, education 10 % (no explicit subscores) | `score` ∈ {0, 1, 2, 3} + verdict ∈ {Skip, Consider, Apply} |
| **P2** | Rubric + a 0–100 production scale | `score` (0–3) + `score_100` (0–100) + verdict |

---

## 3. What it isolates

| Comparison | Question |
|---|---|
| P0 vs P1 | Does adding an explicit rubric move agreement with human judgment? |
| P0/P1 vs P2 | Does the 0–100 scale carry information that the 0–3 scale collapses? |

---

## 4. Dataset

* **JD pool** — 75 unique JDs drawn from a 200-JD reserve (32 + 18 + 25). Three subsets are used:
  * `main` — 32 psychometric JDs (P0, P1, P2)
  * `hr_extra` — 18 HR JDs (P2 only)
  * `engineer_extra` — 25 engineering JDs (P2 only)
* **CVs** — three profiles: CV-1 *Assessment Scientist* (`cv_primary`), CV-2 *Head of People* (`cv_hr`), CV-3 *Solution Architect* (`cv_engineer`).
* **Runs** — 3 per (JD × CV × prompt).
* **Human labels** — `cv_primary × main` only, n = 32, single rater (the project author), labels ∈ {0, 1, 2, 3}.
* **Filtered main subset for discriminant validity** — the 16 JDs in `main` where `cv_primary` received `human_holistic_label ≥ 2`. This removes deliberately irrelevant items that compress effect sizes through floor overlap.

---

## 5. Status

Complete. Results loaded into `notebooks/analysis.ipynb` §1.

---

## 6. Findings (detailed)

This section mirrors the notebook (§1.2 — §1.5) with the corresponding tables and figures, and adds a token-spending section that is not in the notebook.

### 6.1 Score distributions

All three prompts show similar distribution shapes. P0 and P1 were run on a sample of 32 JDs (the `main` pool), most of which are aligned with the Assessment Scientist profile. Across both prompts the same pattern is visible: `cv_primary` receives systematically higher scores than the other two CVs on these JDs. P2 extends the pool to 75 JDs by adding HR-related and engineering-related positions, which gives `cv_hr` and `cv_engineer` a domain where they can score in the upper range.

The clustering at 0 on irrelevant pairs is informative for the reliability/validity reads further down: a system that mostly outputs 0 on non-fits will earn high reliability on that majority class without demonstrating sensitivity to fit gradations.

Reference figure: `results/figures/L1_score_dist_curves.png` (3-panel score curves per prompt, with sample sizes per panel).

### 6.2 Reliability (test–retest, 3 runs)

Two ICC indices are reported side-by-side because the operational unit matters:

| Index | What it measures | When to use |
|---|---|---|
| **ICC(A,1)** — single-measure, two-way random, absolute agreement | Reproducibility of one LLM call | Production decisions ride on one call → this is the operational metric |
| **ICC(A,k)**, k = 3 | Reproducibility of the *mean* of 3 calls | Textbook reporting form; conservatively reported alongside (A,1) |

Why two-way *random*: each JD × CV × prompt cell is one of many possible items the system could be presented with; we want to generalise over items. The three "raters" are three stochastic re-prompts of the *same* model — not three independent judges. ICC(A,1) is the more honest figure; ICC(A,k) is always ≥ (A,1) by construction.

**Results (pooled across CVs):**

| Prompt | Scale | n_subjects | ICC(A,1) | ICC(A,k) | SEM |
|---|---|---|---|---|---|
| P0 | 0–3 | 96 | 0.953 | 0.984 | 0.124 |
| P1 | 0–3 | 96 | 0.971 | 0.990 | 0.097 |
| P2 | 0–3 | 228 | 0.984 | 0.995 | 0.070 |
| P2 | 0–100 | 228 | 0.992 | 0.997 | 1.355 |

n_subjects differs by prompt because P0 and P1 were run only on the Psychometric pool (32 JDs × 3 CVs = 96 (JD, CV) pairs), while P2 covers all three pools. The reliability ICC for P2 was computed on the full L1 master sample (76 JDs × 3 CVs = 228 pairs, including one HR JD that was later dropped from L2 for a Gemini 500 error). All cross-level analyses use the matched 75-JD subset stated in §4; the 1-row difference between 228 and 225 does not materially affect the ICC.

**Verdict reliability (Fleiss κ over 3 runs):**

| Prompt | n_subjects | Fleiss κ | % unanimous | % majority |
|---|---|---|---|---|
| P0 | 96 | 0.913 | 94 % | 100 % |
| P1 | 96 | 0.956 | 97 % | 100 % |
| P2 | 228 | 0.968 | 98 % | 100 % |

All values exceed the "excellent" threshold (ICC ≥ 0.90) and "almost perfect" (κ > 0.80) on standard cutoffs.

**Key reads:**

* Single-call reliability is excellent across all prompts; ICC(A,1) increases by +0.018 from P0→P1 and by +0.013 from P1→P2.
* Verdict consistency is highest under P2 (Fleiss κ = 0.968, 98 % unanimous).
* **Caveat — restricted variance.** Most engineer-CV × non-engineer-JD pairs cluster at 0. "The LLM agrees on doesn't-fit three times" produces high ICC under low variance. ICCs on the 0–3 scale should be read as upper bounds; the ordering P0 < P1 < P2 carries more information than the absolute values.

Reference figure: `results/analysis/icc_heatmap.png` (ICC(A,1) by prompt × CV).

### 6.3 Criterion validity (vs human labels)

n = 32, single rater, on `cv_primary × main` only. Treated as a sanity check, not a confirmatory study.

**Summary table:**

| Prompt | Spearman ρ | Weighted κ | MAE ↓ | Bias (LLM−human) | Exact match |
|---|---|---|---|---|---|
| L1 P0 | 0.823 [0.67, 0.91] | 0.520 [0.32, 0.67] | 0.562 | +0.44 (d = 0.65, medium) | 46.9 % |
| L1 P1 | 0.821 [0.66, 0.91] | 0.514 [0.32, 0.67] | 0.594 | +0.47 (d = 0.70, medium) | 43.8 % |
| L1 P2 | **0.849** [0.71, 0.92] | **0.623** [0.44, 0.77] | **0.469** | +0.41 (d = 0.66, medium) | **56.2 %** |

All bias estimates significant under Wilcoxon (p < 0.005 in each prompt), with Cohen's d in the medium range — the LLM systematically rates slightly higher than the human rater. The bias is roughly constant across prompts (0.41–0.47), so prompt choice does not eliminate it.

**Additional reading:**

* "% within ±1" = 96.9 % in all three prompts, but on a 4-level scale ±1 covers 67 % of all possible distances; this metric is lenient by construction and is reported only to show that no large mismatches occur.
* On the absolute-agreement metrics, P2 shows the highest κ and exact match and the lowest (best) MAE; P0 is intermediate; P1 is worst on all three. This suggests that the 0–100 production prompt produces the most aligned 0–3 output, even though all three prompts share the same 0–3 codomain.

**How to read these results — caveat on the human gold labels.**

All criterion-validity numbers in this section should be interpreted with caution: the human labels come from a single rater who is also the owner of `cv_primary`. The protocol applied several mitigations to reduce rater noise and self-judgment artefacts:

* every JD was seen for the first time during labeling (no prior exposure during system development);
* company names were removed from the JD text before rating, to limit halo effects from familiar employers;
* a written labeling rubric on a 4-point scale was used, with definitions of each level, to reduce within-rater inconsistency.

Even so, what this section measures is best framed as **"does the LLM rate fit the way the CV owner rates fit on her own profile"**, not as agreement with a population of independent expert raters. The ρ ≈ 0.82–0.85 and κ ≈ 0.51–0.62 values are evidence that the LLM produces a stable signal that tracks one informed person's holistic judgment, not evidence of agreement with a generic "true fit" criterion. Independent multi-rater annotation, performed at the rubric/dimension level by raters who do not own the candidate CV, is the planned next step and is required before any confirmatory claim about criterion validity can be made.

**Confusion pattern** (mode of 3 runs per JD, mapped to Skip / Consider / Apply):

* Misses are concentrated in the "Consider when human said Skip" cell — mild over-rating, consistent with the +0.4 bias.
* No "Apply" verdicts when the human said "Skip" in any prompt: the LLM does not fabricate strong fits.
* No "Skip" verdicts when the human said "Apply" in any prompt: the LLM does not miss strong fits either.

Reference figure: `results/analysis/L1_confusion_by_prompt.png` (3-panel confusion heatmap by prompt).

### 6.4 Discriminant validity (between-CV differences)

**Question:** does the system assign different scores to different CVs on the same JD, and does the ranking flip with JD domain?

**Method:** Friedman χ² + Kendall's *W* as the omnibus test (non-parametric, appropriate for ordinal scales), pairwise Wilcoxon signed-rank with rank-biserial *r* effect size for follow-up contrasts.

**Bonferroni correction** per JD-pool family:

| Family | Contrasts | α |
|---|---|---|
| `main` (P0 + P1 + P2 × 3 CV-pairs) | 9 | 0.0056 |
| `hr_extra` / `engineer_extra` (P2 × 3 CV-pairs each) | 3 | 0.0167 |

**Friedman omnibus + Kendall's W:**

| Pool | Prompt | n | χ² | p | Kendall's W | Interpretation |
|---|---|---|---|---|---|---|
| Psychometric pool | P0 | 16 | 26.14 | < 0.001 | **0.817** | large |
| Psychometric pool | P1 | 16 | 25.30 | < 0.001 | **0.791** | large |
| Psychometric pool | P2 | 16 | 25.30 | < 0.001 | **0.791** | large |
| HR pool | P2 | 18 | 25.59 | < 0.001 | 0.674 | large |
| Engineering pool | P2 | 25 | 27.79 | < 0.001 | 0.556 | large |

All omnibus tests are significant at p < 0.001 in all conditions.

**Pairwise contrasts (P2 only, for compactness — same pattern in P0 and P1):**

| Pool | Contrast | Δ mean | p | rank-biserial r | Significant under Bonferroni? |
|---|---|---|---|---|---|
| Psychometric pool (main) | cv_primary − cv_hr | +1.625 | 0.0010 | 0.950 | ✓ |
| Psychometric pool (main) | cv_primary − cv_engineer | +2.312 | 0.0003 | 1.000 | ✓ |
| Psychometric pool (main) | cv_hr − cv_engineer | +0.688 | 0.013 | 0.855 | ✗ (both non-fits, floor overlap expected) |
| HR pool (hr_extra) | cv_primary − cv_hr | −0.368 | 0.241 | −0.324 | ✗ (both plausible HR fits, expected) |
| HR pool (hr_extra) | cv_primary − cv_engineer | +1.263 | 0.0002 | 1.000 | ✓ |
| HR pool (hr_extra) | cv_hr − cv_engineer | +1.632 | 0.0002 | 1.000 | ✓ |
| Engineering pool (engineer_extra) | cv_primary − cv_hr | +0.480 | 0.0027 | 0.857 | ✓ |
| Engineering pool (engineer_extra) | cv_primary − cv_engineer | −0.600 | 0.0079 | −0.771 | ✓ |
| Engineering pool (engineer_extra) | cv_hr − cv_engineer | −1.080 | 0.0001 | −1.000 | ✓ |

**3-way flip test (mean LLM score by CV per pool, P2):**

| Pool | cv_primary | cv_hr | cv_engineer | Expected leader | Observed leader |
|---|---|---|---|---|---|
| main | **2.56** | 0.94 | 0.25 | cv_primary | ✓ cv_primary |
| hr_extra | 1.42 | **1.79** | 0.16 | cv_hr | ✓ cv_hr |
| engineer_extra | 0.56 | 0.08 | **1.16** | cv_engineer | ✓ cv_engineer |

The expected leader matches the observed leader in all three pools. The CV ordering reverses correctly with the JD domain.

**Key reads:**

* Kendall's *W* falls between 0.56 and 0.82 across the three pools — large discrimination effects by Cohen's convention (W ≥ 0.5).
* The expected leader CV scores highest in every pool, and the ordering reverses with JD domain (Psychometric → Assessment Scientist CV; HR → HR CV; Engineering → Engineering CV). Together with the omnibus effect sizes, this is the within-Level-1 evidence that the score responds to CV–JD fit rather than to a fixed CV identity. A test of difference against an alternative explanation (e.g., constant CV-level priors) was not performed.
* The two non-significant pairwise contrasts under Bonferroni — cv_hr − cv_engineer in the Psychometric pool, and cv_primary − cv_hr in the HR pool — are pairs where both CVs are either non-fits or both plausible fits, so floor / ceiling overlap is expected by the dataset design (see §4 and §6.4 "Note on pool asymmetry").

**Note on pool asymmetry.** The Psychometric JD pool was filtered post-hoc to retain only JDs that received a human label of 2 or 3 (Consider / Apply on `cv_primary`), so it is the most curated and most representative of "real fit candidates" of the three pools. The HR and Engineering pools were selected by role-title keywords without human labels and without filtering by content, so they contain a wider mix of relevance levels and a smaller proportion of strong-fit positions. This asymmetry partly explains why the Psychometric pool shows the cleanest separation between CVs.

Even with that caveat, the cross-pool overlap patterns are themselves informative and intentional — the three CV profiles were designed to create partial cross-domain overlap, not full separation:

* On **HR JDs**, Assessment Scientist CV scores meaningfully (mean 1.42 / 0–3; 49.6 / 0–100), only marginally below HR CV (1.79 / 60.1). Psychometrics and HR share people-process and assessment vocabulary, so this partial overlap is expected.
* On **Engineering JDs**, HR CV collapses to near zero (0.08 / 0–3; 8.5 / 0–100), while Assessment Scientist CV holds a higher mean (0.56 / 0–3; 29.3 / 0–100). Psychometrics shares Python and automation skills with engineering; HR does not. The system reflects this asymmetry correctly.

The fact that the system reproduces these expected overlap patterns — not just the trivial winners — is additional evidence that the score is reading content rather than CV identity.

Reference figure: `results/figures/L1P2_mean_per_cv.png` (2-panel mean score per CV across pools, 0–3 and 0–100 scales).

### 6.5 Token spending and efficiency

| Prompt | Mean tokens / call | Mean latency / call (s) | Notes |
|---|---|---|---|
| L1 P0 | 2,103 | 8.24 | minimal prompt, smallest payload |
| L1 P1 | 2,215 | **5.96** | adds rubric instructions, fastest in practice |
| L1 P2 | 2,593 | 7.76 | adds 0–100 scale and verdict block |

**Read:**

* Adding the rubric (P0 → P1) raises tokens by ~5 % and *lowers* latency in the measured runs. Latency is driven more by output structure than input length within this prompt range; the rubric appears to constrain output length.
* Adding the 0–100 scale (P1 → P2) raises tokens by ~17 % and latency by ~30 %, reflecting the longer verdict block P2 emits.

These numbers are an order-of-magnitude reference, not a benchmark — they depend on tokenizer, retry behavior, and concurrency settings at run time.


## 7. Limitations

### 7.1 Measurement-level (affect how to read the numbers)

1. **Scale compression inflates reliability.** The 0–3 scale has only four points and many JD × non-fit-CV pairs cluster at 0 (cv_engineer on psychometric JDs, cv_hr on engineering JDs, etc.). High ICCs partly reflect agreement on a majority class with low variance, not sensitivity to fit gradations. The 0–100 scale (P2) is the more conservative reading (ICC(A,1) = 0.992 vs ICC(A,1) = 0.984 on 0–3 for P2). Interpret the 0–3 ICCs as upper bounds.

2. **"% within ±1" is lenient by construction.** On a 4-level scale, ±1 covers 67 % of all possible distances. The 96.9 % values reported in §6.3 should not be read as agreement; exact match (44–56 %) and Weighted κ (0.51–0.62) are the stricter views.

3. **The three "raters" in ICC are stochastic re-runs of one model**, not independent judges. Two-way random ICC is still appropriate (JDs are sampled from a population of possible items), but the model-side variance is by construction smaller than what cross-rater ICCs assume. ICC(A,1) is reported alongside ICC(A,k) for this reason; ICC(A,1) is the operational metric.

4. **Pool-construction asymmetry in discriminant validity.** The three JD pools were not built by the same rule. The Psychometric pool was filtered post-hoc to retain only the 16 JDs where `cv_primary` received a human label ≥ 2 (Consider / Apply) — this filtering was applied because the full 32-JD `main` pool deliberately includes irrelevant items that compress effect sizes through floor overlap (see §4 of this doc). The HR (n = 18) and Engineering (n = 25) pools were selected by role-title keywords without human labels and without any content-based filtering, so they retain a wider mix of relevance levels and a smaller proportion of strong-fit positions. As a result, the Psychometric pool is more curated than the other two, and its cleaner CV separation in §6.4 is partly a consequence of the filtering, not only of the system's behavior. A more symmetric design would either apply the same human-label filter to all three pools (requires human labels on HR and Engineering JDs, currently unavailable) or skip filtering entirely (and accept the floor-overlap effect). Until that is in place, cross-pool comparisons of effect sizes should be read as descriptive, not as confirmatory evidence that the system discriminates equally well in every domain.


### 7.2 Design-level (affect how strong the inference can be)

5. **Single rater for the criterion labels** (n = 32). The author is both the system developer and the human annotator. Confirmation bias cannot be ruled out. Independent re-annotation by an external rater is planned; current criterion-validity claims are descriptive, not confirmatory.

6. **Criterion labels only on `cv_primary × main`.** No human labels exist for `cv_hr × hr_extra` or `cv_engineer × engineer_extra`. The 3-way flip test substitutes for cross-CV human labels by checking that the *expected leader* CV scores highest on each pool — but a flip test is a weaker form of evidence than rated agreement with humans on every cell.

7. **No held-out test set.** The same 32 JDs drive reliability, criterion validity, and (post-filtering) discriminant validity. Over-fitting risk is low because no parameters are learned, but the absence of held-out data means we cannot quantify out-of-sample generalisation.

8. **Single LLM checkpoint** (Gemini 3.1 Flash Lite preview, temperature = 1.0). Whether the reliability, validity, and discriminant patterns reported here generalise to other LLMs is unknown. Cross-model robustness is planned future work.

9. **Construct definition is informal.** "Overall fit" is operationalised by the rubric in P1 (skills 60 % / experience 30 % / education 10 %), drawn from internal product-expert heuristics. It is not derived from a published competency framework or a job-analysis study. The rubric is a working definition, not a measurement standard. The same weights are used in Level 2 and planned Level 3, so any construct misalignment propagates across levels.

### 7.3 Coverage gaps in validity evidence

10. **Content validity not assessed.** Whether the rubric (skills / experience / education) and the prompts cover the relevant facets of job-candidate fit is asserted, not tested. A formal content-validity study would require expert review of the rubric against actual job-analysis output for each role.

11. **Convergent / divergent validity not assessed.** The system is not compared against *other* fit measures (e.g., a structured interview score, a hiring decision outcome, a separate competency match score). Without an external convergent benchmark, agreement-with-a-single-human-rater is the only criterion in play.

12. **Predictive validity not assessed.** The score is not linked to any downstream hiring outcome (interview pass-through, offer rate, retention). Predictive validity is the gold-standard form of criterion evidence for selection instruments under EEOC / EU AI Act guidance; this project does not have access to such outcomes.

13. **Fairness / subgroup analysis absent.** No analysis of company-name bias, role-seniority bias, gendered language in JDs, or other demographic proxies. The system is not yet evaluated against the fairness criteria required for high-risk AI under the EU AI Act and NYC Local Law 144.

### 7.4 Scope and data caveats

14. **English only.** No multilingual JDs or CVs were tested.

15. **Training-data contamination cannot be ruled out.** Public JDs and the author's own public CV may have appeared in the LLM's pre-training data. There is no way to detect or quantify this with a closed-weight commercial model.

