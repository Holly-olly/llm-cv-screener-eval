# Level 2 — Guided Evaluation (Bridge)

---

## 1. Purpose

Level 1 treats the LLM as a black box — CV and JD enter together, one score comes out. Level 3 removes the LLM from judgment entirely.

Level 2 sits between them: the LLM still performs the evaluation, but is constrained to output **explicit categorical labels per dimension** rather than a holistic score. Aggregation is then done by code using a fixed formula.

This isolates two questions:
- Can structured output labels from an LLM better align with human judgment than a single holistic score?
- Does the source of the judgment (LLM vs similarity) matter, when the aggregation formula is identical?

The deeper motivation is **auditability**: in Level 1 the LLM's reasoning is opaque; in Level 2 you can inspect *which dimension caused* a low score, and re-weight the formula without re-querying the LLM.

---

## 2. Method

```
CV + JD
    ↓
  Prompt (L2_P0)
    ↓
  LLM → categorical labels per dimension:

    SKILLS:              YES | PARTIAL | NO
    EXPERIENCE
      role_relevance:    SAME | SIMILAR | DIFFERENT
      domain_relevance:  SAME | RELATED | UNRELATED
    EDUCATION:           YES | PARTIAL | NO
    HOLISTIC (independent): STRONG | MODERATE | WEAK | NO FIT
                            + HIGH | LOW confidence

    ↓
  Scoring maps applied by code:
    YES → 1.0 / PARTIAL → 0.5 / NO → 0.0
    SAME → 1.0 / SIMILAR → 0.6 / DIFFERENT → 0.0
    SAME → 1.0 / RELATED → 0.5 / UNRELATED → 0.0

    ↓
  score_0-100 = 100 × ( 0.6 · skill + 0.3 · (0.4 · role + 0.6 · domain) + 0.1 · edu )
```

`HOLISTIC` is collected independently — it is not derived from the dimension labels. This gives us two scores per pair: a code-computed `fit_score_100` (further referred to as `score_0-100`) and an LLM-self-reported `holistic_score` (further referred to as `score_0-3`, mapped from STRONG/MODERATE/WEAK/NO FIT) — same convention as at Level 1.

**Construct.** Same construct as Level 1: overall candidate-to-role fit, decomposed into skills, role similarity, domain similarity, and education. L1 lets the LLM weight these implicitly; L2 makes the weights explicit and code-deterministic.

**Weights footnote.** The formula weights (0.6 / 0.12 / 0.18 / 0.1) follow internal product-expert heuristics consistent with the rubric used in L1 P1. **No empirical optimisation has been performed.** Sensitivity analysis is listed under future work.

---

## 3. What it isolates

| Comparison | Question |
|---|---|
| `score_0-3` vs `score_0-100` | Does the LLM's holistic judgment agree with the code-aggregated formula on the same labels? Where they disagree, which one tracks human judgment better? |
| L2 `score_0-3` vs L1 P2 `score_0-3` | Does the structured-label prompt change measurement properties relative to the holistic prompt? (Cross-design comparison — see §6.5.) |

---

## 4. Dataset

* **JD pools** — 75 unique valid JDs (32 + 18 + 25). At Level 1 the same JD set has 19 HR JDs; at Level 2 one HR JD failed with a Gemini 500 error and was not retried, so the L2 sample is 75 rather than 76:
  * `main` — 32 JDs, majority psychometric roles.
  * `hr_extra` — 18 HR JDs (one fewer than L1).
  * `engineer_extra` — 25 engineering JDs.
* **CVs** — same three as Level 1: CV-1 *Assessment Scientist* (`cv_primary`), CV-2 *Head of People* (`cv_hr`), CV-3 *Solution Architect* (`cv_engineer`).
* **Runs** — 3 per (JD × CV); single prompt (no P0/P1/P2 variants at this level).
* **Total valid calls** — 674 of 675.
* **Human labels** — `cv_primary × main` only, n = 32, single rater (same set as Level 1).
* **Filtered main subset for discriminant validity** — the 16 JDs in `main` where `cv_primary` received `human_holistic_label ≥ 2`. Same filtering rule as Level 1, to remove deliberately irrelevant items.

---

## 5. Status

Complete. Results loaded into `notebooks/analysis.ipynb` §2 and §3 (comparison).

---

## 6. Findings (detailed)

### 6.1 Score distributions

The system was queried on the full 75-JD pool with all three CV profiles. Two distributions matter at this level: the categorical `holistic` label that the LLM emits directly, and the continuous `score_0-100` that code computes from the four dimension labels.

* **`holistic` label** (NO FIT 0 / WEAK 1 / MODERATE 2 / STRONG 3) — strongly tied to the CV–JD match. On the Assessment Scientist CV × Psychometric JDs, the mode is MODERATE / STRONG. On the Engineering CV × Psychometric JDs, almost all labels are NO FIT / WEAK. The flip pattern is visible at the label level before any aggregation.
* **`score_0-100`** — bimodal in practice: clusters at ~10–20 (when skills are NO and role is DIFFERENT, the formula forces a low number) and at ~60–90 (when skills are YES and role/domain align). Few middle scores around 30–50.

Reference figure: `results/figures/L2_score_dist_curves.png` (2-panel score curves — score_0-3 and score_0-100, by CV).

### 6.2 Reliability (test–retest, 3 runs)

Same ICC framework as Level 1 (single-measure ICC(A,1) and average-measure ICC(A,k)). Pooled across CVs:

| Output | n_subjects | ICC(A,1) [95 % CI] | ICC(A,k) [95 % CI] |
|---|---|---|---|
| `score_0-3` (0–3) | 224 | 0.967 [0.96; 0.97] | 0.989 [0.99; 0.99] |
| `score_0-100` (continuous) | 224 | 0.968 [0.96; 0.97] | 0.989 [0.99; 0.99] |

**Categorical label reliability (Fleiss κ over 3 runs):**

| Label | Fleiss κ |
|---|---|
| `holistic` (NO FIT / WEAK / MODERATE / STRONG) | 0.905 |
| `skills` (YES / PARTIAL / NO) | 0.91 |
| `role_relevance` (SAME / SIMILAR / DIFFERENT) | 0.87 |
| `domain_relevance` (SAME / RELATED / UNRELATED) | 0.92 |
| `education` (YES / PARTIAL / NO) | 0.96 |

All values exceed the "excellent" threshold (ICC ≥ 0.90) and "almost perfect" (κ > 0.80).

**Key reads:**

* The four dimension labels reproduce at κ ≈ 0.87–0.96 across runs.
* `score_0-100` is a deterministic linear transform of those four labels. Its ICC is mathematically bound to the label-level reliability — it is not an independent empirical check, only a transparency confirmation that the code aggregation introduces no extra noise.
* **`confident` flag is degenerate.** Of 674 valid rows, 673 are `confident = True` (99.85 %). Not informative in this dataset; kept in the schema for possible future routing use.


### 6.3 Criterion validity (vs human labels)

Same 32 pairs as Level 1 (Assessment Scientist CV × Psychometric JDs), same single rater. Metrics computed on `score_0-3` (the LLM holistic label mapped to integer 0–3), comparable to Level 1's 0–3 metrics.

**Summary table:**

| Metric | Value (95 % CI) | Interpretation |
|---|---|---|
| Spearman ρ | **0.869** [0.75, 0.93] | very strong |
| Weighted κ (linear) | 0.517 [0.34, 0.67] | moderate |
| MAE | 0.594 | — |
| Bias (LLM − human) | **+0.594** | Cohen's d = 0.97, **large** |
| % exact match | 46.9 % | — |
| % within ±1 | 93.8 % | — |

All values are on the 0–3 scale, directly comparable with Level 1.

**Key reads:**

* Rank correlation with humans is the highest in the project (ρ = 0.869, above all three Level 1 prompts). The 0.020 gap to L1 P2 (ρ = 0.849) sits within overlapping bootstrap 95 % CIs; no formal test of difference between systems was performed, so this is a point-estimate ordering, not a confirmatory claim.
* Absolute agreement is moderate (κ = 0.517) and lower than L1 P2 (κ = 0.623). The drop is concentrated in the bias term.
* **Systematic positive bias is large**. The LLM rates ~0.6 of a scale point higher than the human rater on average. This is roughly 50 % larger than Level 1's bias (which sits at d ≈ 0.65 across all three prompts).
* "% within ±1" = 93.8 % is lenient on a 4-level scale — ±1 covers 67 % of all possible distances on a 4-level criterion, so this metric is lenient by construction; the stricter exact-match metric (46.9 %) is the more honest view.

**How to read these results — caveat on the human gold labels.**

All criterion-validity numbers in this section should be interpreted with caution: the human labels come from a single rater who is also the owner of the Assessment Scientist CV. The protocol applied several mitigations to reduce rater noise and self-judgment artefacts:

* every JD was seen for the first time during labeling (no prior exposure during system development);
* company names were removed from the JD text before rating, to limit halo effects from familiar employers;
* a written labeling rubric on a 4-point scale was used, with definitions of each level, to reduce within-rater inconsistency.

Even so, what this section measures is best framed as **"does the LLM rate fit the way the CV owner rates fit on her own profile"**, not as agreement with a population of independent expert raters. Independent multi-rater annotation, performed at the rubric/dimension level by raters who do not own the candidate CV, is the planned next step and is required before any confirmatory claim about criterion validity can be made.

**Confusion pattern** (mode of 3 runs per JD, LLM holistic label mapped identically with human labels):

* Exact diagonal: 20/32 = 62.5 %.
* All errors are above the diagonal — the LLM never assigns a lower verdict than the human.
* The dominant error cell is "Consider when human said Skip" (n = 8), driving the positive bias.
* All 6 human-Apply cases are caught (no human-Apply → LLM-Skip transitions).

Reference figure: `results/analysis/L2_confusion.png`.

### 6.4 Discriminant validity (between-CV differences)

Same method as Level 1 (Friedman + Kendall's *W* omnibus, pairwise Wilcoxon with rank-biserial *r*, RM-ANOVA on `score_0-100` as parametric sensitivity check).

**Bonferroni correction** per JD-pool family — 6 contrasts per family (2 score columns × 3 CV-pairs) → α = 0.05 / 6 = **0.0083**.

**Friedman omnibus + Kendall's W:**

| Pool | Score | n | χ² | p | Kendall's W | Interpretation |
|---|---|---|---|---|---|---|
| Psychometric JDs | score_0-3 | 16 | 21.97 | < 0.001 | **0.686** | large |
| Psychometric JDs | score_0-100 | 16 | 20.10 | < 0.001 | 0.628 | large |
| HR pool | score_0-3 | 18 | 26.36 | < 0.001 | 0.732 | large |
| HR pool | score_0-100 | 18 | 28.33 | < 0.001 | **0.787** | large |
| Engineering pool | score_0-3 | 25 | 35.22 | < 0.001 | **0.704** | large |
| Engineering pool | score_0-100 | 25 | 20.51 | < 0.001 | 0.410 | medium |

**RM-ANOVA sensitivity check (`score_0-100`, Greenhouse–Geisser corrected):**

| Pool | n | F | p | partial η² |
|---|---|---|---|---|
| Psychometric JDs | 16 | 26.5 | < 0.001 | 0.576 — large |
| HR pool | 18 | 38.2 | < 0.001 | 0.619 — large |
| Engineering pool | 25 | 17.2 | < 0.001 | 0.294 — large |

Friedman and RM-ANOVA agree in magnitude class.

**3-way flip test (mean `score_0-100` per CV per pool):**

| Pool | cv_primary | cv_hr | cv_engineer | Expected leader | Observed leader |
|---|---|---|---|---|---|
| Psychometric JDs | **75.1** | 31.7 | 19.4 | cv_primary | ✓ cv_primary |
| HR pool | 54.9 | **63.4** | 14.5 | cv_hr | ✓ cv_hr |
| Engineering pool | 30.7 | 12.7 | **42.8** | cv_engineer | ✓ cv_engineer |

The expected leader matches the observed leader in all three pools, as it does at Level 1.

**Pairwise contrasts of interest (failing Bonferroni):**

* Psychometric JDs on `score_0-100`, HR CV − Engineering CV (Δ = +12.3, *p* = 0.17) — both are non-fits for psychometric JDs, expected floor overlap.
* HR pool on `score_0-100`, Assessment Scientist CV − HR CV (Δ = −8.5, *p* = 0.18) — both profiles are plausible HR fits, expected ceiling overlap. The Assessment Scientist CV led here under `score_0-3` but the formula-derived `score_0-100` reverses the leader. This is informative: the LLM's holistic judgment finds a subtle ordering that the fixed-weight formula collapses.
* Engineering pool on `score_0-100`, Assessment Scientist CV − Engineering CV (Δ = −12.2, *p* = 0.08) — the Engineering CV leads, but not by enough to clear Bonferroni at α = 0.0083.

**Key reads:**

* All omnibus tests significant; all flip-test orderings correct.
* `score_0-3` (LLM-judged) effect sizes are uniformly large (W = 0.69–0.73).
* `score_0-100` (formula-aggregated) effect sizes are more variable (W = 0.41–0.79). The drop on the Engineering pool (0.704 → 0.410) is the largest, and it correlates with the Assessment Scientist CV − Engineering CV contrast failing Bonferroni. This suggests the fixed weights may be sub-optimal — the LLM's holistic call separates these CVs cleanly, but the code-aggregated score does not. A formal weight-sensitivity analysis is pending; until then this is suggestive, not confirmatory.

**Note on pool asymmetry.** The Psychometric JD pool was filtered post-hoc to retain only the 16 JDs that received a human label of 2 or 3 (Consider / Apply on the Assessment Scientist CV at Level 1), so it is the most curated and most representative of "real fit candidates" of the three pools. The HR (n = 18) and Engineering (n = 25) pools were selected by role-title keywords without human labels and without filtering by content, so they contain a wider mix of relevance levels and a smaller proportion of strong-fit positions. This asymmetry partly explains why the Psychometric pool shows the cleanest separation between CVs on `score_0-3`, and contributes to the inconsistent `score_0-100` effect sizes across pools.

Even with that caveat, the cross-pool overlap patterns are themselves informative and intentional — the three CV profiles were designed to create partial cross-domain overlap, not full separation:

* On **HR pool**, Assessment Scientist CV scores meaningfully on `score_0-3` (mean 1.78), only marginally below HR CV (2.11). Psychometrics and HR share people-process and assessment vocabulary, so this partial overlap is expected. The `score_0-100` formula partially reverses this ordering, which is one of the markers that the fixed weights are sub-optimal.
* On **Engineering pool**, HR CV collapses to near zero (`score_0-3` 0.12, `score_0-100` 12.7), while Assessment Scientist CV holds a meaningfully higher value (1.12, 30.7). Psychometrics shares Python and automation skills with engineering; HR does not. The system reflects this asymmetry correctly on both score outputs.

The fact that the system reproduces these expected overlap patterns — not just the trivial winners — is consistent with the score responding to content rather than only to CV identity. Alternative explanations (e.g., constant CV-level priors interacting with JD vocabulary) are not formally ruled out.

Reference figure: `results/figures/L2_mean_per_cv.png` (2-panel mean score per CV across pools — `score_0-3` and `score_0-100`).

### 6.5 Comparison with Level 1 (cross-design)

L1 P2 is the closest analogue to L2 P0 — both produce a 0–3 score and (for P2) a 0–100 score, both target the same construct. Comparisons computed on all overlapping JD × CV pairs (n = 225 for 0–100, 228 for 0–3) regardless of human labels.

**Method-to-method agreement (L1 P2 vs L2 P0):**

| Metric | Value (95 % CI) |
|---|---|
| ICC(A,1) on 0–100 scale | **0.906** [0.880, 0.930] |
| Pearson r on 0–100 | 0.907 |
| Spearman ρ on 0–100 | 0.861 [0.81, 0.90] |
| Weighted κ on 0–3 | 0.699 |
| Paired t-test on 0–100 | Δ = −0.95, *p* = 0.21 |
| MAE on 0–100 | 9.30 |

**Reads:**

* ICC(A,1) = 0.906 between methods is excellent — the two designs produce highly aligned scores at the JD × CV level.
* No significant mean shift between L1 P2 and L2 P0 on 0–100 (paired *t*, *p* = 0.21).
* Bland–Altman plot (`results/analysis/L1P2_vs_L2P0_BlandAltman.png`) shows mean difference ≈ −1 and limits of agreement covering ±20 — the two methods agree closely on average but with notable spread on individual JDs.

**Validity comparison against the same human labeled set (n = 32, Assessment Scientist CV × Psychometric JDs):**

| System | Spearman ρ | Weighted κ | MAE ↓ | Bias | Exact match |
|---|---|---|---|---|---|
| L1 P0 | 0.823 | 0.520 | 0.562 | +0.44 | 46.9 % |
| L1 P1 | 0.821 | 0.514 | 0.594 | +0.47 | 43.8 % |
| **L1 P2** | 0.849 | **0.623** | **0.469** | +0.41 | **56.2 %** |
| **L2 P0** | **0.869** | 0.517 | 0.594 | +0.59 | 46.9 % |

**Discriminant comparison on filtered main (n = 16):**

| System | Kendall's W |
|---|---|
| L1 P0 | 0.817 |
| L1 P1 | 0.791 |
| **L1 P2** | **0.791** |
| **L2 `score_0-3`** | 0.686 |
| L2 `score_0-100` | 0.628 |

**Bottom line.**

* **Rank-order alignment with humans is highest at L2** (ρ = 0.869), but the difference vs L1 P2 (ρ = 0.849) is small relative to the bootstrap CIs.
* **Absolute agreement with humans is highest at L1 P2** (κ = 0.623 vs L2's 0.517). L2's larger bias (+0.59 vs L1 P2's +0.41) accounts for most of the gap.
* **Discriminant effect sizes are higher at L1** on the Psychometric pool (filtered W = 0.791 vs L2's 0.686 on `score_0-3`, 0.628 on `score_0-100`). L2 wins on the HR pool and ties on the Engineering pool for `score_0-3`, but loses on the Engineering pool for `score_0-100`. This is again consistent with the fixed weights being sub-optimal.
* On this dataset L2 does not improve measurement quality over L1. The case for L2 rests on **auditability and cheap re-tuning of the formula**, not on stronger statistics.

### 6.6 Token spending and efficiency


| System | Mean tokens / call | Mean latency / call (s) | Notes |
|---|---|---|---|
| L1 P0 | 2,103 | 8.24 | minimal prompt, smallest payload |
| L1 P1 | 2,215 | 5.96 | adds rubric instructions, fastest in practice |
| L1 P2 | 2,593 | 7.76 | adds 0–100 scale and verdict block |
| **L2 P0** | **2,674** | **17.63** | structured-label output (6 categorical fields) |

**Reads:**

* L2 P0 uses ~3 % more tokens than L1 P2, driven by the longer structured-label output schema.
* L2 P0 is **~2.3× slower than L1 P2** at comparable token count (17.6 s vs 7.8 s mean per call). The latency premium is the cost of generating six well-formed categorical fields per call instead of one number.

These numbers are an order-of-magnitude reference, not a benchmark — they depend on tokenizer, retry behavior, schema-validation overhead, and concurrency settings at run time.

---

## 7. Limitations

### 7.1 Measurement-level (affect how to read the numbers)

1. **Ad-hoc aggregation weights are the dominant limitation specific to Level 2.** `score_0-100 = 100 × (0.6·skill + 0.3·(0.4·role + 0.6·domain) + 0.1·edu)` uses product-expert heuristics, not empirical optimisation against any criterion. The clearest signal that the weights are sub-optimal: the LLM's `score_0-3` discriminates cleanly between CVs on the Engineering pool (Kendall W = 0.704), while the formula-aggregated `score_0-100` on the same data only reaches W = 0.410. Weight-sensitivity analysis (grid search over the simplex, re-correlate with human labels) is the single most important next-step analysis.

2. **`score_0-100` reliability is not an independent empirical check.** It is a deterministic linear transform of four labels that themselves reproduce at Fleiss κ ≈ 0.87–0.96. Its ICC is mathematically bound to the label-level reliability. The ICC value reported in §6.2 is a transparency confirmation that the code aggregation introduces no extra noise — not evidence of system reliability beyond what the labels already establish.

3. **Two correlations in §6.3 are not independent tests.** `score_0-100` and `score_0-3` come from the same six LLM labels on the same 32 JDs. When both are reported (as in §6.5), they are two views of one set of judgments; no multiple-comparison correction is applied because we are not testing two distinct hypotheses.

4. **Scale compression inflates reliability.** The 0–3 scale has only four points and many JD × non-fit-CV pairs cluster at 0 (Engineering CV on Psychometric JDs, HR CV on Engineering JDs, etc.). High ICCs partly reflect agreement on a majority class with low variance, not sensitivity to fit gradations. The L2 `score_0-100` ICC = 0.968 is on a continuous scale and partially mitigates this; the `score_0-3` ICC = 0.967 should be read as an upper bound.

5. **"% within ±1" is lenient by construction.** On a 4-level scale, ±1 covers 67 % of all possible distances. The 93.8 % value reported in §6.3 should not be read as agreement; exact match (46.9 %) and Weighted κ (0.517) are the stricter views.

6. **The three "raters" in ICC are stochastic re-runs of one model**, not independent judges. Two-way random ICC is still appropriate (JDs are sampled from a population of possible items), but the model-side variance is by construction smaller than what cross-rater ICCs assume. ICC(A,1) is reported alongside ICC(A,k) for this reason; ICC(A,1) is the operational metric.

7. **Pool-construction asymmetry in discriminant validity.** The three JD pools were not built by the same rule. The Psychometric pool was filtered post-hoc to retain only the 16 JDs where the Assessment Scientist CV received a human label ≥ 2 (Consider / Apply) — this filtering was applied because the full 32-JD pool deliberately includes irrelevant items that compress effect sizes through floor overlap. The HR (n = 18) and Engineering (n = 25) pools were selected by role-title keywords without human labels and without any content-based filtering, so they retain a wider mix of relevance levels and a smaller proportion of strong-fit positions. The Psychometric pool is therefore more curated than the other two, and any apparent difference in discrimination quality between pools could partly reflect this filtering rather than the system's behavior. A more symmetric design would either apply the same human-label filter to all three pools (requires human labels on the HR and Engineering pools, currently unavailable) or skip filtering entirely (and accept the floor-overlap effect).

### 7.2 Design-level (affect how strong the inference can be)

8. **Single rater for the criterion labels** (n = 32). The author is both the system developer and the human annotator. Confirmation bias cannot be ruled out. The protocol applied several mitigations (first-time JD exposure during labeling, company names removed, written 4-level rubric — see §6.3), but the central limitation remains: what the criterion-validity section measures is alignment with one informed person's holistic judgment on her own CV profile, not with a population of independent expert raters. Independent multi-rater annotation, performed at the rubric/dimension level by raters who do not own the candidate CV, is the planned next step.

9. **Criterion labels only on Assessment Scientist CV × Psychometric JDs.** No human labels exist for the HR or Engineering CVs and their JD pools. The 3-way flip test substitutes for cross-CV human labels by checking that the *expected leader* CV scores highest on each pool — but a flip test is a weaker form of evidence than rated agreement with humans on every cell.

10. **No held-out test set.** The same 76 JDs drive reliability, criterion validity (on 32 of them), and discriminant validity (on the filtered 16). Over-fitting risk is low because no parameters are learned, but the absence of held-out data means we cannot quantify out-of-sample generalisation.

11. **Single LLM checkpoint** (Gemini 3.1 Flash Lite preview, temperature = 1.0). Whether the reliability, validity, discriminant patterns, and the ~2.3× latency cost of structured generation reported here generalise to other LLM providers is unknown. Cross-model robustness is planned future work.

12. **Construct definition is informal.** "Overall fit" is operationalised by the rubric weights `(0.6 / 0.12 / 0.18 / 0.1)` in the `score_0-100` formula, drawn from internal product-expert heuristics. L2 makes the decomposition (skill / role / domain / education) explicit. The same weights are used at L1 P1 implicitly and will be used at planned L3, so any construct misalignment propagates across levels.

13. **`confident` flag is degenerate** (99.85 % `True`, 1 row `False` out of 674). The flag was intended as a routing or gating signal but is uninformative in this dataset. Kept in the schema for possible future use cases (highly ambiguous JDs); current analysis ignores it.

### 7.3 Coverage gaps in validity evidence

14. **Content validity not assessed.** Whether the four dimensions (skills / role / domain / education) cover the relevant facets of job-candidate fit, and whether YES/PARTIAL/NO is the appropriate granularity, is asserted rather than tested. A formal content-validity study would require expert review against actual job-analysis output for each role.

15. **Convergent / divergent validity not assessed.** The system is not compared against *other* fit measures (a structured interview score, a separate competency match score, a hiring decision outcome). Agreement-with-a-single-human-rater is the only criterion in play.

16. **Predictive validity not assessed.** The score is not linked to any downstream hiring outcome (interview pass-through, offer rate, retention). Predictive validity is the gold-standard form of criterion evidence for selection instruments under EEOC and EU AI Act guidance; this project does not have access to such outcomes.

17. **Fairness / subgroup analysis absent.** No analysis of company-name bias, role-seniority bias, gendered language in JDs, or other demographic proxies. The label-level structure of L2 in principle enables more targeted fairness checks (e.g. whether `skills = NO` rates differ across protected groups in the underlying CV pool), but no such analysis has been done. Required under high-risk AI rules (EU AI Act, NYC Local Law 144).

18. **L2 systematic bias is larger than L1's.** §6.3 shows Bias = +0.59 (Cohen's d = 0.97, large), compared with L1 P2's +0.41 (d = 0.66, medium). The L2 design adds structure but does not reduce the over-rating tendency — and in fact amplifies it. The mechanism is not yet identified; one hypothesis is that the categorical labels (YES / PARTIAL / NO) are coarser than human gradations and round upward on borderline cases.

### 7.4 Scope and data caveats

19. **English only.** No multilingual JDs or CVs were tested.

20. **Training-data contamination cannot be ruled out.** Public JDs and the author's own public CV may have appeared in the LLM's pre-training data. There is no way to detect or quantify this with a closed-weight commercial model.

21. **One Gemini-side 500 error not retried.** The HR pool has 18 valid JDs at Level 2 vs 19 at Level 1. This is a 1-row drop on a 76-JD dataset and does not materially affect any of the reported statistics, but it is the reason for the pool-size discrepancy between levels.

22. **No formal between-design significance test.** Differences between L1 P2 and L2 P0 on criterion validity (ρ, κ, MAE, bias) and on discriminant validity (Kendall's W) are reported descriptively. Bootstrap 95 % CIs overlap heavily on most metrics; claims about which design is "better" rest on point estimates, not on tests of difference. Confirmatory between-design comparison would require a larger sample and a pre-registered test (e.g. Steiger's *Z* for correlated correlations, or a Hotelling-style test for paired effect sizes).

