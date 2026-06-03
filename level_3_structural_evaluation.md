# Level 3 — Structured Evaluation

---

## 1. Goal

Remove the LLM from the judgment step entirely. Compute fit through explicit segmentation, embedding, and similarity — making every step inspectable and reproducible.

Compare against human gold labels and against Levels 1 and 2.

---

## 2. Method

```
CV                                  JD
 ↓                                   ↓
segment + tag                  segment + tag
 ↓                                   ↓
embed each segment            embed each segment
 ↓                                   ↓
        pairwise cosine matrix
                  ↓
       MaxSim per construct
       (skills, experience, education)
                  ↓
       normalise each block to [0, 1]
       using corpus anchors
                  ↓
       fit_score = 100 × ( 0.6 · skills
                         + 0.3 · experience
                         + 0.1 · education )
```

**Segmentation:** LLM tagging at line level into a closed five-tag taxonomy (`skills`, `experience`, `education`, `mixed`, `other`).

**Embeddings:** sentence-transformer, mean-pooled and L2-normalised. CV embeddings cached — no re-embedding when comparing one CV against multiple JDs.

**Similarity:** per-construct MaxSim — for each JD segment, take its best cosine match against the CV's allowed segments; combine those best-matches via a weighted mean.

**Aggregation:** per-construct linear-adjust normalisation against corpus anchors, then the Level-2 weighted formula applied by code. Same weights as Level 2 so the final 0–100 score is on the same grid.

---

## 3. Dataset

**CVs — 3 candidates:**

| CV | Profile | Role in design |
|---|---|---|
| CV-1 | Assessment Scientist (psychometrics) | Primary profile; runs at all levels |
| CV-2 | Head of People (HR) | Cross-domain: people processes overlap |
| CV-3 | Solution Architect (engineering) | Cross-domain: technical overlap |

**JD pools — same 75 JDs as Level 2** 

**Total pairs:** 3 CVs × 75 JDs = 225 (CV, JD) pairs.

---

## 4. Ground Truth — Human Annotation

**Two-phase design:**

*Phase 1 — JD profiling (researcher only, before annotation):*
Each JD gets a structured profile: must-have skills (3–5), role level, experience minimum, education requirement. Fixed before any CV annotation begins.

*Phase 2 — CV annotation (raters, per pair):*
Raters see the JD profile and CV. Annotate using the same schema as Level 2.

**Cross-annotator design (noise reduction):**
Each rater annotates only pairs where their own CV is not included.


---

## 5. Pipeline

> 🚧 **Draft — editing in progress.** This section is a short procedural summary of the current (block-level) pipeline. Details, numbers and figures will be expanded. Full reproducible scripts live in `scripts/level3/` (`PIPELINE.md` there is the source of truth).

The pipeline turns raw JD and CV text into a 0–100 fit score. The LLM is used only to *structure* the text (tagging and normalising) — the judgment itself is deterministic embedding similarity. Processually, six steps:

**1. Segment and tag.** Each JD and CV is split into content lines; an LLM assigns every line one tag from a closed taxonomy — `skills`, `experience`, `education`, `mixed`, `other`. The model returns labels by `line_id` only (never echoes text). `other` is dropped from all downstream work.

**2. Normalise.** A second LLM pass reads each non-`other` segment and extracts the **canonical skill / experience** it expresses (free text → short comparable labels). A `mixed` line can yield both a skill and an experience label. Education is handled separately: one degree level per document — *highest held* for a CV, *minimum required* for a JD.

**3. Embed.** All of a document's skill labels are concatenated and embedded as **one** vector; the same is done for experience. Model: `all-MiniLM-L6-v2`, mean-pooled, L2-normalised (cosine = dot product). Two vectors per document. Education is **not** embedded.

**4. Per-pair similarity.** For each (CV, JD) pair, cosine of the two skills vectors and, separately, of the two experience vectors → `skills_sim`, `experience_sim`.

**5. Education match (discrete).** No requirement → automatic match; otherwise the CV's highest degree must reach the JD's required level. Ranked `none < high_school < associate < bachelor < master ≈ mba < phd`.

**6. Transform to 0–100.** Each cosine is rescaled by a fixed **anchor-linear** window `clip((cos − 0.30) / (0.80 − 0.30), 0, 1) × 100`, where 0.30 / 0.80 are the empirical floor (off-domain "space-nurse" anchor) and ceiling (CV-mirroring anchor). The sub-scores are combined with the Level-2 weights `0.6 · skills + 0.3 · experience + 0.1 · education`; when a JD has no degree requirement, education's 0.1 is redistributed onto skills and experience in their own 0.6 : 0.3 ratio. Final `clip(0, 100)`.

**Validation done so far:** (a) tag-vs-prototype construct-validity check; (b) agreement between the coarse 5-tag pass and the fine normalisation pass; (c) discriminant check — each CV scores highest on its matching JD pool; (d) distribution comparison against Levels 1 and 2 on the same pairs.

> ⚠️ **Known limitation (deferred).** Degree extraction uses a US-centric taxonomy and misses non-US credentials (e.g. post-Soviet 5-year *Specialist* degrees read as "none"). Education results are reported with this caveat.

---

## 6. Evaluation

Results will be evaluated against human gold labels and against Levels 1 and 2 using:

| Metric | What it measures |
|---|---|
| Correlation with human fit_score | Overall alignment with human judgment |
| Cohen's κ per dimension | Label-level agreement (skills / role / edu) |
| NDCG@k | Ranking quality relative to human ordering |
| Calibration | Score distribution vs human distribution |
| Discriminant validity | Score separation across CV profiles |


Findings will be added after analysis.

---

## 7. Status

Pipeline complete; mitigation of the similarity over-rating issue described in §5.3 is in progress. Evaluation against human gold labels pending the cross-annotator round described in §4.
