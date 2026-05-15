# Level 3 — Structured Evaluation (Experiment 2)

---

## 1. Goal

Remove the LLM from the judgment step entirely. Compute fit through explicit segmentation, embedding, and similarity — making every step inspectable and reproducible.

Compare against human gold labels and against Levels 1 and 2.

---

## 2. Method

```
CV                              JD
 ↓                               ↓
segment                        segment
 │                               │
Skills ──── cosine similarity ── Skills
Exp    ──── cosine similarity ── Exp
Edu    ──── cosine similarity ── Edu
 │
 ↓
score per section
 ↓
fit_score = 0.6 × skills_sim
          + 0.3 × experience_sim
          + 0.1 × education_sim
```

**Segmentation:** LLM + rules — splits CV and JD into three sections (Skills, Experience, Education).

**Embeddings:** sentence-transformers (e.g. `all-MiniLM-L6-v2`). Each section embedded separately. CV embeddings cached — no re-embedding when comparing one CV against multiple JDs.

**Similarity:** cosine similarity per matched section pair.

**Aggregation:** same formula as Level 2. Weights applied by code.

---

## 3. Dataset

**CVs — 3 candidates, 2 versions each:**

| CV | Profile | Role in design |
|---|---|---|
| CV-1 | Assessment Scientist (psychometrics) | Primary profile; runs at all levels |
| CV-2 | Head of People (HR) | Cross-domain: people processes overlap |
| CV-3 | Solution Architect (engineering) | Cross-domain: technical overlap |

Each CV has a real version and a paraphrased variant. Paraphrased versions are used for surface robustness testing within levels.

**Total pairs:** ~80 CV–JD pairs across 3 CVs and a selected JD subset.

---

## 4. Ground Truth — Human Annotation

**Two-phase design:**

*Phase 1 — JD profiling (researcher only, before annotation):*
Each JD gets a structured profile: must-have skills (3–5), role level, experience minimum, education requirement. Fixed before any CV annotation begins.

*Phase 2 — CV annotation (raters, per pair):*
Raters see the JD profile and CV. Annotate using the same schema as Level 2.

**Cross-annotator design (noise reduction):**
Each rater annotates only pairs where their own CV is not included.

| Rater | CV profile | Annotates |
|---|---|---|
| Researcher (Assessment Scientist) | CV-1 | CV-2 × JD pairs + CV-3 × JD pairs |
| SME (HR) | CV-2 | CV-1 × JD pairs + CV-3 × JD pairs |
| SME (Engineer) | CV-3 | CV-1 × JD pairs + CV-2 × JD pairs |

**IRR thresholds before using labels as ground truth:**
- Cohen's κ ≥ 0.60 per skill / role / edu dimension
- ICC(2,1) ≥ 0.70 on final fit_score
- ≥ 70% exact agreement on holistic (4-point)

---

## 5. Evaluation

Results will be evaluated against human gold labels using:

| Metric | What it measures |
|---|---|
| Correlation with human fit_score | Overall alignment with human judgment |
| Cohen's κ per dimension | Label-level agreement (skills / role / domain / edu) |
| NDCG@k | Ranking quality relative to human ordering |
| Calibration | Score distribution vs human distribution |
| Discriminant validity | Score separation across CV profiles |

Cross-level comparison:

| Comparison | Question |
|---|---|
| Level 3 vs Level 2 | Does judgment source matter when the formula is identical? |
| Level 3 vs Level 1 | How much does full structural decomposition gain over holistic LLM? |

Key findings to be added after analysis.

---

## 6. Status

Design complete.

