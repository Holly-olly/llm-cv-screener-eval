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

The pipeline below turns raw JD and CV text into a 0–100 fit score. No LLM is used for judgment — every step is deterministic and inspectable once the segments are tagged.

### 5.1 Segmentation and tagging

Each JD and each CV is split into content lines. Every line is then assigned exactly one tag from a closed five-tag taxonomy:

| Tag | Meaning |
|---|---|
| `skills` | "Can do": concrete capabilities — tools, frameworks, methods, language fluency, specific soft skills. Excludes vague aspirational traits ("drive", "passion", "grit") which are `other`. |
| `experience` | "Has done": demonstrated application of skills — years of experience, prior industries, role types, day-to-day duties. |
| `education` | Formal qualifications: degrees, certifications, universities, academic background. |
| `mixed` | A single line covering BOTH a named skill AND demonstrated experience (e.g. "5+ years of Python development experience"). Replaces multi-tagging — a line is never assigned two tags. |
| `other` | Everything else: benefits, compensation, perks, company description, mission, culture, location, application instructions, LinkedIn UI noise. Excluded from the similarity computation. |

**How it's done.** A small low-temperature LLM tags lines through a structured prompt. The model receives a few-shot prompt containing three hand-labelled JDs plus the canonical tag definitions, and returns labels by `line_id` only — it never echoes the input text. This makes the output cheap, deterministic, and resistant to text-rewriting noise.

**Pre-processing.** Before sending a JD or CV to the LLM:
- Blank lines and surrounding whitespace are stripped.
- Any single line longer than 600 characters (a scraped "wall of text" with no paragraph breaks) is split by sentence boundary so each sentence can be tagged independently.
- The remaining lines are numbered sequentially as a JSON array.

**Post-processing.** A validator checks that every input `line_id` appears exactly once in the output with a tag drawn from the allowed vocabulary. If the LLM omits any lines (rare — observed on a small number of JDs with non-Latin UI strings at the tail), they are auto-filled with tag = "other" and flagged for transparency.


### 5.2 Embedding

Every non-other segment is embedded with a sentence-transformer model, mean-pooled and L2-normalised. The same model is used for CV segments, JD segments, and the tag prototypes — keeping the geometry of the comparison consistent on both sides.

CV embeddings are computed once per CV and cached. When the same CV is scored against multiple JDs, no re-embedding occurs — only the per-JD work and the matrix multiplication.

Because embeddings are L2-normalised, cosine similarity between any two segments is their dot product.

### 5.3 Similarity — per-construct MaxSim

For every (CV, JD) pair, the pairwise cosine matrix between non-`other` segments on both sides is materialised. From this matrix we derive one similarity value per construct (skills, experience, education) using the **MaxSim** rule.

For every JD segment that contributes to a construct *C*, find its best cosine match against the CV segments allowed for *C*. The block similarity for *C* is the weighted mean of those best-matches across all contributing JD segments:

| Construct | Allowed CV tags | JD segments that contribute (with weight) |
|---|---|---|
| skills | `skills`, `mixed` | `skills` (1.0), `mixed` (0.5) |
| experience | `experience`, `mixed` | `experience` (1.0), `mixed` (0.5) |
| education | `education` | `education` (1.0) |

The `mixed` weight of 0.5 prevents double-counting: a `mixed` JD line contributes evidence to both the skills block and the experience block, but at half strength on each side, so it cannot single-handedly inflate either.

**Why MaxSim and not averaged cosine.** A JD requirement is *met* when some part of the CV is a strong semantic match for it — taking the best-matching CV segment per JD line captures this "evidence anywhere" notion. Averaging cosines across all segment pairs (the alternative) penalises long CVs that contain unrelated content alongside the relevant evidence.

**Education fallback rule.**
- If the JD has no education segments, the education block is filled from the mean of (skills_block, experience_block) and the row is flagged as having used the fallback.
- If the JD requires education AND the CV has no education segments, the education block is NaN and propagates to the final fit score — the candidate fails the requirement explicitly.

> ⚠️ **Known issue — similarity over-rating on a few pairs (in progress).** Some (CV, JD) pairs receive an inflated MaxSim block score that is not matched by Level 1 / Level 2 judgement. Two mechanisms have been identified so far: (1) **tiny-JD noise** — when a JD has only a handful of non-`other` segments, MaxSim is averaging very few best-cell cosines and a single high-overlap cell can spike the block; (2) **`mixed`-tag concentration on the CV side** — a CV with many `mixed` lines contributes to both the skills and the experience block at weight 0.5, structurally amplifying its score against `mixed`-heavy JDs.
>

### 5.4 Aggregation

The three block similarities live in different empirical ranges per construct (skills narrower than experience; education spans wider). Applying the Level-2 weighting formula directly to the raw cosines would silently re-weight the constructs because wider-range inputs dominate the weighted sum. The aggregation step therefore has two parts.

**Step 1 — per-construct normalisation to [0, 1] (linear-adjust).** Each block similarity is mapped onto a common 0-to-1 scale using per-construct corpus anchors as the normalisation window:

- The *lower anchor* is the block similarity reached by an extreme off-domain JD — a synthetic "Registered Nurse on a space station" JD that shares no domain with any candidate.
- The *upper anchor* is the block similarity reached by a CV-mirroring synthetic JD — generated by an LLM to mirror a candidate's own recent role, approximating the empirical ceiling.

```
adj_max = lower_anchor + ADJ_FACTOR · (upper_anchor − lower_anchor)    (ADJ_FACTOR = 0.8)
norm(x)  = clip( (x − lower_anchor) / (adj_max − lower_anchor),  0,  1 )
```

The 0.8 shrinkage acknowledges that the synthetic upper anchor sits at the upper *edge* of what cosine can produce — not at the centre of the "strong fit" band. Treating 80 % of the anchor range as the realistic ceiling lets a genuinely strong-fit real-world JD reach values close to 1.0 after normalisation; without the shrinkage even in-domain matches cap below 1.0 and the final 0–100 score never approaches 100.

After Step 1 each block sits on the same [0, 1] scale: 0 means "as off-domain as the synthetic floor anchor", 1 means "as close as a CV-mirroring JD (or closer)".

**Step 2 — Level-2 weighted formula on the normalised sub-scores.** The Level-2 weights are reused unchanged so that the L3 final score sits on the same 0–100 grid as the L2 `fit_score_100`:

```
fit_score_0_100 = 100 × ( 0.6 · skills_norm
                        + 0.3 · experience_norm
                        + 0.1 · education_norm )
```

**Design choice — linear-adjust vs other monotone mappings.** The transformation is one specific choice; sigmoid, corpus-percentile, or piecewise approaches could be substituted without changing the underlying MaxSim aggregator. Linear-adjust is the simplest defensible option on this corpus — anchors define an interpretable window, no extra parameters need fitting, and the mapping preserves the rank ordering of the raw cosines exactly.

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
