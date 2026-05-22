#!/usr/bin/env python3
"""
Level 3 — Step 5: Embed non-`other` JD segments and compare against
construct prototype embeddings.

Construct prototypes are the SAME definitions the LLM was given in Stage 1
(see DEFINITIONS in `03_llm_label_json.py`). This is a construct-validity
check: do segments tagged `skills` actually sit closer (by cosine sim) to
the Skills prototype than to Experience / Education? If yes, the LLM's
tagging is internally consistent with the semantic anchors it was given.

Reads:
  - results/level3/llm_labelled_json/{jd_id}.csv

Writes:
  - results/level3/segment_embeddings.npz       embeddings + chunk metadata
  - results/level3/segment_similarities.csv     per-segment table:
        chunk_id, jd_id, line_id, tag, text,
        sim_skills, sim_experience, sim_education,
        predicted_tag (argmax over the three sims)

Prints:
  - mean similarity per (assigned_tag × prototype) — the construct validity
    matrix.
  - confusion table: assigned_tag vs predicted_tag (argmax).
  - mixed-tag specific check: distribution of (sim_skills - sim_experience).

Run:
    cd ~/Documents/cool-cohen/llm_evaluation
    python3 scripts/level3/05_embed_and_similarity.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


ROOT       = Path(__file__).parent.parent.parent
LBL_DIR    = ROOT / "results" / "level3" / "llm_labelled_json"
EMB_OUT    = ROOT / "results" / "level3" / "segment_embeddings.npz"
SIM_OUT    = ROOT / "results" / "level3" / "segment_similarities.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Construct prototype texts — lifted verbatim from the Stage 1 prompt so that
# the embedding-space "ground truth" matches what the LLM was told.
PROTOTYPES = {
    "skills":
        '"Can do". Ability to apply knowledge and competencies to perform '
        'tasks and solve problems. Concrete, demonstrable capabilities: '
        'technologies, tools, frameworks, methods, language fluency, '
        'specific soft skills like active listening, conflict resolution, '
        'attention to detail.',
    "experience":
        '"Has done". Demonstrated application of skills in a role. '
        'Domain similarity and role relevance. Years of experience, '
        'seniority, prior industries, day-to-day responsibilities, '
        'track record, background in a domain.',
    "education":
        'Formal qualifications: degrees, certifications, universities, '
        'academic backgrounds. Bachelor, Master, PhD, MBA, academic field '
        'of study, certifications equivalent to formal education.',
}


def load_segments() -> pd.DataFrame:
    """Concatenate all per-JD CSVs into one DataFrame; keep non-other rows."""
    frames = []
    for csv_path in sorted(LBL_DIR.glob("*.csv")):
        jd_id = csv_path.stem
        df = pd.read_csv(csv_path, keep_default_na=False)
        df["jd_id"] = jd_id
        df["chunk_id"] = df.apply(lambda r: f"{jd_id}__line{r['line_id']}", axis=1)
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    # Keep only segments with a construct tag — `other` is dropped.
    keep = full[full["tag"].isin(["skills", "experience", "education", "mixed"])].copy()
    return keep.reset_index(drop=True)


def main() -> None:
    print(f"Loading model {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    segs = load_segments()
    print(f"Segments to embed: {len(segs)}  "
          f"(of which mixed={int((segs['tag']=='mixed').sum())}, "
          f"skills={int((segs['tag']=='skills').sum())}, "
          f"experience={int((segs['tag']=='experience').sum())}, "
          f"education={int((segs['tag']=='education').sum())})")

    print("\nEmbedding construct prototypes ...")
    proto_names = list(PROTOTYPES.keys())     # ['skills', 'experience', 'education']
    proto_vecs = model.encode([PROTOTYPES[n] for n in proto_names],
                              normalize_embeddings=True,
                              show_progress_bar=False)

    print(f"Embedding {len(segs)} segments ...")
    seg_vecs = model.encode(segs["text"].tolist(),
                            normalize_embeddings=True,
                            show_progress_bar=True,
                            batch_size=64)

    sims = seg_vecs @ proto_vecs.T              # cosine on L2-normalised vectors
    for i, name in enumerate(proto_names):
        segs[f"sim_{name}"] = np.round(sims[:, i], 4)

    pred_idx = sims.argmax(axis=1)
    segs["predicted_tag"] = [proto_names[i] for i in pred_idx]

    # ── Persist ─────────────────────────────────────────────────────────────
    EMB_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        EMB_OUT,
        embeddings=seg_vecs.astype(np.float32),
        proto_embeddings=proto_vecs.astype(np.float32),
        proto_names=np.array(proto_names),
        chunk_ids=segs["chunk_id"].to_numpy(),
        jd_ids=segs["jd_id"].to_numpy(),
        tags=segs["tag"].to_numpy(),
    )
    print(f"\n✓ Embeddings saved → {EMB_OUT.relative_to(ROOT)}  "
          f"({len(segs)} × {seg_vecs.shape[1]} float32)")

    out_cols = ["chunk_id", "jd_id", "line_id", "tag",
                "sim_skills", "sim_experience", "sim_education",
                "predicted_tag", "text"]
    segs[out_cols].to_csv(SIM_OUT, index=False)
    print(f"✓ Per-segment similarities → {SIM_OUT.relative_to(ROOT)}")

    # ── Construct validity matrix ───────────────────────────────────────────
    print("\n" + "=" * 78)
    print("Construct-validity matrix — mean cosine similarity per (tag × prototype)")
    print("=" * 78)
    matrix = segs.groupby("tag")[["sim_skills", "sim_experience", "sim_education"]].mean()
    matrix = matrix.round(3)
    # Ensure consistent row order
    matrix = matrix.reindex(["skills", "experience", "education", "mixed"]).dropna()
    print(matrix.to_string())

    print("\nReading aid: each row is one assigned tag. The cell highlighted")
    print("should be the LARGEST in its row.  For `mixed`, sim_skills and")
    print("sim_experience should both be high and similar; sim_education low.")

    # ── Confusion: assigned vs argmax predicted ────────────────────────────
    print("\n" + "=" * 78)
    print("Confusion table — rows = assigned tag, cols = argmax predicted tag")
    print("=" * 78)
    # Map mixed → argmax (skills or experience) for this table — `mixed` is
    # not a prototype, so it has no diagonal cell. Show counts as-is.
    conf = pd.crosstab(segs["tag"], segs["predicted_tag"], margins=True, margins_name="TOTAL")
    print(conf.to_string())

    # Per-row purity (excluding mixed since it has no own prototype).
    print("\nPer-tag share that lands on the matching prototype:")
    for tag in ["skills", "experience", "education"]:
        sub = segs[segs["tag"] == tag]
        if len(sub) == 0:
            continue
        purity = (sub["predicted_tag"] == tag).mean()
        print(f"  {tag:11s}: {purity*100:5.1f}%  (n={len(sub)})")

    # ── Mixed-tag specific check ────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("`mixed` segments — balance between skills and experience prototypes")
    print("=" * 78)
    mixed = segs[segs["tag"] == "mixed"].copy()
    if len(mixed):
        mixed["delta_skills_minus_exp"] = mixed["sim_skills"] - mixed["sim_experience"]
        print(f"n = {len(mixed)}")
        print(f"  mean sim_skills     = {mixed['sim_skills'].mean():.3f}")
        print(f"  mean sim_experience = {mixed['sim_experience'].mean():.3f}")
        print(f"  mean sim_education  = {mixed['sim_education'].mean():.3f}")
        print(f"  mean |skills-exp|   = {mixed['delta_skills_minus_exp'].abs().mean():.3f}  "
              "(small = balanced, large = skewed)")
        print(f"  share closer to skills:     "
              f"{(mixed['sim_skills'] > mixed['sim_experience']).mean()*100:.1f}%")
        print(f"  share closer to experience: "
              f"{(mixed['sim_experience'] > mixed['sim_skills']).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
