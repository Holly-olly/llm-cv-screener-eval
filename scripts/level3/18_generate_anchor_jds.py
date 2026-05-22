#!/usr/bin/env python3
"""
Level 3 — Step 18: Generate synthetic ANCHOR job descriptions.

We need a per-CV anchor scale to convert raw cosine fit_score into a
0–100 score that has a transparent psychometric meaning:

  max_anchor[cv]  =  fit_score of (cv × a JD that perfectly matches the
                     candidate's last described work experience)
  min_anchor      =  fit_score of (cv × a JD that has zero meaningful
                     overlap with any candidate profile in the study —
                     a "Registered Nurse for a long-duration space station
                     mission")

Both ends are obtained by feeding the candidate's most recent role text
to GPT-4o-mini and asking it to produce a JD that "asks for exactly what
this person already did". The minimum JD is generated once and reused
across all three CVs.

Why this is psychometrically defensible:
  - The anchor band is constructed from the same population (LLM-style
    JD text) that the real corpus comes from, so the cosine geometry is
    comparable.
  - The maximum is per-candidate (each CV gets its own ceiling), so
    cosine compression — which is candidate-specific — is calibrated
    out before the linear stretch.
  - The minimum is corpus-shared and obviously out-of-domain — it gives
    a stable floor.

Outputs:
  - data/synthetic_anchors/max_{cv_id}.txt   (one per CV)
  - data/synthetic_anchors/min_space_nurse.txt
  - data/synthetic_anchors/_generation_log.json   (prompts + token usage)

Run:
    cd ~/Documents/cool-cohen/llm_evaluation
    python3 scripts/level3/18_generate_anchor_jds.py
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

from openai import OpenAI


ROOT       = Path(__file__).parent.parent.parent
CV_DIR     = ROOT / "data" / "cv"
OUT_DIR    = ROOT / "data" / "synthetic_anchors"
LOG_PATH   = OUT_DIR / "_generation_log.json"

MODEL       = "gpt-4o-mini"
TEMPERATURE = 0.4         # slight creativity for natural JD wording
TOP_P       = 1.0

# Which CVs need a max-anchor. cv_paraphrased is excluded from L3 analysis.
CV_IDS = ["cv_primary", "cv_hr", "cv_engineer"]


# ── Prompts ─────────────────────────────────────────────────────────────────
SYSTEM_MAX = """You are a senior technical recruiter writing a realistic
LinkedIn-style job description for an evaluation experiment.

The candidate's CV is provided. You will:
1. Identify the MOST RECENT role described in the CV (the current or
   latest-dated role).
2. Generate a realistic JD that asks for EXACTLY the work this person
   has been doing in that role — same scope, same methods, same
   responsibilities, same tools, same seniority level, same domain.

Strict rules:
  - The JD must read like a real LinkedIn posting, not a CV mirror.
    Use recruiter voice ("you'll lead", "we're looking for", "must have").
  - Include realistic LinkedIn sections: a short company intro paragraph,
    "About the Role", "Responsibilities", "Requirements / Must-have",
    "Nice to have / Preferred", and a brief "What we offer" block.
  - Cover EVERY substantive bullet from the most recent role — skills,
    methods, deliverables, scale numbers, team sizes, domain context.
  - Years of experience and seniority must match what the CV demonstrates.
  - Length: 35–55 lines (one bullet or sentence per line).
  - NEVER mention the candidate by name. NEVER reveal this is synthetic.
  - Output the JD text only — no preamble, no commentary, no markdown
    code fences.
"""


SYSTEM_MIN = """You are writing a realistic LinkedIn-style job description
for an evaluation experiment. The role: Registered Nurse on a long-
duration commercial space-station mission (private orbital research lab).

Required content:
  - Short company / mission introduction.
  - "About the Role" — provide bedside care to astronauts during a
    6-month orbital rotation, manage in-flight medical emergencies in
    microgravity, support crew well-being under isolation and confinement.
  - "Responsibilities": triage and stabilise injuries in microgravity,
    administer medications via established space-medicine protocols,
    monitor vitals continuously, coordinate with ground-control medical
    teams via delayed comms, maintain on-board pharmacy and medical
    inventory, run regular health screens, conduct telemedicine relays
    for ground-side specialists, maintain detailed shift logs.
  - "Requirements": RN licensure in good standing (US or EU); 5+ years
    of acute / emergency / ICU nursing; BLS, ACLS, PALS certifications;
    completion of an accredited space-medicine programme; proven
    composure in isolated, high-stress environments.
  - "Nice to have": prior aviation or military medicine experience;
    altitude-physiology background; second language (Russian or Mandarin)
    for international crew coordination.
  - "What we offer": competitive compensation, post-mission rehabilitation
    package, public-mission profile.

Style:
  - Realistic LinkedIn voice, recruiter wording.
  - Length: 35–50 lines.
  - Output JD text only — no preamble, no markdown fences.
"""


# ── Helpers ─────────────────────────────────────────────────────────────────
def load_openai_key() -> str:
    env_path = ROOT.parent / ".env.local"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip()
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found")
    return key


def call_openai(client: OpenAI, system: str, user: str) -> tuple[str, dict]:
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    text = resp.choices[0].message.content.strip()
    # Strip markdown fences if the model emitted any
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    usage = {
        "prompt":     resp.usage.prompt_tokens,
        "completion": resp.usage.completion_tokens,
        "total":      resp.usage.total_tokens,
    }
    return text.strip(), usage


def n_content_lines(text: str) -> int:
    return sum(1 for ln in text.splitlines() if ln.strip())


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    client = OpenAI(api_key=load_openai_key())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log: list[dict] = []
    total_usage = {"prompt": 0, "completion": 0, "total": 0}

    # ── 3 × max anchors ─────────────────────────────────────────────────────
    print("Generating MAX anchor JDs (one per CV) ...\n")
    for cv_id in CV_IDS:
        cv_path = CV_DIR / f"{cv_id}.txt"
        if not cv_path.exists():
            print(f"  ✗ missing CV: {cv_path}")
            continue
        cv_text = cv_path.read_text(encoding="utf-8")
        user_msg = (
            f"CV under review (find the most recent role):\n\n"
            f"=== CV START ===\n{cv_text}\n=== CV END ===\n\n"
            "Now generate the synthetic JD."
        )

        try:
            jd_text, usage = call_openai(client, SYSTEM_MAX, user_msg)
        except Exception as e:
            print(f"  ✗ {cv_id}: {e}")
            continue

        out_path = OUT_DIR / f"max_{cv_id}.txt"
        out_path.write_text(jd_text, encoding="utf-8")
        for k in total_usage:
            total_usage[k] += usage[k]

        n_lines = n_content_lines(jd_text)
        print(f"  ✓ {cv_id:<14}  →  {out_path.relative_to(ROOT)}")
        print(f"     {n_lines} content lines   "
              f"tokens: {usage['prompt']:,} + {usage['completion']:,} "
              f"= {usage['total']:,}")
        log.append({
            "kind":      "max",
            "cv_id":     cv_id,
            "out":       str(out_path.relative_to(ROOT)),
            "n_lines":   n_lines,
            "tokens":    usage,
        })

    # ── 1 × universal min anchor (space nurse) ──────────────────────────────
    print("\nGenerating UNIVERSAL MIN anchor JD (space station nurse) ...\n")
    try:
        jd_text, usage = call_openai(client, SYSTEM_MIN,
                                     "Generate the synthetic JD now.")
    except Exception as e:
        print(f"  ✗ {e}")
        jd_text, usage = "", {"prompt": 0, "completion": 0, "total": 0}

    if jd_text:
        out_path = OUT_DIR / "min_space_nurse.txt"
        out_path.write_text(jd_text, encoding="utf-8")
        for k in total_usage:
            total_usage[k] += usage[k]
        n_lines = n_content_lines(jd_text)
        print(f"  ✓ min_space_nurse  →  {out_path.relative_to(ROOT)}")
        print(f"     {n_lines} content lines   "
              f"tokens: {usage['prompt']:,} + {usage['completion']:,} "
              f"= {usage['total']:,}")
        log.append({
            "kind":      "min",
            "cv_id":     None,
            "out":       str(out_path.relative_to(ROOT)),
            "n_lines":   n_lines,
            "tokens":    usage,
        })

    # ── Log ─────────────────────────────────────────────────────────────────
    LOG_PATH.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model":        MODEL,
        "temperature":  TEMPERATURE,
        "top_p":        TOP_P,
        "entries":      log,
        "total_tokens": total_usage,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"Total tokens across all generations: {total_usage['total']:,}")
    cost = (total_usage["prompt"]    * 0.150
            + total_usage["completion"] * 0.600) / 1_000_000
    print(f"Run cost: ~${cost:.4f} on gpt-4o-mini")
    print(f"\nReview the .txt files in {OUT_DIR.relative_to(ROOT)} before "
          "running the anchor-calibration script.")


if __name__ == "__main__":
    main()
