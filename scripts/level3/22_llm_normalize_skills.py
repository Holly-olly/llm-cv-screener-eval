#!/usr/bin/env python3
"""
Level 3 v2 — Step 22: LLM-based skill / experience normalisation.

For every CV and JD in our labelled-JSON library, send the document's
non-`other` segments to Gemini in ONE batched call and ask for
canonicalised skill / experience annotations per segment.

Adapted from `llm_skill_normalizer.py` (which sent one call per segment).
The batched design cuts API calls from N-per-document to 1-per-document,
removing the repeated system prompt and the per-call delay.

Input  (already produced by script 03):
  - results/level3/llm_labelled_json/{jd_id}_input.json + _output.json
  - results/level3/llm_labelled_json_cv/{cv_id}_input.json + _output.json

Output:
  - results/level3/llm_skill_normalization/{doc_id}.json
  - results/level3/llm_skill_normalization/_run_summary.csv

Per document JSON contains:
  - document_id, document_type ('cv' | 'jd'), model
  - n_segments_input, tokens {input, output, total}, latency_s, cost_usd
  - items: [{ chunk_id, tag_original, text, annotations[], rationale }]

CLI:
    # Single-pair test (cheap dry-run on tokens only)
    python3 scripts/level3/22_llm_normalize_skills.py \
        --cv cv_primary --jd maki_people_senior_psychometrician \
        --estimate-only

    # Full run on the whole library
    python3 scripts/level3/22_llm_normalize_skills.py

    # Re-run a specific pair (overwrite existing output)
    python3 scripts/level3/22_llm_normalize_skills.py \
        --cv cv_primary --jd maki_people_senior_psychometrician --force
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any

# google-genai SDK (matches the reference normalizer)
from google import genai
from google.genai import types


ROOT       = Path(__file__).resolve().parent.parent.parent
# Unified labelled JSON (single file per doc, includes tag + text + embedding):
LBL_CV     = ROOT / "results" / "level3" / "labelled" / "cv"
LBL_JD     = ROOT / "results" / "level3" / "labelled" / "jd"
LBL_ANCHOR = ROOT / "results" / "level3" / "labelled" / "anchors"
# Legacy split format (kept as fallback for any doc only present here):
JD_DIR_LEGACY = ROOT / "results" / "level3" / "llm_labelled_json"
CV_DIR_LEGACY = ROOT / "results" / "level3" / "llm_labelled_json_cv"
OUT_BASE   = ROOT / "results" / "level3" / "segments_normalisation"
OUT_DIRS   = {"cv": OUT_BASE / "cv_norm",
              "jd": OUT_BASE / "jd_norm",
              "anchor": OUT_BASE / "jd_norm"}   # anchors are JD-like
SUMMARY_CSV = OUT_BASE / "_run_summary.csv"

DEFAULT_MODEL = "gemini-3.1-flash-lite"

# Gemini 3.1 Flash Lite pricing (per 1M tokens, May 2026 paid tier):
#   input  = $0.075   output = $0.30
# Free tier is also available but per-min/per-day quotas apply. Cost figure
# reported is a worst-case for the paid tier so the user can decide whether
# to throttle or run.
PRICE_PER_M_INPUT_USD  = 0.075
PRICE_PER_M_OUTPUT_USD = 0.30


# ──────────────────────────────────────────────────────────────────────────
# .env discovery (matches the project pattern used by tag-labelling scripts)
# ──────────────────────────────────────────────────────────────────────────
def load_api_key() -> str:
    if k := os.environ.get("GEMINI_API_KEY"):
        return k
    if k := os.environ.get("VITE_GEMINI_API_KEY"):
        return k
    # Walk up from this script searching for .env.local — first hit wins.
    here = Path(__file__).resolve()
    for candidate in (
        here.parents[3] / ".env.local",   # cool-cohen/.env.local
        here.parents[2] / ".env.local",   # llm_evaluation/.env.local
        here.parents[1] / ".env",         # level3/.env
    ):
        if not candidate.exists():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key in ("GEMINI_API_KEY", "VITE_GEMINI_API_KEY") and value:
                return value
    raise SystemExit("GEMINI_API_KEY not found in env or .env.local")


# ──────────────────────────────────────────────────────────────────────────
# Document loading — merge `_input.json` + `_output.json` into per-segment view
# ──────────────────────────────────────────────────────────────────────────
def _load_unified(path: Path) -> list[dict]:
    """Unified format: single JSON with {id, segments:[{line_id, text, tag, embedding}]}."""
    rec = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for s in rec.get("segments", []):
        tag = s.get("tag")
        if tag == "other" or not tag:
            continue
        text = (s.get("text") or "").strip()
        if not text:
            continue
        out.append({"chunk_id": int(s["line_id"]), "tag": tag, "text": text})
    return out


def _load_split(in_path: Path, out_path: Path) -> list[dict]:
    """Legacy split format: _input.json + _output.json."""
    lines  = json.loads(in_path.read_text(encoding="utf-8"))
    labels = {l["line_id"]: l["tag"]
              for l in json.loads(out_path.read_text(encoding="utf-8"))["labels"]}
    out = []
    for row in lines:
        lid = row["line_id"]
        tag = labels.get(lid, "other")
        if tag == "other":
            continue
        text = (row.get("text") or "").strip()
        if not text:
            continue
        out.append({"chunk_id": lid, "tag": tag, "text": text})
    return out


def load_document(doc_id: str, doc_type: str) -> list[dict]:
    """Return non-'other' segments [{chunk_id, tag, text}] for one doc.

    Tries the unified `labelled/{cv|jd|anchors}/{doc_id}.json` first,
    falls back to the legacy split `llm_labelled_json[_cv]/`.
    """
    if doc_type == "cv":
        candidates_unified = [LBL_CV / f"{doc_id}.json"]
        legacy_in   = CV_DIR_LEGACY / f"{doc_id}_input.json"
        legacy_out  = CV_DIR_LEGACY / f"{doc_id}_output.json"
    elif doc_type == "anchor":
        candidates_unified = [LBL_ANCHOR / f"{doc_id}.json"]
        legacy_in = legacy_out = None
    else:  # "jd"
        candidates_unified = [LBL_JD / f"{doc_id}.json"]
        legacy_in   = JD_DIR_LEGACY / f"{doc_id}_input.json"
        legacy_out  = JD_DIR_LEGACY / f"{doc_id}_output.json"

    for p in candidates_unified:
        if p.exists():
            return _load_unified(p)
    if legacy_in and legacy_in.exists() and legacy_out.exists():
        return _load_split(legacy_in, legacy_out)
    raise FileNotFoundError(
        f"Could not find labelled JSON for {doc_type}='{doc_id}'. "
        f"Tried: {[str(p) for p in candidates_unified]}"
        + (f" and {legacy_in}/{legacy_out}" if legacy_in else "")
    )


def list_all_docs() -> tuple[list[str], list[str], list[str]]:
    """List CV / JD / anchor ids present on disk (unified format preferred,
    legacy fallback for split format)."""
    cv_ids = sorted({p.stem for p in LBL_CV.glob("*.json")}) if LBL_CV.exists() else []
    if not cv_ids and CV_DIR_LEGACY.exists():
        cv_ids = sorted({p.stem.removesuffix("_input").removesuffix("_output")
                         for p in CV_DIR_LEGACY.glob("*.json")})
    jd_ids = sorted({p.stem for p in LBL_JD.glob("*.json")}) if LBL_JD.exists() else []
    if not jd_ids and JD_DIR_LEGACY.exists():
        jd_ids = sorted({p.stem.removesuffix("_input").removesuffix("_output")
                         for p in JD_DIR_LEGACY.glob("*.json")})
    anchor_ids = sorted({p.stem for p in LBL_ANCHOR.glob("*.json")}) if LBL_ANCHOR.exists() else []
    return cv_ids, jd_ids, anchor_ids


# ──────────────────────────────────────────────────────────────────────────
# Prompt + response schema (batched: all segments in one call)
# ──────────────────────────────────────────────────────────────────────────
SYSTEM_INSTRUCTIONS = """\
You are a skill / experience normalisation assistant for CVs and job descriptions.

DEFINITIONS
- A "skill" is the ability to apply knowledge or competencies to perform tasks
  ("can do"): tools, frameworks, methods, language fluency, named soft skills.
- "Experience" is the demonstrated application of skills in a role: years of
  experience, seniority, role titles, day-to-day duties.

RULES
1. For EACH input segment, decide whether it expresses skill(s), experience,
   or both. A single segment can produce multiple annotations.
2. Use tag = "skill" or "experience" for each annotation (lowercase).
3. `normalized_label` is the concise canonical name of the skill or
   experience marker. Strip surrounding prose. Examples: "Python",
   "factor analysis", "5+ years experience", "team leadership".
4. `evidence` is the exact substring from the segment that supports the
   annotation. Do not paraphrase.
5. `confidence` is in [0, 1].
6. If a segment is purely structural ("Education", "Experience"), return an
   empty annotations array for it — do NOT fabricate annotations.
7. Output STRICTLY a JSON object matching the schema. No commentary.
"""


# ──────────────────────────────────────────────────────────────────────────
# JD-side education extraction — degree level (not embeddings) + fields
# ──────────────────────────────────────────────────────────────────────────
# Education is matched against the CV by degree LEVEL rather than embedding
# similarity, so we extract structured degree info from the JD's
# `education`-tagged segments. The minimum REQUIRED level and the IDEAL
# PREFERRED level are returned separately so a candidate with the required
# but not the preferred degree can still be flagged as a partial-fit.
EDU_LEVELS = ["none", "high_school", "associate", "bachelor", "master", "mba", "phd"]

EDU_INSTRUCTIONS_JD = """\
You are an education-requirement extractor for a job description.

Read the EDUCATION-tagged segments from the JD and return ONE structured
object with the MINIMUM required degree level.

OUTPUT FIELDS
- required_degree : the MINIMUM degree the JD strictly requires.
- evidence        : the exact substring(s) that support your decision.

ALLOWED DEGREE VALUES (case-sensitive, lowest → highest):
  none, high_school, associate, bachelor, master, mba, phd

RULES
- If JD says "Bachelor's required"            → required = "bachelor"
- If JD says "Bachelor's or Master's"         → required = "bachelor" (minimum)
- If JD says "Master's required"              → required = "master"
- If JD says only "Master's preferred"        → required = "none"
- If JD says "PhD" (only level mentioned)     → required = "phd"
- If JD says "MBA"                            → required = "mba"
- If JD says "MD" / "JD" / "EdD" / "DSc"      → required = "phd"
- Certifications (PMP, AWS, etc.)             → NOT degrees, ignore.
- No degree mentioned anywhere                → required = "none"
- Noise (UI text, logos, section headers)     → required = "none", evidence = ""

Output STRICTLY the JSON object. No commentary.
"""

EDU_INSTRUCTIONS_CV = """\
You are an education-level extractor for a CV.

Read the EDUCATION-tagged segments from the CV and return ONE structured
object with the HIGHEST degree the candidate holds.

OUTPUT FIELDS
- highest_degree : the highest degree the candidate has earned (or is currently
                   enrolled in IF they have no completed degree).
- evidence       : the exact substring(s) that support your decision.

ALLOWED DEGREE VALUES (case-sensitive, lowest → highest):
  none, high_school, associate, bachelor, master, mba, phd

RULES
- If CV lists "M.A. in Psychology" → highest = "master"
- If CV lists both Bachelor's and Master's → highest = "master"
- If CV lists "MBA" specifically → highest = "mba"
- If CV lists PhD / Doctorate / EdD / DSc → highest = "phd"
- If CV lists "MD" / "JD" → highest = "phd"
- Certifications (PMP, AWS, ICAgile, etc.) are NOT degrees; ignore them.
- If CV only mentions training, courses, or certifications without a formal
  degree, return highest = "none".
- If the segments are noise → highest = "none", evidence = "".

Output STRICTLY the JSON object. No commentary.
"""

# KNOWN LIMITATION (deferred): the rules above are US-centric and miss several
# non-English / non-US credential formats, which are extracted as "none":
#   - post-Soviet "Specialist" / "специалитет" (a 5-year programme, ISCED 7,
#     master-equivalent) — e.g. cv_engineer's "Information Security Specialist
#     (2010-2015)" is read as highest_degree="none".
#   - German "Dipl.-Ing.", generic "диплом о высшем образовании", 5-year
#     "Specialty + Qualification" wording, etc.
# Fix when revisited: add explicit mapping rules (Specialist/специалитет →
# master; 4-5 year Specialty/Qualification → bachelor) and re-run affected docs.


def education_schema_jd() -> dict:
    return {
        "type": "object",
        "properties": {
            "required_degree": {"type": "string", "enum": EDU_LEVELS},
            "evidence":        {"type": "string"},
        },
        "required": ["required_degree", "evidence"],
    }


def education_schema_cv() -> dict:
    return {
        "type": "object",
        "properties": {
            "highest_degree": {"type": "string", "enum": EDU_LEVELS},
            "evidence":       {"type": "string"},
        },
        "required": ["highest_degree", "evidence"],
    }


def build_education_prompt(edu_segments: list[dict], doc_type: str) -> str:
    compact = [{"chunk_id": s["chunk_id"], "text": s["text"]} for s in edu_segments]
    base = EDU_INSTRUCTIONS_CV if doc_type == "cv" else EDU_INSTRUCTIONS_JD
    label = "CV" if doc_type == "cv" else "JD"
    return (
        base
        + f"\nINPUT (JSON array of education-tagged segments from the {label}):\n"
        + json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        + "\n\nReturn the JSON object now."
    )


def call_gemini_education(client: genai.Client, model: str,
                          edu_segments: list[dict], doc_type: str,
                          max_retries: int = 3) -> tuple[dict, int, int, float]:
    prompt = build_education_prompt(edu_segments, doc_type)
    schema = education_schema_cv() if doc_type == "cv" else education_schema_jd()
    cfg = types.GenerateContentConfig(
        temperature=0.0,
        maxOutputTokens=512,
        responseMimeType="application/json",
        responseSchema=schema,
    )
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            t0 = time.time()
            r = client.models.generate_content(model=model, contents=prompt, config=cfg)
            latency = time.time() - t0
            parsed = getattr(r, "parsed", None)
            if not parsed:
                txt = (r.text or "").strip()
                if txt.startswith("```"):
                    txt = re.sub(r"^```(?:json)?\s*", "", txt)
                    txt = re.sub(r"\s*```$", "", txt)
                parsed = json.loads(txt)
            usage = getattr(r, "usage_metadata", None)
            in_tok  = int(getattr(usage, "prompt_token_count",     0) or 0)
            out_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
            return parsed, in_tok, out_tok, latency
        except Exception as exc:
            last_err = exc
            msg = str(exc)
            if "RESOURCE_EXHAUSTED" not in msg and "429" not in msg:
                raise
            wait_s = 5.0 * (attempt + 1)
            time.sleep(wait_s)
    raise RuntimeError(f"Gemini (education) failed after {max_retries + 1} attempts: {last_err}")


def response_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "integer"},
                        "annotations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "tag":              {"type": "string", "enum": ["skill", "experience"]},
                                    "normalized_label": {"type": "string"},
                                    "confidence":       {"type": "number"},
                                    "evidence":         {"type": "string"},
                                },
                                "required": ["tag", "normalized_label", "confidence", "evidence"],
                            },
                        },
                        "rationale": {"type": "string"},
                    },
                    "required": ["chunk_id", "annotations"],
                },
            },
        },
        "required": ["items"],
    }


def build_prompt(segments: list[dict]) -> str:
    """One prompt covering all the document's non-other segments."""
    compact = [{"chunk_id": s["chunk_id"], "tag": s["tag"], "text": s["text"]}
               for s in segments]
    return (
        SYSTEM_INSTRUCTIONS
        + "\nINPUT (JSON array of segments):\n"
        + json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        + "\n\nReturn the JSON object now."
    )


# ──────────────────────────────────────────────────────────────────────────
# Token counting + price model
# ──────────────────────────────────────────────────────────────────────────
def count_input_tokens(client: genai.Client, model: str, prompt: str) -> int:
    """Use Gemini's count_tokens endpoint (free) to get exact input tokens."""
    try:
        r = client.models.count_tokens(model=model, contents=prompt)
        return int(r.total_tokens)
    except Exception:
        # Conservative fallback: ~4 chars per token.
        return max(1, len(prompt) // 4)


def estimate_cost(in_tokens: int, out_tokens_est: int) -> float:
    return (in_tokens     * PRICE_PER_M_INPUT_USD  / 1_000_000
          + out_tokens_est * PRICE_PER_M_OUTPUT_USD / 1_000_000)


# ──────────────────────────────────────────────────────────────────────────
# Gemini call + retry
# ──────────────────────────────────────────────────────────────────────────
def call_gemini(client: genai.Client, model: str, prompt: str,
                max_retries: int = 3) -> tuple[dict, int, int, float]:
    """Call Gemini once (batched). Returns (parsed_json, in_tokens, out_tokens, latency_s)."""
    cfg = types.GenerateContentConfig(
        temperature=0.0,
        maxOutputTokens=8192,
        responseMimeType="application/json",
        responseSchema=response_schema(),
    )
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            t0 = time.time()
            r = client.models.generate_content(model=model, contents=prompt, config=cfg)
            latency = time.time() - t0
            parsed = getattr(r, "parsed", None)
            if not parsed:
                txt = (r.text or "").strip()
                if txt.startswith("```"):
                    txt = re.sub(r"^```(?:json)?\s*", "", txt)
                    txt = re.sub(r"\s*```$", "", txt)
                parsed = json.loads(txt)
            usage = getattr(r, "usage_metadata", None)
            in_tok  = int(getattr(usage, "prompt_token_count",     0) or 0)
            out_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
            return parsed, in_tok, out_tok, latency
        except Exception as exc:
            last_err = exc
            msg = str(exc)
            if "RESOURCE_EXHAUSTED" not in msg and "429" not in msg:
                raise
            wait_s = 5.0 * (attempt + 1)
            m = re.search(r"retry in ([0-9.]+)s", msg, flags=re.IGNORECASE)
            if m:
                wait_s = max(wait_s, float(m.group(1)))
            print(f"  rate-limited; sleeping {wait_s:.1f}s (attempt {attempt + 1})")
            time.sleep(wait_s)
    raise RuntimeError(f"Gemini failed after {max_retries + 1} attempts: {last_err}")


# ──────────────────────────────────────────────────────────────────────────
# Per-document driver
# ──────────────────────────────────────────────────────────────────────────
def process_one(client: genai.Client | None, model: str,
                doc_id: str, doc_type: str,
                estimate_only: bool, force: bool) -> dict:
    """Run normalisation on one document. Returns a summary row."""
    out_dir = OUT_DIRS.get(doc_type, OUT_BASE)
    out_path = out_dir / f"{doc_id}.json"
    if out_path.exists() and not force and not estimate_only:
        return {"document_id": doc_id, "document_type": doc_type,
                "skipped": True, "reason": "already exists (use --force to refresh)"}

    all_segments = load_document(doc_id, doc_type)
    if not all_segments:
        return {"document_id": doc_id, "document_type": doc_type,
                "skipped": True, "reason": "no non-other segments"}

    # Split: skills/experience/mixed go to the main batched call;
    # education segments go to a separate JD-only education extraction.
    segments    = [s for s in all_segments if s["tag"] != "education"]
    edu_segments = [s for s in all_segments if s["tag"] == "education"]

    # ── Main pass (skills + experience batched call) ──────────────────────
    prompt = build_prompt(segments) if segments else ""
    in_tok_main_est = 0
    if segments:
        if client is not None:
            in_tok_main_est = count_input_tokens(client, model, prompt)
        else:
            in_tok_main_est = max(1, len(prompt) // 4)
    out_tok_main_est = max(200, 30 * len(segments) + 100) if segments else 0

    # ── Education pass (CV + JD + anchor; only when edu-tagged segments exist) ──
    run_edu = bool(edu_segments)
    in_tok_edu_est = 0
    out_tok_edu_est = 0
    if run_edu:
        edu_prompt = build_education_prompt(edu_segments, doc_type)
        if client is not None:
            in_tok_edu_est = count_input_tokens(client, model, edu_prompt)
        else:
            in_tok_edu_est = max(1, len(edu_prompt) // 4)
        out_tok_edu_est = 150   # small fixed object

    in_tok_est = in_tok_main_est + in_tok_edu_est
    out_tok_est = out_tok_main_est + out_tok_edu_est
    cost_est = estimate_cost(in_tok_est, out_tok_est)

    if estimate_only:
        return {
            "document_id": doc_id, "document_type": doc_type,
            "n_segments_input": len(segments),
            "n_edu_segments":   len(edu_segments),
            "prompt_chars":     len(prompt),
            "tokens_in_est":    in_tok_est,
            "tokens_out_est":   out_tok_est,
            "cost_usd_est":     round(cost_est, 5),
            "skipped":          True, "reason": "estimate-only mode",
        }

    assert client is not None
    in_tok_total = 0
    out_tok_total = 0
    latency_total = 0.0

    # Main batched call (may be a no-op if doc has only education segments)
    if segments:
        parsed, in_tok, out_tok, latency = call_gemini(client, model, prompt)
        in_tok_total  += in_tok
        out_tok_total += out_tok
        latency_total += latency
        items_by_chunk = {it["chunk_id"]: it for it in parsed.get("items", [])}
    else:
        items_by_chunk = {}

    full_items: list[dict] = []
    for seg in segments:
        it = items_by_chunk.get(seg["chunk_id"], {})
        full_items.append({
            "chunk_id":     seg["chunk_id"],
            "tag_original": seg["tag"],
            "text":         seg["text"],
            "annotations":  it.get("annotations", []),
            "rationale":    it.get("rationale", ""),
        })

    # Education pass — CV returns highest_degree; JD/anchor returns required_degree.
    education_payload: dict | None = None
    if run_edu:
        edu_parsed, edu_in, edu_out, edu_latency = call_gemini_education(
            client, model, edu_segments, doc_type,
        )
        in_tok_total  += edu_in
        out_tok_total += edu_out
        latency_total += edu_latency
        if doc_type == "cv":
            education_payload = {
                "highest_degree": edu_parsed.get("highest_degree", "none"),
                "evidence":       edu_parsed.get("evidence", ""),
                "source_chunk_ids": [s["chunk_id"] for s in edu_segments],
                "n_edu_segments":   len(edu_segments),
            }
        else:  # jd or anchor
            education_payload = {
                "required_degree": edu_parsed.get("required_degree", "none"),
                "evidence":        edu_parsed.get("evidence", ""),
                "source_chunk_ids": [s["chunk_id"] for s in edu_segments],
                "n_edu_segments":   len(edu_segments),
            }
    else:
        # No edu-tagged segments → explicit "none"
        if doc_type == "cv":
            education_payload = {"highest_degree":  "none", "evidence": "",
                                 "source_chunk_ids": [], "n_edu_segments": 0}
        else:
            education_payload = {"required_degree": "none", "evidence": "",
                                 "source_chunk_ids": [], "n_edu_segments": 0}

    cost_actual = estimate_cost(in_tok_total, out_tok_total)
    payload: dict = {
        "document_id":      doc_id,
        "document_type":    doc_type,
        "model":            model,
        "n_segments_input": len(segments),
        "n_edu_segments":   len(edu_segments),
        "tokens": {"input": in_tok_total, "output": out_tok_total,
                   "total": in_tok_total + out_tok_total},
        "latency_s":        round(latency_total, 2),
        "cost_usd":         round(cost_actual, 5),
        "items":            full_items,
    }
    if education_payload is not None:
        payload["education"] = education_payload

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    n_skill = sum(1 for it in full_items for a in it["annotations"] if a["tag"] == "skill")
    n_exp   = sum(1 for it in full_items for a in it["annotations"] if a["tag"] == "experience")
    return {
        "document_id": doc_id, "document_type": doc_type,
        "n_segments_input": len(segments),
        "n_edu_segments":   len(edu_segments),
        "tokens_in": in_tok_total, "tokens_out": out_tok_total,
        "cost_usd": round(cost_actual, 5),
        "latency_s": round(latency_total, 2),
        "n_skill_annotations": n_skill,
        "n_experience_annotations": n_exp,
        "education_required":  (education_payload or {}).get("required_degree"),
        "education_highest":   (education_payload or {}).get("highest_degree"),
        "skipped": False,
    }


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cv", default=None,
                   help="Single CV id (e.g. cv_primary). Omit to process all.")
    p.add_argument("--jd", default=None,
                   help="Single JD id (e.g. maki_people_senior_psychometrician). "
                        "Omit to process all.")
    p.add_argument("--anchor", default=None,
                   help="Single anchor id (e.g. max_cv_hr, min_space_nurse). "
                        "Anchors live in labelled/anchors/ and behave like JDs.")
    p.add_argument("--only", choices=["cv", "jd", "anchor"], default=None,
                   help="Process only docs of this type "
                        "(use with single id, or to scope a full-library run).")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--estimate-only", action="store_true",
                   help="Count tokens + estimate cost, do NOT call the API.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing outputs (default is to skip).")
    args = p.parse_args()

    cv_ids_all, jd_ids_all, anchor_ids_all = list_all_docs()
    # If a single id is given, drop the other categories unless --only says otherwise.
    any_single = args.cv or args.jd or args.anchor
    if args.cv:     cv_ids     = [args.cv]
    elif any_single and args.only != "cv":     cv_ids = []
    else:           cv_ids     = cv_ids_all
    if args.jd:     jd_ids     = [args.jd]
    elif any_single and args.only != "jd":     jd_ids = []
    else:           jd_ids     = jd_ids_all
    if args.anchor: anchor_ids = [args.anchor]
    elif any_single and args.only != "anchor": anchor_ids = []
    else:           anchor_ids = anchor_ids_all
    if args.only:
        cv_ids     = cv_ids     if args.only == "cv"     else []
        jd_ids     = jd_ids     if args.only == "jd"     else []
        anchor_ids = anchor_ids if args.only == "anchor" else []
    total_docs = len(cv_ids) + len(jd_ids) + len(anchor_ids)
    print(f"Scope: {len(cv_ids)} CV(s) + {len(jd_ids)} JD(s) "
          f"+ {len(anchor_ids)} anchor(s) = {total_docs} documents to normalise")
    print(f"Model: {args.model}    estimate_only={args.estimate_only}    force={args.force}")
    print()

    client: genai.Client | None = None
    if not args.estimate_only:
        client = genai.Client(api_key=load_api_key())
    else:
        # Even in estimate-only we instantiate a client so count_tokens works;
        # if the key isn't there we fall back to a char-based proxy.
        try:
            client = genai.Client(api_key=load_api_key())
        except SystemExit:
            print("  (no API key — falling back to ~4 chars/token proxy)\n")

    rows: list[dict] = []
    print(f"{'doc_id':<55} {'type':<3} {'segs':>4} {'in_tok':>7} {'out_tok':>7} {'cost$':>7}  status")
    print("─" * 105)

    # ── Open the summary CSV up-front and flush after every row so that
    #    a mid-batch crash still leaves a valid, partially-populated CSV. ──
    summary_fh = None
    summary_writer = None
    if not args.estimate_only:
        OUT_BASE.mkdir(parents=True, exist_ok=True)
        fields = ["document_id", "document_type", "n_segments_input",
                  "n_edu_segments", "tokens_in", "tokens_out", "cost_usd",
                  "latency_s", "n_skill_annotations", "n_experience_annotations",
                  "education_required", "education_highest",
                  "skipped", "reason"]
        # Append mode: re-running after a partial crash adds new rows without
        # nuking what's already there. Header is written only when the file
        # is empty (fresh run, or rotated by the user).
        fresh = not SUMMARY_CSV.exists() or SUMMARY_CSV.stat().st_size == 0
        summary_fh = SUMMARY_CSV.open("a", newline="", encoding="utf-8")
        summary_writer = csv.DictWriter(summary_fh, fieldnames=fields,
                                        extrasaction="ignore")
        if fresh:
            summary_writer.writeheader()
            summary_fh.flush()

    def _do(doc_id: str, doc_type: str) -> None:
        row = process_one(client, args.model, doc_id, doc_type,
                          estimate_only=args.estimate_only, force=args.force)
        rows.append(row)
        print_row(row)
        if summary_writer is not None:
            summary_writer.writerow(row)
            summary_fh.flush()             # disk-flush after every row

    try:
        for doc_id in cv_ids:     _do(doc_id, "cv")
        for doc_id in jd_ids:     _do(doc_id, "jd")
        for doc_id in anchor_ids: _do(doc_id, "anchor")
    finally:
        if summary_fh is not None:
            summary_fh.close()
            print(f"\nSummary CSV → {SUMMARY_CSV.relative_to(ROOT)}")

    # Totals
    real = [r for r in rows if not r.get("skipped")]
    est  = [r for r in rows if r.get("reason") == "estimate-only mode"]
    if real:
        print(f"\nProcessed {len(real)} docs.  "
              f"in_tokens={sum(r['tokens_in'] for r in real):,}  "
              f"out_tokens={sum(r['tokens_out'] for r in real):,}  "
              f"cost=${sum(r['cost_usd'] for r in real):.4f}")
    if est:
        print(f"\nEstimate-only across {len(est)} docs.  "
              f"projected in_tokens={sum(r['tokens_in_est'] for r in est):,}  "
              f"projected out_tokens={sum(r['tokens_out_est'] for r in est):,}  "
              f"projected cost=${sum(r['cost_usd_est'] for r in est):.4f}")


def print_row(row: dict) -> None:
    doc_id = row.get("document_id", "")
    typ    = row.get("document_type", "")
    segs   = row.get("n_segments_input", "")
    if row.get("skipped"):
        if row.get("reason") == "estimate-only mode":
            print(f"{doc_id:<55} {typ:<3} {segs:>4} "
                  f"{row.get('tokens_in_est', 0):>7} "
                  f"{row.get('tokens_out_est', 0):>7} "
                  f"{row.get('cost_usd_est', 0):>7.5f}  estimate")
        else:
            print(f"{doc_id:<55} {typ:<3} {segs:>4} "
                  f"{'-':>7} {'-':>7} {'-':>7}  skip ({row.get('reason')})")
    else:
        print(f"{doc_id:<55} {typ:<3} {segs:>4} "
              f"{row.get('tokens_in', 0):>7} "
              f"{row.get('tokens_out', 0):>7} "
              f"{row.get('cost_usd', 0):>7.5f}  "
              f"skill={row.get('n_skill_annotations', 0)} "
              f"exp={row.get('n_experience_annotations', 0)}")


if __name__ == "__main__":
    main()
