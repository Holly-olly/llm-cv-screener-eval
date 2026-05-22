#!/usr/bin/env python3
"""
Level 3 — Step 3: Structured JSON line-tagging.

Single canonical taxonomy: skills | experience | education | mixed | other.

Design:
  - LLM does NOT echo line text. Input is a JSON list of {line_id, text};
    output is a JSON object {"labels": [{line_id, tag}]}.
  - Single string `tag` per line (no arrays, no multi-tagging — `mixed`
    is the canonical form for a line covering both skill and experience).
  - Compact JSON (no indent) to minimise prompt tokens.
  - Few-shot examples are stratified: up to N_PER_TAG lines per tag from
    each example file, instead of the full JD.
  - Retry on rate limits / 5xx / malformed JSON with exponential backoff.

Definitions used in the prompt (aligned with Level 1 P2):
  skills      — "can do": ability to apply knowledge and competencies.
                Technologies, tools, frameworks, methods, soft skills,
                language fluency.
  experience  — "has done": demonstrated application of skills in a role.
                Domain similarity + role relevance.
  education   — formal qualifications: degrees, certifications, universities,
                academic backgrounds.
  mixed       — a single line covering BOTH a named skill AND demonstrated
                experience (e.g., "5+ years of Python development").
  other       — everything else: benefits, salary, company info, culture,
                generic statements, application instructions, LinkedIn UI.

Reads:
  - data/labeled_rag_jd/*.txt           hand-labelled few-shot examples
  - data/labeled-jds/*.txt              candidate JDs to test on
  - data/unlabeled-jds/*.txt            ditto

Writes per JD:
  - results/level3/llm_labelled_json/{jd_id}_input.json
  - results/level3/llm_labelled_json/{jd_id}_output.json
  - results/level3/llm_labelled_json/{jd_id}.csv     (joined)

Run:
    cd ~/Documents/cool-cohen/llm_evaluation
    python3 scripts/level3/03_llm_label_json.py
"""

import csv
import json
import os
import random
import re
import time
from pathlib import Path

from openai import OpenAI


# ── Config ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent.parent
EXAMPLES  = ROOT / "data" / "labeled_rag_jd"
JD_DIRS   = [ROOT / "data" / "labeled-jds", ROOT / "data" / "unlabeled-jds"]
OUT_DIR   = ROOT / "results" / "level3" / "llm_labelled_json"

MODEL       = "gpt-4o-mini"
TEMPERATURE = 0.1            # 0–0.2, deterministic generation
TOP_P       = 1.0
SEED        = 42

# Run scope:
#   N_LIMIT = None     → process every JD in JD_DIRS that isn't a few-shot example
#   N_LIMIT = integer  → random sample (seed=SEED) for quick tests
#   ONLY_LABELED = True → restrict to data/labeled-jds (the main / psychometric pool)
ONLY_LABELED  = False        # include hr_extra + engineer_extra (unlabeled-jds)
N_LIMIT       = None
SKIP_EXISTING = True         # skip JDs whose _output.json already exists

# Few-shot strategy:
#   - Files in FULL_EXAMPLES are kept whole (most ambiguous cases — meta).
#   - All other example files are stratified by STRATIFIED_QUOTAS.
FULL_EXAMPLES      = {"meta_data_analyst", "Senior_AI Engineer_MoveUp"}
STRATIFIED_QUOTAS  = {"skills": 3, "experience": 3, "other": 3}

MAX_RETRIES          = 4
BASE_BACKOFF_S       = 2.0

ALLOWED_TAGS = ["skills", "experience", "education", "mixed", "other"]
ALLOWED_SET  = set(ALLOWED_TAGS)
TAG_ALIAS    = {"skill": "skills", "exp": "experience", "edu": "education"}
TAG_RE       = re.compile(r"^((?:\[[a-z]+\])+)\s*(.*)$")


# ── Prompt blocks ───────────────────────────────────────────────────────────
DEFINITIONS = """\
TAGS — assign EXACTLY ONE tag per line, drawn from this closed vocabulary:

- skills      : "can do". Concrete capabilities: tools, frameworks,
                methods, language fluency, specific soft skills.
                NOT vague aspirational traits ("drive", "passion",
                "mindset", "hunger", "grit") — those are `other`.
- experience  : "has done". Demonstrated application of skills in a role.
                Consider as DOMAIN similarity + ROLE relevance.
                Years, seniority, prior industries, day-to-day duties.
- education   : Formal qualifications: degrees, certifications, universities,
                academic backgrounds.
- mixed       : A single line covering BOTH a named skill AND demonstrated
                experience. Example: "5+ years of Python development
                experience" (Python = named skill + 5+ years = demonstrated).
                Use `mixed` instead of two tags.
- other       : Everything else — benefits, compensation, perks, company
                description, mission, culture, location, work schedule,
                application instructions, LinkedIn UI noise.
"""


SYSTEM_PROMPT = """You are a careful data-annotation assistant.

INPUT FORMAT
  A JSON array of objects, each {"line_id": int, "text": string}.

OUTPUT FORMAT
  A JSON object: {"labels": [{"line_id": int, "tag": string}, ...]}
  - `tag` is a SINGLE string. Never an array. Never two tags.
  - Allowed values: "skills", "experience", "education", "mixed", "other".

ABSOLUTE RULES
1. Do NOT echo input text. Output contains ONLY line_id and tag.
2. Every input line_id must appear EXACTLY ONCE in the output, in the same
   order as the input.
3. Each line gets exactly ONE tag from the closed vocabulary above. If a
   line covers both skill and experience, use "mixed" — never two tags.
4. Output strictly valid JSON. No commentary, no markdown fences, no
   trailing prose.
"""


# ── Few-shot example construction ───────────────────────────────────────────
def parse_labelled_line(line: str) -> tuple[str, str] | None:
    """Parse '[tag1][tag2] content' → (tag, content). Returns None for blanks.

    Single-tag enforcement:
      - normalises `[skill]` → `[skills]` etc. via TAG_ALIAS.
      - if multiple tags include both skills and experience → collapses to "mixed".
      - if multiple non-canonical tags remain → picks the first valid one.
    """
    m = TAG_RE.match(line)
    if not m:
        return None
    tags_str, content = m.group(1), m.group(2)
    if not content.strip():
        return None
    raw_tags = re.findall(r"\[([a-z]+)\]", tags_str)
    tags = [TAG_ALIAS.get(t, t) for t in raw_tags]
    constructs = {t for t in tags if t in {"skills", "experience"}}
    if constructs == {"skills", "experience"}:
        return "mixed", content
    # First valid tag wins.
    for t in tags:
        if t in ALLOWED_SET:
            return t, content
    return None


def stratified_sample(pairs: list[tuple[str, str]],
                      tag_quotas: dict[str, int]) -> list[tuple[str, str]]:
    """Keep at most `tag_quotas[tag]` lines per tag, preserving original order.
    Tags not in `tag_quotas` are dropped entirely.
    """
    seen: dict[str, int] = {t: 0 for t in tag_quotas}
    out: list[tuple[str, str]] = []
    for tag, content in pairs:
        if tag not in tag_quotas:
            continue
        if seen[tag] < tag_quotas[tag]:
            out.append((tag, content))
            seen[tag] += 1
    return out


def load_examples() -> list[tuple[str, list[dict], list[dict]]]:
    """For each labelled file, build (jd_id, input_json, output_json).

    Files in FULL_EXAMPLES are kept whole (preserves nuance / edge cases).
    All others are stratified by STRATIFIED_QUOTAS.
    """
    out = []
    for path in sorted(EXAMPLES.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        all_pairs: list[tuple[str, str]] = []
        for line in text.splitlines():
            parsed = parse_labelled_line(line)
            if parsed is not None:
                all_pairs.append(parsed)
        if path.stem in FULL_EXAMPLES:
            pairs = all_pairs
        else:
            pairs = stratified_sample(all_pairs, STRATIFIED_QUOTAS)
        input_json:  list[dict] = []
        output_json: list[dict] = []
        for lid, (tag, content) in enumerate(pairs):
            input_json.append({"line_id": lid, "text": content})
            output_json.append({"line_id": lid, "tag": tag})
        out.append((path.stem, input_json, output_json))
    return out


def compact(obj) -> str:
    """JSON serialisation tuned for prompt-token efficiency."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def build_user_prompt(examples: list[tuple[str, list[dict], list[dict]]],
                      target_input: list[dict]) -> str:
    parts = [DEFINITIONS, "\n---\nEXAMPLES (stratified — short, balanced):\n"]
    for i, (jd_id, inp, outp) in enumerate(examples, 1):
        parts.append(f"\n### EXAMPLE {i} — {jd_id}")
        parts.append("INPUT:  " + compact(inp))
        parts.append("OUTPUT: " + compact({"labels": outp}))
    parts.append("\n---\nNow label this JD using the same rules. "
                 "Return ONLY the JSON object, no commentary.\n")
    parts.append("INPUT: " + compact(target_input))
    return "\n".join(parts)


MAX_LINE_CHARS = 600
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


# ── Helpers ─────────────────────────────────────────────────────────────────
def strip_blanks_to_json(text: str) -> list[dict]:
    """Drop blanks, strip whitespace, number remaining lines.

    If a line exceeds MAX_LINE_CHARS (a scraped wall of text with no
    newlines), split it further by sentence boundary so the LLM can
    tag each sentence individually instead of forcing one tag on the
    whole blob.
    """
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if len(s) <= MAX_LINE_CHARS:
            out.append({"line_id": len(out), "text": s})
            continue
        for sentence in SENTENCE_SPLIT.split(s):
            sentence = sentence.strip()
            if sentence:
                out.append({"line_id": len(out), "text": sentence})
    return out


def load_openai_key() -> str:
    env_path = ROOT.parent / ".env.local"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip()
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found in .env.local or env")
    return key


def load_candidate_jds(exclude: set[str]) -> list[Path]:
    dirs = [ROOT / "data" / "labeled-jds"] if ONLY_LABELED else JD_DIRS

    # Restrict to the 75 JDs in the Level 2 master CSV — unlabeled-jds/
    # actually contains a larger pool; we only want the L2 universe.
    l2_master = ROOT / "results" / "level2_master.csv"
    l2_ids: set[str] | None = None
    if l2_master.exists():
        import pandas as pd
        l2_ids = set(pd.read_csv(l2_master)["jd_id"].unique())

    out = []
    for d in dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.txt")):
            if p.stem in exclude:
                continue
            if l2_ids is not None and p.stem not in l2_ids:
                continue
            out.append(p)
    return out


def call_openai_with_retry(client: OpenAI, system: str, user: str) -> tuple[dict, dict]:
    """Call OpenAI with retry on rate limits / 5xx / malformed JSON.

    Exponential backoff: 2, 4, 8, 16 seconds.
    """
    last_err: str | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            raw = resp.choices[0].message.content
            data = json.loads(raw)   # raises JSONDecodeError if malformed
            usage = {
                "prompt":     resp.usage.prompt_tokens,
                "completion": resp.usage.completion_tokens,
                "total":      resp.usage.total_tokens,
            }
            return data, usage
        except json.JSONDecodeError as e:
            last_err = f"malformed JSON (attempt {attempt + 1}): {e}"
        except Exception as e:
            last_err = f"{type(e).__name__} (attempt {attempt + 1}): {e}"
        if attempt < MAX_RETRIES - 1:
            sleep_s = BASE_BACKOFF_S * (2 ** attempt)
            print(f"    retry in {sleep_s:.0f}s — {last_err}")
            time.sleep(sleep_s)
    raise RuntimeError(f"OpenAI call failed after {MAX_RETRIES} attempts: {last_err}")


def validate(target_input: list[dict], output: dict) -> dict:
    """Strict validator. Returns a report dict with `all_ok` flag."""
    expected_ids = [row["line_id"] for row in target_input]
    labels       = output.get("labels", [])
    output_ids   = [row.get("line_id") for row in labels]

    report = {
        "n_input":          len(expected_ids),
        "n_output":         len(labels),
        "missing_ids":      sorted(set(expected_ids) - set(output_ids)),
        "extra_ids":        sorted(set(output_ids)   - set(expected_ids)),
        "duplicate_ids":    sorted({i for i in output_ids if output_ids.count(i) > 1}),
        "order_ok":         output_ids == expected_ids,
        "bad_tag_lines":    [],
        "multi_tag_lines":  [],
        "tag_distribution": {t: 0 for t in ALLOWED_TAGS},
    }

    for row in labels:
        lid = row.get("line_id")
        tag = row.get("tag")
        if isinstance(tag, list):
            report["multi_tag_lines"].append((lid, tag))
            if len(tag) >= 1 and isinstance(tag[0], str):
                tag = tag[0]
            else:
                report["bad_tag_lines"].append((lid, repr(row.get("tag"))))
                continue
        if not isinstance(tag, str) or tag not in ALLOWED_SET:
            report["bad_tag_lines"].append((lid, repr(row.get("tag"))))
            continue
        report["tag_distribution"][tag] += 1

    report["all_ok"] = (
        report["n_input"] == report["n_output"]
        and not report["missing_ids"]
        and not report["extra_ids"]
        and not report["duplicate_ids"]
        and report["order_ok"]
        and not report["bad_tag_lines"]
        and not report["multi_tag_lines"]
    )
    return report


def autofill_missing(target_input: list[dict], output: dict) -> set[int]:
    """Fill any line_ids the LLM forgot with tag='other'. Returns the set
    of line_ids that were auto-filled (for the CSV `auto_filled` column).
    """
    expected = [row["line_id"] for row in target_input]
    labels   = output.setdefault("labels", [])
    present  = {row.get("line_id") for row in labels}
    filled: set[int] = set()
    by_id = {row["line_id"]: row for row in labels}
    rebuilt: list[dict] = []
    for lid in expected:
        if lid in by_id:
            rebuilt.append(by_id[lid])
        else:
            rebuilt.append({"line_id": lid, "tag": "other"})
            filled.add(lid)
    output["labels"] = rebuilt
    return filled


def write_outputs(jd_id: str, target_input: list[dict], output: dict,
                  auto_filled: set[int]) -> tuple[Path, Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    in_path  = OUT_DIR / f"{jd_id}_input.json"
    out_path = OUT_DIR / f"{jd_id}_output.json"
    csv_path = OUT_DIR / f"{jd_id}.csv"

    in_path.write_text(json.dumps(target_input, ensure_ascii=False, indent=2), encoding="utf-8")
    out_path.write_text(json.dumps(output,       ensure_ascii=False, indent=2), encoding="utf-8")

    text_by_id = {row["line_id"]: row["text"] for row in target_input}
    tag_by_id  = {row["line_id"]: row.get("tag") for row in output.get("labels", [])}
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["line_id", "tag", "auto_filled", "text"])
        for lid in sorted(text_by_id):
            w.writerow([lid, tag_by_id.get(lid, ""),
                        "Y" if lid in auto_filled else "",
                        text_by_id[lid]])

    return in_path, out_path, csv_path


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    client   = OpenAI(api_key=load_openai_key())
    examples = load_examples()
    if not examples:
        raise SystemExit(f"No labelled examples found in {EXAMPLES}")

    candidates = load_candidate_jds(exclude={ex[0] for ex in examples})
    if N_LIMIT is None:
        targets = candidates
    else:
        random.seed(SEED)
        targets = random.sample(candidates, min(N_LIMIT, len(candidates)))

    print(f"Few-shot examples: {len(examples)} JDs from {EXAMPLES.relative_to(ROOT)}")
    print(f"  Strategy: FULL for {sorted(FULL_EXAMPLES)}, "
          f"stratified for the rest (quotas: {STRATIFIED_QUOTAS})")
    for jd_id, inp, outp in examples:
        tag_counts: dict[str, int] = {}
        for row in outp:
            tag_counts[row["tag"]] = tag_counts.get(row["tag"], 0) + 1
        mode = "FULL" if jd_id in FULL_EXAMPLES else "stratified"
        print(f"  · [{mode:<10}] {jd_id}  ({len(inp)} lines)  tags={tag_counts}")

    if N_LIMIT is None:
        scope = f"ALL {len(targets)} JDs from " + ("data/labeled-jds" if ONLY_LABELED else "JD_DIRS")
    else:
        scope = f"{len(targets)} random JDs (seed={SEED})"
    print(f"\nProcessing {scope}:")
    for p in targets[:10]:
        print(f"  · {p.stem}")
    if len(targets) > 10:
        print(f"  ... and {len(targets) - 10} more")
    print()

    distribution_rows = []
    total_usage = {"prompt": 0, "completion": 0, "total": 0}
    n_ok = 0
    n_skipped = 0

    for target in targets:
        jd_id        = target.stem
        out_json     = OUT_DIR / f"{jd_id}_output.json"
        if SKIP_EXISTING and out_json.exists():
            n_skipped += 1
            print(f"── {jd_id} ──  (skip — output exists)")
            continue

        raw          = target.read_text(encoding="utf-8")
        target_input = strip_blanks_to_json(raw)

        print(f"── {jd_id} ──  ({len(raw.splitlines())} raw → "
              f"{len(target_input)} content lines)")
        user_prompt = build_user_prompt(examples, target_input)

        try:
            data, usage = call_openai_with_retry(client, SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            print(f"  ✗ {e}\n")
            continue

        for k in total_usage:
            total_usage[k] += usage[k]

        auto_filled = autofill_missing(target_input, data)
        v = validate(target_input, data)
        if v["all_ok"]:
            n_ok += 1
        in_path, out_path, csv_path = write_outputs(jd_id, target_input, data, auto_filled)
        if auto_filled:
            print(f"    ⓘ auto-filled {len(auto_filled)} missing line_ids with tag=other "
                  f"(ids: {sorted(auto_filled)[:5]}{'...' if len(auto_filled) > 5 else ''})")

        status = "✓" if v["all_ok"] else "✗"
        print(f"  {status} input={v['n_input']} → output={v['n_output']}  "
              f"order_ok={v['order_ok']}")
        if v["missing_ids"]:
            print(f"    ! missing line_ids:   {v['missing_ids'][:10]}")
        if v["extra_ids"]:
            print(f"    ! extra line_ids:     {v['extra_ids'][:10]}")
        if v["duplicate_ids"]:
            print(f"    ! duplicate line_ids: {v['duplicate_ids'][:10]}")
        if not v["order_ok"]:
            print(f"    ! output order does not match input")
        if v["multi_tag_lines"]:
            print(f"    ! {len(v['multi_tag_lines'])} lines returned an array tag")
            for lid, t in v["multi_tag_lines"][:3]:
                print(f"        line_id {lid}: {t!r}")
        if v["bad_tag_lines"]:
            print(f"    ! {len(v['bad_tag_lines'])} lines have bad/empty tags")
            for lid, t in v["bad_tag_lines"][:3]:
                print(f"        line_id {lid}: {t}")
        print(f"    tokens: {usage['prompt']:,} + {usage['completion']:,} = {usage['total']:,}")
        print(f"    in/out/csv: {in_path.relative_to(ROOT)} / "
              f"{out_path.name} / {csv_path.name}\n")

        row = {"jd_id": jd_id, "lines": v["n_input"]}
        row.update({t: v["tag_distribution"].get(t, 0) for t in ALLOWED_TAGS})
        distribution_rows.append(row)

    # ── Cross-JD tag distribution ───────────────────────────────────────────
    if distribution_rows:
        print("Tag distribution across JDs:")
        cols = ["jd_id", "lines"] + ALLOWED_TAGS
        widths = {c: max(len(c), max((len(str(r[c])) for r in distribution_rows), default=0))
                  for c in cols}
        widths["jd_id"] = min(widths["jd_id"], 50)
        print("  " + "  ".join(c.ljust(widths[c]) for c in cols))
        print("  " + "  ".join("-" * widths[c] for c in cols))
        for r in distribution_rows:
            print("  " + "  ".join(str(r[c])[:widths[c]].ljust(widths[c]) for c in cols))

    # ── Token cost summary + 76-JD projection ──────────────────────────────
    print()
    n_processed = len(targets) - n_skipped
    print(f"Run summary: {n_ok}/{n_processed} validated cleanly  "
          f"(skipped {n_skipped} already-processed)")
    print(f"Total tokens: prompt={total_usage['prompt']:,}  "
          f"completion={total_usage['completion']:,}  "
          f"total={total_usage['total']:,}")
    if n_processed > 0:
        avg = total_usage["total"] / n_processed
        avg_in  = total_usage["prompt"]     / n_processed
        avg_out = total_usage["completion"] / n_processed
        # gpt-4o-mini pricing: $0.150 / 1M input, $0.600 / 1M output (May 2026)
        cost_in  = total_usage["prompt"]     * 0.150 / 1_000_000
        cost_out = total_usage["completion"] * 0.600 / 1_000_000
        print(f"Per-JD average: {avg:,.0f} tokens ({avg_in:,.0f} in + {avg_out:,.0f} out)")
        print(f"This run cost: ~${cost_in + cost_out:.3f} on gpt-4o-mini")

    print(f"\nReview outputs in: {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
