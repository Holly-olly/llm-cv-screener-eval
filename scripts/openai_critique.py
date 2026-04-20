#!/usr/bin/env python3
"""
OpenAI Critique — External AI Judge for Career Pilot Evaluation
Model: o3 with reasoning_effort="high"

Strategy: 3-pass chunked review + synthesis
-----------------------------------------------
Sending 19k tokens in one shot to any LLM produces shallow coverage —
the model focuses on the start/end and drifts in the middle. Worse, it
will make confident claims about things that are merely absent from its
current attention window.

This script splits the report into 3 focused chunks and makes 4 API calls:

  Pass 1 — Framework & Design     (Sections 1–6)
  Pass 2 — Results & Evidence     (Sections 7–12)
  Pass 3 — Synthesis & Readiness  (Sections 13–18)
  Pass 4 — Cross-cutting synthesis of all three critiques

Each pass includes:
  - Verified data facts from labels.json as ground truth
  - Specific questions calibrated to that content type
  - Instruction to cite exact text and flag "not in this chunk" vs "missing"

Anti-hallucination measures:
  - Ground truth block at top of every prompt (real numbers from labels.json)
  - Ask model to quote text when criticising
  - Ask model to distinguish "I don't see it in this chunk" vs "it's absent from the report"
  - reasoning_effort="high" — o3 spends more compute before answering
  - Structured output format — no free-form wandering

Setup:
    OPENAI_API_KEY=sk-... must be in .env.local (project root) or exported in shell
    pip install openai

Usage:
    cd ~/Documents/cool-cohen
    python3 evaluation/openai_critique.py
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from openai import OpenAI


# ── Config ─────────────────────────────────────────────────────────────────────

MODEL            = 'o3'
REASONING_EFFORT = 'high'    # 'low' = fast/cheap, 'medium' = default, 'high' = thorough
MAX_TOKENS       = 4000      # per call (completion tokens)

REPORT_PATH = Path('docs/case_study.md')
LABELS_PATH = Path('evaluation/labels.json')
OUTPUT_FILE = Path('evaluation/openai_critique_report.md')

# Section numbers to include in each pass
# (matched against "## N." headers — private section and references always excluded)
# Note: Section 12 is now a placeholder (Track E removed). Reviewer will see this.
PASS_SECTIONS = {
    'pass1': list(range(1, 7)),    # 1–6: Introduction → Data & Annotation
    'pass2': list(range(7, 13)),   # 7–12: Validity → Output Quality (§12 is placeholder)
    'pass3': list(range(13, 19)),  # 13–18: Human Factors → Conclusion
}


# ── Key loading ─────────────────────────────────────────────────────────────────

def load_openai_key() -> str:
    env_path = Path('.env.local')
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith('OPENAI_API_KEY='):
                return line.split('=', 1)[1].strip()
    key = os.environ.get('OPENAI_API_KEY', '')
    if key:
        return key
    raise RuntimeError(
        'No OpenAI API key found.\n'
        'Add OPENAI_API_KEY=sk-... to .env.local or export it in your shell.'
    )


# ── Report chunking ─────────────────────────────────────────────────────────────

def load_report_sections(path: Path) -> dict[str, str]:
    """
    Split report into top-level sections by '## N.' headers.
    Returns dict: {'1': 'full text of section 1 including subsections', ...}
    Sections starting with PRIVATE or References are excluded.
    """
    text = path.read_text(encoding='utf-8')

    # Split on top-level ## headers
    parts = re.split(r'\n(?=## )', text)
    sections = {}

    for part in parts:
        # Match "## N. Title" — numbered section
        m = re.match(r'## (\d+)\.', part.strip())
        if m:
            sections[m.group(1)] = part.strip()

    return sections


def get_chunk(sections: dict[str, str], section_numbers: list[int]) -> str:
    """Return concatenated text for the given section numbers."""
    chunks = []
    for n in section_numbers:
        key = str(n)
        if key in sections:
            chunks.append(sections[key])
        else:
            chunks.append(f'## {n}. [Section not yet written — marked as planned]')
    return '\n\n---\n\n'.join(chunks)


# ── Data grounding ──────────────────────────────────────────────────────────────

def build_ground_truth_block(labels_path: Path) -> str:
    """
    Extract verified empirical facts from labels.json.
    These are injected into every prompt so the model can check
    whether the report's claims match the actual data.
    """
    with open(labels_path) as f:
        data = json.load(f)

    import numpy as np

    all_jds      = data['labels']
    scored       = [j for j in all_jds if j.get('ai_score') is not None]
    human_labeled = [j for j in scored if j.get('human_relevance') is not None]
    ai_only      = [j for j in scored if j.get('human_relevance') is None]

    scores_all = [j['ai_score'] for j in scored]
    scores_h   = [j['ai_score'] for j in human_labeled]

    by_label: dict[int, list] = {}
    for j in human_labeled:
        k = j['human_relevance']
        by_label.setdefault(k, []).append(j['ai_score'])

    verdicts = {}
    for j in scored:
        v = j.get('ai_verdict', '')
        if v:
            verdicts[v] = verdicts.get(v, 0) + 1

    exact65  = sum(1 for s in scores_all if s == 65)
    band6070 = sum(1 for s in scores_all if 60 <= s <= 70)

    lines = [
        '## VERIFIED DATA FACTS (from labels.json — treat as ground truth)',
        f'Total JDs: {len(all_jds)}',
        f'AI-scored: {len(scored)}  |  Human-labeled: {len(human_labeled)}  |  AI-only: {len(ai_only)}',
        f'Human-labeled score stats: mean={sum(scores_h)/len(scores_h):.1f}  '
        f'min={min(scores_h):.0f}  max={max(scores_h):.0f}',
        f'All-scored: mean={sum(scores_all)/len(scores_all):.1f}  '
        f'min={min(scores_all):.0f}  max={max(scores_all):.0f}',
        f'65-cluster: exact score=65 → {exact65} JDs ({100*exact65/len(scores_all):.1f}%)',
        f'            60–70 band  → {band6070} JDs ({100*band6070/len(scores_all):.1f}%)',
        f'Verdict distribution: { {k: f"{v} ({100*v/len(scored):.0f}%)" for k, v in verdicts.items()} }',
        'Human label distribution (0=mismatch … 3=strong fit):',
    ]
    for k in sorted(by_label):
        vs = by_label[k]
        lines.append(
            f'  label={k}: n={len(vs)}  mean_ai={sum(vs)/len(vs):.1f}  '
            f'scores={sorted(vs)}'
        )

    return '\n'.join(lines)


# ── Prompts ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an independent expert reviewer with deep expertise in:
- Psychometrics and measurement theory (IRT, CFA, validity frameworks, reliability)
- LLM evaluation methodology (benchmarking, calibration, AI evaluation design)
- Applied people analytics and HR tech

PUBLICATION CONTEXT — read carefully before responding:
This is a case study report being published on GitHub as a portfolio project.
It will be linked from LinkedIn posts aimed at TWO audiences simultaneously:
  (1) Scientists / practitioners — psychometricians, people data scientists,
      AI evaluation researchers. They will judge methodological rigor.
  (2) Business people — hiring managers, HR directors, recruiters.
      They need to understand findings without a statistics background.

The report must be: clear, honest, scientifically grounded, and accessible.
It documents an evaluation of an LLM-based CV–job fit scoring system using
psychometric principles (validity, reliability, fairness) and AI evaluation methods.

YOUR SPECIFIC JOB:
Find issues that fall into EXACTLY TWO categories:
  (A) SCIENTIFIC CONTRADICTION — something that contradicts standard methodology
      in psychometrics or AI evaluation (wrong interpretation, overclaimed conclusion,
      unjustified inference, missing control, standard error in stats reporting).
  (B) MATURITY SIGNAL — something that looks unfinished or amateur to a scientist
      (unsupported assertion, weak evidence presented as strong, obvious caveat missing,
      claim stated with more confidence than the data warrant).

DO NOT flag:
  - Style preferences or writing quality issues
  - Things that are missing but already marked as "planned" placeholders
  - Improvements that would require running new analysis or writing new code
    UNLESS the existing analysis is fundamentally misleading without it
  - Completeness concerns beyond what the scope explicitly claims

CRITICAL rules for your response:
1. When you criticise a claim, QUOTE the exact text from the report.
2. If something is absent from the chunk you received, say "not in this chunk"
   — do NOT assume it is missing from the full report.
3. Ground all numerical claims against the VERIFIED DATA FACTS block.
   Flag any discrepancy.
4. Every finding must map to category A (scientific contradiction) or B (maturity signal).
5. Be specific. No vague criticism."""


PASS1_QUESTIONS = """
You are reviewing SECTIONS 1–6 of the case study report.
Section map: §1 Introduction | §2 Scope & Use Case | §3 Legal & Ethics |
             §4 Construct Definition | §5 Measurement Design | §6 Data & Annotation

Flag ONLY issues in categories A (scientific contradiction) or B (maturity signal).
Skip anything that requires new analysis or is already marked as a planned placeholder.

### Question 1 — Construct and measurement alignment (§4 + §5)
Does the construct defined in §4 (5-dimensional fit with explicit weights) match
what the system actually measures in §5? Specifically:
- Are the dimension weights (35/25/20/15/5) justified by any evidence or reference,
  or presented as arbitrary choices? Category A or B?
- §4.4 describes a custom 21-item skill vocabulary returned 100% for psychometrician
  roles. Is this limitation adequately characterised given the weight assigned to
  technical skills (25%)?

### Question 2 — Decision-support framing (§2.1)
The report frames Career Pilot as "decision-support, not automated decision-making."
Does any language elsewhere in these sections contradict this framing?
Quote any slip. If consistent, say "consistent."

### Question 3 — Data and annotation quality (§6)
Single annotator, n=32. What specific psychometric concerns does this raise that
are NOT already disclosed in §6.5? Focus only on what is missing that a reviewer
would flag — not what is already acknowledged.
Also: Is the binary threshold (H ≥ 2 = positive) justified? Is it standard practice
or an arbitrary choice that requires more argument?

### Question 4 — Data accuracy
Cross-check all numbers in §1–6 against VERIFIED DATA FACTS. Report any mismatch.

Format:
**Q1 — Construct/Measurement Alignment** — category (A/B) + quoted text + issue
**Q2 — Decision-Support Framing** — consistent / or: quoted slip + issue
**Q3 — Annotation Quality Gaps** — list only what is NOT already disclosed
**Q4 — Data Accuracy** — mismatches found, or "none"
"""


PASS2_QUESTIONS = """
You are reviewing SECTIONS 7–11 of the case study report.
(Section 12 is a placeholder — skip it.)
Section map: §7 Validity Evidence | §8 Reliability | §9 Robustness |
             §10 Fairness & Bias | §11 Performance as Decision Support

Flag ONLY issues in categories A (scientific contradiction) or B (maturity signal).
Do not propose improvements requiring new analysis unless the current framing is
fundamentally misleading.

### Question 1 — Statistical reporting completeness (§7.1–§7.4)
For each of the following claims, check whether sample size, test statistic, and
confidence interval are present. Flag where they are absent and the absence
makes the claim look unsubstantiated:
- Human–AI agreement (§7.1): accuracy, FPR, FNR, κ
- Calibration (§7.2): R², Pearson r
- Discriminant validity (§7.3): mean gap, CI, t-test, Cohen's d
- Structured layer (§7.4): r, CI

### Question 2 — The 65-cluster conclusion (§7.1, §7.4, §15)
The report concludes the 65-cluster cannot be fixed by prompting and is
"task-level ambiguity rather than a model-specific failure."
- Is this conclusion adequately supported by the evidence (3 models, McNemar p=1.0)?
- Quote any text where this conclusion is stated with more confidence than
  the data warrant. Category A or B?

### Question 3 — Ranking metrics (§11.1)
NDCG@5 and MRR are used as primary performance metrics.
- Given n=32 JDs (16 positive, 16 negative), is NDCG@5 the right metric?
  Is the sample large enough to make NDCG@5 = 0.956 a meaningful claim,
  or is this an overreach given sample size?
- Is MRR = 1.0 trivially expected given FNR = 0%? Does the report acknowledge this?

### Question 4 — Fairness section maturity (§10)
The company name bias test (§10.2) is flagged as post-hoc in §10.1.
Is the bias test result still presented as meaningful evidence despite this?
Does the fairness coverage (one company-name test) look immature for a portfolio
claiming psychometric rigour? Category A or B?

### Question 5 — Data accuracy
Cross-check all numbers in §7–11 against VERIFIED DATA FACTS. Report any mismatch.

Format:
**Q1 — Statistical Reporting** — for each metric: OK / or: section + quoted text + what's missing
**Q2 — 65-Cluster Conclusion** — supported / or: quoted text + specific overreach
**Q3 — Ranking Metrics** — sound / or: issue + category
**Q4 — Fairness Maturity** — adequate / or: issue + category (A/B) + recommendation (exclude/improve)
**Q5 — Data Accuracy** — mismatches, or "none"
"""


PASS3_QUESTIONS = """
You are reviewing SECTIONS 13–18 of the case study report.
Section map: §13 Human Factors | §14 Limitations | §15 Synthesis |
             §16 Track B (Future Work) | §17 Reproducibility | §18 Conclusion

Flag ONLY issues in categories A (scientific contradiction) or B (maturity signal).

### Question 1 — Limitations honesty (§14)
Is there a material limitation that is NOT acknowledged in §14?
Specifically: does the report acknowledge that all results are from a single
candidate profile and a single domain? That n=32 is too small to generalise?
If yes, is it discussed at the level of impact on each metric — or just mentioned once?

### Question 2 — Synthesis overstating (§15)
Read §15.2 ("ranking vs classification") and §15.3 ("65-cluster interpretation").
Quote any sentence where the conclusion goes beyond what the evidence supports.
Is the researcher-as-developer conflict of interest adequately surfaced?

### Question 3 — Portfolio readiness
Reading only the Abstract and §1 (Introduction):
- Is it immediately clear to a business reader what the tool does and why the
  evaluation matters?
- Is it immediately clear to a scientist what the methodology is?
- What single change would most improve first-impression clarity?

### Question 4 — What to exclude vs. improve before first publication
For each section that has substantive content (not just placeholders):
Rate it as one of:
  PUBLISH — solid enough as-is
  IMPROVE FIRST — fixable with text edits, no new analysis needed
  EXCLUDE FOR NOW — too weak, immature, or unsupported for first publication

Provide one-line justification per section.

Format:
**Q1 — Limitations** — complete / or: quoted gap + section where it should appear
**Q2 — Synthesis Overstating** — sound / or: quoted text + what's overstated
**Q3 — Portfolio First Impression** — specific single change recommended
**Q4 — Section-by-Section Readiness** — table: Section | Status | Reason
"""


SYNTHESIS_PROMPT = """
You have reviewed a case study report in three passes. Below are the three critiques.

The author is about to publish this on GitHub and link it from LinkedIn.
Audiences: scientists (psychometricians, AI evaluators) + business people (HR, managers).

Your output is a CONCRETE REVISION REPORT that the author will act on directly.
Write it as if you are handing it to the author.

---

## 1. Issues that contradict standard science (Category A)

List only genuine scientific contradictions found in the three critiques.
For each:
- **Location:** Section X.X — "[exact quoted text]"
- **Issue:** what specifically is wrong
- **Fix:** exact text change OR exclude this section/claim

Maximum 5 items. If fewer genuine contradictions, list fewer.

---

## 2. Maturity signals that would undermine credibility (Category B)

List the most visible maturity signals — things that make it look amateur to a
scientist reviewer. For each:
- **Location:** Section X.X — "[exact quoted text]"
- **Issue:** why it looks immature
- **Fix:** exact text change OR exclude this section/claim

Maximum 5 items.

---

## 3. Text changes that improve clarity for business readers

List 3–5 specific places where a business reader would be confused or lost.
For each:
- **Location:** Section X.X
- **Current text:** "[quote]"
- **Suggested text:** [rewritten version — keep same length]

---

## 4. Sections to exclude from first publication

List any full sections or subsections that should NOT appear in the first
published version. For each:
- **Section:** name it
- **Reason:** one sentence (too weak / unsupported / immature)
- **Recommendation:** exclude now and add later, OR remove entirely

---

## 5. What is strong — do not change

Name 2–3 things that are genuinely well done and should not be touched.

---

## 6. Publishing verdict

One paragraph: is this ready to publish after the listed fixes, or does it
need something more substantial? Be direct.
"""


# ── API call ────────────────────────────────────────────────────────────────────

def call_o3(client: OpenAI, system: str, user: str, label: str) -> tuple[str, dict]:
    """Make one o3 API call. Returns (text, usage_dict)."""
    print(f'  Calling o3 [{label}] ...')
    resp = client.chat.completions.create(
        model=MODEL,
        reasoning_effort=REASONING_EFFORT,
        max_completion_tokens=MAX_TOKENS,
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user',   'content': user},
        ],
    )
    text  = resp.choices[0].message.content
    usage = {
        'prompt':     resp.usage.prompt_tokens,
        'completion': resp.usage.completion_tokens,
        'total':      resp.usage.total_tokens,
    }
    print(f'    → {usage["prompt"]:,} + {usage["completion"]:,} tokens')
    return text, usage


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    api_key = load_openai_key()
    client  = OpenAI(api_key=api_key)

    print('Loading report and data...')
    sections   = load_report_sections(REPORT_PATH)
    ground_truth = build_ground_truth_block(LABELS_PATH)
    print(f'  Report sections found: {sorted(sections.keys())}')
    print(f'  Ground truth block: {len(ground_truth)} chars')
    print()

    results   = {}
    all_usage = []

    # ── Pass 1: Sections 1–6 ─────────────────────────────────────────────────
    chunk1 = get_chunk(sections, PASS_SECTIONS['pass1'])
    user1  = f'{ground_truth}\n\n---\n\n{chunk1}\n\n---\n\n{PASS1_QUESTIONS}'
    print(f'Pass 1 chunk: {len(chunk1):,} chars')
    results['pass1'], usage1 = call_o3(client, SYSTEM_PROMPT, user1, 'Pass 1: Sections 1–6')
    all_usage.append(('Pass 1', usage1))

    # ── Pass 2: Sections 7–12 ────────────────────────────────────────────────
    chunk2 = get_chunk(sections, PASS_SECTIONS['pass2'])
    user2  = f'{ground_truth}\n\n---\n\n{chunk2}\n\n---\n\n{PASS2_QUESTIONS}'
    print(f'Pass 2 chunk: {len(chunk2):,} chars')
    results['pass2'], usage2 = call_o3(client, SYSTEM_PROMPT, user2, 'Pass 2: Sections 7–12')
    all_usage.append(('Pass 2', usage2))

    # ── Pass 3: Sections 13–18 ───────────────────────────────────────────────
    chunk3 = get_chunk(sections, PASS_SECTIONS['pass3'])
    user3  = f'{ground_truth}\n\n---\n\n{chunk3}\n\n---\n\n{PASS3_QUESTIONS}'
    print(f'Pass 3 chunk: {len(chunk3):,} chars')
    results['pass3'], usage3 = call_o3(client, SYSTEM_PROMPT, user3, 'Pass 3: Sections 13–18')
    all_usage.append(('Pass 3', usage3))

    # ── Pass 4: Synthesis ────────────────────────────────────────────────────
    synthesis_input = (
        f'## Pass 1 Critique (Sections 1–6)\n\n{results["pass1"]}\n\n'
        f'## Pass 2 Critique (Sections 7–12)\n\n{results["pass2"]}\n\n'
        f'## Pass 3 Critique (Sections 13–18)\n\n{results["pass3"]}'
    )
    results['synthesis'], usage4 = call_o3(
        client, SYSTEM_PROMPT, synthesis_input + '\n\n---\n\n' + SYNTHESIS_PROMPT,
        'Pass 4: Synthesis'
    )
    all_usage.append(('Synthesis', usage4))

    # ── Build output ─────────────────────────────────────────────────────────
    total_tokens = sum(u['total'] for _, u in all_usage)
    timestamp    = datetime.now().strftime('%Y-%m-%d %H:%M')

    token_table = '\n'.join(
        f'| {label:<12} | {u["prompt"]:>8,} | {u["completion"]:>8,} | {u["total"]:>8,} |'
        for label, u in all_usage
    )

    report = f"""# OpenAI Critique — Career Pilot Evaluation Report
*Generated: {timestamp}*
*Model: {MODEL} | reasoning_effort: {REASONING_EFFORT}*

## Token Usage

| Pass         |   Prompt |  Completion |    Total |
|--------------|----------|-------------|----------|
{token_table}
| **Total**    |          |             | **{total_tokens:,}** |

---

## Pass 1 — Framework & Design (Sections 1–6)

{results['pass1']}

---

## Pass 2 — Results & Evidence (Sections 7–12)

{results['pass2']}

---

## Pass 3 — Synthesis & Publication Readiness (Sections 13–18)

{results['pass3']}

---

## Pass 4 — Cross-Cutting Synthesis & Revision Plan

{results['synthesis']}

---
*Generated by {MODEL} (reasoning_effort={REASONING_EFFORT}) acting as independent reviewer.*
*Each pass received: verified data facts + report chunk + focused questions.*
"""

    OUTPUT_FILE.write_text(report, encoding='utf-8')
    print()
    print(f'Report saved: {OUTPUT_FILE}')
    print(f'Total tokens used: {total_tokens:,}')
    print()
    print('--- SYNTHESIS PREVIEW ---')
    print(results['synthesis'][:600])
    print('...')


if __name__ == '__main__':
    main()
