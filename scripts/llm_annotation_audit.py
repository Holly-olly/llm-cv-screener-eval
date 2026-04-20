#!/usr/bin/env python3
"""
Track D — LLM Annotation Audit
================================
Question: Are the human labels correct? Do independent LLM judges agree with
the human 0-3 ratings — especially on the 8 false-positive JDs?

Design: 32 JDs rated independently by two LLM judges using the same 0-3
instruction scale that the human annotator used.

Judges:
  - GPT-4o      (OpenAI API)
  - qwen3:8b    (local via Ollama — free, no rate limit)

Key question: The 8 FP JDs scored ≥50 by Gemini but labeled H=1 by human.
  If LLMs say H=1 → human labels correct, FPR is a real AI failure
  If LLMs say H=2 → human labels may be biased, AI is partially right

Metrics:
  - Correlation: LLM-judge vs human (per judge)
  - Inter-judge agreement: GPT-4o vs qwen3 Cohen's kappa + ICC
  - Cohen's kappa: each LLM vs human (0-3 ordinal → binarised for kappa)
  - FP audit table: what do judges say about the 8 FP cases?
  - Binarised: human ≥2 = relevant, LLM ≥2 = relevant

Output:
  evaluation/data/batch-results/llm_annotation_audit.json
  evaluation/llm_annotation_audit_report.md

Usage:
    cd /path/to/cool-cohen
    python3 evaluation/llm_annotation_audit.py

API calls: 32 × 2 judges = 64 calls (~$0.10 GPT-4o + free qwen3)
"""

import json
import os
import re
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from openai import OpenAI

# ── Model config ────────────────────────────────────────────────────────────

JUDGES = {
    'gpt-4o': {
        'client_kwargs': {},
        'model': 'gpt-4o',
        'sleep': 1,
        'max_tokens': 256,
        'strip_think': False,
    },
    'deepseek-r1:8b': {
        'client_kwargs': {'base_url': 'http://localhost:11434/v1', 'api_key': 'ollama'},
        'model': 'deepseek-r1:8b',
        'sleep': 0,
        'max_tokens': 3000,   # enough for short think + label; tested at ~68s/JD
        'strip_think': True,
        'system_prefix': 'Think briefly, then respond in the required format. ',
    },
}

# ── Company name masking (same dict as multi_model_comparison.py) ───────────

COMPANY_NAMES = {
    'maki_people':           ['Maki People'],
    'Accenture':             ['Accenture'],
    'Arrive':                ['Arrive'],
    'Atlantica':             ['Atlantica Sustainable Infrastructure', 'Atlantica'],
    'Canonical':             ['Canonical'],
    'Dandelion':             ['Dandelion Civilization'],
    'DataAnnotation':        ['DataAnnotation'],
    'G-Research':            ['G-Research', 'G Research'],
    'Johnson_Controls':      ['Johnson Controls'],
    'LinkedIn':              ['LinkedIn'],
    'Proofpoint':            ['Proofpoint'],
    'QIC':                   ['QIC Digital Hub', 'QIC'],
    'angels':                ['Angels.Space', 'Angels Space', 'Angels'],
    'anglian':               ['Anglian Water Services', 'Anglian Water'],
    'bazaarvoice':           ['Bazaarvoice'],
    'bpostgroup':            ['bpostgroup', 'bpost group', 'bpost'],
    'codility':              ['Codility'],
    'freeplay':              ['Freeplay'],
    'lightning':             ['Lightning AI', 'Lightning'],
    'meta':                  ['Meta'],
    'skillvue':              ['Skillvue'],
    'sorare':                ['Sorare'],
    'the_world_bank':        ['The World Bank', 'World Bank'],
    'unicef':                ['UNICEF'],
    'Product_Growth':        ['Wiris'],
    'Applied_AI_Engineer':   ['Impress'],
    'Customer_success':      ['CareWise'],
    'Digital_Learning':      ['Schneider Electric', 'Schneider'],
    'QIT':                   ['QIT Software', 'QIT'],
    'Senior_AI':             ['MoveUp', 'Move Up'],
}

MASK = 'Company_XX'

def mask_jd(jd_text: str, jd_id: str) -> str:
    masked = jd_text
    for prefix, names in COMPANY_NAMES.items():
        if jd_id.lower().startswith(prefix.lower()):
            for name in sorted(names, key=len, reverse=True):
                masked = re.sub(re.escape(name), MASK, masked, flags=re.IGNORECASE)
            break
    return masked

# ── Annotation prompt ───────────────────────────────────────────────────────

SYSTEM_MSG = 'You are an expert job relevance annotator. Rate how relevant a job description is for a specific candidate profile.'

# Candidate profile summary (no CV text — same info the human annotator had)
CANDIDATE_PROFILE = """Candidate profile:
- 10+ years in psychometrics: IRT, CFA, DIF, Rasch modelling
- Scale construction, automated test monitoring, DIF analysis at scale
- TestGorilla: built and maintained 350+ assessments
- Python, R, SQL — data analysis and research
- Transitioning into: People Data Scientist, Data Analyst (HR/people domain), Assessment Scientist
- NOT looking for: pure engineering roles, graphic design, marketing, unrelated domains"""

ANNOTATION_PROMPT = """Rate how relevant this job description is for the candidate below.

{candidate_profile}

---
JOB DESCRIPTION:
{jd}

---
Use this 0-3 scale (same as used by the original human annotator):
  3 = Strong fit — candidate's core skills directly match; would likely apply
  2 = Moderate fit — meaningful overlap; worth considering
  1 = Low relevance — some transferable skills but core mismatch
  0 = Not relevant — different domain entirely or clear dealbreaker

Respond in EXACTLY this format:
LABEL: [0|1|2|3]
REASONING: [1-2 sentences explaining the rating]"""

# ── Helpers ─────────────────────────────────────────────────────────────────

def load_openai_key() -> str:
    env = Path('.env.local')
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith('OPENAI_API_KEY='):
                return line.split('=', 1)[1].strip()
    return os.environ.get('OPENAI_API_KEY', '')

def parse_label(text: str, strip_think: bool = False) -> dict:
    if strip_think:
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'<\|think\|>.*?<\|/think\|>', '', text, flags=re.DOTALL)
    result = {'label': None, 'reasoning': ''}
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith('LABEL:'):
            val = line.split(':', 1)[1].strip()
            digits = ''.join(c for c in val if c.isdigit())
            if digits and 0 <= int(digits[0]) <= 3:
                result['label'] = int(digits[0])
        elif line.startswith('REASONING:'):
            result['reasoning'] = line.split(':', 1)[1].strip()
    return result

def cohen_kappa_binary(a: list, b: list, threshold: int = 2) -> float:
    """Binary kappa: ≥threshold = positive."""
    ab = [(1 if x >= threshold else 0, 1 if y >= threshold else 0) for x, y in zip(a, b)]
    n = len(ab)
    po = sum(x == y for x, y in ab) / n
    pa1 = sum(x for x, _ in ab) / n
    pb1 = sum(y for _, y in ab) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if pe < 1 else 0.0

def icc_2way(matrix: np.ndarray) -> float:
    n, k = matrix.shape
    grand = matrix.mean()
    rows = matrix.mean(axis=1)
    cols = matrix.mean(axis=0)
    SSr = k * np.sum((rows - grand) ** 2)
    SSc = n * np.sum((cols - grand) ** 2)
    SSe = np.sum((matrix - rows[:, None] - cols[None, :] + grand) ** 2)
    dfr, dfc, dfe = n - 1, k - 1, (n - 1) * (k - 1)
    MSr = SSr / dfr
    MSc = SSc / dfc if dfc > 0 else 0
    MSe = SSe / dfe if dfe > 0 else 1e-9
    return float(max(0.0, (MSr - MSe) / (MSr + (k - 1) * MSe + k * (MSc - MSe) / n)))

# ── Main ────────────────────────────────────────────────────────────────────

def main():
    openai_key = load_openai_key()

    with open('evaluation/labels.json') as f:
        labels = [l for l in json.load(f)['labels']
                  if isinstance(l.get('human_relevance'), int)
                  and not l.get('is_synthetic')]

    print(f'Track D — LLM Annotation Audit')
    print(f'{len(labels)} JDs × {len(JUDGES)} judges\n')

    # Build clients
    clients = {}
    for name, cfg in JUDGES.items():
        kwargs = dict(cfg['client_kwargs'])
        if 'base_url' not in kwargs:
            kwargs['api_key'] = openai_key
        clients[name] = OpenAI(**kwargs)

    # ── Load checkpoint ─────────────────────────────────────────────────────
    out_path = Path('evaluation/data/batch-results/llm_annotation_audit.json')
    results = {l['jd_id']: {
        'human_relevance': l['human_relevance'],
        'v1_score': l.get('ai_score'),
        'jd_id': l['jd_id'],
        'labels': {},
    } for l in labels}

    if out_path.exists():
        try:
            saved = json.loads(out_path.read_text())
            for row in saved:
                jid = row.get('jd_id')
                if jid in results and row.get('labels'):
                    results[jid]['labels'] = row['labels']
            loaded = sum(1 for r in results.values() if r['labels'])
            if loaded:
                print(f'[checkpoint] Loaded {loaded} JDs from previous run')
        except Exception:
            pass

    def save_checkpoint():
        out_path.write_text(json.dumps(list(results.values()), indent=2, ensure_ascii=False))

    # ── Score each JD with each judge ───────────────────────────────────────
    for judge_name, cfg in JUDGES.items():
        client = clients[judge_name]

        already = sum(1 for r in results.values()
                      if r['labels'].get(judge_name, {}).get('label') is not None)
        if already == len(labels):
            print(f'\n── {judge_name} — already complete ({already}/{len(labels)}), skipping')
            continue

        print(f'\n── {judge_name} ─────────────────────────────────────────')
        print(f'{"JD":<45} {"H":>2}  {"LLM":>4}  Reasoning snippet')
        print('─' * 80)

        for l in labels:
            jd_id = l['jd_id']
            if results[jd_id]['labels'].get(judge_name, {}).get('label') is not None:
                cached = results[jd_id]['labels'][judge_name]
                print(f'{jd_id[:44]:<45} {l["human_relevance"]:>2}  {cached["label"]:>4}  [cached]')
                continue

            jd_text = (Path('evaluation') / l['jd_file']).read_text()
            jd_masked = mask_jd(jd_text, jd_id)

            prompt = ANNOTATION_PROMPT.format(
                candidate_profile=CANDIDATE_PROFILE,
                jd=jd_masked,
            )

            try:
                sys_msg = cfg.get('system_prefix', '') + SYSTEM_MSG
                resp = client.chat.completions.create(
                    model=cfg['model'],
                    messages=[
                        {'role': 'system', 'content': sys_msg},
                        {'role': 'user', 'content': prompt},
                    ],
                    temperature=0.2,   # low temp for annotation consistency
                    max_tokens=cfg['max_tokens'],
                )
                raw = resp.choices[0].message.content or ''
                parsed = parse_label(raw, strip_think=cfg.get('strip_think', False))
                lbl = parsed['label']
                reasoning_short = parsed['reasoning'][:60] if parsed['reasoning'] else '—'

                agree = '' if lbl is None else (' ✓' if abs(lbl - l['human_relevance']) <= 1 else ' ✗')
                fp_flag = ' ← FP case' if l['human_relevance'] <= 1 and (l.get('ai_score') or 0) >= 50 else ''
                print(f'{jd_id[:44]:<45} {l["human_relevance"]:>2}  {str(lbl):>4}  {reasoning_short}{agree}{fp_flag}')

                results[jd_id]['labels'][judge_name] = {
                    'label': lbl,
                    'reasoning': parsed['reasoning'],
                }

            except Exception as e:
                print(f'{jd_id[:44]:<45}  ERROR: {str(e)[:60]}')
                results[jd_id]['labels'][judge_name] = {'label': None, 'reasoning': ''}

            if cfg.get('sleep', 0):
                time.sleep(cfg['sleep'])

        save_checkpoint()
        print(f'  [checkpoint saved after {judge_name}]')

    # ── Analysis ─────────────────────────────────────────────────────────────
    judge_names = list(JUDGES.keys())
    complete = [r for r in results.values()
                if all(r['labels'].get(j, {}).get('label') is not None for j in judge_names)]
    n = len(complete)

    print(f'\n{"═"*70}')
    print(f'ANNOTATION AUDIT RESULTS  (n={n})')
    print(f'{"═"*70}')

    human_labels = [r['human_relevance'] for r in complete]

    # Per-judge agreement with human
    print(f'\n  Agreement with human labels (binarised: ≥2 = relevant):')
    print(f'  {"Judge":<15}  {"r(Pearson)":>10}  {"kappa":>8}  {"ICC":>8}')
    print('  ' + '─' * 46)
    for j in judge_names:
        jlabels = [r['labels'][j]['label'] for r in complete]
        r_val = float(np.corrcoef(human_labels, jlabels)[0, 1])
        kappa = cohen_kappa_binary(human_labels, jlabels)
        mat = np.array(list(zip(human_labels, jlabels)), dtype=float)
        icc = icc_2way(mat)
        print(f'  {j:<15}  {r_val:>10.3f}  {kappa:>8.3f}  {icc:>8.3f}')

    # Inter-judge agreement
    if len(judge_names) >= 2:
        j1, j2 = judge_names[0], judge_names[1]
        l1 = [r['labels'][j1]['label'] for r in complete]
        l2 = [r['labels'][j2]['label'] for r in complete]
        inter_kappa = cohen_kappa_binary(l1, l2)
        inter_icc = icc_2way(np.array(list(zip(l1, l2)), dtype=float))
        inter_r = float(np.corrcoef(l1, l2)[0, 1])
        print(f'\n  Inter-judge agreement ({j1} vs {j2}):')
        print(f'    Pearson r = {inter_r:.3f}')
        print(f'    Cohen κ   = {inter_kappa:.3f}')
        print(f'    ICC       = {inter_icc:.3f}')

    # ── FP audit — the key question ────────────────────────────────────────
    fp_cases = [r for r in complete if r['human_relevance'] <= 1 and (r.get('v1_score') or 0) >= 50]
    print(f'\n  FP audit — {len(fp_cases)} cases where AI scored ≥50 but human labeled 0-1:')
    print(f'  {"JD":<45} {"H":>2}  {"AI":>4}  ' + '  '.join(f'{j[:8]:>8}' for j in judge_names) + '  Verdict')
    print('  ' + '─' * (55 + 10 * len(judge_names)))

    fp_llm_agree_human = 0
    fp_llm_agree_ai = 0
    for r in sorted(fp_cases, key=lambda x: x.get('v1_score') or 0, reverse=True):
        jlabels = [r['labels'][j]['label'] for j in judge_names]
        llm_mean = np.mean([l for l in jlabels if l is not None])

        # Verdict: do LLMs agree with human (label ≤1) or AI (label ≥2)?
        if llm_mean < 2:
            verdict = 'LLMs→Human ✓ (FPR is real)'
            fp_llm_agree_human += 1
        else:
            verdict = 'LLMs→AI (human may err)'
            fp_llm_agree_ai += 1

        label_str = '  '.join(f'{str(l):>8}' for l in jlabels)
        print(f'  {r["jd_id"][:44]:<45} {r["human_relevance"]:>2}  {r.get("v1_score") or 0:>4}  {label_str}  {verdict}')

    print(f'\n  Summary:')
    print(f'    LLMs agree with human (FPR is real):      {fp_llm_agree_human}/{len(fp_cases)}')
    print(f'    LLMs agree with AI (human label suspect):  {fp_llm_agree_ai}/{len(fp_cases)}')

    # ── Markdown report ────────────────────────────────────────────────────
    report_lines = [
        '# LLM Annotation Audit — Track D',
        f'*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}*',
        f'*n={n} JDs | Judges: {", ".join(judge_names)} | Masked JDs (Company_XX)*',
        '',
        '## Purpose',
        '',
        'Independent LLM judges re-rate all 32 JDs on the same 0-3 scale used by',
        'the human annotator. Key question: are the 8 FP cases (AI≥50, H≤1) truly',
        'low-relevance, or did the human annotator under-rate them?',
        '',
        '## Agreement with Human Labels',
        '',
        '| Judge | Pearson r | Cohen κ | ICC |',
        '|-------|-----------|---------|-----|',
    ]
    for j in judge_names:
        jlabels = [r['labels'][j]['label'] for r in complete]
        r_val = float(np.corrcoef(human_labels, jlabels)[0, 1])
        kappa = cohen_kappa_binary(human_labels, jlabels)
        mat = np.array(list(zip(human_labels, jlabels)), dtype=float)
        icc = icc_2way(mat)
        report_lines.append(f'| {j} | {r_val:.3f} | {kappa:.3f} | {icc:.3f} |')

    if len(judge_names) >= 2:
        report_lines += [
            '',
            '## Inter-Judge Agreement',
            '',
            f'| Metric | {j1} vs {j2} |',
            '|--------|-----------|',
            f'| Pearson r | {inter_r:.3f} |',
            f'| Cohen κ   | {inter_kappa:.3f} |',
            f'| ICC       | {inter_icc:.3f} |',
        ]

    report_lines += [
        '',
        '## FP Case Audit',
        '',
        f'**{len(fp_cases)} FP cases** (AI≥50, Human≤1) — what do independent LLM judges say?',
        '',
        f'- LLMs agree with human (FPR is real AI failure): **{fp_llm_agree_human}/{len(fp_cases)}**',
        f'- LLMs agree with AI (human label may be off):    **{fp_llm_agree_ai}/{len(fp_cases)}**',
        '',
    ]

    if fp_llm_agree_human >= len(fp_cases) * 0.75:
        report_lines.append('**Finding:** LLM judges consistently agree with human labels on FP cases.')
        report_lines.append('The FPR=50% reflects a genuine AI calibration problem, not annotation bias.')
    elif fp_llm_agree_ai >= len(fp_cases) * 0.75:
        report_lines.append('**Finding:** LLM judges side with the AI scores on most FP cases.')
        report_lines.append('Human labels may be systematically conservative — warrants review.')
    else:
        report_lines.append('**Finding:** Mixed — LLM judges split on FP cases.')
        report_lines.append('Some FP cases are genuinely ambiguous; neither AI nor human is clearly right.')

    report_lines += [
        '',
        '*Raw data: `evaluation/data/batch-results/llm_annotation_audit.json`*',
    ]

    report_path = Path('evaluation/llm_annotation_audit_report.md')
    report_path.write_text('\n'.join(report_lines), encoding='utf-8')
    print(f'\n  Report saved: {report_path}')
    print(f'  Data saved:   {out_path}')


if __name__ == '__main__':
    main()
