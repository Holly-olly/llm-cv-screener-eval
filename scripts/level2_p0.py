#!/usr/bin/env python3
"""
Level 2 — P0 prompt: Guided categorical labels (Bridge design).

LLM outputs categorical labels only — no numbers. Code applies scoring maps
and computes fit_score via fixed formula:

    fit_score = 0.6 × skill_score
              + 0.3 × (0.4 × role_relevance + 0.6 × domain_relevance)
              + 0.1 × education_score

Output: llm_evaluation/results/level2_p0_{cv_name}.json
Each record carries raw LLM labels + code-computed scores + token/latency meta
+ human ratings (holistic from H1, structural list from H1/H2/H3 when present).

Usage:
    python3 level2_p0.py                                       # 32 main JDs, cv_primary, 1 run
    python3 level2_p0.py --cv cv_hr --runs 3                   # 32 main, cv_hr, 3 runs
    python3 level2_p0.py --jd Product_Growth_Analyst_wiris     # single JD, 1 run
    python3 level2_p0.py --cv cv_primary --extras              # 44 extra JDs → level2_p0_{cv}_extra.json

The --extras flag reuses the JD list from L1 P2 extras (level1_p2_{cv}_extra.json):
all unique JDs already scored at L1 P2 (engineer-pool + hr-pool) are re-scored with
the L2 prompt. The `source` field (engineer_extra / hr_extra) is preserved from L1.
"""

import json
import os
import sys
import time
from pathlib import Path

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import google.generativeai as genai

# ── Config ────────────────────────────────────────────────────────────────────

GEMINI_MODEL = 'gemini-3.1-flash-lite-preview'
ROOT = Path(__file__).parent.parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)

# ── Prompt ────────────────────────────────────────────────────────────────────

PROMPT_L2_P0 = """You are evaluating candidate fit for a job. 
Your task is to assign categorical labels — not scores. 
Numbers will be computed separately.

CANDIDATE CV:
{cv}

JOB DESCRIPTION:
{jd}

Rate the candidate on four dimensions. Use only the allowed labels for each.

SKILLS — do the candidate's skills cover the must-have requirements of this role?
- YES: strong match across the critical skills required
- PARTIAL: some key skills present but notable gaps or only partial coverage
- NO: skills do not match the core requirements

EXPERIENCE — two sub-dimensions:
- role_relevance: how similar is the candidate's past role type to this role?
  SAME = same role functions and similar role title | SIMILAR = adjacent role | DIFFERENT = different function
- domain_relevance: how similar is the candidate's industry/domain?
  SAME = identical domain | RELATED = adjacent domain | UNRELATED = different domain

EDUCATION — does the candidate meet the stated education requirement?
- YES: meets it | PARTIAL: partially meets it | NO: does not meet it
- If no education requirement is stated: YES

HOLISTIC — overall judgment (anchored to Level 1 score bands for cross-level comparability):
- STRONG (= L1 score 3): Strong fit, candidate should apply with confidence, would shortlist for interview
- MODERATE (= L1 score 2): Partial fit — real overlap but notable gaps or dealbreakers
- WEAK (= L1 score 1): Weak overlap, surface match only, significant misalignment
- NO FIT (= L1 score 0): Not relevant at all

Confidence of judgment: 
- HIGH: signal is clear
- LOW: role is ambiguous or CV is hard to read

Respond in exactly this format — no other text:
SKILLS: [YES|PARTIAL|NO]
ROLE_RELEVANCE: [SAME|SIMILAR|DIFFERENT]
DOMAIN_RELEVANCE: [SAME|RELATED|UNRELATED]
EDUCATION: [YES|PARTIAL|NO]
HOLISTIC: [STRONG|MODERATE|WEAK|NO FIT]
CONFIDENCE: [HIGH|LOW]"""

# ── Scoring maps (applied by code, not LLM) ───────────────────────────────────

SKILL_MAP    = {'YES': 1.0, 'PARTIAL': 0.5, 'NO': 0.0}
ROLE_MAP     = {'SAME': 1.0, 'SIMILAR': 0.6, 'DIFFERENT': 0.0}
DOMAIN_MAP   = {'SAME': 1.0, 'RELATED': 0.5, 'UNRELATED': 0.0}
EDU_MAP      = {'YES': 1.0, 'PARTIAL': 0.5, 'NO': 0.0}
HOLISTIC_MAP = {'STRONG': 3, 'MODERATE': 2, 'WEAK': 1, 'NO FIT': 0}


def labels_to_scores(labels: dict) -> dict:
    """Map nominal labels to numeric scores per dimension. No aggregation, no verdict.

    fit_score and verdict are deliberately NOT computed here — threshold/weighting
    decisions belong in the analysis stage (psychometric calibration) where they
    can be set transparently and compared against the LLM's own HOLISTIC label.
    Missing labels return None (not 0.0) so analysis can distinguish missing
    from genuine zero.
    """
    return {
        'skill_score':    SKILL_MAP.get(labels.get('skills')),
        'role_score':     ROLE_MAP.get(labels.get('role_relevance')),
        'domain_score':   DOMAIN_MAP.get(labels.get('domain_relevance')),
        'edu_score':      EDU_MAP.get(labels.get('education')),
        'holistic_score': HOLISTIC_MAP.get(labels.get('holistic')),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_api_key():
    key = os.environ.get('GEMINI_API_KEY_free') or os.environ.get('GEMINI_API_KEY')
    if key:
        return key
    env_path = ROOT.parent / '.env.local'
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith('GEMINI_API_KEY_free='):
                return line.split('=', 1)[1].strip()
            if line.startswith('GEMINI_API_KEY=') or line.startswith('VITE_GEMINI_API_KEY='):
                key = line.split('=', 1)[1].strip()
    return key


def parse_labels(text: str) -> dict:
    labels = {'skills': None, 'role_relevance': None, 'domain_relevance': None,
              'education': None, 'holistic': None, 'confidence': None}

    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        key, _, val = line.partition(':')
        key = key.strip().upper()
        val = val.strip().upper()

        if key == 'SKILLS' and val in SKILL_MAP:
            labels['skills'] = val
        elif key == 'ROLE_RELEVANCE' and val in ROLE_MAP:
            labels['role_relevance'] = val
        elif key == 'DOMAIN_RELEVANCE' and val in DOMAIN_MAP:
            labels['domain_relevance'] = val
        elif key == 'EDUCATION' and val in EDU_MAP:
            labels['education'] = val
        elif key == 'HOLISTIC' and val in HOLISTIC_MAP:
            labels['holistic'] = val
        elif key == 'CONFIDENCE' and val in ('HIGH', 'LOW'):
            labels['confidence'] = val

    return labels


def score_one(client, cv_text, jd_text, jd_id):
    prompt = PROMPT_L2_P0.format(cv=cv_text, jd=jd_text)
    try:
        t0 = time.time()
        resp = client.generate_content(prompt)
        latency = round(time.time() - t0, 2)
        labels = parse_labels(resp.text)
        scores = labels_to_scores(labels)
        meta = resp.usage_metadata
        return {
            'jd_id': jd_id,
            'labels': labels,
            'scores': scores,
            'error': None,
            'latency_s': latency,
            'prompt_tokens': meta.prompt_token_count,
            'output_tokens': meta.candidates_token_count,
            'total_tokens':  meta.total_token_count,
        }
    except Exception as e:
        return {'jd_id': jd_id, 'labels': None, 'scores': None, 'error': str(e),
                'latency_s': None, 'prompt_tokens': None, 'output_tokens': None, 'total_tokens': None}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    single_jd = None
    cv_name = 'cv_primary'
    runs = 1
    temperature = 1.0
    use_extras = False

    i = 0
    while i < len(args):
        if args[i] == '--jd' and i + 1 < len(args):
            single_jd = args[i + 1]; i += 2
        elif args[i] == '--cv' and i + 1 < len(args):
            cv_name = args[i + 1]; i += 2
        elif args[i] == '--runs' and i + 1 < len(args):
            runs = int(args[i + 1]); i += 2
        elif args[i] == '--temperature' and i + 1 < len(args):
            temperature = float(args[i + 1]); i += 2
        elif args[i] == '--extras':
            use_extras = True; i += 1
        else:
            i += 1

    api_key = load_api_key()
    if not api_key:
        print('ERROR: No API key. Set GEMINI_API_KEY or add to .env.local')
        sys.exit(1)

    genai.configure(api_key=api_key)
    client = genai.GenerativeModel(
        GEMINI_MODEL,
        generation_config=genai.GenerationConfig(temperature=temperature),
    )

    cv_file = DATA / 'cv' / f'{cv_name}.txt'
    if not cv_file.exists():
        print(f'ERROR: CV not found: {cv_file}')
        sys.exit(1)
    cv_text = cv_file.read_text()
    print(f'CV: {cv_file.name}')
    print(f'Model: {GEMINI_MODEL}  temperature={temperature}')
    print(f'Prompt: L2_P0 (categorical labels → code-computed fit_score)')
    print(f'Runs per JD: {runs}\n')

    labels_file = DATA / 'human_labels.json'
    with open(labels_file) as f:
        all_labels = json.load(f)

    if single_jd:
        stem = single_jd.replace('.txt', '').strip()
        entries = [l for l in all_labels if l['jd_id'].strip() == stem]
        if not entries:
            for sub in ('labeled-jds', 'unlabeled-jds'):
                jd_dir = DATA / sub
                matches = [f for f in jd_dir.iterdir() if f.name.strip().replace('.txt', '') == stem]
                if matches:
                    fname = matches[0].name
                    entries = [{'jd_id': fname.strip().replace('.txt', ''),
                                'jd_file': f'data/{sub}/{fname}'}]
                    break
            if not entries:
                print(f'ERROR: JD not found: {stem}')
                sys.exit(1)
    elif use_extras:
        # Reuse JD list from L1 P2 extras for this CV: unique jd_ids + preserve source field.
        extras_path = RESULTS / f'level1_p2_{cv_name}_extra.json'
        if not extras_path.exists():
            print(f'ERROR: extras file not found: {extras_path}')
            print('Run L1 P2 extras scoring first.')
            sys.exit(1)
        extras_rows = json.loads(extras_path.read_text())
        seen = {}
        for r in extras_rows:
            jid = r['jd_id'].strip()
            seen.setdefault(jid, r.get('source', 'extra'))

        ul_dir = DATA / 'unlabeled-jds'
        entries = []
        for jid, src in seen.items():
            jd_path = ul_dir / f'{jid}.txt'
            if not jd_path.exists():
                matches = [f for f in ul_dir.iterdir() if f.name.strip().replace('.txt', '') == jid]
                if matches:
                    jd_path = matches[0]
                else:
                    print(f'  SKIP {jid} (file not found in unlabeled-jds)')
                    continue
            entries.append({
                'jd_id': jid,
                'jd_file': str(jd_path.relative_to(ROOT)),
                'source': src,
            })
        print(f'Loaded {len(entries)} unique extra JDs from {extras_path.name} '
              f'(sources: ' + ', '.join(sorted({e["source"] for e in entries})) + ')\n')
    else:
        entries = all_labels

    print(f'{"JD":<44} {"run":<7} {"H":>3}  {"SK":<7} {"ROLE":<9} {"DOMAIN":<9} {"EDU":<7} {"HOLISTIC":<9} {"CONF"}')
    print('-' * 110)

    results = []
    for entry in entries:
        jd_id = entry['jd_id']
        jd_file_rel = entry.get('jd_file', '')
        jd_path = ROOT / jd_file_rel if jd_file_rel else None
        if not jd_path or not jd_path.exists():
            jd_dir = DATA / 'labeled-jds'
            matches = [f for f in jd_dir.iterdir() if f.name.strip().replace('.txt', '') == jd_id.strip()]
            if matches:
                jd_path = matches[0]
            else:
                print(f'{jd_id[:45]:<46} SKIP (file missing)')
                continue

        jd_text = jd_path.read_text()

        # CV-aware human labels: only attach when current cv matches the cv this label was created for.
        label_cv = entry.get('cv', 'cv_primary')
        if cv_name == label_cv:
            human_holistic     = entry.get('human_holistic_label')
            human_structural   = entry.get('human_structural_ratings', [])
        else:
            human_holistic     = None
            human_structural   = []

        for run_id in range(1, runs + 1):
            result = score_one(client, cv_text, jd_text, jd_id)
            result['run_id'] = run_id
            result['cv'] = cv_name
            result['model'] = GEMINI_MODEL
            result['temperature'] = temperature
            result['prompt'] = 'L2_P0'
            result['source'] = entry.get('source', 'main')
            result['human_holistic_label']     = human_holistic
            result['human_structural_ratings'] = human_structural

            human_str = str(human_holistic) if human_holistic is not None else '—'
            if result['scores']:
                lbl = result['labels']
                sk       = (lbl.get('skills') or '—')[:7]
                role     = (lbl.get('role_relevance') or '—')[:9]
                domain   = (lbl.get('domain_relevance') or '—')[:9]
                edu      = (lbl.get('education') or '—')[:7]
                holistic = (lbl.get('holistic') or '—')[:9]
                conf     = (lbl.get('confidence') or '—')
            else:
                sk = role = domain = edu = holistic = conf = '—'
            run_str = f'[{run_id}/{runs}]' if runs > 1 else ''
            print(f'{jd_id[:43]:<44} {run_str:<7} {human_str:>3}  {sk:<7} {role:<9} {domain:<9} {edu:<7} {holistic:<9} {conf}')

            results.append(result)

            if result['error']:
                err = result['error']
                if 'quota' in err.lower() or '429' in err:
                    print('Rate limit hit — stopping.')
                    break
            else:
                time.sleep(2)

    out_file = RESULTS / (f'level2_p0_{cv_name}_extra.json' if use_extras
                          else f'level2_p0_{cv_name}.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    valid = [r for r in results if r['scores'] is not None]
    print(f'\nScored: {len(valid)}/{len(results)} | Saved to: {out_file}')

    if single_jd and valid:
        r = valid[0]
        print(f'\n--- LLM labels (nominal) ---')
        print(f"  Skills:           {r['labels']['skills']}")
        print(f"  Role relevance:   {r['labels']['role_relevance']}")
        print(f"  Domain relevance: {r['labels']['domain_relevance']}")
        print(f"  Education:        {r['labels']['education']}")
        print(f"  Holistic:         {r['labels']['holistic']} ({r['labels']['confidence']})")
        s = r['scores']
        print(f'\n--- Mapped scores (no aggregation, no verdict) ---')
        print(f"  skill_score:      {s['skill_score']}")
        print(f"  role_score:       {s['role_score']}")
        print(f"  domain_score:     {s['domain_score']}")
        print(f"  edu_score:        {s['edu_score']}")
        print(f"  holistic_score:   {s['holistic_score']}  (HOLISTIC_MAP 0–3)")
        print(f'\n--- Human ratings ---')
        print(f"  holistic 0-3 (H1):    {r['human_holistic_label']}")
        n_struct = len(r['human_structural_ratings'])
        print(f"  structural raters:    {n_struct}" + (
            f'  ({[s.get("rater") for s in r["human_structural_ratings"]]})' if n_struct else ''))


if __name__ == '__main__':
    main()
