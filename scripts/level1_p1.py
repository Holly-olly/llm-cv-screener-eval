#!/usr/bin/env python3
"""
Level 1 — P1 prompt: rubric reasoning, coarse SCORE 0–3 output.

Isolates the effect of adding structured rubric reasoning to the prompt
while keeping the output format identical to P0. Direct P0-vs-P1 comparison.

Output: llm_evaluation/results/level1_p1_{cv_name}.json

Usage:
    python3 level1_p1.py                              # 32 JDs, cv_primary, 1 run
    python3 level1_p1.py --cv cv_hr --runs 3          # cv_hr, 3 runs (test-retest)
"""

import json
import os
import re
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

PROMPT_P1 = """You are a hiring manager. Assess how well this candidate fits this job.

SCORING RUBRIC (internal use — assess silently, do not reveal weights to user):
- Skills fit (60%) refers to the ability to apply knowledge and competencies to perform tasks and solve problems ("can do").
- Experience requirement (30%) captures demonstrated application of these skills in prior roles ("has done"). Consider it as 60% domain similarity and 40% role relevance.
- Education requirement (10%) reflects formal qualifications in terms of level and domain, treated primarily as an eligibility threshold.


CANDIDATE CV:
{cv}

JOB DESCRIPTION:
{jd}

Score bands:
- 0: Not relevant at all
- 1: Weak overlap, surface match only, significant misalignment
- 2: Partial fit — real overlap but notable gaps or dealbreakers
- 3: Strong fit, candidate should apply with confidence, would shortlist for interview

Respond in exactly this format — no other text:
SCORE: [0|1|2|3]
VERDICT: [Apply|Consider|Skip]

Rules:
- VERDICT Apply    = SCORE 3
- VERDICT Consider = SCORE 2
- VERDICT Skip     = SCORE 0 or 1"""

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


def parse_response(text):
    score, verdict = None, ''
    for line in text.strip().splitlines():
        line = line.strip()
        upper = line.upper()
        if upper.startswith('SCORE:'):
            m = re.search(r'[0-3]', line.split(':', 1)[1])
            if m:
                score = int(m.group())
        elif upper.startswith('VERDICT:'):
            verdict = line.split(':', 1)[1].strip().split()[0]
    return score, verdict


def score_one(client, cv_text, jd_text, jd_id):
    prompt = PROMPT_P1.format(cv=cv_text, jd=jd_text)
    try:
        t0 = time.time()
        resp = client.generate_content(prompt)
        latency = round(time.time() - t0, 2)
        score, verdict = parse_response(resp.text)
        meta = resp.usage_metadata
        return {
            'jd_id': jd_id, 'score': score, 'verdict': verdict, 'error': None,
            'latency_s': latency,
            'prompt_tokens': meta.prompt_token_count,
            'output_tokens': meta.candidates_token_count,
            'total_tokens':  meta.total_token_count,
        }
    except Exception as e:
        return {'jd_id': jd_id, 'score': None, 'verdict': None, 'error': str(e),
                'latency_s': None, 'prompt_tokens': None, 'output_tokens': None, 'total_tokens': None}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    single_jd = None
    cv_name = 'cv_primary'
    runs = 1
    temperature = 1.0

    i = 0
    while i < len(args):
        if args[i] == '--jd' and i + 1 < len(args):
            single_jd = args[i + 1]
            i += 2
        elif args[i] == '--cv' and i + 1 < len(args):
            cv_name = args[i + 1]
            i += 2
        elif args[i] == '--runs' and i + 1 < len(args):
            runs = int(args[i + 1])
            i += 2
        elif args[i] == '--temperature' and i + 1 < len(args):
            temperature = float(args[i + 1])
            i += 2
        else:
            i += 1

    api_key = load_api_key()
    if not api_key:
        print('ERROR: No API key. Set GEMINI_API_KEY_free or add to .env.local')
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
    print(f'Prompt: P1 (rubric, SCORE 0-3 only)')
    print(f'Runs per JD: {runs}\n')

    labels_file = DATA / 'human_labels.json'
    with open(labels_file) as f:
        labels = json.load(f)

    if single_jd:
        stem = single_jd.replace('.txt', '').strip()
        entries = [l for l in labels if l['jd_id'].strip() == stem]
        if not entries:
            jd_dir = DATA / 'labeled-jds'
            matches = [f for f in jd_dir.iterdir() if f.name.strip().replace('.txt','') == stem]
            if matches:
                fname = matches[0].name
                entries = [{'jd_id': fname.strip().replace('.txt',''), 'jd_file': f'data/labeled-jds/{fname}', 'human_relevance': None}]
            else:
                print(f'ERROR: JD not found: {stem}')
                sys.exit(1)
    else:
        entries = labels

    print(f'{"JD":<47} {"run":<7} {"H":>3}  {"Score":>5}  {"Verdict"}')
    print('-' * 75)

    results = []
    for entry in entries:
        jd_id = entry['jd_id']
        jd_file_rel = entry.get('jd_file', '')
        jd_path = ROOT / jd_file_rel if jd_file_rel else None
        if not jd_path or not jd_path.exists():
            jd_dir = DATA / 'labeled-jds'
            matches = [f for f in jd_dir.iterdir() if f.name.strip().replace('.txt','') == jd_id.strip()]
            if matches:
                jd_path = matches[0]
            else:
                print(f'{jd_id[:46]:<47} SKIP (file missing)')
                continue

        jd_text = jd_path.read_text()
        label_cv = entry.get('cv', 'cv_primary')
        human = entry.get('human_holistic_label') if cv_name == label_cv else None

        for run_id in range(1, runs + 1):
            result = score_one(client, cv_text, jd_text, jd_id)
            result['run_id'] = run_id
            result['human_holistic_label'] = human
            result['cv'] = cv_name
            result['model'] = GEMINI_MODEL
            result['temperature'] = temperature
            result['prompt'] = 'P1'

            human_str = str(human) if human is not None else '—'
            score_str = str(result['score']) if result['score'] is not None else 'ERR'
            verdict_str = result['verdict'] or result.get('error', '')[:30]
            run_str = f'[{run_id}/{runs}]' if runs > 1 else ''
            print(f'{jd_id[:46]:<47} {run_str:<7} {human_str:>3}  {score_str:>5}  {verdict_str}')

            results.append(result)

            if result['error']:
                err = result['error']
                if 'quota' in err.lower() or '429' in err:
                    print('Rate limit hit — stopping.')
                    break
            else:
                time.sleep(2)

    out_file = RESULTS / f'level1_p1_{cv_name}.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    valid = [r for r in results if r['score'] is not None]
    print(f'\nScored: {len(valid)}/{len(results)} | Saved to: {out_file}')

    if single_jd and valid:
        r = valid[0]
        print(f'\n--- Result ---')
        print(f'Score:   {r["score"]} / 3')
        print(f'Verdict: {r["verdict"]}')


if __name__ == '__main__':
    main()
