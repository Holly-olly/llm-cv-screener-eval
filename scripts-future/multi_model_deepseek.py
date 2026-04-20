#!/usr/bin/env python3
"""
Track C — deepseek-r1:8b scoring
==================================
Runs a simplified matching prompt on all 32 masked JDs through deepseek-r1:8b (local Ollama).
max_tokens=3000 keeps thinking within bounds (~68s per JD, ~37 min total for 32 JDs).

Results are appended to the shared multi_model_comparison.json.

Usage:
    cd /path/to/cool-cohen
    ollama serve   # (if not already running)
    python3 evaluation/multi_model_deepseek.py

Prerequisites:
    ollama pull deepseek-r1:8b
    pip install openai

Cost: free (local), time: ~37 min
"""

import json, re, time
from pathlib import Path
from openai import OpenAI

MODEL_KEY  = 'deepseek-r1:8b'
MODEL_ID   = 'deepseek-r1:8b'
MAX_TOKENS = 3000   # enough for short think + structured answer; tested at ~68s/JD
DATA_FILE  = Path('evaluation/data/batch-results/multi_model_comparison.json')

FOCUS   = 'People Data Scientist, Data Analyst (people/HR domain), Assessment Scientist'
PERSONA = 'talent acquisition and executive search'

# "Think briefly" nudges deepseek to keep reasoning short without changing the evaluation task
SYSTEM_MSG = (f'You are a Senior Hiring Manager specializing in {PERSONA}. '
              f'Think briefly, then respond in the required format.')

USER_TEMPLATE = """Rate how well this candidate's CV matches the job description. Score 0-100.

CANDIDATE CV:
{cv}

---
JOB DESCRIPTION:
{jd}

---
Scoring guide: 80-100 strong fit, 65-79 partial fit, 50-64 adjacent domain, 25-49 weak overlap, 0-24 not relevant.
Apply if score>=70, Consider if 45-69, Skip if below 45.

Candidate focus: {focus}

Respond in EXACTLY this format:
SCORE: [number 0-100]
VERDICT: [Apply|Consider|Skip]
SUMMARY: [1-2 sentences on the main fit signal and key gap]"""

COMPANY_NAMES = {
    'maki_people': ['Maki People'], 'Accenture': ['Accenture'], 'Arrive': ['Arrive'],
    'Atlantica': ['Atlantica Sustainable Infrastructure', 'Atlantica'],
    'Canonical': ['Canonical'], 'Dandelion': ['Dandelion Civilization'],
    'DataAnnotation': ['DataAnnotation'], 'G-Research': ['G-Research', 'G Research'],
    'Johnson_Controls': ['Johnson Controls'], 'LinkedIn': ['LinkedIn'],
    'Proofpoint': ['Proofpoint'], 'QIC': ['QIC Digital Hub', 'QIC'],
    'angels': ['Angels.Space', 'Angels Space', 'Angels'],
    'anglian': ['Anglian Water Services', 'Anglian Water'], 'bazaarvoice': ['Bazaarvoice'],
    'bpostgroup': ['bpostgroup', 'bpost group', 'bpost'], 'codility': ['Codility'],
    'freeplay': ['Freeplay'], 'lightning': ['Lightning AI', 'Lightning'],
    'meta': ['Meta'], 'skillvue': ['Skillvue'], 'sorare': ['Sorare'],
    'the_world_bank': ['The World Bank', 'World Bank'], 'unicef': ['UNICEF'],
    'Product_Growth': ['Wiris'], 'Applied_AI_Engineer': ['Impress'],
    'Customer_success': ['CareWise'],
    'Digital_Learning': ['Schneider Electric', 'Schneider'],
    'QIT': ['QIT Software', 'QIT'], 'Senior_AI': ['MoveUp', 'Move Up'],
}

def mask_jd(jd_text, jd_id):
    masked = jd_text
    for prefix, names in COMPANY_NAMES.items():
        if jd_id.lower().startswith(prefix.lower()):
            for name in sorted(names, key=len, reverse=True):
                masked = re.sub(re.escape(name), 'Company_XX', masked, flags=re.IGNORECASE)
            break
    return masked

def parse_score(text):
    # Strip thinking blocks — both closed and truncated
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    result = {'score': None, 'verdict': '', 'summary': ''}
    score_m = re.search(r'\*{0,2}SCORE\*{0,2}\s*:+\s*\*{0,2}(\d{1,3})\*{0,2}', text, re.IGNORECASE)
    if score_m:
        result['score'] = max(0, min(100, int(score_m.group(1))))
    verdict_m = re.search(r'\*{0,2}VERDICT\*{0,2}\s*:+\s*\*{0,2}(Apply|Consider|Skip)\*{0,2}', text, re.IGNORECASE)
    if verdict_m:
        result['verdict'] = verdict_m.group(1).capitalize()
    summary_m = re.search(r'\*{0,2}SUMMARY\*{0,2}\s*:+\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if summary_m:
        result['summary'] = summary_m.group(1).strip()
    return result

def load_data():
    with open('evaluation/labels.json') as f:
        return [l for l in json.load(f)['labels']
                if isinstance(l.get('human_relevance'), int)
                and not l.get('is_synthetic')
                and l.get('ai_score') is not None]

def load_checkpoint():
    if DATA_FILE.exists():
        try:
            return {r['jd_id']: r for r in json.loads(DATA_FILE.read_text())}
        except Exception:
            pass
    return {}

def save(results):
    DATA_FILE.write_text(json.dumps(list(results.values()), indent=2, ensure_ascii=False))

def main():
    client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
    cv_text = Path('evaluation/cv/cv_primary.txt').read_text()
    labels  = load_data()
    results = load_checkpoint()

    for l in labels:
        if l['jd_id'] not in results:
            results[l['jd_id']] = {
                'jd_id': l['jd_id'],
                'human_relevance': l['human_relevance'],
                'v1_score': l['ai_score'],
                'scores': {},
            }

    already = sum(1 for r in results.values()
                  if r['scores'].get(MODEL_KEY, {}).get('score') is not None)
    if already == len(labels):
        print(f'deepseek-r1:8b: already complete ({already}/{len(labels)}). Nothing to do.')
        print('Run multi_model_analyze.py to generate the report.')
        return

    remaining = len(labels) - already
    print(f'deepseek-r1:8b — {remaining} JDs remaining  (~{remaining*70//60} min)')
    print(f'{"JD":<45} {"H":>2}  {"Score":>5}  {"Verdict"}  {"Time":>5}')
    print('─' * 70)

    for l in labels:
        jd_id = l['jd_id']
        if results[jd_id]['scores'].get(MODEL_KEY, {}).get('score') is not None:
            s = results[jd_id]['scores'][MODEL_KEY]
            print(f'{jd_id[:44]:<45} {l["human_relevance"]:>2}  {s["score"]:>5}  {s["verdict"]}  [cached]')
            continue

        jd_masked = mask_jd((Path('evaluation') / l['jd_file']).read_text(), jd_id)
        user_msg  = USER_TEMPLATE.format(focus=FOCUS, cv=cv_text, jd=jd_masked)

        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=[{'role': 'system', 'content': SYSTEM_MSG},
                           {'role': 'user',   'content': user_msg}],
                temperature=0.4,
                max_tokens=MAX_TOKENS,
            )
            elapsed = time.time() - t0
            raw    = resp.choices[0].message.content or ''
            parsed = parse_score(raw)
            s      = parsed['score']

            fp   = ' FP' if l['human_relevance'] <= 1 and (s or 0) >= 50 else ''
            fn   = ' FN' if l['human_relevance'] >= 2 and (s or 0) < 50  else ''
            warn = ' !' if s is None else ''
            print(f'{jd_id[:44]:<45} {l["human_relevance"]:>2}  {str(s):>5}  '
                  f'{parsed["verdict"]}{fp}{fn}{warn}  {elapsed:.0f}s')
            results[jd_id]['scores'][MODEL_KEY] = parsed

        except Exception as e:
            elapsed = time.time() - t0
            print(f'{jd_id[:44]:<45}  ERROR ({elapsed:.0f}s): {e}')
            results[jd_id]['scores'][MODEL_KEY] = {'score': None, 'verdict': '', 'summary': ''}

        # Save after every JD — safe to Ctrl+C and resume
        save(results)

    done = sum(1 for r in results.values()
               if r['scores'].get(MODEL_KEY, {}).get('score') is not None)
    none_count = len(labels) - done
    print(f'\n✓ Saved → {DATA_FILE}')
    print(f'  Scored: {done}/{len(labels)}  |  None: {none_count}')
    if none_count:
        print(f'  Re-run to retry failed JDs (they will be skipped if already scored).')
    print('Next: run multi_model_analyze.py')

if __name__ == '__main__':
    main()
