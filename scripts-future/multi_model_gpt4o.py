#!/usr/bin/env python3
"""
Track C — GPT-4o scoring
=========================
Runs the P1 prompt on all 32 masked JDs through GPT-4o.
Results are saved to the shared multi_model_comparison.json.
Already complete — GPT-4o results are in the checkpoint.
Re-run only if you want to refresh.

Usage:
    cd /path/to/cool-cohen
    python3 evaluation/multi_model_gpt4o.py

Cost: ~$0.10 (32 calls)
"""

import json, os, re, time, numpy as np
from pathlib import Path
from openai import OpenAI

MODEL_KEY   = 'gpt-4o'
MODEL_ID    = 'gpt-4o'
SLEEP       = 1
MAX_TOKENS  = 512
DATA_FILE   = Path('evaluation/data/batch-results/multi_model_comparison.json')

TALENTS  = ('10+ years psychometrics, IRT, CFA, DIF, Rasch; automated monitoring at scale; '
            'TestGorilla 350+ assessments; transitioning into people data science')
FOCUS    = 'People Data Scientist, Data Analyst (people/HR domain), Assessment Scientist'
HARD_NOS = 'None'
PERSONA  = 'talent acquisition and executive search'
SYSTEM_MSG = f'You are a Senior Hiring Manager specializing in {PERSONA}.'

USER_TEMPLATE = """SCORING RUBRIC (internal use — assess silently):
1. Domain Expertise (35%): depth in candidate's core field
2. Technical Skills (25%): match with required tools and methods
3. Seniority Fit (20%): experience level vs role requirement
4. Education Fit (15%): qualification vs role requirement
5. Strategic Alignment (5%): match with candidate's stated focus

Score bands:
- 80-100: Strong fit — would shortlist with confidence
- 65-79:  Partial fit — genuine overlap, real chance of being shortlisted
- 50-64:  Adjacent domain — transferable skills but core misaligned
- 25-49:  Weak overlap — surface keyword match only
- 0-24:   Not relevant — different domain or dealbreaker

CANDIDATE HIDDEN CONTEXT (weigh heavily):
- Talents: {talents}
- Current Focus: {focus}
- Hard NOs: {hard_nos}

---
CANDIDATE CV:
{cv}

---
JOB DESCRIPTION:
{jd}

---
Respond in EXACTLY this format — no other text:
SCORE: [0-100]
VERDICT: [Apply|Consider|Skip]
SUMMARY: [2-3 sentences on domain match, key fit signal, and main gap]

Rules: Apply>=70, Consider 45-69, Skip<45"""

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
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
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

def load_key():
    env = Path('.env.local')
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith('OPENAI_API_KEY='):
                return line.split('=', 1)[1].strip()
    return os.environ.get('OPENAI_API_KEY', '')

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
    client = OpenAI(api_key=load_key())
    cv_text = Path('evaluation/cv/cv_primary.txt').read_text()
    labels = load_data()
    results = load_checkpoint()

    # Ensure all JDs are in results dict
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
        print(f'GPT-4o: already complete ({already}/{len(labels)}). Nothing to do.')
        print('Run multi_model_analyze.py to generate the report.')
        return

    print(f'GPT-4o — {len(labels)} JDs')
    print(f'{"JD":<45} {"H":>2}  {"Score":>5}  Verdict')
    print('─' * 65)

    for l in labels:
        jd_id = l['jd_id']
        if results[jd_id]['scores'].get(MODEL_KEY, {}).get('score') is not None:
            s = results[jd_id]['scores'][MODEL_KEY]
            print(f'{jd_id[:44]:<45} {l["human_relevance"]:>2}  {s["score"]:>5}  {s["verdict"]}  [cached]')
            continue

        jd_masked = mask_jd((Path('evaluation') / l['jd_file']).read_text(), jd_id)
        user_msg = USER_TEMPLATE.format(talents=TALENTS, focus=FOCUS,
                                        hard_nos=HARD_NOS, cv=cv_text, jd=jd_masked)
        try:
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=[{'role': 'system', 'content': SYSTEM_MSG},
                           {'role': 'user', 'content': user_msg}],
                temperature=0.4, max_tokens=MAX_TOKENS,
            )
            parsed = parse_score(resp.choices[0].message.content or '')
            s = parsed['score']
            fp = ' FP' if l['human_relevance'] <= 1 and (s or 0) >= 50 else ''
            fn = ' FN' if l['human_relevance'] >= 2 and (s or 0) < 50 else ''
            print(f'{jd_id[:44]:<45} {l["human_relevance"]:>2}  {str(s):>5}  {parsed["verdict"]}{fp}{fn}')
            results[jd_id]['scores'][MODEL_KEY] = parsed
        except Exception as e:
            print(f'{jd_id[:44]:<45}  ERROR: {e}')
            results[jd_id]['scores'][MODEL_KEY] = {'score': None, 'verdict': '', 'summary': ''}
        time.sleep(SLEEP)

    save(results)
    print(f'\nSaved → {DATA_FILE}')
    print('Next: run multi_model_analyze.py')

if __name__ == '__main__':
    main()
