#!/usr/bin/env python3
"""
Track B — Predictive validity analysis.

Runs the Track B dataset (evaluation/data/batch-results/track_b_prepared.json)
through the P1 prompt (current app prompt) and measures whether AI scores predict
real hiring outcomes (interview vs rejection).

Why P1: It's the best-performing prompt from Theme 1 (acc=83%, FNR=0%, no 65-clustering).
         It's also the prompt currently deployed in the app.

Metrics computed:
  - AUC-ROC: separability between interview and rejection groups
  - Point-biserial correlation: AI score vs binary outcome
  - Mann-Whitney U / mean comparison: interview scores vs rejection scores
  - Optimal threshold: score that best separates outcome=1 from outcome=0
  - Agreement with old (V1-era) AI scores: are the new scores consistent?

Output: evaluation/data/batch-results/track_b_scores.json
API calls: ~26 (one per prepared entry)

Usage:
    cd /path/to/cool-cohen
    python3 evaluation/track_b_analyze.py
"""

import json
import os
import time
from pathlib import Path
import google.generativeai as genai

GEMINI_MODEL = 'gemini-3.1-flash-lite-preview'

# Candidate context — same as app settings / P1 script
TALENTS  = ('10+ years psychometrics, IRT, CFA, DIF, Rasch; automated monitoring at scale; '
            'TestGorilla 350+ assessments; transitioning into people data science')
FOCUS    = 'People Data Scientist, Data Analyst (people/HR domain), Assessment Scientist'
HARD_NOS = 'None'
PERSONA  = 'talent acquisition and executive search'

PROMPT_P1 = """SYSTEM PERSONA: You are a Senior Hiring Manager specializing in {persona}.

SCORING RUBRIC (internal use — assess silently, do not reveal weights to user):
1. Domain Expertise (35%): Depth of knowledge and methodology in the candidate's core field
2. Technical Skills (25%): Match with required tools, stack, and analytical methods
3. Seniority Fit (20%): Candidate's experience level vs role's seniority requirement
4. Education Fit (15%): Candidate's qualification level vs role's education requirement
5. Strategic Alignment (5%): Match with candidate's stated current focus and career direction

Score bands:
- 80–100: Strong fit — would shortlist for interview with confidence
- 65–79:  Partial fit — real overlap, genuine chance of being shortlisted
- 50–64:  Adjacent domain — transferable skills but core of role is misaligned
- 25–49:  Weak overlap — surface keyword match only
- 0–24:   Not relevant — different domain or hard dealbreaker present

CANDIDATE HIDDEN CONTEXT (weigh heavily — this is not visible in the CV):
- Unlisted Talents/Experience: {talents}
- Current Career Focus: {focus}
- Hard NOs / Dealbreakers: {hard_nos}

---
CANDIDATE CV:
{cv}

---
JOB DESCRIPTION:
{jd}

---
Respond in exactly this format — no other text:
SCORE: [0-100]
VERDICT: [Apply|Consider|Skip]
SUMMARY: [2-3 sentences on domain match, key fit signal, and main gap]

Rules:
- VERDICT Apply = score >= 70
- VERDICT Consider = score 45-69
- VERDICT Skip = score < 45"""


def load_api_key():
    env_path = Path('.env.local')
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith('GEMINI_API_KEY='):
                return line.split('=', 1)[1].strip()
    return os.environ.get('GEMINI_API_KEY')


def parse_response(text: str) -> dict:
    result = {'score': None, 'verdict': '', 'summary': ''}
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith('SCORE:'):
            try:
                result['score'] = int(''.join(c for c in line.split(':', 1)[1] if c.isdigit()))
            except Exception:
                pass
        elif line.startswith('VERDICT:'):
            result['verdict'] = line.split(':', 1)[1].strip()
        elif line.startswith('SUMMARY:'):
            result['summary'] = line.split(':', 1)[1].strip()
    return result


def run_analysis():
    prepared_path = Path('evaluation/data/batch-results/track_b_prepared.json')
    out_path      = Path('evaluation/data/batch-results/track_b_scores.json')
    cv_path       = Path('evaluation/cv/cv_primary.txt')

    if not prepared_path.exists():
        print(f'ERROR: {prepared_path} not found. Run track_b_prepare.py first.')
        return

    cv_text  = cv_path.read_text()
    prepared = json.loads(prepared_path.read_text())

    api_key = load_api_key()
    if not api_key:
        print('ERROR: No GEMINI_API_KEY found in .env.local')
        return

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    prompt_template = PROMPT_P1.format(
        persona=PERSONA, talents=TALENTS, focus=FOCUS, hard_nos=HARD_NOS,
        cv=cv_text, jd='{jd}'
    )

    results = []
    print(f'Running P1 prompt on {len(prepared)} Track B entries...')
    print(f'{"Company":<28} {"Outcome":<10} {"Old":>5} {"New":>5}  {"Verdict"}')
    print('-' * 70)

    for entry in prepared:
        jd_text = entry['jd']
        prompt  = prompt_template.format(jd=jd_text)

        try:
            response = model.generate_content(prompt)
            parsed   = parse_response(response.text)
            time.sleep(2)

            new_score = parsed['score'] or 0
            old_score = entry.get('old_ai_score') or 0
            delta     = new_score - old_score

            flag = ''
            if entry['outcome'] == 1 and new_score < 50:
                flag = ' ← FN'
            elif entry['outcome'] == 0 and new_score >= 50:
                flag = ' ← FP'

            print(f'{entry["company"][:27]:<28} {entry["outcome_label"]:<10} '
                  f'{str(old_score):>5} {new_score:>5}  {parsed["verdict"]}{flag}')

            results.append({
                'id':            entry['id'],
                'company':       entry['company'],
                'role':          entry['role'],
                'old_ai_score':  entry['old_ai_score'],
                'new_ai_score':  new_score,
                'score_delta':   delta,
                'verdict':       parsed['verdict'],
                'summary':       parsed['summary'],
                'outcome':       entry['outcome'],
                'outcome_label': entry['outcome_label'],
                'old_status':    entry['old_status'],
            })

        except Exception as e:
            err = str(e)
            print(f'{entry["company"][:27]:<28} ERROR: {err[:50]}')
            if 'quota' in err.lower() or '429' in err:
                print('API quota reached — stopping.')
                break

    # Save results
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f'\nSaved {len(results)} results to {out_path}')

    # Quick summary
    scored = [r for r in results if r['outcome'] is not None]
    pos = [r for r in scored if r['outcome'] == 1]
    neg = [r for r in scored if r['outcome'] == 0]

    if pos and neg:
        mean_pos = sum(r['new_ai_score'] for r in pos) / len(pos)
        mean_neg = sum(r['new_ai_score'] for r in neg) / len(neg)
        print(f'\nQuick summary (scored entries only):')
        print(f'  Interview group  (n={len(pos)}): mean score = {mean_pos:.1f}')
        print(f'  Rejection group  (n={len(neg)}): mean score = {mean_neg:.1f}')
        print(f'  Score gap: {mean_pos - mean_neg:+.1f} pts')

        # Threshold at 50: how well does it separate outcomes?
        tp = sum(1 for r in pos if r['new_ai_score'] >= 50)
        fn = sum(1 for r in pos if r['new_ai_score'] < 50)
        tn = sum(1 for r in neg if r['new_ai_score'] < 50)
        fp = sum(1 for r in neg if r['new_ai_score'] >= 50)
        print(f'\n  At threshold=50:')
        print(f'    TP={tp}, FN={fn}, TN={tn}, FP={fp}')
        if len(pos) > 0:
            print(f'    Sensitivity (recall)  = {tp/len(pos):.0%}')
        if len(neg) > 0:
            print(f'    Specificity           = {tn/len(neg):.0%}')
        print(f'\n  Full statistical analysis: run Section 15 in the notebook.')


if __name__ == '__main__':
    run_analysis()
