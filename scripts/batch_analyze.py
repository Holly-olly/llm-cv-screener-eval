#!/usr/bin/env python3
"""
Career Pilot — Batch AI Analysis Script (Track A)

Run AFTER label_jds.py. Analyzes each JD with Gemini API one by one.
Saves result to labels.json immediately after each file.
If daily API limit hits — stops cleanly and shows exactly where.

Usage:
    python3 batch_analyze.py --cv cv/cv_primary.txt

Requirements:
    pip install google-generativeai python-dotenv
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path


# Load API key from .env.local in parent app folder
def load_api_key():
    env_path = Path(__file__).parent.parent / '.env.local'
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith('GEMINI_API_KEY='):
                return line.split('=', 1)[1].strip()
    return os.environ.get('GEMINI_API_KEY')

LABELS_FILE = 'labels.json'
BATCH_JOBS_DIR = 'data/batch-jobs'
GEMINI_MODEL = 'gemini-3.1-flash-lite-preview'

PROMPT_TEMPLATE = """You are a senior talent acquisition expert evaluating candidate fit.

=== CANDIDATE CV ===
{cv}

=== JOB DESCRIPTION ===
{jd}

=== INSTRUCTIONS ===
Evaluate how well this candidate fits this job description.

Respond in exactly this format — no other text:

SCORE: [number 0-100]
VERDICT: [Apply | Consider | Skip]
SUMMARY: [2-3 sentences explaining the score. Be specific about matches and gaps.]
TOP_MATCHES: [comma-separated list of 3-5 specific skills/experiences that match]
KEY_GAPS: [comma-separated list of 1-3 important gaps, or "None" if strong fit]

Rules:
- SCORE 75-100 = strong fit, candidate has core required skills
- SCORE 50-74 = partial fit, some relevant skills but notable gaps
- SCORE 25-49 = weak fit, adjacent but significant misalignment
- SCORE 0-24 = not relevant, different domain or dealbreaker present
- VERDICT Apply = score >= 70
- VERDICT Consider = score 45-69
- VERDICT Skip = score < 45
"""

def load_labels():
    with open(LABELS_FILE) as f:
        return json.load(f)

def save_labels(data):
    with open(LABELS_FILE, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def parse_response(text):
    """Extract score, verdict, summary from Gemini response."""
    result = {
        'ai_score': None,
        'ai_verdict': '',
        'ai_summary': '',
        'ai_top_matches': '',
        'ai_key_gaps': '',
        'ai_raw': text
    }
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith('SCORE:'):
            try:
                result['ai_score'] = int(''.join(c for c in line.split(':', 1)[1] if c.isdigit()))
            except:
                pass
        elif line.startswith('VERDICT:'):
            result['ai_verdict'] = line.split(':', 1)[1].strip()
        elif line.startswith('SUMMARY:'):
            result['ai_summary'] = line.split(':', 1)[1].strip()
        elif line.startswith('TOP_MATCHES:'):
            result['ai_top_matches'] = line.split(':', 1)[1].strip()
        elif line.startswith('KEY_GAPS:'):
            result['ai_key_gaps'] = line.split(':', 1)[1].strip()
    return result

def analyze_jd(model, jd_text, cv_text):
    """Call Gemini API and return parsed result."""
    prompt = PROMPT_TEMPLATE.format(cv=cv_text, jd=jd_text)
    response = model.generate_content(prompt)
    return parse_response(response.text)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv', required=True, help='Path to CV text file')
    parser.add_argument('--delay', type=float, default=2.0,
                        help='Seconds to wait between API calls (default: 2)')
    parser.add_argument('--include-unlabeled', action='store_true',
                        help='Also analyze synthetic/unlabeled JDs (human_relevance=null)')
    args = parser.parse_args()

    # Load API key
    api_key = load_api_key()
    if not api_key:
        print('❌ No GEMINI_API_KEY found. Check .env.local in project root.')
        sys.exit(1)

    # Load CV
    cv_path = Path(args.cv)
    if not cv_path.exists():
        print(f'❌ CV file not found: {cv_path}')
        sys.exit(1)
    cv_text = cv_path.read_text(encoding='utf-8')
    print(f'✓ CV loaded: {cv_path} ({len(cv_text)} chars)')

    # Setup Gemini
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        print(f'✓ Gemini model: {GEMINI_MODEL}')
    except ImportError:
        print('❌ Missing package. Run: pip install google-generativeai')
        sys.exit(1)

    # Load labels
    labels_data = load_labels()
    labels = labels_data['labels']

    # Find pending entries
    if args.include_unlabeled:
        # All entries without an AI score (human-labeled + synthetic unlabeled)
        pending = [l for l in labels if l.get('ai_score') is None]
        print('\n⚠️  --include-unlabeled: synthetic JDs included (for distribution / 65-cluster only)')
    else:
        # Default: only human-labeled entries
        pending = [l for l in labels if
                   l.get('human_relevance') is not None and
                   l.get('ai_score') is None]
    done = [l for l in labels if l.get('ai_score') is not None]

    print(f'\n📋 Status: {len(done)} already analyzed | {len(pending)} pending')
    print('=' * 60)

    if not pending:
        print('✓ All JDs already analyzed!')
        return

    for i, entry in enumerate(pending, 1):
        jd_id = entry['jd_id']
        jd_file = Path(entry['jd_file'])

        print(f'\n[{i}/{len(pending)}] {jd_id}')

        if not jd_file.exists():
            print(f'  ⚠️  File not found: {jd_file} — skipping')
            continue

        jd_text = jd_file.read_text(encoding='utf-8')

        try:
            result = analyze_jd(model, jd_text, cv_text)

            # Update entry in labels
            entry['ai_score'] = result['ai_score']
            entry['ai_verdict'] = result['ai_verdict']
            entry['ai_summary'] = result.get('ai_summary', '')
            entry['ai_top_matches'] = result.get('ai_top_matches', '')
            entry['ai_key_gaps'] = result.get('ai_key_gaps', '')
            entry['analysis_date'] = datetime.today().strftime('%Y-%m-%d')

            save_labels(labels_data)

            human = entry.get('human_relevance')
            if human is not None:
                agreement = '✓' if (
                    (human >= 2 and result['ai_score'] and result['ai_score'] >= 50) or
                    (human <= 1 and result['ai_score'] and result['ai_score'] < 50)
                ) else '△'
                print(f'  Human: {human}/3  |  AI: {result["ai_score"]}/100  {result["ai_verdict"]}  {agreement}')
            else:
                print(f'  [unlabeled]  AI: {result["ai_score"]}/100  {result["ai_verdict"]}')
            print(f'  {result.get("ai_summary","")[:100]}')

        except Exception as e:
            err = str(e)
            print(f'  ❌ Error: {err[:120]}')
            if 'quota' in err.lower() or '429' in err or 'limit' in err.lower():
                print(f'\n⛔ API LIMIT REACHED after {i-1} files.')
                print(f'   Last completed: {pending[i-2]["jd_id"] if i > 1 else "none"}')
                print(f'   Remaining: {len(pending) - (i-1)} files')
                print(f'   Resume tomorrow — already-analyzed files will be skipped.')
                save_labels(labels_data)
                sys.exit(0)
            continue

        if i < len(pending):
            time.sleep(args.delay)

    print('\n' + '=' * 60)
    print(f'✅ DONE — all {len(pending)} JDs analyzed')
    print(f'   Results saved to: {LABELS_FILE}')
    print(f'   Next step: open evaluation notebook')

if __name__ == '__main__':
    main()
