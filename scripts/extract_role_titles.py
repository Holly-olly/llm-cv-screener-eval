#!/usr/bin/env python3
"""
Extract job titles and company names from all JD filenames (labeled + unlabeled).

Step 1 — LLM extracts company + job title, fixes typos, normalises casing.
Step 2 — Full data (filename, company, job_title, source) saved to sensitive/role_titles.csv
Step 3 — Public export: job_title + source only → data/role_titles.csv (safe for GitHub)

Usage:
    python3 extract_role_titles.py
"""

import csv
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=FutureWarning)
import google.generativeai as genai

ROOT             = Path(__file__).parent.parent
LABELED_FOLDER   = ROOT / 'data' / 'labeled-jds'
UNLABELED_FOLDER = ROOT / 'data' / 'unlabeled-jds'
SENSITIVE_CSV    = ROOT / 'sensitive' / 'role_titles.csv'
PUBLIC_CSV       = ROOT / 'data' / 'role_titles.csv'
BATCH_SIZE       = 80


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


def build_prompt(filenames: list[str]) -> str:
    listing = '\n'.join(f'{i+1}. {f}' for i, f in enumerate(filenames))
    return f"""Below is a list of job description filenames. Each filename encodes a company name and a job title.

For each filename:
1. Extract the company name and the job title separately.
2. Fix any typos in the job title (e.g. "scientst" → "Scientist").
3. Use proper Title Case for both company and job title.
4. Remove underscores, hyphens, hash-like suffixes (random letters+digits), trailing spaces, and file extensions.

{listing}

Respond as a CSV table, one row per file, no header, no extra text:
number,company,job_title
"""


def parse_response(text: str, filenames: list[str]) -> list[dict]:
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(',', 2)
        if len(parts) == 3:
            idx_str, company, job_title = parts
            try:
                idx = int(idx_str.strip()) - 1
                filename = filenames[idx] if 0 <= idx < len(filenames) else ''
            except ValueError:
                filename = ''
            rows.append({
                'filename': filename,
                'company':  company.strip(),
                'job_title': job_title.strip(),
            })
    if len(rows) != len(filenames):
        print(f'  WARNING: expected {len(filenames)} rows, got {len(rows)}')
    return rows


def extract_batch(model, filenames: list[str]) -> list[dict]:
    prompt = build_prompt(filenames)
    resp = model.generate_content(prompt)
    return parse_response(resp.text, filenames)


def collect_files(folder: Path) -> list[str]:
    return sorted(p.name.strip() for p in folder.iterdir() if p.suffix == '.txt')


def main():
    api_key = load_api_key()
    if not api_key:
        print('ERROR: No API key found.')
        return

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3.1-flash-lite-preview')

    labeled_files   = collect_files(LABELED_FOLDER)
    unlabeled_files = collect_files(UNLABELED_FOLDER)
    print(f'Labeled: {len(labeled_files)}  Unlabeled: {len(unlabeled_files)}')

    all_rows = []

    # ── labeled (32) — one batch ──────────────────────────────────────────────
    print(f'\nProcessing labeled ({len(labeled_files)} files)...')
    rows = extract_batch(model, labeled_files)
    for r in rows:
        all_rows.append({**r, 'source': 'labeled'})
        print(f"  {r['company']:<30} {r['job_title']}")
    time.sleep(3)

    # ── unlabeled (168) — in chunks ───────────────────────────────────────────
    print(f'\nProcessing unlabeled ({len(unlabeled_files)} files)...')
    for start in range(0, len(unlabeled_files), BATCH_SIZE):
        batch = unlabeled_files[start:start + BATCH_SIZE]
        print(f'  Batch {start+1}–{start+len(batch)}...')
        rows = extract_batch(model, batch)
        for r in rows:
            all_rows.append({**r, 'source': 'unlabeled'})
        time.sleep(3)

    # ── Step 2: save full data to sensitive/ ──────────────────────────────────
    with open(SENSITIVE_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'company', 'job_title', 'source'])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'\nFull data saved to: {SENSITIVE_CSV}')

    # ── Step 3: public export — job_title + source only ───────────────────────
    with open(PUBLIC_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['job_title', 'source'])
        writer.writeheader()
        for r in all_rows:
            writer.writerow({'job_title': r['job_title'], 'source': r['source']})
    print(f'Public export saved to: {PUBLIC_CSV}')

    labeled_count   = sum(1 for r in all_rows if r['source'] == 'labeled')
    unlabeled_count = sum(1 for r in all_rows if r['source'] == 'unlabeled')
    print(f'\nTotal: {len(all_rows)} rows — labeled: {labeled_count}  unlabeled: {unlabeled_count}')


if __name__ == '__main__':
    main()
