#!/usr/bin/env python3
"""
Career Pilot — Blind Labelling Script (Track A)

Run BEFORE AI analysis. Shows each JD one by one.
You enter relevance score 0-3. Saved to labels.json immediately.
AI scores are never shown during this step.

Usage:
    python3 label_jds.py

Scale:
    0 = Not relevant   — different domain, no overlap, dealbreaker → would NOT apply
    1 = Weak overlap   — some shared keywords but core is misaligned → would NOT apply
    2 = Partial fit    — adjacent role, some match, clear gaps → would apply ON A CHANCE
    3 = Strong fit     — core domain match (psychometrics/assessment/people data) → would apply WITH CONFIDENCE

Key rules:
    0 vs 1 → any data/research element at all? No → 0
    1 vs 2 → would you actually submit? Yes → 2
    2 vs 3 → is psychometrics/measurement a core requirement? Yes → 3
"""

import json
import os
from datetime import date

BATCH_JOBS_DIR = "data/batch-jobs"
LABELS_FILE = "labels.json"

def load_labels():
    with open(LABELS_FILE) as f:
        return json.load(f)

def save_labels(data):
    with open(LABELS_FILE, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_jd_files():
    files = sorted([
        f for f in os.listdir(BATCH_JOBS_DIR)
        if f.endswith('.txt')
    ])
    return files

def already_labeled(labels_data, jd_id):
    for entry in labels_data['labels']:
        if entry['jd_id'] == jd_id and entry.get('human_relevance') is not None:
            return True
    return False

def find_or_create_entry(labels_data, jd_id, filename):
    for entry in labels_data['labels']:
        if entry['jd_id'] == jd_id:
            return entry
    # Create new entry
    new_entry = {
        "jd_id": jd_id,
        "jd_file": f"data/batch-jobs/{filename}",
        "company": "",
        "role": "",
        "jd_type": "",
        "is_synthetic": False,
        "human_relevance": None,
        "label_notes": "",
        "labeled_date": "",
        "ai_score": None,
        "ai_verdict": "",
        "analysis_date": ""
    }
    labels_data['labels'].append(new_entry)
    return new_entry

def show_jd(filepath):
    with open(filepath) as f:
        text = f.read()
    lines = text.strip().split('\n')
    # Print header (first line = company — role)
    print("\n" + "═" * 70)
    print(f"  {lines[0]}")
    print("═" * 70)
    # Print body — limit to 80 lines to keep readable
    body = '\n'.join(lines[1:])
    body_lines = body.split('\n')
    if len(body_lines) > 80:
        print('\n'.join(body_lines[:80]))
        print(f"\n  ... [{len(body_lines) - 80} more lines — press Enter to see rest or skip]")
        choice = input("  Show rest? (y/n): ").strip().lower()
        if choice == 'y':
            print('\n'.join(body_lines[80:]))
    else:
        print(body)

def get_score():
    while True:
        raw = input("\n  Your relevance score (0-3, or 's' to skip, 'q' to quit): ").strip().lower()
        if raw == 'q':
            return 'quit'
        if raw == 's':
            return 'skip'
        if raw in ('0', '1', '2', '3'):
            return int(raw)
        print("  Enter 0, 1, 2, 3, 's' to skip, or 'q' to quit")

def get_notes():
    notes = input("  Notes (optional, press Enter to skip): ").strip()
    return notes

def main():
    print("\n" + "═" * 70)
    print("  CAREER PILOT — BLIND LABELLING SESSION")
    print("═" * 70)
    print("""
  Scale:
    0 = Not relevant   → would NOT apply (different domain, dealbreaker)
    1 = Weak overlap   → would NOT apply (some keywords, wrong core)
    2 = Partial fit    → would apply ON A CHANCE (adjacent, gaps present)
    3 = Strong fit     → would apply WITH CONFIDENCE (psychometrics/assessment/people data)

  Rules:
    0 vs 1 → any data/research element? No → 0
    1 vs 2 → would you actually submit? Yes → 2
    2 vs 3 → is measurement/psychometrics a core requirement? Yes → 3

  ⚠  Label based on YOUR expert judgment only.
     Do not think about AI scores — they haven't been run yet.
""")
    input("  Press Enter to start...\n")

    labels_data = load_labels()
    files = get_jd_files()

    total = len(files)
    done = sum(1 for f in files if already_labeled(labels_data, f.replace('.txt', '')))
    remaining = [f for f in files if not already_labeled(labels_data, f.replace('.txt', ''))]

    print(f"  Total JDs: {total} | Already labelled: {done} | Remaining: {len(remaining)}\n")

    if not remaining:
        print("  ✓ All JDs already labelled! Run batch_analyze next.")
        return

    for i, filename in enumerate(remaining, 1):
        jd_id = filename.replace('.txt', '')
        filepath = os.path.join(BATCH_JOBS_DIR, filename)

        print(f"\n  [{i} of {len(remaining)}]")
        show_jd(filepath)

        score = get_score()

        if score == 'quit':
            save_labels(labels_data)
            print(f"\n  Session saved. {done + i - 1} labelled total. Resume anytime.")
            return

        if score == 'skip':
            print("  Skipped.")
            continue

        notes = get_notes()

        entry = find_or_create_entry(labels_data, jd_id, filename)
        entry['human_relevance'] = score
        entry['label_notes'] = notes
        entry['labeled_date'] = str(date.today())

        save_labels(labels_data)
        print(f"  ✓ Saved: {jd_id} → {score}")

    print("\n" + "═" * 70)
    print("  ✓ ALL JDs LABELLED")
    print("  Next step: run batch AI analysis")
    print("  → node scripts/batch-analyze.js --input data/batch-jobs --output full")
    print("═" * 70 + "\n")

if __name__ == "__main__":
    main()
