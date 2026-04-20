#!/usr/bin/env python3
"""
Fetch ~168 REAL job descriptions using free, no-auth APIs.
Target: n=32 → n=200 total in labels.json.

APIs used (all free, zero signup required):
    1. Jobicy  — https://jobicy.com/api/v2/remote-jobs
       Tags: data-science, hr, engineering, marketing, business
    2. The Muse — https://www.themuse.com/api/public/jobs
       Categories: Data and Analytics, Science and Engineering

Both return full job description HTML. No API key, no account needed.

Usage:
    cd evaluation/
    python3 fetch_real_jds_free.py
"""

import json
import os
import re
import ssl
import time
import hashlib
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError

# ── Config ────────────────────────────────────────────────────────────────────
LABELS_FILE  = 'labels.json'
BATCH_DIR    = Path('data/batch-jobs')
BATCH_DIR.mkdir(parents=True, exist_ok=True)

TARGET_NEW   = 168
MIN_DESC_LEN = 300

# SSL context — macOS Python 3.14 needs this
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode    = ssl.CERT_NONE

# ── Jobicy searches ───────────────────────────────────────────────────────────
# (industry_tag, expected_zone, max_results)
# Confirmed working tags: data-science, hr, engineering, marketing, business
JOBICY_SEARCHES = [
    ("data-science",  "strong",   20),   # data scientists, analysts → strong/moderate
    ("hr",            "moderate", 20),   # HR roles → moderate/strong
    ("engineering",   "mismatch", 20),   # software/hardware engineers → mismatch
    ("marketing",     "mismatch", 20),   # marketing roles → mismatch
    ("business",      "mismatch", 20),   # biz dev, sales → mismatch
]

# ── The Muse searches ─────────────────────────────────────────────────────────
# (category, start_page, num_pages, expected_zone)
# "Data and Analytics" = broad mix: data scientists, ops analysts, ML, finance
# "Science and Engineering" = mismatch (mostly STEM engineering)
MUSE_SEARCHES = [
    ("Data and Analytics",    1, 5, "moderate"),   # broad data roles
    ("Science and Engineering", 1, 3, "mismatch"), # engineering/science
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def fetch_json(url):
    req = Request(url, headers={'User-Agent': 'CareerPilotEval/1.0'})
    with urlopen(req, timeout=20, context=SSL_CTX) as resp:
        return json.loads(resp.read())


def clean_html(text):
    """Strip HTML tags and normalise whitespace."""
    text = re.sub(r'<[^>]+>', ' ', text)
    for entity, char in [('&amp;','&'), ('&lt;','<'), ('&gt;','>'),
                          ('&nbsp;',' '), ('&#39;',"'"), ('&quot;','"')]:
        text = text.replace(entity, char)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def make_id(company, title, desc):
    slug = re.sub(r'[^a-z0-9]+', '_', f"{company}_{title}".lower()).strip('_')[:55]
    h    = hashlib.md5(desc[:200].encode()).hexdigest()[:6]
    return f"{slug}_{h}"


def try_add(data, existing_ids, existing_descs, added,
            company, title, desc_raw, location, zone, source_note):
    """Validate, deduplicate, and add one JD. Returns updated added count."""
    desc = clean_html(desc_raw)
    if len(desc) < MIN_DESC_LEN:
        return added

    jd_id = make_id(company, title, desc)
    if jd_id in existing_ids:
        return added

    desc_key = desc[:150]
    if desc_key in existing_descs:
        return added
    existing_descs.add(desc_key)

    jd_file = BATCH_DIR / f"{jd_id}.txt"
    jd_file.write_text(f"{company} — {title}\nLocation: {location}\n\n{desc}",
                       encoding='utf-8')

    data['labels'].append({
        "jd_id":           jd_id,
        "jd_file":         str(jd_file),
        "company":         company,
        "role":            title,
        "jd_type":         zone,
        "is_synthetic":    False,
        "human_relevance": None,
        "label_notes":     source_note,
        "labeled_date":    "",
        "ai_score":        None,
        "ai_verdict":      "",
        "analysis_date":   "",
        "ai_summary":      "",
        "ai_top_matches":  "",
        "ai_key_gaps":     "",
    })
    existing_ids.add(jd_id)
    added += 1
    print(f"  [{added:3d}] {company[:35]:35s} — {title[:45]}")
    return added


def save(data):
    with open(LABELS_FILE, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── API fetchers ──────────────────────────────────────────────────────────────
def run_jobicy(tag, zone, max_results, data, existing_ids, existing_descs, added):
    """Fetch Jobicy jobs by industry tag."""
    url = f"https://jobicy.com/api/v2/remote-jobs?industry={tag}&count={max_results}"
    try:
        resp = fetch_json(url)
        jobs = resp.get('jobs', [])
        fetched = 0
        for job in jobs:
            if added >= TARGET_NEW:
                break
            company  = job.get('companyName', 'Unknown').strip()
            title    = job.get('jobTitle', '').strip()
            desc_raw = job.get('jobDescription', '')
            location = job.get('jobGeo', 'Remote')
            source   = f"Jobicy — industry: '{tag}'"
            prev = added
            added = try_add(data, existing_ids, existing_descs, added,
                            company, title, desc_raw, location, zone, source)
            if added > prev:
                fetched += 1
        if fetched == 0:
            print(f"    (no new results)")
        time.sleep(1.5)
    except HTTPError as e:
        print(f"    HTTP {e.code} — skipping")
        time.sleep(3)
    except Exception as e:
        print(f"    ERROR: {e} — skipping")
        time.sleep(3)
    return added


def run_muse(category, start_page, num_pages, zone, data, existing_ids, existing_descs, added):
    """Fetch The Muse jobs by category across multiple pages."""
    cat_enc = category.replace(' ', '+')
    for page in range(start_page, start_page + num_pages):
        if added >= TARGET_NEW:
            break
        url = f"https://www.themuse.com/api/public/jobs?category={cat_enc}&page={page}&descending=true"
        try:
            resp = fetch_json(url)
            jobs = resp.get('results', [])
            fetched = 0
            for job in jobs:
                if added >= TARGET_NEW:
                    break
                company  = job.get('company', {}).get('name', 'Unknown').strip()
                title    = job.get('name', '').strip()
                desc_raw = job.get('contents', '')
                locations = job.get('locations', [])
                location  = locations[0].get('name', 'US') if locations else 'US'
                source    = f"The Muse — category: '{category}' page {page}"
                prev = added
                added = try_add(data, existing_ids, existing_descs, added,
                                company, title, desc_raw, location, zone, source)
                if added > prev:
                    fetched += 1
            if fetched == 0:
                print(f"    page {page}: (no new results)")
            time.sleep(1.0)
        except HTTPError as e:
            print(f"    HTTP {e.code} on page {page} — skipping")
            time.sleep(3)
        except Exception as e:
            print(f"    ERROR on page {page}: {e}")
            time.sleep(3)
    return added


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    with open(LABELS_FILE) as f:
        data = json.load(f)

    existing_ids   = {j['jd_id'] for j in data['labels']}
    existing_descs = set()
    added = 0

    print(f"Target: +{TARGET_NEW} real JDs  |  Current total: {len(data['labels'])}")
    print()

    # ── Phase 1: Jobicy ───────────────────────────────────────────────────────
    print("── Jobicy API ───────────────────────────────────────────────────────")
    for tag, zone, max_r in JOBICY_SEARCHES:
        if added >= TARGET_NEW:
            break
        print(f"  [{tag}] ({zone})")
        added = run_jobicy(tag, zone, max_r, data, existing_ids, existing_descs, added)
        save(data)

    # ── Phase 2: The Muse (fill remaining) ───────────────────────────────────
    print()
    print("── The Muse API ─────────────────────────────────────────────────────")
    for category, start_page, num_pages, zone in MUSE_SEARCHES:
        if added >= TARGET_NEW:
            break
        print(f"  [{category}] pages {start_page}–{start_page+num_pages-1} ({zone})")
        added = run_muse(category, start_page, num_pages, zone,
                         data, existing_ids, existing_descs, added)
        save(data)

    print()
    print(f"Done. Added: {added} real JDs")
    print(f"Total JDs in labels.json: {len(data['labels'])}")

    if added < TARGET_NEW:
        gap = TARGET_NEW - added
        print(f"\nNote: Got {added}/{TARGET_NEW}. Need {gap} more.")
        print("Options:")
        print("  1. Run again (APIs refresh daily)")
        print("  2. Increase num_pages in MUSE_SEARCHES (e.g. 5→10)")
        print("  3. Add more Jobicy tags (check jobi.cy/apidocs for valid slugs)")

    print()
    print("Next step:")
    print("  python3 batch_analyze.py --cv cv/cv_primary.txt --include-unlabeled")


if __name__ == '__main__':
    main()
