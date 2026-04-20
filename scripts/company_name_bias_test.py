#!/usr/bin/env python3
"""
Company Name Bias Test
======================
Question: Does the LLM score differently when it knows the company name vs when
the company name is replaced with "Company_XX"?

This test has two purposes:
  1. Scientific: detect company reputation bias (e.g., Meta/LinkedIn scored higher
     regardless of JD quality because the model associates them with prestige).
  2. Practical: if no meaningful bias found → safe to mask all company names
     before publishing to GitHub.

Design: 24 JDs × 3 runs each
  Run O  (original):   real JD text, company name visible
  Run M1 (masked 1):   company name replaced with "Company_XX"
  Run M2 (masked 2):   same mask, second run → gives masked-only reliability

This gives three quantities per JD:
  bias        = score_M_mean - score_O        (signed: positive = LLM scores higher masked)
  masked_SD   = SD(M1, M2)                   (within-masked noise)
  total_SD    = SD(O, M1, M2)                (full 3-run variability)

Aggregate metrics:
  mean_bias           with 95% CI (bootstrap)
  Cohen's d           effect size for bias
  Wilcoxon signed-rank p-value (non-parametric paired test, O vs M_mean, n=24)
  ICC(2,1) — masked runs only   (M1 vs M2, reliability after removing company signal)
  ICC(2,1) — all 3 runs         (O, M1, M2)

Decision rule (printed at end):
  |mean_bias| < 3 pts AND ICC_masked > 0.90  →  "No meaningful bias — safe to mask"
  |mean_bias| >= 3 pts                        →  "Company name affects scores — document as finding"

Output:
  evaluation/data/batch-results/company_name_bias.json   ← raw scores
  evaluation/company_name_bias_report.md                 ← summary report

Usage:
    cd /path/to/cool-cohen
    python3 evaluation/company_name_bias_test.py

API calls: 24 × 3 = 72 Gemini calls (~3–4 min with 2s sleep)
"""

import json
import os
import re
import time
import numpy as np
from pathlib import Path
import google.generativeai as genai

GEMINI_MODEL = 'gemini-3.1-flash-lite-preview'
MASK_TOKEN   = 'Company_XX'
N_BOOTSTRAP  = 2000
SLEEP_SEC    = 2

# ── Candidate context (P1 prompt — current app prompt) ─────────────────────────

TALENTS  = ('10+ years psychometrics, IRT, CFA, DIF, Rasch; automated monitoring at scale; '
            'TestGorilla 350+ assessments; transitioning into people data science')
FOCUS    = 'People Data Scientist, Data Analyst (people/HR domain), Assessment Scientist'
HARD_NOS = 'None'
PERSONA  = 'talent acquisition and executive search'

PROMPT_TEMPLATE = """SYSTEM PERSONA: You are a Senior Hiring Manager specializing in {persona}.

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

# ── Explicit company name map ───────────────────────────────────────────────────
# Maps jd_id prefix → list of strings to replace in JD text.
# Multiple variants handle different capitalizations / abbreviations in the JD body.

COMPANY_NAMES: dict[str, list[str]] = {
    'maki_people':                          ['Maki People'],
    'Accenture':                            ['Accenture'],
    'Arrive':                               ['Arrive'],
    'Atlantica':                            ['Atlantica Sustainable Infrastructure', 'Atlantica'],
    'Canonical':                            ['Canonical'],
    'Dandelion':                            ['Dandelion Civilization'],
    'DataAnnotation':                       ['DataAnnotation'],
    'G-Research':                           ['G-Research', 'G Research'],
    'Johnson_Controls':                     ['Johnson Controls'],
    'LinkedIn':                             ['LinkedIn'],
    'Proofpoint':                           ['Proofpoint'],
    'QIC':                                  ['QIC Digital Hub', 'QIC'],
    'angels':                               ['Angels.Space', 'Angels Space', 'Angels'],
    'anglian':                              ['Anglian Water Services', 'Anglian Water'],
    'bazaarvoice':                          ['Bazaarvoice'],
    'bpostgroup':                           ['bpostgroup', 'bpost group', 'bpost'],
    'codility':                             ['Codility'],
    'freeplay':                             ['Freeplay'],
    'lightning':                            ['Lightning AI', 'Lightning'],
    'meta':                                 ['Meta'],
    'skillvue':                             ['Skillvue'],
    'sorare':                               ['Sorare'],
    'the_world_bank':                       ['The World Bank', 'World Bank'],
    'unicef':                               ['UNICEF'],
}


def get_company_names_for_jd(jd_id: str) -> list[str]:
    """Return list of name variants to mask for this JD."""
    jd_id_lower = jd_id.lower()
    for prefix, names in COMPANY_NAMES.items():
        if jd_id_lower.startswith(prefix.lower()):
            return names
    # Fallback: first segment of jd_id
    first = re.split(r'[_\-\s]', jd_id)[0]
    return [first] if first else []


def mask_company(jd_text: str, names: list[str]) -> str:
    """Replace all occurrences of company name variants with MASK_TOKEN."""
    masked = jd_text
    # Replace longest names first to avoid partial replacements
    for name in sorted(names, key=len, reverse=True):
        masked = re.sub(re.escape(name), MASK_TOKEN, masked, flags=re.IGNORECASE)
    return masked


def load_api_key() -> str:
    env_path = Path('.env.local')
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith('GEMINI_API_KEY='):
                return line.split('=', 1)[1].strip()
    return os.environ.get('GEMINI_API_KEY', '')


def parse_score(text: str):
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith('SCORE:'):
            try:
                return int(''.join(c for c in line.split(':', 1)[1] if c.isdigit()))
            except Exception:
                pass
    return None


def icc_2way_absolute(matrix: np.ndarray) -> dict:
    """ICC(2,1) — two-way random effects, absolute agreement, single measures."""
    from scipy import stats
    n, k = matrix.shape
    grand  = matrix.mean()
    rows   = matrix.mean(axis=1)
    cols   = matrix.mean(axis=0)

    SSr = k * np.sum((rows - grand) ** 2)
    SSc = n * np.sum((cols - grand) ** 2)
    SSe = np.sum((matrix - rows[:, None] - cols[None, :] + grand) ** 2)

    dfr, dfc, dfe = n - 1, k - 1, (n - 1) * (k - 1)
    MSr = SSr / dfr
    MSc = SSc / dfc if dfc > 0 else 0
    MSe = SSe / dfe if dfe > 0 else 1e-9

    icc = (MSr - MSe) / (MSr + (k - 1) * MSe + k * (MSc - MSe) / n)
    icc = float(max(0.0, icc))

    F = MSr / MSe
    p = float(1 - stats.f.cdf(F, dfr, dfe))
    return {'icc': icc, 'F': F, 'df1': dfr, 'df2': dfe, 'p': p, 'MSr': MSr, 'MSe': MSe}


def bootstrap_ci(values: list[float], n: int = N_BOOTSTRAP, alpha: float = 0.05) -> tuple:
    """Bootstrap 95% CI for the mean."""
    arr = np.array(values)
    means = [np.mean(arr[np.random.randint(len(arr), size=len(arr))]) for _ in range(n)]
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def cohens_d_paired(diffs: list[float]) -> float:
    arr = np.array(diffs)
    return float(arr.mean() / arr.std(ddof=1)) if arr.std(ddof=1) > 0 else 0.0


def main():
    # ── Setup ──────────────────────────────────────────────────────────────────
    api_key = load_api_key()
    if not api_key:
        print('ERROR: No GEMINI_API_KEY in .env.local')
        return

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    cv_text = Path('evaluation/cv/cv_primary.txt').read_text()

    with open('evaluation/labels.json') as f:
        labels = [l for l in json.load(f)['labels'] if not l.get('is_synthetic')]

    prompt_base = PROMPT_TEMPLATE.format(
        persona=PERSONA, talents=TALENTS, focus=FOCUS, hard_nos=HARD_NOS,
        cv=cv_text, jd='{jd}'
    )

    results = []
    total = len(labels)

    print(f'Company Name Bias Test — {total} JDs × 3 runs (1 original + 2 masked)')
    print(f'{"JD":<42} {"H":>2}  {"Orig":>4}  {"M1":>4}  {"M2":>4}  {"Bias":>5}  {"MaskSD":>6}')
    print('─' * 80)

    for i, entry in enumerate(labels):
        jd_id   = entry['jd_id']
        jd_path = Path('evaluation') / entry['jd_file']
        human   = entry.get('human_relevance', '?')

        if not jd_path.exists():
            print(f'{jd_id[:41]:<42}  SKIP — file not found')
            continue

        jd_orig   = jd_path.read_text()
        names     = get_company_names_for_jd(jd_id)
        jd_masked = mask_company(jd_orig, names)

        scores = {}  # 'orig', 'm1', 'm2'
        for run_label, jd_text in [('orig', jd_orig), ('m1', jd_masked), ('m2', jd_masked)]:
            try:
                prompt   = prompt_base.format(jd=jd_text)
                response = model.generate_content(prompt)
                score    = parse_score(response.text)
                scores[run_label] = score
                time.sleep(SLEEP_SEC)
            except Exception as e:
                err = str(e)
                print(f'  ERROR [{run_label}] {jd_id[:30]}: {err[:60]}')
                scores[run_label] = None
                if '429' in err or 'quota' in err.lower():
                    print('API quota reached — stopping early.')
                    break

        s_o  = scores.get('orig')
        s_m1 = scores.get('m1')
        s_m2 = scores.get('m2')

        if None in [s_o, s_m1, s_m2]:
            print(f'{jd_id[:41]:<42}  INCOMPLETE — {scores}')
            continue

        mask_mean = (s_m1 + s_m2) / 2
        bias      = mask_mean - s_o
        mask_sd   = float(np.std([s_m1, s_m2], ddof=1))

        flag = ''
        if abs(bias) >= 10:
            flag = ' ◄ large'
        elif abs(bias) >= 5:
            flag = ' ◄ notable'

        print(f'{jd_id[:41]:<42} {str(human):>2}  {s_o:>4}  {s_m1:>4}  {s_m2:>4}  '
              f'{bias:>+5.1f}  {mask_sd:>6.1f}{flag}')

        results.append({
            'jd_id':            jd_id,
            'company_names':    names,
            'human_relevance':  human,
            'score_orig':       s_o,
            'score_m1':         s_m1,
            'score_m2':         s_m2,
            'score_mask_mean':  round(mask_mean, 2),
            'bias':             round(bias, 2),          # masked_mean − original
            'mask_sd':          round(mask_sd, 2),
            'total_sd':         round(float(np.std([s_o, s_m1, s_m2], ddof=1)), 2),
        })

    if not results:
        print('\nNo complete results — check API key and run again.')
        return

    # ── Save raw results ───────────────────────────────────────────────────────
    out_raw = Path('evaluation/data/batch-results/company_name_bias.json')
    out_raw.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # ── Statistics ────────────────────────────────────────────────────────────
    from scipy import stats as scipy_stats

    biases      = [r['bias'] for r in results]
    orig_scores = [r['score_orig']      for r in results]
    mask_means  = [r['score_mask_mean'] for r in results]

    mean_bias = float(np.mean(biases))
    ci_lo, ci_hi = bootstrap_ci(biases)
    d = cohens_d_paired(biases)

    # Wilcoxon signed-rank (paired, non-parametric)
    wilcox = scipy_stats.wilcoxon(mask_means, orig_scores, alternative='two-sided')

    # ICC for masked runs only (M1 vs M2)
    mat_masked = np.array([[r['score_m1'], r['score_m2']] for r in results], dtype=float)
    icc_masked = icc_2way_absolute(mat_masked)

    # ICC for all 3 runs (O, M1, M2)
    mat_all = np.array([[r['score_orig'], r['score_m1'], r['score_m2']] for r in results], dtype=float)
    icc_all = icc_2way_absolute(mat_all)

    # Verdict
    no_bias = abs(mean_bias) < 3.0 and icc_masked['icc'] > 0.90
    verdict = 'No meaningful bias — safe to mask company names for publication.' if no_bias \
              else 'Company name affects scores — document as a finding.'

    # ── Print stats ───────────────────────────────────────────────────────────
    print(f'\n{"═"*60}')
    print(f'COMPANY NAME BIAS — SUMMARY  (n={len(results)} JDs)')
    print(f'{"═"*60}')
    print(f'\n  Mean bias (masked − original): {mean_bias:+.2f} pts')
    print(f'  95% bootstrap CI:              [{ci_lo:+.2f}, {ci_hi:+.2f}]')
    print(f"  Cohen's d (paired):            {d:.3f}")
    print(f'  Wilcoxon p-value:              {wilcox.pvalue:.4f}')
    print(f'\n  ICC (masked M1 vs M2 only):    {icc_masked["icc"]:.3f}'
          f'  p={icc_masked["p"]:.4f}')
    print(f'  ICC (all 3 runs O+M1+M2):      {icc_all["icc"]:.3f}'
          f'  p={icc_all["p"]:.4f}')
    print(f'\n  Mean masked SD (M1 vs M2):     {np.mean([r["mask_sd"] for r in results]):.2f} pts')
    print(f'  Mean total SD  (all 3):        {np.mean([r["total_sd"] for r in results]):.2f} pts')
    print(f'\n  ▶  VERDICT: {verdict}')

    # ── Top movers (biggest bias) ─────────────────────────────────────────────
    sorted_by_bias = sorted(results, key=lambda r: abs(r['bias']), reverse=True)
    print(f'\n  Top 5 JDs by |bias|:')
    print(f'  {"JD":<42} {"Orig":>4}  {"MaskMean":>8}  {"Bias":>6}')
    for r in sorted_by_bias[:5]:
        print(f'  {r["jd_id"][:41]:<42} {r["score_orig"]:>4}  '
              f'{r["score_mask_mean"]:>8.1f}  {r["bias"]:>+6.1f}')

    # ── Markdown report ───────────────────────────────────────────────────────
    from datetime import datetime
    report_lines = [
        '# Company Name Bias Test — Report',
        f'*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}*',
        f'*Model: {GEMINI_MODEL} | Prompt: P1 (current app) | n={len(results)} JDs × 3 runs*',
        '',
        '## Design',
        '',
        '- **Run O** (original): real JD text, company name visible',
        '- **Run M1** (masked 1): company name → `Company_XX`',
        '- **Run M2** (masked 2): same mask, repeated → within-masked reliability',
        '',
        '**Bias** = mean(M1, M2) − O per JD.  ',
        '**Decision threshold**: |mean bias| < 3 pts AND ICC(masked) > 0.90 → safe to mask.',
        '',
        '## Results',
        '',
        f'| Metric | Value |',
        f'|--------|-------|',
        f'| n JDs | {len(results)} |',
        f'| Mean bias (masked − original) | **{mean_bias:+.2f} pts** |',
        f'| 95% bootstrap CI | [{ci_lo:+.2f}, {ci_hi:+.2f}] |',
        f"| Cohen's d | {d:.3f} |",
        f'| Wilcoxon p-value | {wilcox.pvalue:.4f} |',
        f'| ICC masked (M1 vs M2) | {icc_masked["icc"]:.3f} (p={icc_masked["p"]:.4f}) |',
        f'| ICC all 3 runs | {icc_all["icc"]:.3f} (p={icc_all["p"]:.4f}) |',
        f'| Mean masked SD | {np.mean([r["mask_sd"] for r in results]):.2f} pts |',
        f'| Mean total SD | {np.mean([r["total_sd"] for r in results]):.2f} pts |',
        '',
        f'## Verdict',
        '',
        f'**{verdict}**',
        '',
        '## Per-JD scores',
        '',
        '| JD | H | Orig | M1 | M2 | Bias | MaskSD |',
        '|----|---|------|----|----|------|--------|',
    ]
    for r in sorted(results, key=lambda x: abs(x['bias']), reverse=True):
        report_lines.append(
            f'| {r["jd_id"][:40]} | {r["human_relevance"]} | {r["score_orig"]} | '
            f'{r["score_m1"]} | {r["score_m2"]} | {r["bias"]:+.1f} | {r["mask_sd"]:.1f} |'
        )

    report_lines += [
        '',
        '## Interpretation',
        '',
        '- **Bias direction**: positive = LLM scores higher when company name is masked; '
        'negative = LLM scores higher when company is known.',
        '- **ICC(masked)** measures pure test-retest reliability once company name is removed.',
        '- **ICC(all 3)** measures consistency across the original + masked conditions; '
        'a drop vs ICC(masked) indicates the company name is adding score variance.',
        '',
        '*Raw scores: `evaluation/data/batch-results/company_name_bias.json`*',
    ]

    report_path = Path('evaluation/company_name_bias_report.md')
    report_path.write_text('\n'.join(report_lines), encoding='utf-8')
    print(f'\n  Report: {report_path}')
    print(f'  Raw:    {out_raw}')


if __name__ == '__main__':
    np.random.seed(42)
    main()
