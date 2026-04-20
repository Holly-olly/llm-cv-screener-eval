#!/usr/bin/env python3
"""
Track C — Multi-model comparison: Analysis
==========================================
Reads multi_model_comparison.json (produced by multi_model_gpt4o.py,
multi_model_qwen3.py, etc.) and produces the comparison report.

Works with whatever models are present in the JSON — no hardcoded list.

Usage:
    cd /path/to/cool-cohen
    python3 evaluation/multi_model_analyze.py

Output:
    evaluation/multi_model_report.md
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_FILE = Path('evaluation/data/batch-results/multi_model_comparison.json')

def icc_2way(matrix: np.ndarray) -> float:
    n, k = matrix.shape
    if k < 2:
        return float('nan')
    grand = matrix.mean()
    rows = matrix.mean(axis=1)
    cols = matrix.mean(axis=0)
    SSr = k * np.sum((rows - grand) ** 2)
    SSc = n * np.sum((cols - grand) ** 2)
    SSe = np.sum((matrix - rows[:, None] - cols[None, :] + grand) ** 2)
    dfr, dfc, dfe = n - 1, k - 1, (n - 1) * (k - 1)
    MSr = SSr / dfr
    MSc = SSc / dfc if dfc > 0 else 0
    MSe = SSe / dfe if dfe > 0 else 1e-9
    return float(max(0.0, (MSr - MSe) / (MSr + (k - 1) * MSe + k * (MSc - MSe) / n)))

def main():
    data = json.loads(DATA_FILE.read_text())

    # Detect which models are present and have at least some scores
    all_model_keys = set()
    for row in data:
        all_model_keys.update(row.get('scores', {}).keys())
    model_names = sorted(all_model_keys)

    if not model_names:
        print('No model scores found in data file. Run multi_model_gpt4o.py / multi_model_qwen3.py first.')
        return

    # Only include JDs where ALL present models have a non-None score
    complete = [r for r in data
                if all(r.get('scores', {}).get(m, {}).get('score') is not None
                       for m in model_names)]

    n = len(complete)
    n_neg = sum(1 for r in complete if r['human_relevance'] <= 1)
    n_pos = sum(1 for r in complete if r['human_relevance'] >= 2)

    print(f'Multi-model comparison — {n} complete JDs  (pos={n_pos}  neg={n_neg})')
    print(f'Models: Gemini V1 (baseline) + {", ".join(model_names)}')
    print()

    # Build score arrays: Gemini V1 + all models
    gemini = {r['jd_id']: r['v1_score'] for r in complete}
    all_labels = ['Gemini V1'] + model_names
    score_arrays = {'Gemini V1': np.array([r['v1_score'] for r in complete], dtype=float)}
    for m in model_names:
        score_arrays[m] = np.array([r['scores'][m]['score'] for r in complete], dtype=float)

    # ── Per-model metrics ──────────────────────────────────────────────────
    print(f'{"═"*68}')
    print(f'METRICS PER MODEL')
    print(f'{"═"*68}')
    print(f'{"Model":<18} {"Acc":>6} {"FPR":>6} {"FNR":>6} {"FPs@65":>8} {"MeanScore":>10}')
    print('─' * 56)

    summary_rows = []
    for label in all_labels:
        s_arr = score_arrays[label]
        tp = fp = tn = fn = fps65 = 0
        for r, s in zip(complete, s_arr):
            h = r['human_relevance']
            if h >= 2 and s >= 50:   tp += 1
            elif h <= 1 and s >= 50: fp += 1; fps65 += (1 if s == 65 else 0)
            elif h <= 1 and s < 50:  tn += 1
            elif h >= 2 and s < 50:  fn += 1
        acc = (tp + tn) / n
        fpr = fp / n_neg if n_neg else 0
        fnr = fn / n_pos if n_pos else 0
        tag = '  ← baseline' if label == 'Gemini V1' else ''
        fp_str = f'{fp}/{fp+tn}@{fps65}×65'
        print(f'{label:<18} {acc:>6.1%} {fpr:>6.1%} {fnr:>6.1%} {fp_str:>8} {s_arr.mean():>10.1f}{tag}')
        summary_rows.append({'model': label, 'acc': acc, 'fpr': fpr, 'fnr': fnr,
                              'fp': fp, 'tn': tn, 'fps_at_65': fps65, 'mean_score': s_arr.mean()})

    # ── Inter-model agreement ──────────────────────────────────────────────
    print(f'\nInter-model ICC (score agreement):')
    if len(model_names) >= 2:
        mat_new = np.column_stack([score_arrays[m] for m in model_names])
        icc_new = icc_2way(mat_new)
        print(f'  ICC({", ".join(model_names)}) = {icc_new:.3f}')
    else:
        icc_new = float('nan')

    mat_all = np.column_stack([score_arrays[l] for l in all_labels])
    icc_all = icc_2way(mat_all)
    print(f'  ICC(all incl. Gemini) = {icc_all:.3f}')

    # ── Pairwise correlations ──────────────────────────────────────────────
    print(f'\nPairwise Pearson r:')
    for i, l1 in enumerate(all_labels):
        for l2 in all_labels[i+1:]:
            r_val = float(np.corrcoef(score_arrays[l1], score_arrays[l2])[0, 1])
            print(f'  {l1} vs {l2}: r = {r_val:.3f}')

    # ── FP overlap table ───────────────────────────────────────────────────
    neg_rows = [r for r in complete if r['human_relevance'] <= 1]
    print(f'\nFP cases (H≤1):  * = FP (score≥50)')
    header = f'{"JD":<45} {"H":>2}  {"Gemini":>7}'
    for m in model_names:
        header += f'  {m[:9]:>9}'
    print(header)
    print('─' * (52 + 11 * len(model_names)))

    fp_overlap = {m: 0 for m in model_names}
    for r in sorted(neg_rows, key=lambda x: x['v1_score'], reverse=True):
        v1s = r['v1_score']
        row_str = f'{r["jd_id"][:44]:<45} {r["human_relevance"]:>2}  {v1s:>6}{"*" if v1s>=50 else " "}'
        for m in model_names:
            ms = r['scores'][m]['score']
            is_fp = ms >= 50
            if is_fp:
                fp_overlap[m] += 1
            row_str += f'  {ms:>8}{"*" if is_fp else " "}'
        print(row_str)
    print('  * = FP')

    print(f'\nFP count by model (of {n_neg} negatives):')
    print(f'  Gemini V1: {sum(1 for r in neg_rows if r["v1_score"]>=50)}/{n_neg}')
    for m in model_names:
        print(f'  {m}: {fp_overlap[m]}/{n_neg}')

    # ── Key finding ────────────────────────────────────────────────────────
    fprs = [row['fpr'] for row in summary_rows]
    fpr_range = max(fprs) - min(fprs)
    print(f'\n{"═"*68}')
    print(f'KEY FINDING')
    print(f'{"═"*68}')
    if fpr_range < 0.15:
        print(f'FPR range across models: {fpr_range:.0%}  (<15pp → task-fundamental)')
        print('All models show similar FPR — the borderline JDs are genuinely ambiguous.')
        print('This is a property of the task, not a Gemini-specific calibration failure.')
    else:
        best = min(summary_rows, key=lambda r: r['fpr'])
        print(f'FPR range across models: {fpr_range:.0%}  (≥15pp → model-specific)')
        print(f'{best["model"]} achieves lowest FPR ({best["fpr"]:.0%}) — model choice matters.')

    # ── Markdown report ────────────────────────────────────────────────────
    report = [
        '# Multi-Model Comparison — Track C',
        f'*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}*',
        f'*n={n} JDs (pos={n_pos}, neg={n_neg}) | Gemini V1 baseline + {", ".join(model_names)}*',
        f'*All JDs masked (Company_XX) | P1 prompt (GPT-4o) / simplified prompt (local models)*',
        '',
        '## Research Question',
        '',
        'Is FPR=50% a Gemini-specific calibration failure, or a fundamental limit of the task?',
        'If all models produce the same FPs → the borderline JDs are genuinely ambiguous.',
        'If only Gemini fails → there is a model-specific fix.',
        '',
        '## Metrics by Model',
        '',
        '| Model | Accuracy | FPR | FNR | FPs at 65 | Mean Score |',
        '|-------|---------|-----|-----|-----------|------------|',
    ]
    for row in summary_rows:
        report.append(f'| {row["model"]} | {row["acc"]:.1%} | {row["fpr"]:.1%} | '
                      f'{row["fnr"]:.1%} | {row["fps_at_65"]}/{row["fp"]} FPs | {row["mean_score"]:.1f} |')

    report += [
        '',
        '## Inter-Model Agreement (ICC)',
        '',
        f'| Scope | ICC |',
        f'|-------|-----|',
    ]
    if not np.isnan(icc_new):
        report.append(f'| New models only ({", ".join(model_names)}) | {icc_new:.3f} |')
    report.append(f'| All models incl. Gemini V1 | {icc_all:.3f} |')

    report += ['', '## Key Finding', '']
    if fpr_range < 0.15:
        report += [
            f'**FPR is consistent across all models (range = {fpr_range:.0%}, < 15pp threshold).**',
            '',
            'The false positive cases are inherently ambiguous — every model struggles with them.',
            'This suggests the FPR problem is a property of the task (borderline JDs with partial',
            'overlap), not a Gemini-specific miscalibration. Prompt engineering cannot fix a',
            'labelling problem: these JDs are genuinely on the boundary between H=1 and H=2.',
        ]
    else:
        best = min(summary_rows, key=lambda r: r['fpr'])
        report += [
            f'**FPR varies across models (range = {fpr_range:.0%}).**',
            f'{best["model"]} achieves the lowest FPR ({best["fpr"]:.0%}), suggesting',
            'the problem is partially model-specific and a better-calibrated model could reduce it.',
        ]

    report += [
        '',
        '*Raw data: `evaluation/data/batch-results/multi_model_comparison.json`*',
    ]

    rp = Path('evaluation/multi_model_report.md')
    rp.write_text('\n'.join(report), encoding='utf-8')
    print(f'\nReport saved → {rp}')

if __name__ == '__main__':
    main()
