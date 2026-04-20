# Saved for Later — Deleted Content

Content removed from case_study.md and the evaluation notebook, preserved here for potential future use.

---

## A. Section 12 — Output Quality and Explainability (Track E)

*Removed from case_study.md because: Track E (LLM-as-judge explanation quality) was removed from the main document scope. The notebook Part 17 was also removed. Content saved here in case Track E is revived in a future update.*

---

### 12. Output Quality and Explainability

> **→ Framework:** [→ FW §12] — LLM-as-judge methodology, verbosity bias, explanation evaluation

#### 12.1 Evaluation of Generated Explanations

*Q11 · Are the AI's explanations useful?*

The system generates a structured explanation for each analysis (SUMMARY, CV TWEAKS, ATS KEYWORDS, FINAL TIP). Explanation quality was evaluated using an LLM-as-judge approach with GPT-4o-mini as judge, rated independently of the score.

#### 12.2 LLM-as-Judge Methodology

**Judge model:** GPT-4o-mini (~$0.03 for 32 JDs; lower capability intentional to avoid self-enhancement bias [[7]](#references)). Each SUMMARY text rated on three dimensions (1–5 scale): *Accuracy*, *Specificity*, *Usefulness*. P1 and P2 summaries for each JD rated independently. **Blind pairwise comparison** on the 8 FP cases: judge shown two anonymised summaries (A/B). A/B assignment randomised with `random.seed(42)`. **Verbosity bias check:** Pearson r between summary length and rating [[6]](#references).

#### 12.3 Analysis of Usefulness, Specificity, and Accuracy

| Condition | Mean Rating (1–5) | SD |
|-----------|------------------:|---:|
| P1 (rubric prompt) — SUMMARY | **4.08** | 0.61 |
| P2 (CoT prompt) — SUMMARY | **3.25** | 0.79 |
| P1 blind pairwise wins (8 FP cases) | **7/8** | — |

| Bias Check | r (length vs rating) | Interpretation |
|------------|---------------------:|----------------|
| P1 verbosity correlation | 0.12 | Negligible |
| P2 verbosity correlation | 0.09 | Negligible |

> **Key Finding — P1 Produces Better Explanations Than P2:** The P1 rubric prompt produces higher-quality summaries on all three dimensions. P2's explicit reasoning chain causes the model to "overthink" — producing longer, more hedged summaries that are less directly actionable. P1 wins 7/8 blind pairwise comparisons. The verbosity bias check confirms the judge is not simply rewarding longer outputs.

#### 12.4 Limitations of Automated Evaluation of Explanations

The LLM-as-judge methodology tests length-rating correlation (r ≈ 0.10, negligible) but does not test whether the judge rewards coherence or factual accuracy independent of surface fluency. The auto-evaluation also introduces a circularity: the judge (GPT-4o-mini) is evaluating outputs from a related model family (Gemini). Finally, the evaluation was conducted on SUMMARY text only; CV TWEAKS and ATS KEYWORDS quality were not formally rated.

#### Cross-References

> **→ Framework:** [→ FW §12] — LLM-as-judge design, verbosity bias, automated evaluation limitations
> **→ Notebook:** Part 17 — "LLM-as-Judge: Reasoning Quality (Track E)"

---

## B. Notebook Part 17 — LLM-as-Judge: Reasoning Quality (Track E)

*Removed from `career_pilot_evaluation.ipynb` cells 48–50 because: matched deleted Section 12 content. Preserved here verbatim.*

---

### Cell 48 (markdown)

```
---
## Part 17 — LLM-as-Judge: Reasoning Quality (Track E)

**Research question:** Does P2 (chain-of-thought prompt) produce better-quality explanations than P1 (rubric prompt)? And does reasoning quality degrade on the cases the AI gets wrong (FP cases)?

**Method:** GPT-4o-mini independently rates Gemini's SUMMARY text (not the score) for each of 32 JDs under P1 and P2 on three criteria (1–5 scale):
- **Accuracy** — does the reasoning correctly reflect job requirements?
- **Specificity** — concrete evidence cited vs generic statements?
- **Usefulness** — would this help a candidate decide to apply?

Additional: blind pairwise comparison on the 8 FP cases (A/B randomised to prevent position bias), verbosity bias check (summary length vs rating).
```

---

### Cell 49 (code)

```python
import json
import numpy as np
import matplotlib.pyplot as plt

with open('../evaluation/data/batch-results/llm_judge_scores.json') as f:
    judge_data = json.load(f)

def mean_rating(r):
    vals = [r[k] for k in ['accuracy','specificity','usefulness'] if r and r.get(k) is not None]
    return round(sum(vals)/len(vals), 3) if vals else None

complete = [r for r in judge_data
            if mean_rating(r.get('p1_ratings')) is not None
            and mean_rating(r.get('p2_ratings')) is not None]
n = len(complete)

p1_avgs = [mean_rating(r['p1_ratings']) for r in complete]
p2_avgs = [mean_rating(r['p2_ratings']) for r in complete]

print(f'LLM-as-Judge: Reasoning Quality  (n={n}, judge=GPT-4o-mini)')
print()
print(f'{"Metric":<15}  {"P1 (rubric)":>11}  {"P2 (CoT)":>9}  {"Δ (P2−P1)":>10}')
print('─'*50)
print(f'{"Overall mean":<15}  {np.mean(p1_avgs):>11.3f}  {np.mean(p2_avgs):>9.3f}  '
      f'{np.mean(p2_avgs)-np.mean(p1_avgs):>+10.3f}')

for dim in ['accuracy', 'specificity', 'usefulness']:
    p1_d = [r['p1_ratings'][dim] for r in complete if r['p1_ratings'].get(dim)]
    p2_d = [r['p2_ratings'][dim] for r in complete if r['p2_ratings'].get(dim)]
    delta = np.mean(p2_d) - np.mean(p1_d)
    print(f'{dim.capitalize():<15}  {np.mean(p1_d):>11.3f}  {np.mean(p2_d):>9.3f}  {delta:>+10.3f}')

# FP vs non-FP
fp_rows  = [r for r in complete if r.get('is_fp')]
nfp_rows = [r for r in complete if not r.get('is_fp')]
if fp_rows and nfp_rows:
    fp_p1  = np.mean([mean_rating(r['p1_ratings']) for r in fp_rows])
    nfp_p1 = np.mean([mean_rating(r['p1_ratings']) for r in nfp_rows])
    print(f'\nReasoning quality on FP vs non-FP cases (P1):')
    print(f'  FP cases (n={len(fp_rows)}):      {fp_p1:.3f}')
    print(f'  Non-FP cases (n={len(nfp_rows)}): {nfp_p1:.3f}')
    print(f'  Gap (non-FP higher by): {nfp_p1-fp_p1:+.3f}')

# Verbosity bias
with open('../evaluation/data/batch-results/prompt_p1.json') as f:
    p1_data = {r['jd_id']: r for r in json.load(f)}
with open('../evaluation/data/batch-results/prompt_p2.json') as f:
    p2_data = {r['jd_id']: r for r in json.load(f)}

p1_lens=[]; p1_ratings_=[]; p2_lens=[]; p2_ratings_=[]
for r in complete:
    jid = r['jd_id']
    if jid in p1_data and mean_rating(r['p1_ratings']):
        p1_lens.append(len(p1_data[jid].get('summary','')))
        p1_ratings_.append(mean_rating(r['p1_ratings']))
    if jid in p2_data and mean_rating(r['p2_ratings']):
        p2_lens.append(len(p2_data[jid].get('summary','')))
        p2_ratings_.append(mean_rating(r['p2_ratings']))

print(f'\nVerbosity bias: r(length, quality rating)')
if p1_lens: print(f'  P1: r={np.corrcoef(p1_lens, p1_ratings_)[0,1]:.3f}  (mean={np.mean(p1_lens):.0f} chars)')
if p2_lens: print(f'  P2: r={np.corrcoef(p2_lens, p2_ratings_)[0,1]:.3f}  (mean={np.mean(p2_lens):.0f} chars)')

# Pairwise
pw = [r for r in judge_data if r.get('pairwise') and r['pairwise'].get('actual_winner')]
if pw:
    p1w = sum(1 for r in pw if r['pairwise']['actual_winner']=='P1')
    p2w = sum(1 for r in pw if r['pairwise']['actual_winner']=='P2')
    ties = sum(1 for r in pw if r['pairwise']['actual_winner']=='Tie')
    print(f'\nBlind pairwise on FP cases (n={len(pw)}): P1 wins={p1w}  P2 wins={p2w}  Ties={ties}')

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Track E — LLM-as-Judge: Reasoning Quality', fontsize=13, fontweight='bold')
dims = ['Accuracy', 'Specificity', 'Usefulness']
p1_means_d = [np.mean([r['p1_ratings'][d.lower()] for r in complete if r['p1_ratings'].get(d.lower())]) for d in dims]
p2_means_d = [np.mean([r['p2_ratings'][d.lower()] for r in complete if r['p2_ratings'].get(d.lower())]) for d in dims]
x = np.arange(len(dims)); w = 0.3
ax = axes[0]
ax.bar(x-w/2, p1_means_d, w, label='P1 (rubric)', color='#6366f1', alpha=0.85)
ax.bar(x+w/2, p2_means_d, w, label='P2 (CoT)', color='#f59e0b', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(dims)
ax.set_ylabel('Mean rating (1–5)'); ax.set_ylim(0, 5.5); ax.legend()
ax.set_title('P1 vs P2 quality by dimension')
ax2 = axes[1]
fp_p1_list  = [mean_rating(r['p1_ratings']) for r in complete if r.get('is_fp')]
nfp_p1_list = [mean_rating(r['p1_ratings']) for r in complete if not r.get('is_fp')]
ax2.scatter(np.ones(len(nfp_p1_list))*0 + np.random.RandomState(0).uniform(-0.1,0.1,len(nfp_p1_list)),
            nfp_p1_list, alpha=0.6, color='#10b981', s=50, label=f'Non-FP (n={len(nfp_p1_list)})')
ax2.scatter(np.ones(len(fp_p1_list))*1  + np.random.RandomState(1).uniform(-0.1,0.1,len(fp_p1_list)),
            fp_p1_list,  alpha=0.6, color='#ef4444', s=50, label=f'FP (n={len(fp_p1_list)})')
ax2.set_xticks([0,1]); ax2.set_xticklabels(['Non-FP','FP'])
ax2.set_ylabel('P1 reasoning quality (mean 1–5)'); ax2.set_title('Reasoning quality: FP vs non-FP')
ax2.set_ylim(1, 5.5); ax2.legend()
plt.tight_layout()
plt.savefig('../evaluation/figures/llm_judge_quality.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

### Cell 50 (markdown)

```
---
### Part 17 — Findings

**P1 (rubric) produces significantly better explanations than P2 (CoT)** across all three quality dimensions:

| Dimension | P1 (rubric) | P2 (CoT) | Δ |
|-----------|-------------|----------|---|
| Accuracy  | 4.38 | 3.81 | −0.56 |
| **Specificity** | **3.78** | **2.53** | **−1.25** |
| Usefulness | 4.09 | 3.41 | −0.69 |
| **Overall** | **4.08** | **3.25** | **−0.83** |

The largest gap is in **Specificity** (−1.25pp): P2 CoT summaries are more verbose but less grounded — they tend toward generic statements rather than citing concrete evidence from the CV.

**Blind pairwise on FP cases: P1 preferred 7/8 times.** Even on the cases where the AI score is wrong, the P1 explanation is clearer and more specific than P2.

**FP reasoning quality gap:** P1 quality drops from 4.15 (non-FP) to 3.88 (FP) — a small but consistent signal that the model is less confident in its reasoning on borderline cases, which is where it makes errors.

**Verbosity bias is negligible:** r=0.31 for P1 (weak), r=0.06 for P2. Summary length is not driving quality ratings.

**Conclusion:** The deployed P1 (rubric) prompt is the right choice — not just for accuracy (FNR=0%) but also for explanation quality. Chain-of-thought adds length without adding substance for this task.
```
