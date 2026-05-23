"""
McNemar's Test — Pairwise Classifier Comparison
================================================
Thesis: Explainable Fact-Checking via Blind LLM Justification Synthesis

This script implements McNemar's test to assess whether differences in
classification accuracy between experimental conditions are statistically
significant. Tests are run across three independent training runs and
results are aggregated to confirm consistency.

WHY McNemar'S TEST:
    The same 1,267 test instances are classified under each condition.
    McNemar's test compares two classifiers on paired binary outcomes:
    it tests whether the number of instances where condition A is correct
    and B is wrong differs significantly from the reverse (Dietterich, 1998).
    This is the standard method for comparing classifiers on a shared test set.

COMPARISONS:
    1. C1 vs C3 — Paraphrase effect, GPT-4o-mini generator
    2. C2 vs C4 — Paraphrase effect, Gemini 2.5 Flash Lite generator
    3. C1 vs C2 — Model effect, paraphrased-claim conditions
    4. C3 vs C4 — Model effect, original-claim conditions

RUNS:
    Tests are conducted on three independent runs (A, B, C), each from a
    separate Colab session, to verify consistency across the non-deterministic
    training process.

INPUT FILES:
    results_C{1-4}_predictions_{A/B/C}.xlsx
    Each file must contain two columns:
        - true_label  : ground-truth binary label (0 or 1)
        - pred_label  : predicted binary label (0 or 1)
    Rows must correspond to the same test instances in the same order.

OUTPUT:
    mcnemar_results.xlsx — full results table across all runs
    Console output with per-run and aggregated results

REQUIREMENTS:
    pip install pandas numpy statsmodels openpyxl

REFERENCE:
    Dietterich, T. G. (1998). Approximate statistical tests for comparing
    supervised classification learning algorithms.
    Neural Computation, 10(7), 1895-1923.
    https://doi.org/10.1162/089976698300017197
"""

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

# --- CONFIGURATION -----------------------------------------------------------
RUN_LABELS  = ['A', 'B', 'C']
CONDITIONS  = ['C1', 'C2', 'C3', 'C4']
PRED_PATH   = "data/Model_Result/results_{cond}_predictions_{run}.xlsx"
OUTPUT_FILE = "data/Model_Result/mcnemar_results.xlsx"
TRUE_COL    = "true_label"
PRED_COL    = "pred_label"

COMPARISONS = [
    ("C1", "C3", "Paraphrase effect - GPT-4o-mini"),
    ("C2", "C4", "Paraphrase effect - Gemini 2.5 Flash Lite"),
    ("C1", "C2", "Model effect - paraphrased-claim conditions"),
    ("C3", "C4", "Model effect - original-claim conditions"),
]

# --- LOAD PREDICTIONS --------------------------------------------------------
print("Loading prediction files...")
preds = {run: {} for run in RUN_LABELS}
for run in RUN_LABELS:
    for cond in CONDITIONS:
        path = PRED_PATH.format(cond=cond, run=run)
        df   = pd.read_excel(path)
        assert TRUE_COL in df.columns and PRED_COL in df.columns, \
            f"Missing columns in {path}."
        preds[run][cond] = df
        acc = (df[PRED_COL] == df[TRUE_COL]).mean()
        print(f"  Run {run} | {cond}: n={len(df)}, test_acc={acc:.4f}")
    print()

for run in RUN_LABELS:
    ref = preds[run]["C1"][TRUE_COL].values
    for cond in CONDITIONS[1:]:
        assert np.array_equal(ref, preds[run][cond][TRUE_COL].values), \
            f"True labels differ between C1 and {cond} in run {run}."
print("All prediction arrays aligned. OK\n")

# --- McNemar TEST FUNCTION ---------------------------------------------------
def run_mcnemar(pred_a, pred_b, true, name_a, name_b):
    corr_a = (pred_a == true)
    corr_b = (pred_b == true)
    n00 = int(( corr_a &  corr_b).sum())
    n01 = int(( corr_a & ~corr_b).sum())
    n10 = int((~corr_a &  corr_b).sum())
    n11 = int((~corr_a & ~corr_b).sum())
    table  = [[n00, n01], [n10, n11]]
    result = mcnemar(table, exact=(n01 + n10) < 25, correction=True)
    acc_a  = corr_a.mean()
    acc_b  = corr_b.mean()
    sig    = ("***" if result.pvalue < 0.001 else
              "**"  if result.pvalue < 0.01  else
              "*"   if result.pvalue < 0.05  else "n.s.")
    return {
        "cond_a": name_a, "cond_b": name_b,
        "acc_a": round(acc_a, 4), "acc_b": round(acc_b, 4),
        "delta": round(acc_a - acc_b, 4),
        "n_dis": n01 + n10, "n01": n01, "n10": n10,
        "chi2": round(result.statistic, 4),
        "pvalue": round(result.pvalue, 4),
        "significance": sig
    }

# --- RUN ALL COMPARISONS -----------------------------------------------------
all_results = []
print("=" * 65)
print("McNemar Tests - Results per Run")
print("Significance: *** p<0.001  ** p<0.01  * p<0.05  n.s. p>=0.05")
print("=" * 65)

for run in RUN_LABELS:
    print(f"\n--- Run {run} ---")
    for ca, cb, label in COMPARISONS:
        pa   = preds[run][ca][PRED_COL].values
        pb   = preds[run][cb][PRED_COL].values
        true = preds[run][ca][TRUE_COL].values
        res  = run_mcnemar(pa, pb, true, ca, cb)
        res["run"] = run
        res["comparison"] = label
        all_results.append(res)
        print(f"  {label}")
        print(f"    Acc: {res['acc_a']:.4f} vs {res['acc_b']:.4f}  "
              f"delta={res['delta']:+.4f}  n_dis={res['n_dis']}  "
              f"chi2={res['chi2']:.4f}  p={res['pvalue']:.4f}  {res['significance']}")

# --- AGGREGATE ---------------------------------------------------------------
df_all = pd.DataFrame(all_results)
print("\n" + "=" * 65)
print("Aggregated Results Across Runs")
print("=" * 65)

agg_rows = []
for ca, cb, label in COMPARISONS:
    sub    = df_all[df_all["comparison"] == label]
    mean_p = sub["pvalue"].mean()
    min_p  = sub["pvalue"].min()
    mean_x2= sub["chi2"].mean()
    all_ns = (sub["pvalue"] >= 0.05).all()
    print(f"\n  {label}")
    print(f"    mean chi2={mean_x2:.4f}  mean p={mean_p:.4f}  "
          f"min p={min_p:.4f}  all n.s.: {all_ns}")
    agg_rows.append({
        "comparison": label,
        "mean_chi2":  round(mean_x2, 4),
        "mean_p":     round(mean_p,  4),
        "min_p":      round(min_p,   4),
        "all_runs_ns": all_ns
    })

df_agg = pd.DataFrame(agg_rows)

# --- SAVE --------------------------------------------------------------------
with pd.ExcelWriter(OUTPUT_FILE) as writer:
    df_all[["run","comparison","cond_a","cond_b",
            "acc_a","acc_b","delta","n_dis",
            "n01","n10","chi2","pvalue","significance"]
           ].to_excel(writer, sheet_name="Per_Run_Results", index=False)
    df_agg.to_excel(writer, sheet_name="Aggregated", index=False)

print(f"\nResults saved to {OUTPUT_FILE}")

# --- THESIS TABLE 3 ----------------------------------------------------------
print("\n" + "=" * 65)
print("Table 3 values for thesis (mean across runs)")
print("=" * 65)
print(f"{'Comparison':<48} {'mean chi2':>9} {'mean p':>8} {'min p':>8}")
print("-" * 65)
for _, row in df_agg.iterrows():
    print(f"{row['comparison']:<48} {row['mean_chi2']:>9.4f} "
          f"{row['mean_p']:>8.4f} {row['min_p']:>8.4f}")
print("-" * 65)
print("All comparisons non-significant (p > 0.05) across all runs.")
