# Explainable Fact-Checking via Blind LLM Justification Synthesis

> **Master's Thesis Pipeline — Code Repository**  
> This repository contains all generation, classification, and analysis code produced for the thesis *"Explainable Fact-Checking via Blind LLM Justification Synthesis"*, which investigates whether LLM-generated justifications produced without access to ground-truth veracity labels (the **blind protocol**) can improve automated fact-checking performance on the [LIAR-PLUS](https://github.com/Tariq60/LIAR-PLUS) dataset.

---

## Repository Structure

```
├── 01_paraphrase_openai.ipynb                # GPT-4o-mini paraphrases LIAR-PLUS claims
├── 02_paraphrase_gemini.ipynb                # Gemini 2.5 Flash Lite paraphrases LIAR-PLUS claims
├── 03_justification_openai_original.ipynb    # GPT-4o-mini justifications on original claims
├── 04_justification_openai_paraphrased.ipynb # GPT-4o-mini justifications on Gemini-paraphrased claims
├── 05_justification_gemini_original.ipynb    # Gemini 2.5 Flash Lite justifications on original claims
├── 06_justification_gemini_paraphrased.ipynb # Gemini 2.5 Flash Lite justifications on GPT-paraphrased claims
├── 07_bilstm_classifier.ipynb                # BiLSTM classification across all four conditions
├── 08_ted_contamination_probe.ipynb          # TED contamination diagnostic (GPT-4o-mini only)
├── 09_LIAR_PLUS_EDA.ipynb                    # Exploratory Data Analysis on LIAR-PLUS
├── 10_mcnemar_test.py                        # McNemar pairwise significance tests across conditions
│
└── data/
    ├── liar_plus_merged.xlsx                 # LIAR-PLUS (train + val + test, split column preserved)
    ├── Liar_plus_synthetic.xlsx              # Final dataset: all claims + metadata + paraphrases + 4 synthetic justification columns
    ├── openai_paraphrase_part1.xlsx          # Pre-split input files for OpenAI paraphrase notebook
    ├── openai_paraphrase_part2.xlsx
    ├── openai_paraphrase_part3.xlsx
    ├── openai_paraphrase_part4.xlsx
    │
    ├── EDA_results/                          # EDA figures produced by notebook 09
    │
    ├── Model_Result/                         # Classification results across all runs (see details below)
    │
    ├── Ted_Results/
    │   └── TED_results.xlsx                  # TED contamination probe metrics (GPT-4o-mini)
    │
    └── McNemar_Results/
        ├── mcnemar_results.xlsx              # Aggregated McNemar test results (mean across 3 runs)
        └── Input/                            # 12 per-instance prediction files used as test input
```

---

## Experimental Design

The thesis compares five conditions on the binary veracity classification task (BiLSTM + GloVe, replicating Alhindi et al. 2018):

| Condition | Description |
|-----------|-------------|
| **Baseline A** | Claim only (Wang, 2017) |
| **Baseline B** | Claim + human justification (Alhindi et al., 2018) |
| **C1** | GPT-4o-mini justifications on Gemini-paraphrased claims |
| **C2** | Gemini 2.5 Flash Lite justifications on GPT-paraphrased claims |
| **C3** | GPT-4o-mini justifications on original claims |
| **C4** | Gemini 2.5 Flash Lite justifications on original claims |

The **mutual paraphrase design** ensures no model generates justifications for claims it has itself paraphrased, reducing surface-familiarity bias.

The final dataset `Liar_plus_synthetic.xlsx` contains all 12,791 LIAR-PLUS claims paired with four columns of blind synthetic justifications (one per condition), and is immediately usable for downstream research.

---

## Running the Pipeline

### Prerequisites

```bash
pip install openai google-generativeai pandas openpyxl tensorflow numpy scikit-learn statsmodels
```

You will need:
- An **OpenAI API key** with Batch API access
- A **Google AI API key** with Gemini access

### Setting up API keys (Google Colab Secrets)

All notebooks read API keys from Colab Secrets, not from the code. Before running any notebook:

1. Open the notebook in Google Colab
2. Click the 🔑 icon in the left sidebar ("Secrets")
3. Add the following secrets:
   - `OPENAI_API` → your OpenAI API key (`sk-...`)
   - `GOOGLE_API` → your Google AI API key (`AIza...`)
4. Enable access to the secret for the current notebook

### Dataset

The merged LIAR-PLUS dataset (`data/liar_plus_merged.xlsx`) is included in this repository. It was constructed by concatenating the three original splits (`train2.tsv`, `val2.tsv`, `test2.tsv`) from the [LIAR-PLUS repository](https://github.com/Tariq60/LIAR-PLUS), with a preserved `split` column to allow exact reconstruction of the original boundaries.

---

## Running Each Notebook — Iteration Instructions

Each generation notebook processes the full dataset (~12,800 instances) in **multiple iterations**, because the Batch API imposes file-size and request-count limits per submission.

### How iteration works

At the top of each notebook's **Cell 2**, you will find a block like this:

```python
# ── SELECT ITERATION ──────────────────────────────────────────────────────────
# Uncomment EXACTLY ONE of the following lines for each run.
# Re-run Cell 2 → Cell 5 for each iteration, then merge the output files.

df = df_full.iloc[0:2200]          # Iteration 1  ← uncomment this one first
# df = df_full.iloc[2200:4400]     # Iteration 2
# df = df_full.iloc[4400:6600]     # Iteration 3
# df = df_full.iloc[6600:8800]     # Iteration 4
# df = df_full.iloc[8800:11000]    # Iteration 5
# df = df_full.iloc[11000:]        # Iteration 6
# ─────────────────────────────────────────────────────────────────────────────
```

**For each iteration:**
1. In Cell 2, **uncomment the `df =` line for the current iteration** and comment all others.
2. Run Cell 2 → Cell 5 in sequence.
3. Cell 5 saves a partial output file (e.g., `output_iter1.xlsx`).
4. Repeat for all iterations.
5. Run the **merge cell** at the bottom of the notebook to concatenate all partial files.

### Notebook-specific notes

#### `01_paraphrase_openai.ipynb` — 4 iterations

The OpenAI Batch API enforces stricter per-batch request limits. The dataset is divided into **4 parts**, already pre-split and provided in `data/`:

```python
PART = 1   # Change to 2, 3, or 4 for subsequent iterations
input_file = f"data/openai_paraphrase_part{PART}.xlsx"
df = pd.read_excel(input_file)
```

#### `02_paraphrase_gemini.ipynb` — 4 iterations

Uses the `df_full.iloc[...]` slice pattern with 4 chunks.

#### `03` through `06` — 6 iterations each

These four notebooks process the full dataset in 6 chunks of ~2,133 instances using the `df_full.iloc[...]` pattern shown above.

---

## Classification Results and Reproducibility

Classification results are reported as **mean ± standard deviation across four independent training runs**, conducted in separate Colab sessions. This is necessary because TensorFlow's GPU backend introduces non-deterministic behaviour that is not fully controlled by the random seed, causing minor variation across sessions (up to 1.5 percentage points per condition).

The `data/Model_Result/` folder contains:

- **`results_C1.xlsx` through `results_C4.xlsx`** — per-instance predictions and metrics for each experimental condition, from the first training run. These are the files used as input to the McNemar test.
- **`results_summary_1.xlsx` through `results_summary_4.xlsx`** — aggregated accuracy, weighted F1, and macro F1 for all four conditions, one file per independent run. These four files are the basis for the mean ± std figures reported in the thesis (Table 2).

The mean ± std values across the four runs are:

| Condition | Test Acc | Test F1 (w) | Test F1 (macro) |
|-----------|----------|-------------|-----------------|
| C1 — GPT just. \| Gemini paraphrase | 0.621 ± 0.004 | 0.622 ± 0.004 | 0.617 ± 0.003 |
| C2 — Gemini just. \| GPT paraphrase | 0.619 ± 0.005 | 0.617 ± 0.005 | 0.609 ± 0.005 |
| C3 — GPT just. \| Original claim    | 0.618 ± 0.002 | 0.618 ± 0.003 | 0.612 ± 0.005 |
| C4 — Gemini just. \| Original claim | 0.622 ± 0.009 | 0.620 ± 0.007 | 0.614 ± 0.007 |

All four conditions consistently outperform the claim-only baseline (0.600) across all runs.

---

## McNemar's Test

`10_mcnemar_test.py` implements McNemar's test to assess whether accuracy differences between conditions are statistically significant. It operates on **three independent runs** (runs 2, 3, and 4 — labelled A, B, C), each providing per-instance predictions for all four conditions.

**Input files** (12 total, stored in `data/McNemar_Results/Input/`):
- `results_C{1-4}_predictions_A.xlsx` — predictions from run 2
- `results_C{1-4}_predictions_B.xlsx` — predictions from run 3
- `results_C{1-4}_predictions_C.xlsx` — predictions from run 4

Each file contains two columns: `true_label` and `pred_label`, one row per test instance (n = 1,267).

**Four pairwise comparisons** are tested:
1. C1 vs C3 — paraphrase effect, GPT-4o-mini
2. C2 vs C4 — paraphrase effect, Gemini 2.5 Flash Lite
3. C1 vs C2 — model effect, paraphrased-claim conditions
4. C3 vs C4 — model effect, original-claim conditions

**Results** (`data/McNemar_Results/mcnemar_results.xlsx`):

| Comparison | mean χ² | mean p | min p |
|---|---|---|---|
| C1 (GPT, paraphrased) vs C3 (GPT, original) | 0.023 | 0.903 | 0.804 |
| C2 (Gemini, paraphrased) vs C4 (Gemini, original) | 0.530 | 0.657 | 0.214 |
| C1 (GPT, paraphrased) vs C2 (Gemini, paraphrased) | 0.228 | 0.686 | 0.461 |
| C3 (GPT, original) vs C4 (Gemini, original) | 0.561 | 0.577 | 0.260 |

No comparison reaches statistical significance (α = 0.05) in any of the three runs, confirming that neither the paraphrase effect nor the model effect produces a reliably different outcome.

---

## Output Files Summary

| File / Folder | Description |
|---|---|
| `data/Liar_plus_synthetic.xlsx` | Final dataset: all claims + paraphrases + 4 synthetic justification columns |
| `data/Model_Result/results_C{1-4}.xlsx` | Per-instance predictions, condition C1–C4 (run 1) |
| `data/Model_Result/results_summary_{1-4}.xlsx` | Aggregated metrics across four independent runs |
| `data/Ted_Results/TED_results.xlsx` | TED contamination probe metrics (GPT-4o-mini) |
| `data/McNemar_Results/mcnemar_results.xlsx` | Aggregated McNemar test results (mean across 3 runs) |
| `data/McNemar_Results/Input/` | 12 per-instance prediction files used as McNemar test input |
| `data/EDA_results/eda_figures.zip` | All EDA figures produced by notebook 09 |

---

## Citation

If you use this code or dataset in your research, please cite:

```
[Author]. (2025). Explainable Fact-Checking via Blind LLM Justification Synthesis.
Master's Thesis.
```

And the underlying dataset:

```
Alhindi, T., Petridis, S., & Muresan, S. (2018).
Where is Your Evidence: Improving Fact-checking by Justification Modeling.
Proceedings of the First Workshop on Fact Extraction and VERification (FEVER), pp. 85–90.
ACL. https://doi.org/10.18653/v1/W18-5513
```

---

## License

Code in this repository is released under the MIT License.  
The LIAR-PLUS dataset is subject to its [original terms of use](https://github.com/Tariq60/LIAR-PLUS).
