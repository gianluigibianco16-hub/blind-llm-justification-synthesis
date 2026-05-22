# Explainable Fact-Checking via Blind LLM Justification Synthesis

> **Master's Thesis Pipeline — Code Repository**  
> This repository contains all generation, classification, and analysis code produced for the thesis *"Explainable Fact-Checking via Blind LLM Justification Synthesis"*, which investigates whether LLM-generated justifications produced without access to ground-truth veracity labels (the **blind protocol**) can improve automated fact-checking performance on the [LIAR-PLUS](https://github.com/Tariq60/LIAR-PLUS) dataset.

---

## Repository Structure

```
├── 01_paraphrase_openai.ipynb               # GPT-4o-mini paraphrases LIAR-PLUS claims
├── 02_paraphrase_gemini.ipynb               # Gemini 2.5 Flash Lite paraphrases LIAR-PLUS claims
├── 03_justification_openai_original.ipynb   # GPT-4o-mini justifications on original claims
├── 04_justification_openai_paraphrased.ipynb# GPT-4o-mini justifications on Gemini-paraphrased claims
├── 05_justification_gemini_original.ipynb   # Gemini 2.5 Flash Lite justifications on original claims
├── 06_justification_gemini_paraphrased.ipynb# Gemini 2.5 Flash Lite justifications on GPT-paraphrased claims
├── 07_bilstm_classifier.ipynb               # BiLSTM classification across all four conditions
├── 08_ted_contamination_probe.ipynb         # TED contamination diagnostic (GPT-4o-mini only)
├── 09_LIAR_PLUS_EDA.ipynb                   # Exploratory Data Analysis on LIAR-PLUS
│
└── data/
    ├── liar_plus_merged.xlsx                # LIAR-PLUS (train + val + test, split column preserved)
    ├── Liar_plus_synthetic.xlsx             # Final dataset: all claims + metadata + Paraphrase + 4 synthetic justification columns
    ├── openai_paraphrase_part1.xlsx         # Pre-split input files for OpenAI paraphrase notebook (to be upload manually)
    ├── openai_paraphrase_part2.xlsx
    ├── openai_paraphrase_part3.xlsx
    ├── openai_paraphrase_part4.xlsx
    │
    ├── EDA_results/
    │   └── eda_figures.zip                  # All EDA figures produced by notebook 09
    │
    ├── Model_Result/
    │   ├── results_C1.xlsx                  # Per-instance predictions and metrics, condition C1
    │   ├── results_C2.xlsx                  # Per-instance predictions and metrics, condition C2
    │   ├── results_C3.xlsx                  # Per-instance predictions and metrics, condition C3
    │   ├── results_C4.xlsx                  # Per-instance predictions and metrics, condition C4
    │   └── results_summary.xlsx             # Accuracy, weighted F1, macro F1 across all conditions
    │
    └── Ted_Results/
        └── TED_results.xlsx                 # TED contamination probe metrics (GPT-4o-mini)
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

---

### Notebook-specific notes

#### `01_paraphrase_openai.ipynb` — 4 iterations

The OpenAI Batch API enforces stricter per-batch request limits. The dataset is divided into **4 parts**, which are already pre-split and provided in `data/`:

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

## Output Files

| Output file | Description |
|---|---|
| `data/Liar_plus_synthetic.xlsx` | Final dataset with all 4 synthetic justification columns |
| `data/Model_Result/results_C1.xlsx` | Per-instance predictions, condition C1 |
| `data/Model_Result/results_C2.xlsx` | Per-instance predictions, condition C2 |
| `data/Model_Result/results_C3.xlsx` | Per-instance predictions, condition C3 |
| `data/Model_Result/results_C4.xlsx` | Per-instance predictions, condition C4 |
| `data/Model_Result/results_summary.xlsx` | Accuracy, weighted F1, macro F1 — all conditions |
| `data/Ted_Results/TED_results.xlsx` | TED contamination probe metrics |
| `data/EDA_results/eda_figures.zip` | All EDA figures |

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
