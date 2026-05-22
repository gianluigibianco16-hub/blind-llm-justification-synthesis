# Explainable Fact-Checking via Blind LLM Justification Synthesis

> **Master's Thesis Pipeline — Code Repository**  
> This repository contains all generation and classification code produced for the thesis *"Explainable Fact-Checking via Blind LLM Justification Synthesis"*, which investigates whether LLM-generated justifications produced without access to ground-truth veracity labels (the **blind protocol**) can improve automated fact-checking performance on the [LIAR-PLUS](https://github.com/Tariq60/LIAR-PLUS) dataset.

---

## Repository Structure

```
├── 01_paraphrase_openai.ipynb          # GPT-4o-mini paraphrases LIAR-PLUS claims
├── 02_paraphrase_gemini.ipynb          # Gemini 2.5 Flash Lite paraphrases LIAR-PLUS claims
├── 03_justification_openai_original.ipynb    # GPT-4o-mini justifications on original claims
├── 04_justification_openai_paraphrased.ipynb # GPT-4o-mini justifications on Gemini-paraphrased claims
├── 05_justification_gemini_original.ipynb    # Gemini 2.5 Flash Lite justifications on original claims
├── 06_justification_gemini_paraphrased.ipynb # Gemini 2.5 Flash Lite justifications on GPT-paraphrased claims
├── 07_bilstm_classifier.ipynb          # BiLSTM classification across all four conditions
├── 08_ted_contamination_probe.ipynb    # TED-based contamination diagnostic (GPT-4o-mini only)
│
├── data/
│   ├── liar_plus_merged.xlsx           # Full LIAR-PLUS (train + val + test, split column preserved)
│   ├── openai_paraphrase_part1.xlsx    # Pre-split input files for OpenAI paraphrase notebook
│   ├── openai_paraphrase_part2.xlsx
│   ├── openai_paraphrase_part3.xlsx
│   └── openai_paraphrase_part4.xlsx
│
└── outputs/                            # Generated justifications and classification results
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

---

## Running the Pipeline

### Prerequisites

```bash
pip install openai google-generativeai pandas openpyxl tensorflow numpy scikit-learn
```

You will need:
- An **OpenAI API key** with Batch API access
- A **Google AI API key** with Gemini access
- The LIAR-PLUS dataset (see below)

### Dataset

Download LIAR-PLUS from the [official repository](https://github.com/Tariq60/LIAR-PLUS). Place the three split files (`train2.tsv`, `val2.tsv`, `test2.tsv`) in the `data/` folder.

Run `01_paraphrase_openai.ipynb` Cell 1 to merge the splits into `liar_plus_merged.xlsx` with a preserved `split` column.

---

## Running Each Notebook — Iteration Instructions

Each generation notebook processes the full dataset (≈12,800 instances) in **multiple iterations**, because the Batch API imposes file-size and request-count limits. Each iteration processes one chunk of the dataset.

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

1. Open the notebook.
2. In Cell 2, **uncomment the `df =` line for the current iteration** and comment all others.
3. Run Cell 2 → Cell 5 in sequence.
4. Cell 5 saves a partial output file (e.g., `output_iter1.xlsx`).
5. Repeat steps 2–4 for iterations 2 through N.
6. After all iterations complete, run the **merge cell** at the bottom of the notebook to concatenate all partial files into a single output.

> **Tip:** Run one iteration per Colab session, or use separate Colab tabs. Batch jobs can take several hours; Cell 4 polls for completion and will wait until the job finishes.

---

### Notebook-specific notes

#### `01_paraphrase_openai.ipynb` — 4 iterations

The OpenAI Batch API enforces a stricter per-batch request limit for free/low-tier accounts. The dataset is therefore divided into **4 parts** rather than 6.

> **Pre-split files provided:** `data/openai_paraphrase_part1.xlsx` through `part4.xlsx` are already split and ready to use.  
> Cell 2 of this notebook loads one part file per iteration rather than slicing `df_full`. Change the `PART` constant at the top of Cell 2:

```python
PART = 1   # Change to 2, 3, or 4 for subsequent iterations
input_file = f"data/openai_paraphrase_part{PART}.xlsx"
df = pd.read_excel(input_file)
```

#### `02_paraphrase_gemini.ipynb` — 4 iterations

Same split logic as above, but uses the Google AI Batch API. Uses the `df_full.iloc[...]` slice pattern with 4 chunks.

#### `03_justification_openai_original.ipynb` through `06_justification_gemini_paraphrased.ipynb` — 6 iterations each

These four notebooks each process the full ≈12,800-instance dataset in 6 chunks of ≈2,133 instances. Use the `df_full.iloc[...]` slice pattern shown above.

---

## Output Files

After all iterations complete and the merge cell is run, each notebook produces one Excel file:

| Output file | Used as input to classifier |
|---|---|
| `paraphrases_openai.xlsx` | C2, C4 input (Gemini justification notebook) |
| `paraphrases_gemini.xlsx` | C1, C3 input (OpenAI justification notebook) |
| `justifications_c1_openai_geminiparaphrase.xlsx` | Condition C1 |
| `justifications_c2_gemini_openaiparaphrase.xlsx` | Condition C2 |
| `justifications_c3_openai_original.xlsx` | Condition C3 |
| `justifications_c4_gemini_original.xlsx` | Condition C4 |

---

## Classification

`07_bilstm_classifier.ipynb` trains and evaluates the BiLSTM classifier for all four experimental conditions and the claim-only baseline. It requires:

- GloVe 6B 100d embeddings: download from [nlp.stanford.edu/projects/glove](https://nlp.stanford.edu/projects/glove/) and place `glove.6B.100d.txt` in the `data/` folder.
- The four justification output files above.

The notebook reports accuracy, weighted F1, and macro F1 on validation and test sets, and runs McNemar's tests between all condition pairs (see thesis Section 5.2).

---

## Contamination Assessment

`08_ted_contamination_probe.ipynb` implements the TED-based contamination diagnostic described in thesis Section 4.7. It requires an OpenAI API key with `logprobs` support and runs synchronous (non-batch) calls on a 400-instance stratified sample.

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
In Proceedings of the First Workshop on Fact Extraction and VERification (FEVER), pp. 85–90.
ACL. https://doi.org/10.18653/v1/W18-5513
```

---

## License

Code in this repository is released under the MIT License.  
The LIAR-PLUS dataset is subject to its [original terms of use](https://github.com/Tariq60/LIAR-PLUS).
