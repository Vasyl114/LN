# Group 26 — code

Python 3.10, CPU only. Install: `pip install -r requirements.txt` and
`pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu`.
Run every script from this folder. Every script is seeded (42) and documents itself in its header.

**Data.** `train.csv` and `test_no_labels.csv` (given by the teachers) are read from `data/` next to the scripts, or from `../../StudentsPack/` if that folder exists. `train_augmented.csv` is the augmented training set (step 2 regenerates it identically).
**Internet** is needed once: M3 downloads Bio_ClinicalBERT (the exact release is pinned in `M3_bert_model.py`) and M1 downloads the NLTK tokenizer data.
**Output folders** (`../figures/`) are created when missing; `../results.txt` is written next to `code/`.

| Step | Command | Time | Output |
|---|---|---|---|
| 1 | `python3 data_analysis.py` | seconds | statistics, `../figures/fig1_text_lengths.png` |
| 2 | `python3 data_augmentation.py` | 25 s | `train_augmented.csv` (already included) |
| 3 | `python3 M1_baseline_model.py` | 20 s | `results/m1_baseline_original.json` |
| 4 | `python3 M2_classical_model.py` | 10 min | `results/m2*` |
| 5 | `python3 M3_bert_model.py` | 50 min | `results/m3_heldout_scores.npy` |
| 6 | `python3 M4_final_system.py` | seconds | `results/m3.json`, `m4.json`, `summary_table.csv` |
| 7 | `python3 M4_error_analysis.py` | seconds | `../figures/fig2_confusion_matrix.png`, error report |
| 8 | `python3 M3_bert_model.py --full` | 55 min | `results/m3_test_scores.npy` (trained on all records) |
| 9 | `python3 generate_results.py` | seconds | `../results.txt` |

Shared modules: `data_utils.py` (loading, repair of `train.csv`, split, first keyword),
`evaluation.py` (metrics), `model_utils.py` (M2 pipeline, keyword rule of M4).
`M3_bert_model.py --smoke` is a one-minute check that writes nothing in this folder.
