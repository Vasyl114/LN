# Project Status — Group 26 (NLP "Help the Doctor")

> Hand-over document: what the project is, how the files are organised, what has been built, the numbers, where work stopped and what is left. Rewritten on **2026-10-11** after the model line-up was simplified (no layers, no copy-aware decoding). Deadline: **2026-10-16, 23:59** (Fenix, `26.zip`).

---

## 1. The project in six lines

- **Task:** predict the medical specialty (12 classes) of a clinical record (`train.csv` → `test_no_labels.csv`).
- **Deliverables:** `26.zip` with `26.pdf` (max 3 pages, template-based), `results.txt` (398 labels, one per line, no header, same order as the test file) and `code/` (no model weights).
- **Grading:** 4 points automatic (accuracy above 49% **and** no class F1 below 25% on the hidden test labels) + 16 for the paper (section scores in `Report_Plan.md`). The 49% is only a threshold, not a number to reproduce.
- **Research question (paper, Introduction):** *does a fine-tuned clinical language model that reads only a little text of each record outperform a classical classifier that reads all of it?*
- **Teachers' answer (2026-10-10):** no restriction on the approach (rule-based, ChatGPT, fine-tuning all allowed); what matters is a coherent narrative. So the first keyword may be used. They also published a clean test file (398 records) and allowed training on the entire training data.
- **Rules:** the test file is only for `results.txt`; files in `StudentsPack/` are never modified; LLM use must be declared in the paper.

---

## 2. File-system architecture

```
03_Main_Project_LN/
├── StudentsPack/        GIVEN, never edited (task PDF, LaTeX template, train.csv, test_no_labels.csv)
├── Project_Guide/       our documentation
│   ├── 01_project_overview_and_plan.md   digest of the PDF/template/CSVs (Section 9 = verified facts)
│   ├── 02_..., 03_...   ORIGINAL guides written before the models existed (their model plans are superseded by this file)
│   ├── Report_Plan.md   section-by-section plan of the paper
│   ├── Project_Status.md  THIS FILE
│   └── notes/project_storyline.md   non-technical story of the project (partly outdated, kept as background)
└── 26/                  OUR WORK STATION = what becomes 26.zip
    ├── 26.tex, biblio.bib, figures/   the paper and its two appendix figures
    ├── results.txt                    398 predicted labels (made by code/07_generate_results.py)
    └── code/
        ├── data_utils.py     SHARED: load_train() (repairs the 7 corrupted rows in memory), load_test(),
        │                     split_train_heldout(), remove_heldout(), first_keyword_label()
        ├── evaluation.py     SHARED: summarize() = the one evaluation used by every model
        ├── model_utils.py    SHARED: TF-IDF + linear SVM pipeline, add_first_keyword() (the Model 4 rule, weight 5)
        ├── M1_baseline_model.py     M1  Naive Bayes + TF-IDF on description (the given baseline)
        ├── 02_data_analysis.py      statistics, the two observations, figures
        ├── 03_data_augmentation.py  writes train_augmented.csv (rule-based, seeded)
        ├── M2_classical_model.py    M2  SVM on all fields, CV-tuned
        ├── M3_bert_model.py         M3  Bio_ClinicalBERT on the description; --full = train on all records, score the test file
        ├── M4_final_system.py       M4  = M3 + first keyword; summary table (reads saved scores, no training)
        ├── 07_generate_results.py   writes ../results.txt from M4
        ├── train_augmented.csv      5,648 records (2,669 original + 2,979 generated)
        ├── requirements.txt         pinned library versions
        └── results/                 small result files (see §3)
```

---

## 3. The models (one version each, no layers)

| Model | Reads | Technique | Held-out acc / macro-F1 / lowest class F1 (%) |
|---|---|---|---|
| M1 baseline | description | lowercase + Porter, TF-IDF, Multinomial NB, defaults | 38.8 / 13.7 / 0.0 |
| M2 classical | description, sample name, transcription, keywords | TF-IDF (1–2 words), class-balanced linear SVM; C = 250 and n-grams by 5-fold CV | 86.3 / 81.5 / 50.0 |
| M3 language model | description (128 tokens) | Bio_ClinicalBERT + linear layer, fine-tuned, lr 3e-5, batch 16, 3 epochs, seed 42 | 48.9 / 51.0 / 20.8 |
| **M4 final** | M3 + first keyword | first keyword's specialty added to M3's standardised scores with weight 5 (no score gap can outvote it); no training of its own | **92.1 / 89.6 / 70.0** |
| ablation | M2 + first keyword | the same rule on M2 (information only) | 89.1 / 87.2 / 75.0 |

- Held-out = 534 records (20%, stratified, seed 42, `split_train_heldout`); never trained on, never an augmentation source; nothing tuned on it. One accuracy carries about ±4 points.
- M4 is 100% right on held-out records with a first keyword (440) and 55.3% on those without (94, decided by M3).
- M2 reaches 88–89% on description + sample name + keywords alone (no transcription): the long transcription does not help it.
- Augmentation makes no clear difference for M2 (86.7% original vs 86.0%/86.3% augmented).
- The best C of M2 sits at the edge of the grid (250); the gain from 50 to 250 was about 1 point in CV.
- M3's held-out numbers come from the saved scores of the run of 2026-10-08 (same code and data; training log: accuracy per epoch 53.4 → 50.2 → 48.9, the model memorises notes listed under other labels).
- Removed on purpose (2026-10-11): the L0/L1/L2 layers, copy-aware decoding (it relies on the dataset never repeating a (label, note) pair, checkable only against test labels), the fusion of M2 and M3, and the frozen-embedding model.

`results/` holds: `m1_baseline_original.json`, `m2.json`, `m3.json`, `m4.json` and their held-out prediction CSVs, `m2_config.json` (chosen hyperparameters), `m2_heldout_scores.npy`, `m3_heldout_scores.npy`, `m3_test_scores.npy` (after the `--full` run), `m2_input_ablation.csv`, `summary_table.csv`.

---

## 4. How to run everything

From `26/code/` (Python 3.10, CPU only; versions in `requirements.txt`; PyTorch CPU: `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu`):

| Order | Command | Time (4-core laptop) | Output |
|---|---|---|---|
| 1 | `python3 02_data_analysis.py` | seconds | statistics, `../figures/*.png` |
| 2 | `python3 03_data_augmentation.py` | ~25 s | `train_augmented.csv` |
| 3 | `python3 M1_baseline_model.py` | ~20 s | `results/m1_baseline_original.*` |
| 4 | `python3 M2_classical_model.py` | ~10 min (large C is slow) | `results/m2*` |
| 5 | `python3 M3_bert_model.py` (`--smoke` = 2-min check) | ~50 min | `results/m3_heldout_scores.npy` |
| 6 | `python3 M4_final_system.py` | seconds | `results/m3.json`, `m4.json`, `summary_table.csv` |
| 7 | `python3 M3_bert_model.py --full` | ~55 min | `results/m3_test_scores.npy` |
| 8 | `python3 07_generate_results.py` | seconds | `../results.txt` |

Every script is seeded (42). `05` downloads Bio_ClinicalBERT pinned to revision `d5892b39a4adaed74b92212a44081509db72f87b`. LaTeX: `~/.local/bin/tectonic -X compile 26.tex` in a scratch copy of `26.tex`, `biblio.bib`, `figures/` (or Overleaf).

---

## 5. Key decisions

1. **Data repair in memory, files untouched.** 7 rows of `train.csv` swallowed the following records: 2,616 parsed rows → 2,669 records (noise kept as given).
2. **One fixed split** (80/20) used by every script; hyperparameters by CV inside the training portion or fixed in advance.
3. **Augmentation:** 15 rule-based EDA-inspired variants (no language model), generated from training records only until every class has 400; first keyword kept first; ratio 36:1 → 2:1.
4. **Two dataset properties** drive the design and are disclosed as a limitation: the first keyword is the specialty (2,219 of 2,219 records with keywords) and the same note appears under several specialties (58%), so text alone cannot determine the label (Observation 1) and the keywords can (Observation 2). The final model relies on the first.

---

## 6. Where we stopped

**Done:** the model scripts were renamed to the report's names (`M1_baseline_model.py` … `M4_final_system.py`); data analysis, loader, split, augmentation; models M1–M4 with a shared evaluation; paper Introduction, Data (2.1, 2.2), Models, Experimental Setup with the hyperparameter table.

**To do**
- [x] `results.txt` for M4 written on 2026-10-11 (`M3_bert_model.py --full`, then `07_generate_results.py`): 398 lines, trained on all 5,648 records, 320 records decided by the first keyword and 78 by M3; identical on rerun (md5 49fb157b…); 363 of 398 lines equal the old SVM file.
- [ ] **Check with the teacher (user's note):** whether the replicability details must appear in the 3 pages. They were removed from the paper to save space: seed 42, library versions, pinned Bio_ClinicalBERT revision, that `results.txt` is trained on all 5,648 records, the extra hyperparameters (TF-IDF `min_df` 2, 150,000 features, sublinear tf; BERT warm-up 10%, weight decay 0.01, AdamW). All of it is in the code headers and `requirements.txt`.
- [x] **M3 verification rerun** (2026-10-11): the edited `M3_bert_model.py` gives the same accuracy per epoch as the first run (53.4 → 50.2 → 48.9) and the same predicted label for all 534 held-out records; the raw scores differ by at most 0.003 (floating-point order on the CPU), so every number in §3 stands. `M4_final_system.py` was rerun: M4 92.13 / 89.58 / 70.00, unchanged.
- [x] `M4_error_analysis.py` written and run: confusion matrix (`../figures/fig3_confusion_matrix.png`), per-class table, error breakdown (all 42 errors are among the 94 records without keywords; 22 are notes listed under several specialties), examples.
- [x] Paper: Results, Discussion (answer to the question, errors, 3 examples in Table 5, limitations), Future Work and the LLM declaration written on 2026-10-11; libraries and the Kaggle source are cited. Per-label table and confusion matrix are in Appendix A (Figure 1 chart removed, Table 1 carries the numbers).
- [ ] **Space (user said to leave it for later):** with everything written the body is about 3.55 pages (Discussion and Future Work spill onto page 4) and the appendix needs 2 pages. Cut about 0.55 page of body text/tables and fit the appendix on one page. Ideas: merge Tables 1 and 2, shorten the Introduction roadmap and Models paragraphs, shrink the confusion matrix.
- [ ] Check the wording of the LLM declaration in the paper (Bibliography section) and the Kaggle reference (only `https://www.kaggle.com`, the task statement gives no dataset URL).
- [ ] Final packaging: compile `26.pdf`, clean run of all scripts, build `26.zip` (`26.pdf`, `results.txt`, `code/`; no weights), submit before the deadline.

---

## 7. Gotchas

- Both CSVs are `;`-separated with `"` quoting; the clean test file has no header (use `data_utils.load_test()`).
- Scripts must be run from `26/code/` (relative paths).
- PyTorch results depend on the thread count (fixed to 4 in `05`) and library versions.
- Held-out accuracy is noisy (±4 points); do not over-interpret differences of 1–3 points.
- Consider a `.gitignore` with `__pycache__/` and `*.pyc`.
- `train_clean.csv` (the repaired `train.csv` for inspection) was deleted; `python3 data_utils.py` regenerates it.
