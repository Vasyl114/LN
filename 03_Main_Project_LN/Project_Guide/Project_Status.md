# Project Status — Group 26 (NLP "Help the Doctor")

> Hand-over document: what the project is, how the files are organised, what has been built, the numbers so far, where work stopped, and what is left. Written on **2026-10-08**. Deadline: **2026-10-16, 23:59** (Fenix, `26.zip`).

---

## 1. The project in five lines

- **Task:** predict the medical specialty (12 classes) of a clinical record from `train.csv` / `test_no_labels.csv`.
- **Deliverables:** `26.zip` containing `26.pdf` (max 3 pages, template-based), `results.txt` (one label per line, no header, same order as the test file) and the code (no model weights).
- **Grading:** 4 points automatic (accuracy above 49% **and** no class F1 below 25% on the hidden test labels) + 16 points for the paper. Section scores are in `Report_Plan.md`.
- **Our research question:** *does a clinical language model fine-tuned only on the short description beat a classical classifier that reads all the text?*
- **Rule reminders:** the test file is only for generating `results.txt`; the given files in `StudentsPack/` are never modified; LLM use must be declared in the paper; 3 points are lost if any instruction is not followed.

---

## 2. File-system architecture

```
03_Main_Project_LN/
├── StudentsPack/                  GIVEN by the teachers. Never edited.
│   ├── Project-2026-Description.pdf   task statement, rules, grading
│   ├── Project-Template(.zip / -extracted/)   LaTeX template (template.tex) + biblio.bib
│   ├── train.csv                  2,616 parsed rows, ';'-separated (7 corrupted rows, see §4)
│   └── test_no_labels.csv         409 records, NO header row, NO labels
│
├── Project_Guide/                 Our documentation
│   ├── 01_project_overview_and_plan.md   digest of the PDF/template/CSVs (Section 9 = verified facts)
│   ├── 02_what_is_given_and_what_to_build.md
│   ├── 03_group26_complete_guide.md       old step-by-step guide (Phase 3 is superseded by the plan below)
│   ├── Report_Plan.md             section-by-section plan of the paper (subparts, word caps, floats)
│   ├── Project_Status.md          THIS FILE
│   └── notes/project_storyline.md
│
├── 26/                            OUR WORK STATION = what becomes 26.zip
│   ├── 26.tex, biblio.bib         the paper (see §7)
│   ├── figures/                   fig1_class_distribution.png, fig2_text_lengths.png (made by script 02)
│   └── code/
│       ├── data_utils.py          SHARED: load_train() repair, split_train_heldout(), remove_heldout(),
│       │                          note_key(), seen_labels(), copy_aware_scores(), first_keyword_label()
│       ├── evaluation.py          SHARED: summarize() = the one evaluation used by every model
│       ├── model_utils.py         SHARED: the classical pipeline (TF-IDF + linear SVM) used by 04 and 07
│       ├── 01_baseline_model.py   Model 1: Naive Bayes + TF-IDF on description (the given baseline)
│       ├── 02_data_analysis.py    statistics + the two paper observations + figures
│       ├── 03_data_augmentation.py  writes train_augmented.csv (run once, seeded)
│       ├── 04_model_classical.py  Model 2: TF-IDF + linear SVM on all the text (CV-tuned)
│       ├── 05_model_bert.py       Model 3: Bio_ClinicalBERT fine-tuned on the description (~50 min CPU)
│       ├── 06_model_bert_embeddings.py  Model 4 (OPTIONAL, NOT RUN): frozen BERT on the full note (~1 h CPU)
│       ├── 07_final_system.py     Model 5: layers L0/L1/L2 + fusion + the summary table
│       ├── train_augmented.csv    5,648 records (2,669 original + 2,979 generated)
│       ├── train_clean.csv        the repaired train.csv, ONLY for inspection (no script reads it)
│       ├── requirements.txt       pinned library versions
│       └── results/               every model's metrics (.json), held-out predictions (.csv),
│                                  saved scores (.npy), m3_training_log.txt, summary_table.csv
└── __MACOSX/                      junk from unzipping on a Mac (ignore)
```

Not in the repo: the LaTeX plan lives in `~/.claude/plans/` (Claude Code's own folder). Its content is summarised in §5 and in `Report_Plan.md`.

---

## 3. How to run everything

From `26/code/` (Python 3.10, CPU only; versions in `requirements.txt`; PyTorch CPU: `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu`):

| Order | Command | Time (4-core laptop) | Output |
|---|---|---|---|
| 1 | `python3 02_data_analysis.py` | seconds | prints statistics, writes `../figures/*.png` |
| 2 | `python3 03_data_augmentation.py` | ~25 s | `train_augmented.csv` |
| 3 | `python3 01_baseline_model.py [train_augmented.csv]` | ~20 s | `results/m1_baseline_*` |
| 4 | `python3 04_model_classical.py` | ~1 min | `results/m2_*`, `m2_heldout_scores.npy` |
| 5 | `python3 05_model_bert.py` (`--smoke` = 2-min check) | ~50 min | `results/m3_*`, `m3_heldout_scores.npy` |
| 6 | `python3 07_final_system.py` | ~2 min | `results/m5_*`, `summary_table.csv` |
| opt. | `python3 06_model_bert_embeddings.py` | ~1 h | `results/m4_*` |

- Every script is seeded (42). Scripts 02, 03, 04 and the smoke test of 05 were re-run and gave identical outputs. The full run of 05 was run once.
- `05` downloads Bio_ClinicalBERT pinned to revision `d5892b39a4adaed74b92212a44081509db72f87b` (needs internet the first time).
- LaTeX: `tectonic` was installed to `~/.local/bin` (static build; it downloads packages on first use). Compile: copy `26.tex`, `biblio.bib`, `figures/` to a scratch folder and run `~/.local/bin/tectonic -X compile 26.tex`. Anyone else can use Overleaf.

---

## 4. Key design decisions (and why)

1. **Data repair in memory, files untouched.** 7 rows of `train.csv` have a `description` cell that swallowed the following records (3 are cut at 32,759 characters). `load_train()` splits them back: 2,616 parsed rows → **2,669 records**. Treated as realistic scraping noise; `test_no_labels.csv` has 4 such records too (not yet handled, see §8).
2. **One fixed split.** 80% train (2,135) / 20% held out (534), stratified, seed 42, defined once in `data_utils.split_train_heldout()` and used by every script. Held-out records are never trained on and never used as an augmentation source. Nothing is tuned on the held-out set; hyperparameters are chosen by 5-fold CV inside the training portion (Model 2) or fixed in advance from the literature (Model 3).
3. **Augmentation** (`03`): 15 variants inspired by EDA (curated synonym map, guarded WordNet, deletions, swaps, abbreviation expansion, etc.), generated only from training records, until every class has 400. Protected: upper-case abbreviations, numbers/dosages, spinal levels, laterality and negation words. The first keyword (the specialty) always stays first. Result: ratio largest:smallest class 36:1 → 2:1.
4. **Models and why** (full reasoning is in the script headers):
   - **M1** Naive Bayes + TF-IDF on description: the given reference.
   - **M2** linear SVM on description + sample name + transcription: the classical side ("all the text"). Keywords deliberately not in its text input.
   - **M3** Bio_ClinicalBERT on description only: the language-model side. Pre-trained on clinical notes (same genre). lr 3e-5, batch 16, 3 epochs, max length 128, length-bucketed batches, seed 42, last epoch reported (not tuned).
   - **M4 (optional)** frozen BERT on the full note: separates model type from amount of text.
   - **M5** final system: layers below + fusion of M2 and M3 (scores rescaled per record and added with equal weight, not tuned).
5. **Layers** (reported separately so the fair comparison is not hidden by the shortcuts):
   - **L0** model alone (text only).
   - **L1** + copy-aware decoding: the dataset has no repeated (label, note) pair, so a held-out note that already exists in train almost always has a *different* label (242 of 243). The labels the training data carries for that note are removed from the candidates. Uses training labels only.
   - **L2** + first-keyword feature: the first keyword is the specialty itself in 100% of records with keywords; added as its own one-hot feature block (weight 3, chosen by CV). The classifier learns to trust it; no hand-written rule, but *we* chose to expose that field.

---

## 5. Results so far (held-out, 534 records, accuracy / macro-F1 / lowest class F1, %)

| Model | L0 text only | L1 + copy-aware | L2 + keywords |
|---|---|---|---|
| Always answering "Surgery" | 33.5 | | |
| M1 Naive Bayes, original / augmented data | 38.8 / 13.7 / 0.0 · 41.8 / 41.7 / 21.1 | | |
| M2 SVM on all text (augmented) | 41.8 / 46.7 / 19.5 | 78.3 / 78.1 / 66.7 | 93.3 / 91.3 / 76.2 |
| M3 BERT on description (augmented) | 48.9 / 51.0 / 20.8 | 75.7 / 73.7 / 57.1 | not defined alone |
| Fusion M2 + M3 | 44.0 / 49.1 / 19.5 | 79.4 / 79.9 / 70.6 | **94.8 / 93.6 / 80.0** (current final system) |

Other measured facts:
- **Input fields (M2, L0 → L1):** description 36.7 → 65.2, + sample name 40.8 → 71.9, + transcription 42.7 → 77.9 (original data). More text helps.
- **Augmentation:** within about ±1 point for the linear model (class-weighted already). Report honestly.
- **M3 per epoch (L0 accuracy):** 53.4 → 50.2 → 48.9. It memorises notes whose copies carry other labels, so it gets worse. The epoch was fixed in advance (3), so 48.9 is the reported number.
- **Final system by kind of record:** with keywords 99.5–100% (440 records); without keywords 88.9% if a copy is in train (27), 64.2% if not (67).
- **Noise:** one accuracy on 534 records carries about ±4 points.

Dataset facts used in the paper: 731 distinct notes (1,530 records, 58% of those with a transcription) appear under more than one specialty; 2,219 records have keywords (all with the label first), 450 (16.9%) have none; descriptions average 18.9 words, transcriptions 457.0.

---

## 6. Where we stopped

**Done**
- [x] Data analysis, figures, the two observations (script 02).
- [x] Data repair loader, fixed split, augmentation (scripts 03, `data_utils.py`).
- [x] Models 1, 2, 3 and the final-system comparison (scripts 01, 04, 05, 07), with a shared evaluation.
- [x] Paper: Introduction, Data Analysis, Data Augmentation written (3 pages compile, citations resolve, no white-space gaps).
- [x] Documentation: `Report_Plan.md`, guide fixes, this file.

**Not done**
- [ ] `Models_Rationale.md` (short "why this over that" per model; the same reasoning is already in the script headers).
- [ ] Model 4 (script written, never run; optional).
- [ ] **`results.txt` generator** (`08_...`): loader for the test file, retrain the chosen system on all 2,669 records + augmentation, predict.
- [ ] Confusion matrix figure (abbreviated labels) and a script that lists misclassified held-out examples (needed for the Discussion).
- [ ] Paper sections: Models, Experimental Setup (+ 4.1 hyperparameter table), Results (model table, per-class table, confusion matrix), Discussion, Future Work, LLM-use declaration, missing references (PyTorch etc.).
- [ ] Final packaging: compile `26.pdf` (max 3 pages), run all scripts from clean, build `26.zip` (no weights).

---

## 7. The paper (`26/26.tex`) — current state

Template headings kept exactly. Written: Introduction (research question stated), 2.1 Data Analysis (Table 1, Observations 1 and 2), 2.2 Data Augmentation. Everything from Models onward is a bare heading on purpose (the plan lives in `Report_Plan.md`). Appendix holds the two figures. Uses `\raggedbottom` to avoid stretched spacing; acronyms defined with short forms.

---

## 8. Open decisions and questions

1. **Teachers' opinion** on the two shortcuts (keywords field, copy-aware decoding). Draft email: ask whether using the first keyword as an input feature is acceptable, state the two properties found, promise results with and without it. Send to `meic-ln@disciplinas.tecnico.ulisboa.pt`, subject "Project". If the answer is no: submit L1 without keywords (about 78–79% held-out).
2. **Which system generates `results.txt`:** M2 SVM only (93.3%, reproducible in minutes) or Fusion (94.8%, needs the 50-minute BERT fine-tune to reproduce). We cannot ship weights.
3. **Without any shortcut** the best model alone is M3 at 48.9% (just under 49%; lowest class F1 20.8%), so the automatic points would be borderline.
4. **Test-file alignment:** the file has 409 parsed records but 452 physical lines (4 corrupted records). The statement says "same line number"; which counting the grader uses is unknown. Also ask the teachers.
5. **Unexplained gap:** our baseline scores 38.8% on held-out while the statement implies about 49% on the test set. Cause unknown (cannot be checked without test labels).
6. **Risk to disclose:** copy-aware decoding assumes test notes behave like held-out notes (240 of 398 non-empty test notes also occur in train).
7. **Model 4:** run (about 1 hour) or skip.

---

## 9. Gotchas

- Both CSVs are `;`-separated with `"` quoting. `test_no_labels.csv` has **no header**: read it with `header=None` (the default drops the first record and gives 408).
- Scripts must be run from `26/code/` (relative paths).
- `train_clean.csv` is only a viewing aid (run `python3 data_utils.py` to regenerate it). Nothing depends on it.
- `__pycache__` and `.pyc` files were committed once by mistake; consider a `.gitignore` (`__pycache__/`, `*.pyc`). The `results/` folder is small enough to commit (the `.npy` score files are needed by script 07).
- PyTorch results depend on the thread count (fixed to 4 in script 05) and library versions; the numbers above come from the versions in `requirements.txt`.
- Held-out accuracy is noisy (±4 points); do not over-interpret differences of 1–3 points.

---

## 10. Suggested next steps (in order)

1. Send the email to the teachers (§8.1, §8.4).
2. Write the `results.txt` generator, the confusion matrix and the error-example script.
3. Write `Models_Rationale.md`, then the Models, Experimental Setup and Results sections.
4. Pick 3 misclassified examples, write Discussion and Future Work, add the LLM-use declaration and missing references.
5. Compile, check 3 pages, clean run of all scripts, build `26.zip`, submit before **October 16, 23:59**.
