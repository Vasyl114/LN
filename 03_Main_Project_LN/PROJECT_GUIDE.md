# Project Guide — Group 26, NLP "Help the Doctor"

> The only document about the project. It replaces all earlier guides, plans and status files. Last update: **2026-10-12**. Deadline: **2026-10-16, 23:59**, Fenix.
> Read this first when resuming. Anything not here is in the code headers (`26/code/*.py`) or in the paper (`26/26.tex`).

---

## 1. The task and the rules

**Task:** predict the medical specialty (12 classes) of a clinical record. Train on `StudentsPack/train.csv`, predict `StudentsPack/test_no_labels.csv` (398 records, no header, no labels).

**Deliverable:** `26.zip` (not rar) containing exactly `26.pdf` (max 3 pages), `results.txt` and `code/` (code only, no model weights; extra datasets the code needs are allowed). Members: André Maçorano (110203), David Fonseca (109494), Miguel Ferreira (110268), Vasyl Podvirnyy (109545), 25% each (this goes in the author line).

**Grading (20 points):**
- **Automatic, 4 points:** accuracy above 49% gives 2; additionally no per-class F1 below 25% gives 2. The 49% is a threshold, not a number to reproduce. If the group cannot replicate its own work, the automatic component is 0.
- **Paper, 16 points:** Introduction 1.5, Data Analysis 1.5, Data Augmentation 2.0 (mandatory), Models 4.0 (at least 3 models 1.0, replicability 1.0, creativity 2.0), Experimental Setup 1.0, Results 1.5, Discussion 3.0, Future Work 0.5, References 1.0. The "General quality / Replicability / Creativity" list after `\end{document}` in the template is leftover text, not extra points.
- **3 points are discounted** if any instruction is not followed. Only the first 3 pages are read, even if page 1 is a cover page.

**What each section must contain (template):**
- **Introduction:** an explicit research question that guides the whole paper.
- **Data Analysis:** class distribution plus at least 2 concrete observations backed by numbers; generic statements score zero ("the dataset is unbalanced" needs facts).
- **Models:** at least 3 models, enough detail to replicate (hyperparameters, seeds, preprocessing, versions).
- **Experimental Setup:** datasets, splits (the group's "own test set"), metrics, hyperparameters each explained (4.1).
- **Results:** results of the best models, results by label, and a confusion matrix of at least one model (readable, abbreviated labels).
- **Discussion:** at least 3 concrete misclassified examples from the own test set (text excerpt, predicted label, correct label, interpretation), the most common errors, an answer to the research question, limitations of the data and of the system.
- **Future Work:** tied to the research question.
- **References:** papers, libraries, pre-trained models, the data source, and an explicit declaration of all LLM use.

**Template format:** `\documentclass[twocolumn,10pt]{article}` with `authblk`, `bera`, `inputenc`; apalike bibliography; fixed section skeleton (1 Introduction; 2 Data with 2.1 Data Analysis and 2.2 Data Augmentation; 3 Models; 4 Experimental Setup with 4.1 Parameters/Hyperparameters; 5 Results; 6 Discussion; 7 Future Work; Bibliography; Appendix A). No abstract, no cover page. Appendix A: at most one page, extra figures and tables only; the paper must be understandable without it. Bibliography and appendix do not count toward the 3 pages.

**Style rules:** formal English, no contractions, no subjective adjectives; every figure and table labelled and cited in the text; each acronym defined once and used consistently; LaTeX quotes ` ``like this'' `; cite sources; the "Evaluation and tips" block of the template removed.

**Other rules from the statement:**
- The test file is never used for training, only for `results.txt`. Line N of `results.txt` is the prediction for line N of the test file.
- Pre-processing applied to training data must also be applied to test data. Do not remove stop words blindly.
- The labels are not to be changed; questionable labels must be discussed in the paper. "Look at the input and output, not just the numbers."
- Any project may get an oral discussion; every member must be able to explain every part of the code and of the paper.
- Questions: meic-ln@disciplinas.tecnico.ulisboa.pt (subject "Project"). Fenix may publish FAQs; check it before submitting.
- **Teachers' answers (2026-10-09/10):** the approach is unrestricted (rule-based, LLM, fine-tuning all allowed; what matters is a coherent narrative); training on the entire training data is allowed; the clean test file (398 records) is only for `results.txt`. The teacher also said that replicability means listing what each model contains, so the work can be redone exactly (done in Table 2 of the paper).

---

## 2. The data (verified by parsing)

- Both CSVs: `;` separator, `"` quote, CRLF, UTF-8. Fields: `medical_specialty` (label), `description` (short), `sample_name`, `transcription` (long note), `keywords`. The test file has the last four and no header. Source: adapted from a Kaggle dataset (the statement links only https://www.kaggle.com).
- **`train.csv` is intentionally noisy and must never be edited.** 7 rows have a `description` cell that swallowed the following records (3 are cut at 32,759 characters and spill into the next row). `data_utils.load_train()` repairs this in memory: 2,616 parsed rows become **2,669 records**.
- **Classes (2,669):** Surgery 893, Cardiovascular-Pulmonary 309, Orthopedic 300, Radiology 222, General Medicine 212, Gastroenterology 195, Neurology 184, Obstetrics-Gynecology 136, Neurosurgery 79, Ophthalmology 73, Psychiatry-Psychology 41, Dermatology 25 (ratio about 36:1; always answering Surgery gives 33.5%).
- **Lengths:** descriptions average 19 words, transcriptions 457. Of the descriptions, only about 0.1% exceed 128 BERT tokens (median 33).
- **Observation 1 (copies):** 731 of 1,838 distinct transcriptions are listed under more than one label (1,530 records, 57%), always with the same description and sample name. Text alone cannot determine the label there; the best single label per note gives a text-only ceiling of about 70%.
- **Observation 2 (keywords):** in all 2,219 records with keywords the first keyword is the specialty itself; 450 records have none.
- **Test file:** 398 records, 320 with a first-keyword specialty, 78 without. The old 409-record file (corrupted) was replaced by the teachers on 2026-10-09; ignore anything about 409 or 408 lines.

---

## 3. The story and the design (decisions taken, with reasons)

**Research question (grey box in the Introduction):** *Does a fine-tuned clinical language model that reads only a little text of each record outperform a classical classifier that reads all of it?*

**Line-up (one version each, one held-out number each):**
- **M1 baseline:** description only, lowercase + Porter stemming, TF-IDF, Multinomial Naive Bayes, defaults, original data.
- **M2 classical:** TF-IDF + class-balanced linear SVM on description, sample name, transcription and keywords; no extra feature, no post-processing.
- **M3 language model:** Bio_ClinicalBERT + linear layer, fine-tuned on the description only. A stepping stone; reported plainly.
- **M4 final (writes `results.txt`):** M3 plus the first keyword: the specialty named by the first keyword is added to M3's standardised scores with weight 5, which no score gap can outvote, so the keyword decides when it exists and M3 decides otherwise. No training of its own.
- **Comparison row:** the same keyword rule applied to M2, so the paper can say how much of the gap is the keyword.

**Held-out results** (534 records, 20% stratified split, seed 42; accuracy / macro-F1 / lowest class F1, %):

| Model | Acc. | Macro-F1 | Min. F1 |
|---|---|---|---|
| M1 | 38.76 | 13.66 | 0.0 (Dermatology) |
| M2 | 86.33 | 81.55 | 50.0 (Dermatology) |
| M3 | 48.88 | 51.01 | 20.78 (Radiology) |
| **M4** | **92.13** | **89.58** | **70.0 (Psychiatry-Psychology)** |
| M2 + first keyword | 89.14 | 87.23 | 75.0 (General Medicine) |

**Findings to keep in mind:**
- One accuracy on 534 records carries about ±3–4 points; do not over-read differences of 1–3 points.
- M4 is 100% right on the 440 held-out records with a first keyword and 55.3% on the 94 without. All 42 errors are in the second group; 22 of them are notes listed under several specialties, and 17 have a copy in the training portion.
- The answer to the question is "yes, on this dataset, but mainly because one short field carries the label": the keyword rule on M2 gives 89.1%, so the language model adds about 3 points, within the noise. The paper says this plainly, and states the dependence on the first keyword as a limitation (it would not transfer to notes without keywords).
- M2 input ablation (accuracy, original/augmented training): description 32.2/32.4; + sample name 33.9/33.9; + keywords 88.8/89.0; description + sample name + transcription 38.0/36.5; all four fields 87.6/86.3. The transcription does not help once keywords are present, and augmentation gives no clear gain for M2.
- M3 held-out accuracy per epoch: 53.4, 50.2, 48.9. It memorises copies carrying other labels, and the last epoch is reported anyway (the epoch is never chosen on the held-out set).
- Text-only models cannot exceed about 70%, and the best C of M2 sat at the edge of its grid (250), so the M2 to M4 gap is smaller than first expected.

**Rejected on purpose (do not reintroduce without a reason):**
- Layers L0/L1/L2 and copy-aware decoding (relied on the dataset never repeating a (label, note) pair, checkable only against test labels).
- Fusion of M2 and M3 (+1.5 points, within noise).
- Dropping ambiguous notes from training (loses about half the data, no gain).
- An in-network keyword head (two extra BERT runs).
- A frozen-embedding model, and full-note BERT (the median note has about 650 tokens).
- An LLM for augmentation. Augmentation is rule-based and uses no language model.

---

## 4. Exact content of each model (what the paper's Table 2 must say)

- **Global:** split seed 42; Python 3.10.12; pinned versions in `26/code/requirements.txt` (pandas 2.3.3, numpy 1.26.4, scipy 1.15.3, scikit-learn 1.7.2, nltk 3.10.3, matplotlib 3.10.9, transformers 5.19.0, torch 2.14.1 CPU). Held-out portion: 534 records (20%, stratified), never trained on, never an augmentation source. The 80% training portion has 2,135 original records, or 5,114 augmented records.
- **Augmentation:** rule-based, 15 variants inspired by EDA (synonym replacement with a clinical term map and WordNet, word and sentence deletion, adjacent-word swaps, abbreviation expansion, number rewriting, filler removal); keywords after the first are subsampled and shuffled; negations, laterality, numbers and dosages are never altered. Every class below 400 records is enlarged, generated from training records only: +2,979 records, 5,648 in total, ratio 36:1 to 2:1. Deterministic (the file regenerates byte-identical).
- **M1:** description; NLTK `word_tokenize`, lowercase, Porter stemmer; `TfidfVectorizer()` and `MultinomialNB()` defaults; original training portion.
- **M2:** fields joined with a space; TF-IDF 1–2 words, `min_df` 2, 150,000 features, sublinear tf; `LinearSVC` with C=250, `class_weight='balanced'`, seed 42. C and the n-gram range were chosen by 5-fold stratified CV on the original training portion (macro-F1) over n-grams {(1,1),(1,2)} and C {0.1, 0.5, 2, 10, 50, 250}. Trained on the augmented training portion.
- **M3:** `emilyalsentzer/Bio_ClinicalBERT` revision `d5892b39a4adaed74b92212a44081509db72f87b` plus the standard linear classification head; input description cut at 128 tokens; AdamW lr 3e-5, weight decay 0.01, 10% warm-up then linear decay, gradient clipping 1.0, batch 16, 3 epochs, plain cross-entropy (the augmented data is already balanced), length-bucketed batches (50 batches per bucket), 4 CPU threads, seed 42; the model after the last epoch is used. All hyperparameters fixed in advance from the usual BERT range; nothing tuned on the held-out set.
- **M4:** M3 log-probabilities standardised per record (mean 0, standard deviation 1) plus 5.0 on the class named by the first keyword (`KEYWORD_WEIGHT`, fixed in advance). For `results.txt` M3 is trained on all 5,648 records (`M3_bert_model.py --test`, then `M4_final_system.py --test`).
- **Reproducibility check (2026-10-12):** M1, M2, M4, error analysis, augmentation and `results.txt` reproduce exactly; M3 reproduces the same predicted labels (raw scores differ by at most 0.003 from CPU float order). `M3_bert_model.py --smoke` runs. After the code reorganisation of 2026-10-14 (each model script holds its own model; shared code in `model_utils.py`) every result file, `results.txt` and the confusion matrix were checked byte-identical, and the smoke runs give identical scores.

---

## 5. Repository (what exists and why)

```
03_Main_Project_LN/                (git root is one level up: /home/duckycfx/Work/LN)
├── PROJECT_GUIDE.md               this file
├── .gitignore                     __pycache__/ and *.pyc
├── StudentsPack/                  GIVEN by the teachers; never modified (statement PDF, template, both CSVs)
└── 26/                            our work; the zip is built from here
    ├── 26.tex, biblio.bib         the paper source
    ├── figures/                   fig1_text_lengths.png, fig2_confusion_matrix.png (both in Appendix A)
    ├── results.txt                398 predicted labels, made by `code/M4_final_system.py --test`
    └── code/
        ├── README.md              run order, times, where the data is read from
        ├── data/                  NOT in the repository: created only inside the zip (copy of the two given CSVs)
        ├── requirements.txt       pinned versions
        ├── data_utils.py          loading, in-memory repair of train.csv, split, first keyword
        ├── model_utils.py         shared, not a model: the one evaluation (summarize), save/load of score tables, writing results.txt, BERT helpers
        ├── data_analysis.py       statistics and the numbers of the two observations; draws fig1
        ├── data_augmentation.py   writes train_augmented.csv
        ├── M1_baseline_model.py, M2_classical_model.py, M3_bert_model.py
        ├── M4_final_system.py     M4 = M3 + keyword rule, no training: default = held-out evaluation, --test = writes results.txt
        ├── summary_table.py       reads the results/*.json and writes summary_table.csv
        ├── M4_error_analysis.py   confusion matrix (fig2), per-class report, error breakdown, examples
        ├── train_augmented.csv    5,648 records (19 MB), the exact training data; regenerable
        └── results/               small outputs the later scripts read: m1/m2/m3/m4 json, m2_config.json,
                                   m2_heldout_scores.npy, m3_heldout_scores.npy, m3_test_scores.npy,
                                   m2_input_ablation.csv, m2_first_keyword.json, m4_heldout_predictions.csv, summary_table.csv
```

The zip holds only `26.pdf`, `results.txt` and `code/` (without `__pycache__`). The teacher sees nothing else: no git, no `StudentsPack/`, no guide. The code therefore reads `train.csv` and `test_no_labels.csv` from `code/data/` first and from `../../StudentsPack/` second (`data_utils.find_data_file`), and creates `../figures/` when missing. Packaging: copy `26/` to a scratch folder, compile `26.pdf` there, add `code/data/` with the two given CSVs, delete `__pycache__`, then zip `26.pdf`, `results.txt`, `code/` from that folder as `26.zip`. Check by unzipping into an empty folder and running `M4_final_system.py` and `M4_final_system.py --test` (both reproduce exactly; verified on 2026-10-13). `26.tex`, `biblio.bib` and `figures/` stay in the repository. `results/` is kept in `code/` so that M4 and the paper's numbers can be reproduced without the 50-minute BERT run.

Compile the paper (the `tectonic` engine is at `~/.local/bin/tectonic`; `pdflatex` is not installed): copy `26.tex`, `biblio.bib` and `figures/` to a scratch folder and run `~/.local/bin/tectonic -X compile 26.tex`, or use Overleaf.

---

## 6. The paper (`26/26.tex`) — state and writing agreements

**State (2026-10-13):** all sections written; the group edited the Introduction and Observations 1 and 2 on 2026-10-12 (kept as is). Compiled with tectonic: body ends on page 3 with about 4 lines to spare, bibliography on page 4, Appendix A alone on page 5 (`\clearpage` before it). Floats: Table 1 class distribution (2 blocks), Table 2 configuration of each model, Table 3 results; appendix: Figure 1 text lengths, Table 4 results by label of M4, Figure 2 confusion matrix. Citations: Kaggle (no year, prints "nd"), He and Garcia, scikit-learn, Transformers, Bio_ClinicalBERT, NLTK, EDA, WordNet, PyTorch. The LLM declaration (Claude through Claude Code as coding and writing assistant; the group defined the question, chose the models and checked results) sits under "Bibliography".

**Agreements:**
- **Self-contained paper:** the reader has only `26.pdf`, `results.txt` and `code/`. Do not cite git hashes, repository paths, Hugging Face revision hashes or files outside the zip; point to files inside the zip with their `code/` path (for example `code/requirements.txt`). Explain every number the reader cannot derive from the given CSVs (for example the 2,669 records versus the 2,616 rows pandas parses).
- No mixing of content between sections: split details only in Experimental Setup; no examples inside the augmentation text; interpretation in Discussion, facts in Results.
- Replicability lives in Table 2 only; the model paragraphs give the reasons, not the numbers.
- Honest reporting: the keyword dependence is a stated limitation; the comparison row shows the keyword explains most of the gap.
- Space is a hard rule. Any addition must be paid for with a cut; recompile and check page 3 after every change.
- Discussion examples (records 34, 547, 768 of the held-out analysis) are quoted verbatim in the text, with correct label, prediction and interpretation.

---

## 7. How to work with the user

- Short, organised explanations. When asked for "the models in bullet presentation": Reads / Technique / Training / Importance / Imbalance handling / Held-out result.
- Honest opinions, including disagreement; the user dislikes long tables and long explanations.
- Files in `StudentsPack/` are never modified. Do not commit or push; the user pushes.
- Every modelling choice must be explainable to the teacher (oral discussion).
- The runs use 3 epochs for M3 (the user once wrote "4 epochs").

---

## 8. What is left (as of 2026-10-12)

1. Compile once in Overleaf and confirm the body still ends on page 3 (tectonic and pdflatex may break lines differently; slack is only about 2 lines, so any addition needs a cut).
2. The group checks the wording of the LLM declaration and the Kaggle reference.
3. Final proofreading: typos, contractions, every figure and table cited.
4. Build `26.pdf`, then `26.zip` as described in section 5; check the zip opens, `results.txt` has 398 lines and the code runs from the unzipped folder.
5. Check Fenix for FAQs, then submit before the deadline.
6. Prepare for the oral discussion: each member can explain Sections 2–4 above.
