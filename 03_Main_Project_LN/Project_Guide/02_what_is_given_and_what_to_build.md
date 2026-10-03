# Group 26 — What Is Given, What to Build, What to Deliver

---

## GROUP 1 — What Has Been Given to You (by the professors)

These files are already in your [`StudentsPack/`](file:///home/duckycfx/Work/LN/03_Main_Project_LN/StudentsPack/) folder. You do not create these.

| File | What it is | How you use it |
|---|---|---|
| [`train.csv`](file:///home/duckycfx/Work/LN/03_Main_Project_LN/StudentsPack/train.csv) | 2,613 medical transcription records with labels | Training and validation of all your models |
| [`test_no_labels.csv`](file:///home/duckycfx/Work/LN/03_Main_Project_LN/StudentsPack/test_no_labels.csv) | 409 medical transcription records WITHOUT labels and without a header row (see `01`, Section 9.3) | Input for your best model to generate predictions |
| [`Project-2026-Description.pdf`](file:///home/duckycfx/Work/LN/03_Main_Project_LN/StudentsPack/Project-2026-Description.pdf) | Full project specification | Read it — it defines rules, grading, submission |
| [`template.tex`](file:///home/duckycfx/Work/LN/03_Main_Project_LN/StudentsPack/Project-Template-extracted/Project-Template/template.tex) | LaTeX template for the paper | Fill it in — it defines the paper structure |
| [`biblio.bib`](file:///home/duckycfx/Work/LN/03_Main_Project_LN/StudentsPack/Project-Template-extracted/Project-Template/biblio.bib) | BibTeX bibliography file | Add your references here |

### What the training data looks like

Each row in `train.csv` is semicolon-separated with 5 columns:

```
medical_specialty ; description ; sample_name ; transcription ; keywords
```

- **medical_specialty** → the label (12 possible classes: Surgery, Radiology, Neurology, etc.)
- **description** → short 1-2 sentence summary of the case (clean, short)
- **sample_name** → name of the procedure (e.g., "Chest Pain - Cardiac Consult")
- **transcription** → the full clinical note (very long free text — the richest signal)
- **keywords** → comma-separated medical keywords (clean, discriminative)

### The 12 labels (classes)

| # | Label | Training samples | % of data |
|---|---|---|---|
| 1 | Surgery | 876 | 33.5% |
| 2 | Cardiovascular-Pulmonary | 299 | 11.4% |
| 3 | Orthopedic | 292 | 11.2% |
| 4 | Radiology | 217 | 8.3% |
| 5 | General Medicine | 207 | 7.9% |
| 6 | Gastroenterology | 192 | 7.3% |
| 7 | Neurology | 181 | 6.9% |
| 8 | Obstetrics-Gynecology | 135 | 5.2% |
| 9 | Neurosurgery | 78 | 3.0% |
| 10 | Ophthalmology | 71 | 2.7% |
| 11 | Psychiatry-Psychology | 40 | 1.5% |
| 12 | Dermatology | 25 | 1.0% |

> [!WARNING]
> The data is severely imbalanced. Surgery has 35x more samples than Dermatology. This must be handled in your models and discussed in your paper.

---

## GROUP 2 — What You Need to Create (Code & Tools)

These are the scripts/notebooks your group builds from scratch. Organized in logical order:

### Step 1: Data Analysis Script
**Purpose:** Understand the dataset deeply before modelling. Required for the paper's Data section.

What it must produce:
- Class distribution chart (bar chart with counts and %)
- Text length statistics per class (average word count of description, transcription, keywords)
- Top-N most frequent words per class (TF-IDF or simple frequency)
- At least 2 concrete, non-obvious observations to write about in the paper
- Identification of any duplicate or noisy entries

**Output:** Figures/tables to paste into the LaTeX paper.

---

### Step 2: Data Augmentation Script
**Purpose:** Artificially increase minority class samples. Mandatory for 2.0 pts.

What it must produce:
- An augmented version of `train.csv` with more samples for underrepresented classes (especially Dermatology=25 and Psychiatry=40)
- A clear, documented strategy that can be described in the paper

Suggested approach (easy to implement, good to describe):
- **For the weakest classes:** Use an LLM (e.g., GPT-4o or Claude) to generate 50-100 synthetic clinical notes in the same style as the real data, then add them to training
- **Alternatively:** Random oversampling (duplicate minority rows with small text perturbations like synonym swap)
- **Even simpler but still valid:** Class-weighted training (no new samples, but the model penalises errors on minority classes more heavily) — describe this as your augmentation/balancing strategy

---

### Step 3: Model Scripts (at least 3 models)

#### Model 1 — Baseline (Multinomial Naive Bayes + TF-IDF)
- Input: `description` field only
- Preprocessing: lowercase + stemming (use NLTK PorterStemmer)
- This is the professors' own baseline. Replicating it validates your setup.
- Expected accuracy: ~49% (you need to beat this)

#### Model 2 — Improved Classical ML (SVM or Logistic Regression + TF-IDF)
- Input: combine description + keywords + sample_name into one text field
- Preprocessing: lowercase, remove stopwords, TF-IDF (the PDF warns: "Attention to blindly removing stop words" — stop-word removal must be a justified choice, see `01`, Section 9.1)
- Tune regularization parameter C with cross-validation
- Expected accuracy: ~70-80%

#### Model 3 — Pre-trained Language Model (fine-tuned BERT variant)
- Input: description field (short enough for BERT's 512 token limit)
- Model: `emilyalsentzer/Bio_ClinicalBERT` or `dmis-lab/biobert-base-cased-v1.2` (both free on HuggingFace)
- Fine-tune for classification with a linear head on top
- Expected accuracy: ~80-90%+

> [!TIP]
> If computing power is limited, fine-tune only on the `description` field (short text = fast). The `transcription` field is much richer but requires truncation and more GPU memory.

---

### Step 4: Evaluation & Results Script
**Purpose:** Compare all models fairly on a held-out validation set.

What it must produce:
- Accuracy and macro-F1 for each model
- Per-class F1 for the best model
- A confusion matrix (for at least one model — can be any model, not necessarily the best)
- A list of misclassified examples from your validation set (you need >= 3 specific ones for the Discussion)

---

### Step 5: Prediction Script (generate results.txt)
**Purpose:** Apply your best model to `test_no_labels.csv` and produce the submission file.

What it must produce:
- A file named exactly `results.txt`
- 409 lines, one predicted label per line
- No header line
- Line N in `results.txt` corresponds to line N in `test_no_labels.csv`

Example `results.txt`:
```
Surgery
Neurology
Radiology
Cardiovascular-Pulmonary
...
```

---

### Step 6: The LaTeX Paper
**Purpose:** The scientific report. Worth 16 out of 20 points.

Based on [`template.tex`](file:///home/duckycfx/Work/LN/03_Main_Project_LN/StudentsPack/Project-Template-extracted/Project-Template/template.tex), fill in these sections:

| Section | What to write |
|---|---|
| Title | Something descriptive (e.g., "Medical Specialty Classification from Clinical Text: A Comparative Study") |
| Authors | "Group 26, Name1 (X%), Name2 (X%), Name3 (X%), Name4 (X%)" |
| Introduction | Background + your research question |
| Data > Data Analysis | Dataset stats + 2+ concrete observations |
| Data > Data Augmentation | Your augmentation strategy |
| Models | Describe all 3+ models with enough detail to reproduce |
| Experimental Setup | Split strategy, metrics used, hyperparameters |
| Results | Table of results + confusion matrix + per-label F1 |
| Discussion | 3+ misclassified examples analysed + answer to research question |
| Future Work | What you would do with more time |
| Bibliography | All papers, tools, models used + LLM declaration |
| Appendix A (optional) | Extra figures/tables (max 1 page, does not count toward the 3-page limit) |

---

## GROUP 3 — What Must Be Delivered (Final Outputs)

Everything inside a ZIP file named **`26.zip`**, submitted via Fenix by **October 16, 2026 at 23:59**.

```
26.zip
├── 26.pdf              ← The short paper (max 3 pages, no cover page)
├── results.txt         ← 409 lines of predictions for test_no_labels.csv
└── code/               ← All your scripts (Python files or notebooks)
    ├── 01_data_analysis.py (or .ipynb)
    ├── 02_data_augmentation.py
    ├── 03_model_baseline.py
    ├── 04_model_svm.py
    ├── 05_model_bert.py
    ├── 06_evaluation.py
    └── 07_generate_results.py
```

> [!IMPORTANT]
> Do NOT include trained model weights (checkpoint files are too large). Only include the code. The professors must be able to run it and get the same results.

> [!CAUTION]
> The ZIP must be named `26.zip`, not `group26.zip` or anything else. The paper must be named `26.pdf`. The results file must be named `results.txt`. These are strict.

---

## Suggested Research Questions for Group 26

Choose one of these — or combine elements from multiple:

---

### Option A (Recommended - Well-rounded, covers most grading criteria)
> **"How does the choice of input representation — from TF-IDF bag-of-words to domain-specific pre-trained language models — affect medical specialty classification performance, and what is the role of data augmentation in addressing severe class imbalance?"**

**Why it's good:** It lets you compare classical ML vs. BERT (Models section), motivates augmentation experiments (mandatory section), and gives you clear material for Discussion (which model works and why, which classes are hardest).

---

### Option B (Field-focused — very original, easier to do)
> **"Which input fields of a clinical record — the short description, the full transcription, or the extracted keywords — are most informative for medical specialty classification, and can combining them outperform any single field alone?"**

**Why it's good:** You build the same model (e.g., SVM or BERT) but vary the input. Very clean experimental design, easy to interpret results, highly replicable. Gives you a concrete story: "keywords alone get X%, description gets Y%, combining all three gets Z%."

---

### Option C (Imbalance-focused — directly addresses the hardest challenge)
> **"What is the impact of different data augmentation strategies on the classification of rare medical specialties in a highly imbalanced clinical text dataset?"**

**Why it's good:** Directly targets the most visible problem in the data (1% Dermatology vs 33% Surgery). Lets you compare augmentation methods head-to-head. Very strong Discussion section material since you can show before/after per-class F1.

---

> [!NOTE]
> **My recommendation: Option A or B.** Option A gives you the most natural story and covers all grading dimensions cleanly. Option B is the most original and easiest to execute with limited compute — you don't even need a GPU since the same model is reused with different inputs.
