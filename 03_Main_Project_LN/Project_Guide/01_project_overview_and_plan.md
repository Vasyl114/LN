# NLP Project - "Help the Doctor": Overview & Plan

> **Sources:** [Project-2026-Description.pdf](file:///home/duckycfx/Work/LN/03_Main_Project_LN/StudentsPack/Project-2026-Description.pdf) - [template.tex](file:///home/duckycfx/Work/LN/03_Main_Project_LN/StudentsPack/Project-Template-extracted/Project-Template/template.tex) - [train.csv](file:///home/duckycfx/Work/LN/03_Main_Project_LN/StudentsPack/train.csv) - [test_no_labels.csv](file:///home/duckycfx/Work/LN/03_Main_Project_LN/StudentsPack/test_no_labels.csv)

---

## 1. What the Project Is About

This is a **medical text classification** task. Given a clinical text entry (from a medical transcription), your group must predict which **medical specialty** it belongs to.

It simulates a **shared task** competition: you get a training set, build your models, and apply your best model to a held-out test set where a ranking will be produced. But equally important — **you write a short scientific paper** describing your work (worth 16 out of 20 points).

### The Dataset (`train.csv` and `test_no_labels.csv`)

Each entry in the dataset has 5 fields (semicolon-separated CSV):

| Field | Description |
|---|---|
| `medical_specialty` | **The label** - one of 12 medical specialties |
| `description` | A short description of the clinical case or procedure |
| `sample_name` | Name of the procedure or consultation type |
| `transcription` | The full clinical note (free text, very long) |
| `keywords` | List of associated keywords |

**Training set:** 2,613 samples across **12 classes**

| Specialty | Count | % |
|---|---|---|
| Surgery | 876 | 33.5% |
| Cardiovascular-Pulmonary | 299 | 11.4% |
| Orthopedic | 292 | 11.2% |
| Radiology | 217 | 8.3% |
| General Medicine | 207 | 7.9% |
| Gastroenterology | 192 | 7.3% |
| Neurology | 181 | 6.9% |
| Obstetrics-Gynecology | 135 | 5.2% |
| Neurosurgery | 78 | 3.0% |
| Ophthalmology | 71 | 2.7% |
| Psychiatry-Psychology | 40 | 1.5% |
| Dermatology | 25 | 1.0% |

> [!WARNING]
> The dataset is **highly imbalanced**: Surgery alone is 33.5% while Dermatology is only 1%. This is a key challenge you must address and discuss.

**Test set:** 409 records with **no labels** and **no header row** (pandas' default header handling silently consumes the first record and reports 408 — see Section 9.3). Your best model must generate predictions for all 409 records into `results.txt` (one label per line, no header).

---

## 2. What You Must Deliver (`Project-2026-Description.pdf`)

### 2.1 Automatic Evaluation (4 points)

Run your best model on `test_no_labels.csv` and produce `results.txt`.

| Score | Condition |
|---|---|
| 2.0 pts | Beat the weak baseline: Multinomial Naive Bayes + TF-IDF on `description` field with lowercasing + stemming → **accuracy > 49%** |
| +2.0 pts | Additionally, **no per-class F1 score below 25%** (model must predict all categories, not collapse to the dominant Surgery class) |

> [!IMPORTANT]
> The baseline uses only the `description` field. Using more fields and better models should be enough to beat it.

### 2.2 Short Paper (16 points)

- **Max 3 pages** (bibliography and appendix do NOT count toward the limit)
- **Language:** Portuguese or English
- **Format:** Follow the provided LaTeX template exactly
- **Filename:** `NUM.pdf` (where NUM is your group number)

> [!CAUTION]
> **3 points are discounted** if any instruction is not followed. If the paper exceeds 3 pages, only the first 3 are read. If page 1 is a cover page with names, those count as evaluated pages - **do not use a cover page**.

---

## 3. Paper Structure & Grading Rubric (`template.tex`)

### Section Scores

| Section | Points | Key Requirements |
|---|---|---|
| **Introduction** | 1.5 | Must include a **research question** (e.g., "How does classical ML compare to current language models on clinical text?"). This question guides all other sections and defines the story of the paper. |
| **Data Analysis** | 1.5 | Class distribution + **at least 2 concrete observations** about patterns or anomalies. Generic descriptions score **zero**. |
| **Data Augmentation** | 2.0 | **Mandatory**. Clearly describe your augmentation strategy. |
| **Models** | 4.0 | >= 3 models described (1.0), replicability (1.0), creativity (2.0) |
| **Experimental Setup** | 1.0 | Datasets used, splits, evaluation metrics, hyperparameters |
| **Results** | 1.5 | Results of best models + **per-label results** + **confusion matrix** of at least one model |
| **Discussion** | 3.0 | >= 3 concrete misclassified examples (text excerpt + predicted + correct + interpretation). Results must align with research question. Generic = **zero**. |
| **Future Work** | 0.5 | What you would do with more time, tied to research question |
| **References** | 1.0 | All papers, models, libraries, + **LLM usage declared explicitly** |
**Paper total: 16 points** — the nine rows above sum to exactly 16. Together with the 4 points of the automatic evaluation (Section 2.1), the project total is **20 points**.

> [!NOTE]
> `template.tex` also contains a leftover list titled "How the project will be evaluated" placed *after* `\end{document}` (lines 84-89), so it is not rendered in `template.pdf`: General quality of the paper (1.5: sound methods, clearness, zero typos, illustrative examples, pictures and figures), Replicability (1.5), Creativity (2.0), References (1.0). These are **not extra points** on top of the table (replicability 1.0 and creativity 2.0 are already inside the 4.0 of Models). Read them only as a hint of what the graders value.

### Critical Formatting Tips (from `template.tex`)

- **No cover page** - it eats into the 3-page limit
- Indicate **contribution % per group member** in the author line (mandatory)
- Use APA citation style (`\bibliographystyle{apalike}`)
- Use formal English - no contractions ("it's" -> "it is", "they're" -> "they are")
- No subjective adjectives ("amazing", "nice", "great" are not scientific)
- Label ALL figures and tables and always reference them in the text body
- Use correct LaTeX quotation marks: ` ``like this'' ` not `"like this"`
- Confusion matrix labels must be readable (abbreviate specialty names if needed)
- Cite sources or add footnote URLs for computational resources
- Declare ALL LLM usage explicitly in References
- Remove the "Evaluation and tips" section before submitting

---

## 4. Submission Requirements (`Project-2026-Description.pdf`)

**Deadline:** October 16th, 2026 - 23:59 via Fenix

Submit a **ZIP file** (NOT RAR) named `NUM.zip` containing:
1. `NUM.pdf` - the short paper
2. Project code (not trained model weights, just the scripts) + any extra datasets gathered
3. `results.txt` - 409 lines, one predicted label per line, no header, same order as `test_no_labels.csv`

> [!IMPORTANT]
> You must be able to re-run your code and reproduce results. If you cannot replicate your own results, you get **0 on the automatic component**.

---

## 5. Project Plan - Phase by Phase

### Phase 0: Define the Research Question (Everyone - Day 1)

**This is the most important step.** Define your research question together before anything else. It drives everything else in the paper.

Good examples from the project description:
- *"How does classical machine learning compare to current language models on clinical text?"*
- *"Which data augmentation strategy is most effective for highly imbalanced medical datasets?"*
- *"Which input fields (description vs. transcription vs. keywords) contribute most to classification performance?"*

> [!NOTE]
> The research question must appear in the Introduction and must be answered in the Discussion. It defines the story of the paper. All model choices, experimental setup, and discussion points must revolve around answering it.

---

### Phase 1: Data Analysis (Person A - 2-3 days)

**Goal:** Understand the data deeply. The rubric explicitly punishes generic descriptions.

Tasks:
- [ ] Properly parse `train.csv` (semicolon-separated, double-quote escaped fields)
- [ ] Compute and visualize class distribution (bar chart)
- [ ] Analyze text length statistics per class for description, transcription, and keywords fields
- [ ] Look at vocabulary per class - which words are most discriminative?
- [ ] Identify at least 2 **concrete** anomalies or patterns, for example:
  - "Surgery transcriptions are on average 3x longer than Dermatology notes"
  - "Psychiatry-Psychology entries rarely contain anatomical terms unlike Surgery"
  - "Radiology entries consistently mention imaging modalities (CT, MRI, X-ray) absent in other classes"
- [ ] Identify potentially noisy or ambiguous cases (e.g., Neurosurgery vs. Surgery overlap)

**Output for paper:** Data section with statistics tables, figures (class distribution bar chart, text length boxplot per class), and 2+ concrete non-obvious observations with supporting numbers.

---

### Phase 2: Data Augmentation (Person B - 2-3 days)

**Goal:** Mandatory (2.0 pts). Must clearly describe the strategy and justify it.

Options to explore:
- **Random oversampling** of minority classes (Dermatology=25, Psychiatry=40 samples)
- **Back-translation:** Translate minority class samples to another language and back to English using an MT system
- **Synonym replacement** using WordNet or a medical thesaurus (UMLS)
- **LLM-based synthetic generation:** Prompt an LLM to generate new clinical notes for underrepresented classes with realistic format
- **Paraphrasing** with T5 or a similar seq2seq model

> [!TIP]
> Tie augmentation to the research question. If your RQ is about imbalance, compare model performance with and without augmentation as a controlled experiment.

---

### Phase 3: Model Building (Persons C + D - 3-5 days)

**Goal:** Build >= 3 models. The models section is worth 4.0 pts.

Suggested model progression:

| Model | Type | Input Fields | Notes |
|---|---|---|---|
| Model 1 | Baseline: MNB + TF-IDF | `description` only | Replicate official baseline to validate your evaluation setup |
| Model 2 | Classical ML: SVM or Logistic Regression + TF-IDF | Combined fields | Use description + keywords + sample_name |
| Model 3 | Pre-trained embeddings: BioBERT/FastText features | description or transcription | Static embeddings, mean pooling |
| Model 4 | Fine-tuned Transformer: BioBERT or ClinicalBERT | description or transcription | End-to-end fine-tuning, expected best performance |

> [!NOTE]
> **Replicability (1.0 pt):** Specify ALL hyperparameters, random seeds, library versions, preprocessing steps. Someone must be able to reproduce your results just by reading the Models + Experimental Setup sections.

> [!TIP]
> **Creativity (2.0 pts):** Go beyond standard fine-tuning. Ideas: ensemble of classical + neural models, multi-field fusion (concatenate description + keywords embeddings), prompt-based zero-shot classification, hierarchical classification (broad category first), class-weighted loss to handle imbalance.

---

### Phase 4: Experimental Setup & Evaluation (Everyone - 1-2 days)

- [ ] Create a stratified train/validation split from `train.csv` (e.g., 80/20)
- [ ] `test_no_labels.csv` is **ONLY** for generating `results.txt` - never train on it
- [ ] Choose and report evaluation metrics: Accuracy, macro-F1, per-class F1
- [ ] Document ALL hyperparameters with brief explanations of each
- [ ] Run best model on `test_no_labels.csv` and generate `results.txt` (409 lines, no header)

---

### Phase 5: Results & Discussion (Everyone - 2 days)

**Results section:**
- Table comparing all models (accuracy, macro-F1, per-class F1)
- Confusion matrix for at least one model (use abbreviated label names)
- Highlight per-label results of your best model

**Discussion section (3.0 pts - highest scoring single section):**
- Find >= 3 **concrete misclassified examples** from your own validation set:
  - Include the text excerpt (a short representative snippet)
  - Show the predicted label and the true label
  - Explain WHY the model likely failed (tie to data patterns, class similarity, etc.)
  - Example: "This Neurosurgery note was classified as Surgery because the transcription describes a general operative procedure without mentioning brain or spine anatomy. This aligns with the high lexical overlap between Surgery and Neurosurgery observed during data analysis."
- Relate all findings back to your research question
- Discuss class imbalance impact and augmentation effectiveness
- Discuss which fields were most informative and why

---

### Phase 6: Paper Writing (Everyone - 2-3 days)

- Use [`template.tex`](file:///home/duckycfx/Work/LN/03_Main_Project_LN/StudentsPack/Project-Template-extracted/Project-Template/template.tex) and [`biblio.bib`](file:///home/duckycfx/Work/LN/03_Main_Project_LN/StudentsPack/Project-Template-extracted/Project-Template/biblio.bib)
- Assign one lead writer per section
- **Remove** the "Evaluation and tips" section (lines 53-79 of template.tex) before compiling the final PDF
- Proofread collectively - zero typos is part of the grade
- Run through the checklist below before submission

---

## 6. Critical Submission Checklist

### Paper Quality
- [ ] Paper is <= 3 pages (bibliography and appendix excluded)
- [ ] No cover page used
- [ ] Author line includes contribution % per member (e.g., "Group 5, Ana Silva (30%), Joao Costa (25%), ...")
- [ ] Research question explicitly stated in Introduction
- [ ] >= 2 concrete data observations with supporting numbers
- [ ] Data augmentation strategy clearly described
- [ ] >= 3 models described with enough detail to replicate
- [ ] Confusion matrix present with readable (abbreviated) labels
- [ ] Per-label F1 of best model reported
- [ ] >= 3 concrete misclassified examples with text excerpt + predicted + correct + explanation
- [ ] All figures and tables are labeled and referenced in the text
- [ ] All hyperparameters mentioned and briefly explained
- [ ] All acronyms introduced and used consistently
- [ ] Formal English only (no contractions)
- [ ] No subjective adjectives
- [ ] All papers, libraries, pre-trained models cited
- [ ] LLM usage explicitly declared in References
- [ ] "Evaluation and tips" section removed before compiling PDF

### Submission ZIP
- [ ] File named `NUM.zip` (not RAR)
- [ ] Contains `NUM.pdf`
- [ ] Contains all runnable code (no model weights)
- [ ] Contains `results.txt` with 409 lines, one label per line, no header
- [ ] Line order in `results.txt` matches line order in `test_no_labels.csv`
- [ ] Submitted via Fenix by **October 16, 2026 at 23:59**

---

## 7. Suggested Work Division (Group of 4)

| Person | Primary | Secondary |
|---|---|---|
| Person 1 | Data Analysis + "Data" paper section | Help with Discussion examples |
| Person 2 | Data Augmentation + "Data Augmentation" paper section | Results table |
| Person 3 | Models 1 & 2 (classical ML) + Experimental Setup section | Introduction |
| Person 4 | Models 3 & 4 (transformers) + generate results.txt | Future Work section |

All 4 review paper together before submission.

---

## 8. Key Nuances & Gotchas

1. **Research question is the spine of the paper** - define it Day 1. It gates 3.0 pts (Discussion) + 1.5 pts (Introduction) + 0.5 pts (Future Work) = 5 pts directly.
2. **Never train on test_no_labels.csv** - it is only for generating the final results.txt.
3. **3 points penalty** for not following instructions - read the rules twice.
4. **Appendix is extra figures only** and the paper must be fully understandable without it.
5. **Generic statements score zero** in Data Analysis and Discussion - always back claims with numbers and text examples.
6. **Replicability is graded** - random seeds, library versions, all hyperparameters must be in the paper.
7. **LLM usage must be declared** - even using ChatGPT to fix grammar counts and must be stated.
8. **Data augmentation is mandatory** - missing it costs 2 pts directly.
9. **Know your fields**: `description` is short and clean (great for fast experiments); `transcription` is very long and rich (best signal, but slower); `keywords` is a compact discriminative signal. Think about which combinations to use.
10. **Class imbalance is the central challenge**: Surgery=33.5%, Dermatology=1%. Per-class F1 below 25% costs 2 pts in automatic evaluation. Your augmentation and loss function must account for this.

---

## 9. Source Details Verified Against the PDF, Template and CSVs (added 2026-10-03)

> This section is the result of checking this guide line by line against `Project-2026-Description.pdf`, `template.tex` / `template.pdf` and the two CSV files. It records what Sections 1-8 omit, so that the guide can be used without re-reading the sources. Text in quotation marks is verbatim from the sources.

### 9.1 Rules and tips from the PDF that are not mentioned above

- **Oral discussion (PDF §6):** "All projects may be subject to an oral discussion. If students are unable to demonstrate a clear understanding of the work they have submitted, the final grade may be adjusted to reflect the demonstrated level of knowledge." Every member must be able to explain every part of the code and of the paper.
- **Limitations (PDF §1):** the paper must "highlight limitations associated with the given data and your system" — both kinds of limitation, explicitly.
- **Look at data and outputs (PDF §1, §7):** "You should convince us that you have looked at the data and at the returned outputs (not just at the evaluation scores)." and "Look at the input and output!!! (not just to numbers)".
- **Label disagreements (PDF §7):** "The dataset is adapted from a 'real' dataset from Kaggle. Datasets in NL have errors and are usually imbalanced. You will also probably find many labels that you don't agree with. You are probably right, but the datasets will not be changed. Discuss these situations in your paper." Questionable labels are expected and must be discussed in the paper.
- **Stop words (PDF §7):** "Attention to blindly removing stop words." Stop-word removal has to be a justified choice, not a default.
- **Pre-processing parity (PDF §7):** "Pre-processing applied to the training set should also be applied to the test set."
- **Systematic work (PDF §7):** "Remember what you have learned during the class about data and evaluation: try to do a systematic work."
- **MSc-level problem (PDF §7):** "there is a clearly identified problem that you need to solve in the best possible way, but we do not tell you how to do it."
- **No perfect score (PDF §7):** "There is no 100% accuracy (this is a research problem)."
- **Zip contents (PDF §5):** "the project code (not the models, just the code) and eventual datasets gathered to the project" — any extra or augmented dataset that the code needs belongs in the zip.
- **Group size (PDF §6):** "groups up to 4 students".
- **Questions (PDF §6):** `meic-ln@disciplinas.tecnico.ulisboa.pt`, subject "Project".
- **FAQs (PDF §6):** "We might release FAQs about the project – again, Fenix (Section Projects)." Check Fenix before submitting.

### 9.2 Template details that are not mentioned above

- **Page format:** `\documentclass[twocolumn,10pt]{article}` with packages `authblk`, `bera` and `inputenc` (utf8). Bibliography via `\bibliographystyle{apalike}` and `\bibliography{biblio}`.
- **Exact section skeleton:** 1 Introduction; 2 Data (2.1 Data Analysis, 2.2 Data Augmentation); 3 Models; 4 Experimental Setup (4.1 Parameters/Hyperparameters); 5 Results; 6 Discussion; 7 Future Work; Bibliography; Appendix A: Extra Figures and Tables. The template has **no abstract**.
- **Appendix:** "Maximum one page (it does not count for the 3 pages limit): just extra figures and tables. Notice that we should be able to understand the 3 pages paper without these figures and tables."
- **"Your own test set":** the Experimental Setup rubric asks for "splits (which result in your own test set(s))" and the Discussion examples must come "from your own test set". These guides call that held-out part of `train.csv` the "validation set"; the template's term is "own test set".
- **Discussion** also asks to "Try to explain the most common errors".
- **Results** asks for "the results of your best models" (plural), "results by label of your best model" and "a comprehensive confusion matrix of one of your models (not necessarily the best one)".
- **Tips missing from Section 3:** "If you say things such as 'The dataset is unbalanced', explain why (facts)." and, for acronyms, "use it consistently (check how to do it in latex)".
- **Leftovers in `template.tex` that conflict with the PDF** (the PDF prevails): the comment on line 45 says "Only the first two pages of the paper will be read" (the PDF says three); the Results rubric calls the test file `test_no_labels.txt` (the file is `.csv`); the list after `\end{document}` is covered by the note under the table in Section 3.

### 9.3 Facts about the given CSV files (verified by parsing them)

Both files use `;` as separator, `"` as quote character, CRLF line endings and UTF-8. The five field names match the MTSamples "Medical Transcriptions" dataset on Kaggle (an inference: the PDF only says "a 'real' dataset from Kaggle").

**`train.csv`** — has a header row.

| Count | Meaning |
|---|---|
| 2,673 | physical lines (1 header + 2,672) |
| 2,616 | rows returned by `pd.read_csv(sep=';', quotechar='"')` |
| 2,613 | rows carrying one of the 12 valid labels (the number used throughout these guides; class counts as in the table of Section 1) |
| 7 | of those 2,613 are corrupted: the `description` cell swallowed the records that followed it (0-based pandas rows 159, 710, 931, 1058, 2051, 2280, 2337) |
| 3 | rows whose label is a fragment of text: the cut-off tails of rows 931, 2051 and 2280 (rows 932, 2052, 2281) |
| 2,606 | fully clean rows |

- The 7 corrupted cells hold 56 swallowed lines, every one starting with a valid label; 49 of them are complete 5-field records. Three of the cells are cut at exactly 32,759 characters.
- In the 2,606 clean rows `transcription` is empty in 24 and `keywords` is empty in 396.
- **How the code handles this:** `26/code/data_utils.py` (`load_train()`) leaves the given file untouched and repairs these rows in memory, splitting each corrupted cell back into its records and rejoining the three cut ones. The result is **2,669 records** (the 2,606 clean rows, the 7 corrupted rows reduced to their own description, and 56 recovered records), which is the dataset size used by the scripts and the paper. On those 2,669 records the first keyword equals the label in all 2,219 rows that have keywords, and 731 transcriptions appear under more than one label (1,530 rows).

**`test_no_labels.csv`** — has **no header row** (its first line is already a record) and 4 columns.

| Count | Meaning |
|---|---|
| 452 | physical lines |
| 409 | records returned by `pd.read_csv(sep=';', quotechar='"', header=None)` |
| 408 | what pandas returns with its default `header=0`, because the first record is consumed as column names |

- 4 records are corrupted in the same way as in train (0-based rows 17, 241, 324, 357, i.e. `results.txt` lines 18, 242, 325, 358). Their `description` swallowed 43 lines in total, **each still starting with its label**. The test file must never be used for training (PDF §3), which includes these embedded labelled lines.
- The record after each corrupted one (lines 19, 243, 326, 359) is its cut-off tail: a keyword list sitting in the `description` column with the other fields empty. Line 80 is an empty record (`;;;`).
- Empty fields over the 409 records: `description` 1, `sample_name` 9, `transcription` 11, `keywords` 83.

> [!CAUTION]
> **`results.txt` must be aligned with 409 records, not 408.** Every "408" in earlier versions of these guides came from reading the file with the default header, which drops the first record and shifts every prediction by one line. Because of the 4 corrupted records, "line number" is also ambiguous (409 parsed records versus 452 physical lines). 409 is the count a CSV parser gives; since the 4 automatic points depend on this alignment, it is worth confirming with the professors (contact in Section 9.1).

**Patterns in the 2,606 clean train rows**

- The same clinical note is listed under several specialties: there are 1,815 distinct transcriptions for 2,582 non-empty ones, and 701 of them appear with more than one label (1,467 rows). Most frequent label pairs for an identical transcription: Orthopedic/Surgery 145, Cardiovascular-Pulmonary/Surgery 114, Gastroenterology/Surgery 97, Obstetrics-Gynecology/Surgery 67, Neurology/Radiology 57, Neurosurgery/Surgery 55.
- Share of each class whose transcription also exists under another label: Neurosurgery 91%, Radiology 72%, Orthopedic 70%, Obstetrics-Gynecology 64%, Surgery 59%, Gastroenterology 59%, Cardiovascular-Pulmonary 56%, Ophthalmology 55%, Neurology 52%, Dermatology 48%, Psychiatry-Psychology 18%, General Medicine 7%.
- In 2,171 of the 2,210 rows with non-empty `keywords` (98.2%) the first keyword is the specialty label itself (for example `cardiovascular / pulmonary, ...`).
- 237 of the 398 non-empty test transcriptions also occur verbatim in `train.csv`.
