# Group 26 — Complete Step-by-Step Project Guide
### Natural Language Processing — Medical Specialty Classification

> **Deadline:** October 16, 2026 at 23:59 via Fenix
> **Submission:** `26.zip` containing `26.pdf`, `results.txt`, and `code/`
> **What you already have:** The baseline model script at `26/code/01_baseline_model.py` ✅

---

## HOW TO READ THIS GUIDE

- 📝 **WRITE** → Stop and write something down (notes for the paper)
- 💻 **CODE** → Something that needs to be programmed
- ▶️ **RUN** → Execute the code and collect the output
- 📊 **RECORD** → Save the output number or figure to use in the paper
- 🤔 **THINK** → A decision or conclusion must be made
- ✅ **DONE** → Checkpoint — the step is complete
- 🔴 **CRITICAL** → Do not skip or change this, it affects the grade directly

---
---

# PHASE 0 — SETUP AND RESEARCH QUESTION
*Time estimate: 1–2 hours | Who: Everyone together (group meeting)*

---

## STEP 0.1 — Confirm the Folder Structure

Your working folder for the entire project is:
```
/home/duckycfx/Work/LN/03_Main_Project_LN/26/
```

This folder is what will become the `26.zip` file at the end. Inside it should have:
```
26/
├── code/
│   └── 01_baseline_model.py   ← already exists ✅
├── 26.tex                     ← the paper (will be created in Phase 5)
├── biblio.bib                 ← the references file (will be created in Phase 5)
├── results.txt                ← the final predictions (will be created in Phase 4)
└── 26.pdf                     ← the compiled paper (final output)
```

> 🔴 **CRITICAL:** The file `results.txt` must be at the root of the `26/` folder, not inside `code/`. The paper must be named `26.pdf`, not `paper.pdf` or anything else.

✅ **STEP 0.1 DONE** when: You have verified the folder exists and understand its structure.

---

## STEP 0.2 — Define Your Research Question

**What is the goal of this step?**

The research question is the central argument of your scientific paper. It is not about figuring out what to code — it is about deciding what *story* you want to tell. Everything else in the paper (the models you choose, the experiments you run, what you discuss, what future work you propose) must directly connect to answering this one question. Without it, your paper is just a technical report. With it, your paper is a scientific argument.

A good research question for this project must be:
- **Specific to this dataset** (mention clinical text, medical specialty, or the imbalance problem)
- **Answerable by your experiments** (you can run code and get numbers that answer it)
- **Narrow enough to discuss in 3 pages**

**🔴 CRITICAL:** The professors give you 1.5 points for the Introduction section, but ALSO reward the research question indirectly in the Discussion (3 pts) and Future Work (0.5 pts). Choosing the right question is worth approximately 5 points. (An earlier version of this guide also counted a "General Quality mark (1.5 pts)"; that item comes from leftover text after `\end{document}` in `template.tex` and is not a separate mark — see `01`, Section 3.)

**Recommended research question for Group 26:**

> *"Which combination of clinical record fields — the short description, the full transcription, and the extracted keywords — is most informative for medical specialty classification, and does combining multiple fields outperform a fine-tuned language model trained on a single field alone?"*

**Why this question is ideal for you:**
- It lets you run multiple versions of the SAME model (or a simple model) with different inputs. This is fast, clean, and creative.
- The answer will be clearly visible in the results table (each field combination = one row in the table).
- It explains why you build multiple models: because you are testing different inputs, not just chasing the highest number.
- It connects directly to the imbalance problem (certain fields will hurt minority classes more than others).
- It is original — most groups will just compare Naive Bayes vs. BERT and stop there.

**🤔 GROUP DECISION:** Agree on the research question now, before any code is written. Write it down in one sentence.

📝 **WRITE:** Copy the final agreed research question into a shared document or notes file. Everyone must be able to quote it from memory.

✅ **STEP 0.2 DONE** when: The group agrees on one final research question written in one clear sentence.

---
---

# PHASE 1 — DATA ANALYSIS
*Time estimate: 2–3 days | Who: Person A*

**Goal:** Understand the dataset deeply. Discover things that are not obvious just from looking at the CSV. The professors give 1.5 points for the Data Analysis section, but **explicitly state that generic descriptions score zero**. You must produce specific numbers and observations.

---

## STEP 1.1 — Load the Data Correctly

💻 **CODE:** Create a new file `26/code/02_data_analysis.py`.

Inside it, load the data the correct way:
```
pd.read_csv('train.csv', sep=';', quotechar='"', engine='python')
```

Then immediately print the shape of the DataFrame (how many rows and columns you have) and the column names, to confirm it loaded correctly.

▶️ **RUN** it. You should see exactly 5 columns and approximately 2,613 rows.

✅ **STEP 1.1 DONE** when: The data loads without errors and shows 5 columns.

---

## STEP 1.2 — Compute Class Distribution

💻 **CODE:** Count how many times each `medical_specialty` label appears. Then compute what percentage each class represents out of the total.

▶️ **RUN** it and get a table showing: label name, count, percentage.

📊 **RECORD:** Save this exact table — you will put it in the paper.

**Expected output you should get:**

| Label | Count | % |
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

📊 **RECORD:** Also generate a **bar chart** of this distribution. Save it as a `.png` image. You will use this figure in the paper. Label the axes. Title it "Class Distribution of Training Set".

📝 **WRITE — Observation 1 (for the paper):**
Look at the numbers above. Notice that Surgery has 876 samples and Dermatology only has 25. That is a ratio of 35:1. Write this observation in your notes in exact language you could use in the paper: *"The dataset is highly imbalanced: the majority class (Surgery, 33.5%) contains 35 times more samples than the minority class (Dermatology, 1.0%). This poses a significant risk of classifiers defaulting to frequent classes, especially for minority classes."*

✅ **STEP 1.2 DONE** when: You have a table, a saved chart, and Observation 1 written down.

---

## STEP 1.3 — Analyse Text Length Per Class

**What this tells you:** Some medical specialties might be much easier to identify simply because their clinical notes are written in a very different length or style. This is useful information to discuss.

💻 **CODE:** For each of the 5 columns (`description`, `sample_name`, `transcription`, `keywords`), compute the average word count per class. A word count is simply `len(text.split())`.

▶️ **RUN** it. Look especially at:
- How much longer is `transcription` compared to `description`?
- Are some classes consistently shorter or longer than others?

📊 **RECORD:** Save the table showing average word count per class for `description` and `transcription`. These two are the most relevant.

📝 **WRITE — Observation 2 (for the paper):**
Look at what you find. A likely observation: *"Radiology notes are significantly shorter in the transcription field (average X words) compared to Surgery (average Y words). This may indicate that the transcription field is less informative for Radiology, and that the description or keywords fields may be more useful for that class."* Write your actual version using the real numbers you compute.

✅ **STEP 1.3 DONE** when: You have a word-length comparison table and Observation 2 written down.

---

## STEP 1.4 — Look at the Most Frequent Words Per Class

**What this tells you:** This is where you find the "signal" in the data — the words that actually separate one specialty from another. It also helps identify problems (like noise words appearing everywhere).

💻 **CODE:** For each class, take all the `description` text, combine it into one big block of text, and count which words appear most often. You can use `CountVectorizer` from scikit-learn with `max_features=20` to get the top 20 words per class.

▶️ **RUN** it. Look for two things:
1. Words that appear in ONE class but not others (these are strong discriminating signals). Example: "cataract" only in Ophthalmology.
2. Words that appear in EVERY class (these are noise). Example: "patient", "procedure".

📝 **WRITE — Observation 3 (optional but valuable):**
Describe one specific finding. Example: *"Radiology notes consistently contain imaging-specific terminology (CT, MRI, X-ray, contrast) that is largely absent from other specialties. Conversely, Surgery notes share significant vocabulary with Neurosurgery (e.g., incision, closure, anesthesia), suggesting these two classes may be difficult to separate based on vocabulary alone."*

📊 **RECORD (optional):** Generate a word cloud image for 2–3 interesting classes. Save as `.png` files for the appendix of the paper.

✅ **STEP 1.4 DONE** when: You have identified at least 2 discriminating patterns and 1 noise problem, all written down in paper-ready language.

---

## STEP 1.5 — Check for Noisy or Problematic Rows

**What this tells you:** Real-world data is never clean. Finding and reporting specific problems makes your paper more credible and specific.

💻 **CODE:** Check for:
- Empty `transcription` values (some rows may have no clinical note at all)
- Very short `description` values (less than 5 words)
- Any `medical_specialty` labels that look corrupted (very long strings — you already handled this in the baseline, but inspect what those rows were)

▶️ **RUN** it. Count how many rows have each problem.

📝 **WRITE — Note for the paper:**
Example: *"We observed that X rows contain empty transcription fields, and Y rows have descriptions of fewer than 5 words. These rows were retained in training but may introduce noise, particularly for models relying on longer text inputs."*

✅ **STEP 1.5 DONE** when: You know the exact count of problematic rows and have written about them.

---

## PHASE 1 PAUSE — Write the Data Section of the Paper

📝 **WRITE:** Before moving on, Person A should now draft the **"Data"** section of the paper in the LaTeX template. This section has two subsections: **Data Analysis** and **Data Augmentation** (leave the second one blank for now — Person B will fill it in Phase 2).

For the Data Analysis subsection, write:
1. One paragraph describing the dataset (what it is, where it comes from, how many rows, how many classes, what the 5 fields are)
2. The class distribution table
3. Observation 1 (class imbalance with exact numbers)
4. Observation 2 (text length differences — use the real numbers you computed)
5. Reference the bar chart figure

🔴 **CRITICAL REMINDER:** Use formal English. Do not write "the dataset is really unbalanced" — write "the dataset exhibits significant class imbalance". No contractions. No subjective adjectives.

✅ **PHASE 1 DONE** when: All observations are written, all charts are saved, and the Data Analysis subsection of the paper has a first draft.

---
---

# PHASE 2 — DATA AUGMENTATION
*Time estimate: 2–3 days | Who: Person B*

**Goal:** Add new training samples to increase the size of minority classes. This is mandatory (2.0 points). You must also clearly describe your strategy in the paper, because that description itself is what is graded.

---

## STEP 2.1 — Decide on Your Augmentation Strategy

🤔 **THINK:** Before writing any code, decide what you will do. The two most practical approaches for this project are:

**Option 1 — LLM-Generated Synthetic Samples (Recommended)**
- Use ChatGPT or a similar LLM (which you must declare in the paper's References section)
- Write a prompt asking it to generate new clinical notes in the same style as the real data, for the underrepresented classes
- Target the 4 smallest classes: Dermatology (25), Psychiatry-Psychology (40), Ophthalmology (71), Neurosurgery (78)
- Aim to bring each of them up to approximately 100–150 samples total
- The new rows must have all 5 fields filled: `medical_specialty`, `description`, `sample_name`, `transcription`, `keywords`

**Option 2 — Class Weights (No new data, but valid)**
- Instead of creating new rows, you tell the model to penalize mistakes on minority classes more heavily
- This is done by adding `class_weight='balanced'` to your model
- Simpler to implement but less powerful and less creative
- **This option is valid but will score lower on creativity**

📝 **WRITE your strategy decision down.** You will need to describe it in detail in the paper.

---

## STEP 2.2 — Generate the Augmented Data

If you chose **Option 1 (LLM-generated):**

📝 **WRITE:** Draft a prompt to use in ChatGPT. Something like: *"Generate 5 clinical medical transcription records for the specialty 'Dermatology'. Each record must have the following fields: a short description of 1-2 sentences, a sample name (the type of procedure or visit), a full clinical transcription note in the style of a real medical report (at least 100 words), and a list of relevant medical keywords. Format them as semicolon-separated rows matching this header: medical_specialty;description;sample_name;transcription;keywords"*

▶️ **DO:** Run this prompt for each of the minority classes. Collect the outputs.

💻 **CODE:** Create a new file `26/code/02_data_augmentation.py`. Inside it:
1. Load the original `train.csv`
2. Load your synthetic data (either from a separate CSV you create from the LLM outputs, or by hardcoding the new rows as Python dictionaries)
3. Concatenate the original and the synthetic data
4. Shuffle the combined dataset
5. Print the new class distribution to verify it improved
6. Save the augmented data to a NEW file called `train_augmented.csv` inside the `code/` folder

▶️ **RUN** it. Print the new class distribution.

📊 **RECORD:** Save both the old and new class distribution tables. You will show the "before and after" in the paper.

📝 **WRITE — Data Augmentation subsection (for the paper):**
Describe exactly what you did: how many samples you generated, for which classes, using which tool (cite it!), and what the final class distribution looks like after augmentation. Example: *"To address class imbalance, we generated X synthetic clinical records for the four most underrepresented classes using [LLM name]. The synthetic records were generated with a prompt instructing the model to produce realistic clinical notes matching the style and format of the training data. Following augmentation, the Dermatology class increased from 25 to 80 samples."*

🔴 **CRITICAL:** You must declare that you used an LLM for augmentation in the References section. The professors explicitly require this.

✅ **PHASE 2 DONE** when: You have `train_augmented.csv`, the Data Augmentation paper subsection is drafted, and the "before and after" table is saved.

---
---

# PHASE 3 — BUILD THE MODELS
*Time estimate: 3–5 days | Who: Person C (Models 1 & 2) and Person D (Model 3)*

**Goal:** Build at least 3 models. The models section is worth 4.0 points, split as: description of models (1.0), replicability (1.0), creativity (2.0).

**What you already have:**
- Model 1 (Naive Bayes baseline) → `01_baseline_model.py` ✅ — **Already done!**
- Results from Model 1: **38.76% accuracy, 13.66% macro-F1** on the held-out set (534 records, fixed split) ✅
  (the earlier 37.09% used a different row set and split; the models phase now follows `Report_Plan.md` and `Models_Rationale.md`, not Phase 3 below)

---

## STEP 3.1 — Model 2: Improved Classical ML

*Who: Person C*

**What it is:** The same type of approach as Model 1 (TF-IDF + a classical classifier), but improved in two ways: a better algorithm (Logistic Regression or SVM), and a better input (using multiple fields instead of just `description`).

**What new parameters to try:**

1. **Change the input:** Instead of only `description`, combine multiple fields into one text string per row: `description + " " + sample_name + " " + keywords`. Try this combination and also try `description` only, `keywords` only, and all three combined. Record the accuracy for each. This directly tests your research question.

2. **Change the algorithm:** Replace `MultinomialNB` with `LogisticRegression(max_iter=1000, random_state=42)`. Logistic Regression is almost always stronger than Naive Bayes on text classification and handles class imbalance better.

3. **Handle class imbalance:** Add `class_weight='balanced'` to the Logistic Regression. This makes the model pay more attention to rare classes like Dermatology. You can compare with and without this parameter to see the difference.

4. **Improve TF-IDF:** Add these parameters to `TfidfVectorizer`: `max_features=50000`, `ngram_range=(1,2)`. The `ngram_range=(1,2)` means the model will also look at pairs of consecutive words (e.g., "heart failure" as one token, not just "heart" and "failure" separately). This captures more meaning.

5. **Use the augmented data:** Load `train_augmented.csv` instead of `train.csv` for training.

💻 **CODE:** Create `26/code/02_model_improved_ml.py`. Run experiments in this order:
- Version A: `description` only + Logistic Regression (compare to the baseline)
- Version B: `description + keywords + sample_name` (combined) + Logistic Regression
- Version C: Version B + `class_weight='balanced'` + `ngram_range=(1,2)`

▶️ **RUN** each version. For each one, record the accuracy and the full classification report.

📊 **RECORD all results in a table like this:**

| Model | Input | Algorithm | Class Weight | Accuracy | Macro F1 |
|---|---|---|---|---|---|
| M1 Baseline | description | Naive Bayes | None | 38.76% | 13.66% |
| M2-A | description | Logistic Regression | None | ? | ? |
| M2-B | desc+keywords+name | Logistic Regression | None | ? | ? |
| M2-C | desc+keywords+name | Logistic Regression | Balanced | ? | ? |

📝 **WRITE — CONCLUSIONS FROM MODEL 2 (before moving to Model 3):**
After running all versions, stop and answer these questions in writing:
- Did combining fields help? By how much?
- Did class weighting improve minority class F1 scores?
- Which classes are still being predicted poorly?
- Did we beat the 49% baseline? Yes or No?

These written conclusions are not just notes — they are the material for your Discussion section.

✅ **STEP 3.1 DONE** when: All 3 versions of Model 2 are run, results are recorded in the table, and written conclusions are done.

---

## STEP 3.2 — Model 3: Pre-trained Language Model (BERT)

*Who: Person D*

**What it is:** A much more powerful approach where instead of counting words (TF-IDF), we use a model that was pre-trained on millions of medical texts and already "understands" medical language. We then fine-tune it to classify our 12 specialties.

**Before you start — check computing power:**
- If you have a GPU (even a laptop GPU), use it. Training will take ~30–60 minutes.
- If you have NO GPU (CPU only), training will take 2–4 hours but will still work. Use `description` field only (it's short enough).
- If you have access to Google Colab (free), use it — it provides a free GPU. Upload your data file and run the script there.

**What model to use:** `dmis-lab/biobert-base-cased-v1.2` or `emilyalsentzer/Bio_ClinicalBERT`. Both are free on HuggingFace and are trained on medical text. Do NOT use a general BERT model — the medical domain vocabulary matters here.

**What new parameters to try:**

1. **Input field:** Run two versions: one with `description` only, one with `description + " " + keywords` combined. Choose whichever gives better results for your final predictions.

2. **Epochs:** Train for 3–5 epochs. More epochs = better learning up to a point. Record which epoch gave the best validation accuracy (it usually peaks around epoch 3–4 and then starts to overfit).

3. **Learning rate:** Use `2e-5` or `3e-5`. These are standard values for fine-tuning BERT. Write this in the paper under hyperparameters.

4. **Batch size:** Use 16 or 32 depending on available memory.

5. **Class weights:** Also apply weighted loss here, the same way as Model 2.

💻 **CODE:** Create `26/code/03_model_bert.py`. Use the `transformers` library from HuggingFace:
- Load the tokenizer and model from HuggingFace
- Tokenize the text using the model's tokenizer (with `max_length=128` for `description` — it is short enough)
- Fine-tune the model on the training set
- Evaluate on the validation set at the end of each epoch
- Save the final accuracy and classification report

▶️ **RUN** it. This will take longer than the previous models.

📊 **RECORD:**
- Accuracy per epoch (to show learning curve)
- Final accuracy and macro-F1 on validation set
- Full classification report (per-class F1)
- Add Model 3 to your results table

📝 **WRITE — CONCLUSIONS FROM MODEL 3 (before moving to Phase 4):**
Answer these questions in writing:
- Did BERT significantly outperform Model 2? By how much?
- Which classes improved the most with BERT?
- Which classes are STILL being predicted poorly even with BERT?
- Does this align with or contradict what you expected based on the research question?
- Can you explain WHY BERT is better (or not much better) than the classical model?

✅ **STEP 3.2 DONE** when: Model 3 is trained, results are in the table, and written conclusions are done.

---

## STEP 3.3 — Complete the Results Table

📊 **RECORD:** Your final results table should now look something like this:

| Model | Input | Algorithm | Accuracy | Macro F1 |
|---|---|---|---|---|
| M1 — Baseline | description | Naive Bayes + TF-IDF | 38.76% | 13.66% |
| M2-A | description | Logistic Regression + TF-IDF | ?% | ? |
| M2-B | desc + keywords + name | Logistic Regression + TF-IDF | ?% | ? |
| M2-C | desc + keywords + name | LR + TF-IDF + class weight | ?% | ? |
| M3 — BERT | description | Bio_ClinicalBERT fine-tuned | ?% | ? |

This table goes directly into the Results section of the paper.

✅ **STEP 3.3 DONE** when: The full results table is populated with real numbers.

---
---

# PHASE 4 — EVALUATION, DISCUSSION MATERIAL, AND FINAL PREDICTIONS
*Time estimate: 2 days | Who: Everyone*

**Goal:** Collect all the material you need for the Results and Discussion sections of the paper, and generate the final `results.txt` file for submission.

---

## STEP 4.1 — Generate the Confusion Matrix

📊 **RECORD:** Generate a confusion matrix for **at least one model** (it does not have to be the best model — it can be whichever one produces the most interesting or readable matrix).

💻 **CODE:** Use `sklearn.metrics.ConfusionMatrixDisplay` with `matplotlib` to create and save the confusion matrix as a `.png` image.

**Important formatting rules for the paper:**
- Abbreviate the class names so they fit on the axes (e.g., "Cardiovascular-Pulmonary" → "CV-Pulm", "Obstetrics-Gynecology" → "OB-Gyn", "Psychiatry-Psychology" → "Psych")
- Make sure the labels are readable — do not use the full long names
- Save the image at high resolution (`dpi=200` at minimum)
- Title it "Confusion Matrix — [Model Name]"

📊 **RECORD:** Save the confusion matrix image. It goes in the Results section of the paper.

✅ **STEP 4.1 DONE** when: At least one confusion matrix image is saved and readable.

---

## STEP 4.2 — Collect Misclassified Examples

**This is the most important step for the Discussion section (3.0 points).**

The professors require **at least 3 concrete misclassified examples** from your own validation set. Each example must include:
1. A text excerpt from the clinical note (a short representative sentence, not the full text)
2. The correct label (what it should be)
3. The model's predicted label (what the model guessed wrong)
4. Your interpretation: WHY do you think the model failed here?

💻 **CODE:** In your best model's script, after computing `y_pred`, find the rows where `y_pred != y_val`. Print those rows (or save them to a file) showing the description text, the true label, and the predicted label.

📝 **WRITE — Find 3 specific misclassified cases and analyse each one:**

For each case, ask yourself:
- *"Does this text look like it could reasonably belong to both the true class and the predicted class?"*
- *"Which specific words or phrases might have confused the model?"*
- *"Is this error consistent with what I know about the class (e.g., Neurosurgery being confused with Surgery because both involve operations)?"*

Example of a good case analysis for the paper:
> *"A Neurosurgery note was misclassified as Surgery. The excerpt reads: 'Patient underwent general anesthesia. A standard incision was made...' This error is expected: both Neurosurgery and Surgery involve operative procedures, and the shared vocabulary (incision, anesthesia, hemostasis) provides little discriminative signal. The model would require neurological anatomical terms ('dura', 'craniotomy') to distinguish them — terms absent from the description field but present in the transcription."*

Write this level of detail for at least 3 cases.

📊 **RECORD:** Save the 3 (or more) misclassified cases in your notes. These examples go directly into the Discussion section.

✅ **STEP 4.2 DONE** when: You have 3 concrete misclassified examples, each with a written explanation.

---

## STEP 4.3 — Generate `results.txt` (The Final Submission File)

**🔴 CRITICAL: Only do this AFTER you have finished all model experiments. Use ONLY your best performing model.**

💻 **CODE:** Create `26/code/04_generate_results.py`. This script must:
1. Load the original `train.csv` and train your best model on **the entire training set** (all 2,613 rows — NOT just the 80% split, because you don't need to hold any data back anymore for validation)
2. Load `test_no_labels.csv` (the 398 unlabelled records of the clean test file). The file has **no header row**: load it with `header=None` (`data_utils.load_test()` does this)
3. Apply the same preprocessing as your best model
4. Run the model on those 398 rows
5. Output the predictions as a file named `results.txt` in the `26/` folder root

**Format of `results.txt`:**
- Exactly 398 lines
- One label per line (just the label text, nothing else)
- No header line
- The label on line N corresponds to row N of `test_no_labels.csv`

Example:
```
Surgery
Neurology
Radiology
Cardiovascular-Pulmonary
Surgery
Dermatology
...
```

▶️ **RUN** it. After it runs, check the file:
- Count the lines: it must be exactly 398
- Check that every line is one of the 12 valid label names
- Check that there are no empty lines

🔴 **CRITICAL:** Never train on `test_no_labels.csv`. It is only ever used as input to your best trained model to generate predictions.

✅ **STEP 4.3 DONE** when: `results.txt` exists, has exactly 398 lines, every line is a valid label name.

---
---

# PHASE 5 — WRITE THE PAPER
*Time estimate: 3–4 days | Who: Everyone — each person writes assigned sections*

**Goal:** Compile everything you have collected across Phases 1–4 into a scientific paper of at most 3 pages using the LaTeX template.

**🔴 CRITICAL rules before starting:**
- The paper must be **at most 3 pages** (bibliography and appendix do not count)
- **No cover page** — the title and authors start on page 1, followed directly by the Introduction (the template has no abstract)
- The author line must include the **contribution percentage for each member** (e.g., "Group 26, Ana Silva (30%), João Costa (25%), ...")
- Use the provided LaTeX template structure exactly
- **Remove the "Evaluation and tips" section** (lines 53–79 of `template.tex`) before compiling the PDF — it is only for your reference, never submitted

---

## STEP 5.1 — Copy and Set Up the LaTeX Template

Copy `template.tex` and `biblio.bib` from the `StudentsPack/Project-Template-extracted/` folder into your `26/` folder. Rename `template.tex` to `26.tex`.

Open `26.tex` and make the following immediate changes:
- Change the `\title{}` from "Title" to your actual paper title
- Change the `\author{}` line to include all group member names and their contribution percentages

✅ **STEP 5.1 DONE** when: `26.tex` compiles to a PDF without errors (even if empty sections).

---

## STEP 5.2 — Fill in Each Section (Assign Per Person)

Work through the sections in this order. Each person writes their assigned section based on the notes they have been collecting throughout the project.

---

### SECTION: Introduction — Person C

**Write:**
1. First paragraph: What is the problem? (Medical text classification. Clinical notes are long and complex. Doctors write in jargon. Automatically identifying the specialty from a clinical note has practical applications for health record management.)
2. Second paragraph: What is the dataset? (Briefly introduce `train.csv` — 2,613 medical transcriptions, 12 specialties, highly imbalanced.)
3. Third paragraph: **The research question** — State it explicitly. Use the exact sentence your group agreed on in Step 0.2. Then explain how the rest of the paper is structured to answer it.

**Scoring focus:** 1.5 points. The research question must be explicit.

---

### SECTION: Data Analysis — Person A

**Write:** Use everything from Phase 1. Include:
1. What the dataset is
2. The class distribution table
3. Observation 1 (the imbalance)
4. Observation 2 (text length differences)
5. Embed the bar chart figure (reference it in the text: "as shown in Figure 1...")

**Scoring focus:** 1.5 points. Generic = 0. Must have specific numbers.

---

### SECTION: Data Augmentation — Person B

**Write:** Use everything from Phase 2. Include:
1. Why augmentation was needed (minority classes had too few samples)
2. Exactly what you did (which method, which tool, how many samples generated, for which classes)
3. The before/after class distribution comparison (a small table or numbers inline)
4. If you declared an LLM: state it explicitly here AND in the References section

**Scoring focus:** 2.0 points. Must clearly describe the strategy — not just say "we augmented the data".

---

### SECTION: Models — Person C and Person D

**Write:** For each model (at least 3), include:
1. What the model is (Naive Bayes, Logistic Regression, Bio_ClinicalBERT, etc.)
2. What the input is (which fields, any preprocessing)
3. Key hyperparameters (TF-IDF max_features, ngram_range, learning rate, batch size, epochs, random seed)
4. Why you chose this model (tie to research question — e.g., "To investigate the impact of field selection, we ran this model with three different input configurations...")
5. Any specific implementation decisions (class weighting, etc.)

**Scoring focus (4.0 pts):**
- 1.0: Are at least 3 models described?
- 1.0: Could someone reproduce your results from reading this section?
- 2.0: Did you make creative, thoughtful choices?

🔴 **REPLICABILITY CHECKLIST — before submitting this section:**
- [ ] Every hyperparameter is listed
- [ ] The random seed (`random_state=42`) is mentioned
- [ ] The library and version is mentioned (e.g., "scikit-learn 1.x", "transformers 4.x")
- [ ] The train/val split ratio (80/20) is mentioned
- [ ] Which fields are used as input is clearly stated for each model

---

### SECTION: Experimental Setup — Person C

**Write:**
1. Which dataset was used for training (the augmented version)
2. How you split it (80/20 stratified split, random_state=42)
3. What the test set is (`test_no_labels.csv`, 398 samples, never used for training)
4. Which evaluation metrics you used and why (Accuracy, Macro-F1, Per-class F1; explain that Macro-F1 is more informative than Accuracy for imbalanced datasets)

**Scoring focus:** 1.0 point. Must be complete enough to replicate.

---

### SECTION: Results — Person D

**Write:**
1. The full results table (all model versions, accuracy, macro-F1)
2. Per-label F1 of your best model (a table with one row per class)
3. The confusion matrix figure (embed it, reference it in text: "as shown in Figure 2...")
4. A brief factual description of what the table shows (do not interpret yet — save interpretation for Discussion)

**Scoring focus:** 1.5 points. Must include: results table + per-label F1 + confusion matrix.

---

### SECTION: Discussion — Everyone (most important section)

**Write:** This is 3.0 points — the highest single section. It must be specific.

1. Answer the research question directly: *"Our results show that combining the description, keywords, and sample_name fields (Model 2-C) improved accuracy from 37% (baseline) to X%, supporting the hypothesis that multi-field representations carry more discriminative information than any single field alone..."*

2. Discuss the 3 misclassified examples from Step 4.2. Each one must have:
   - A short text excerpt (1–2 sentences from the note)
   - The true label
   - The predicted label
   - Your interpretation of why the model failed

3. Discuss class imbalance: which classes are still predicted poorly even after augmentation? Why?

4. Discuss the difference between Model 2 and Model 3: when did the complexity of BERT help, and when was it not worth it?

🔴 **CRITICAL:** Every statement must be supported by numbers or specific examples. Do not write "the model did well" — write "the model achieved 85% macro-F1 on the Surgery class." Do not write "some classes were hard" — write "Dermatology achieved the lowest per-class F1 of X%, likely due to the limited number of training samples even after augmentation."

**Scoring focus:** 3.0 points. Generic = 0.

---

### SECTION: Future Work — Person A

**Write:** What would you do if you had more time? Must connect to the research question.

Suggestions:
- Try the `transcription` field with a model that handles long text (Longformer, BigBird)
- Explore more sophisticated augmentation (back-translation, paraphrasing)
- Run error analysis on the full test set once labels are released
- Try a hierarchical classifier (first classify broad groups like "surgical vs. non-surgical", then fine-grained)

**Scoring focus:** 0.5 points. Brief (4–6 sentences). Must connect to the research question.

---

### SECTION: References — Everyone

**Write (in `biblio.bib`):**
- The BioBERT paper (if you used it)
- The HuggingFace transformers library
- The scikit-learn paper
- The NLTK book or paper
- Any other external source you referenced
- **🔴 CRITICAL: A declaration of LLM use.** The professors require this explicitly. Add a reference entry like: "ChatGPT (OpenAI, 2024) was used to generate synthetic clinical notes for data augmentation and to assist with grammar corrections in the manuscript." Use a footnote or bibliography entry.

**Scoring focus:** 1.0 point.

---

## STEP 5.3 — Review and Polish the Paper (Everyone)

After all sections have been drafted, all 4 members should read the complete paper together and check:

**Content checks:**
- [ ] Research question is explicitly stated in the Introduction
- [ ] Every claim in the paper is supported by a number or a specific example
- [ ] The confusion matrix figure has readable labels and is referenced in the text
- [ ] The per-label F1 table is present in the Results section
- [ ] The 3 misclassified examples are present in the Discussion
- [ ] Future Work connects to the research question

**Formatting checks:**
- [ ] Paper is at most 3 pages (compile it and count)
- [ ] No cover page
- [ ] Author line has all names and contribution percentages
- [ ] All figures are labeled (Figure 1, Figure 2, ...) and referenced in the text body
- [ ] All tables are labeled (Table 1, Table 2, ...) and referenced in the text body
- [ ] Formal English only — search for contractions ("it's", "they're", "we've") and remove them
- [ ] No subjective adjectives (search for "amazing", "great", "nice", "interesting")
- [ ] All acronyms are introduced the first time they are used (e.g., "TF-IDF (Term Frequency–Inverse Document Frequency)")
- [ ] LaTeX quotation marks used correctly (` ``like this'' ` not `"like this"`)
- [ ] "Evaluation and tips" section is removed
- [ ] Bibliography compiles correctly

**Replicability check:**
- [ ] Every model has all hyperparameters listed
- [ ] Random seed is mentioned (`random_state=42`)
- [ ] Library versions are mentioned
- [ ] The train/val split is described

✅ **PHASE 5 DONE** when: The paper compiles to `26.pdf`, is at most 3 pages, and every checklist item above is ticked.

---
---

# PHASE 6 — FINAL SUBMISSION
*Time estimate: 1–2 hours | Who: Person C or D (whoever submitted correctly before)*

---

## STEP 6.1 — Final Verification of `results.txt`

Before zipping:
- Open `results.txt` and count the lines manually: must be exactly 398
- Load `test_no_labels.csv` with `header=None` and confirm it also has 398 records (one per physical line)
- Spot-check 5–10 lines: are the labels valid class names?
- Confirm no header row, no empty lines, no extra spaces

---

## STEP 6.2 — Verify the Code Runs

Even if the professors probably won't run your code, you must be prepared if they do. Do a clean run:
- Delete any generated files (`.pkl` model files, `.csv` intermediaries)
- Run your scripts in order: `01_baseline_model.py`, `02_data_augmentation.py`, `02_model_improved_ml.py`, `03_model_bert.py`, `04_generate_results.py`
- Confirm all scripts run without errors on a clean environment

---

## STEP 6.3 — Create the ZIP File

The ZIP must contain:
```
26.zip
├── 26.pdf             ← the compiled paper
├── results.txt        ← 398 lines of predictions
├── biblio.bib         ← bibliography source file
├── 26.tex             ← LaTeX source file
└── code/
    ├── 01_baseline_model.py
    ├── 02_data_augmentation.py
    ├── 02_model_improved_ml.py
    ├── 03_model_bert.py
    └── 04_generate_results.py
```

🔴 **CRITICAL naming rules:**
- The ZIP file must be named exactly `26.zip`
- The paper inside must be named exactly `26.pdf`
- The predictions file must be named exactly `results.txt`
- Do NOT include trained model weights or checkpoint files (they are too large and not needed)
- Do NOT submit a `.rar` file — only `.zip`

---

## STEP 6.4 — Submit via Fenix

- Log into Fenix with your IST credentials
- Find the Natural Language course
- Submit `26.zip` before **October 16, 2026 at 23:59**
- Confirm submission was accepted — you will get a confirmation receipt

✅ **PROJECT COMPLETE**

---
---

# TEAM DELEGATION SUMMARY

| Person | Owns | Phases |
|---|---|---|
| **Person A** | Data Analysis, Word clouds, Paper: Data Analysis section, Paper: Future Work | Phase 1, Part of Phase 5 |
| **Person B** | Data Augmentation script, Synthetic data generation, Paper: Data Augmentation section | Phase 2, Part of Phase 5 |
| **Person C** | Model 2 (Logistic Regression experiments), Paper: Introduction + Experimental Setup + Models (partial) | Phase 3.1, Part of Phase 5 |
| **Person D** | Model 3 (BERT), Confusion matrix, Misclassified examples, `results.txt` generator, Paper: Results + Models (partial) | Phase 3.2, Phase 4, Part of Phase 5 |
| **Everyone** | Research question (Phase 0), Discussion section, Final review, Submission (Phase 5.3 + Phase 6) | — |

---

# TIMELINE SUGGESTION (19 days total before Oct 16)

| Days | Task |
|---|---|
| Sep 28 (today) | Phase 0: Research question decision (group meeting) |
| Sep 29–Oct 1 | Phase 1: Data analysis (Person A) |
| Oct 1–Oct 3 | Phase 2: Data augmentation (Person B) |
| Oct 2–Oct 6 | Phase 3: Model 2 (Person C) + Model 3 (Person D) — parallel |
| Oct 7–Oct 8 | Phase 4: Confusion matrix + misclassified examples + results.txt |
| Oct 9–Oct 13 | Phase 5: Paper writing — each person writes their sections |
| Oct 14 | Full group review of paper — fix everything |
| Oct 15 | Final code clean-up + zip creation |
| Oct 16 | Submit before 23:59 |
