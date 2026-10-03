# Report Plan — Group 26 (`26.tex`)

> Reference for writing the paper: what goes in each section, in what order, and how much space it gets. The LaTeX file itself only holds the template headings and the text already written; everything about what is still to be written lives here.

**Research question (RQ):** does a domain-specific language model, fine-tuned on the short clinical description, outperform a classical classifier trained on the full available text?

**Constraints that shape everything below**
- At most 3 pages, two-column, 10pt. Bibliography and Appendix A (max one page) do not count.
- The section headings are fixed by the template and may not be renamed, reordered or dropped.
- The paper must be understandable without the appendix.
- The paper is worth 16 points; the per-section points below sum to 16.

---

## 1. How subparts are marked

- The template's numbered headings stay exactly as they are: 1 Introduction, 2 Data (2.1 Data Analysis, 2.2 Data Augmentation), 3 Models, 4 Experimental Setup (4.1 Parameters/Hyperparameters), 5 Results, 6 Discussion, 7 Future Work, Bibliography, Appendix A.
- Subparts inside a section are run-in bold labels (`\textbf{Label.}` at the start of a paragraph). They cost no vertical space, unlike extra `\subsection`s.
- Where a pattern repeats, the order repeats: every model is described with the same five slots, every error example with the same four.

---

## 2. The thread: one step per section

Each section answers one question and hands something to the next. This is what makes the paper read as a process and keeps the RQ present throughout.

| Step | Section | Question it answers | Hands over to the next |
|---|---|---|---|
| 1 | 1 Introduction | What is the problem and what do we want to find out? | The RQ and the pipeline in one sentence |
| 2 | 2.1 Data Analysis | What does the data look like, and what in it will matter? | The imbalance and two observations that later sections reuse |
| 3 | 2.2 Data Augmentation | What did we do about the imbalance? | The training set every model uses |
| 4 | 3 Models | Which systems are compared, and why these? | One system per side of the RQ, plus a reference point |
| 5 | 4 Experimental Setup | How is the comparison made fair and repeatable? | Own test set, metrics, hyperparameters |
| 6 | 5 Results | What numbers came out? (facts only) | Tables and the confusion matrix |
| 7 | 6 Discussion | What do the numbers and the outputs mean? | The answer to the RQ, the errors, the limitations |
| 8 | 7 Future Work | What would sharpen the answer? | — |

Read-through test: the last sentence of each section should lead into the next one, and Discussion subpart 1 must answer the sentence written in Introduction subpart 2.

---

## 3. Section-by-section subdivision

Budgets assume about 400 words per column and about 5.5 usable columns after the title block. They are caps, to be checked by compiling.

### 1 Introduction — 1.5 pts — about 150 words — *written*
1. **Task and motivation** (2 sentences).
2. **Research question**, stated explicitly in one sentence, followed by why the answer is not obvious.
3. **Approach in one sentence**: data inspection, augmentation, three models, own test set, error analysis. This sentence doubles as the roadmap of the paper.

Kept out: dataset statistics (they belong to 2.1) and model details (Section 3).

### 2.1 Data Analysis — 1.5 pts — about 200 words + Table 1 — *written*
1. **Dataset**: origin, size, the five fields, and how the file was read and repaired (seven cells had absorbed the records that followed them).
2. **Class distribution**: Table 1 and the imbalance stated with numbers.
3. **Observation 1 — the same note is listed under several specialties.** Identical text, different label. It bounds what any text-only model can reach and explains confusions with Surgery in advance.
4. **Observation 2 — the label is inside the keywords.** The first keyword is the specialty itself in almost every row that has keywords. It bears directly on the RQ, because only the full-text model can see it.
5. One sentence on text length, pointing to the appendix figure.

Kept out: any remedy (2.2) and any model consequence beyond a forward pointer.

### 2.2 Data Augmentation — 2.0 pts — about 200 words — *written*
1. **Motivation**: one sentence linking back to the imbalance.
2. **Strategy**: which transformations, on which fields, what is protected from change.
3. **Procedure**: which classes, target size, seed, and that records are generated only from the training portion so the own test set stays untouched.
4. **Outcome**: the "after" column of Table 1. The effect on performance is reported in Results.

### 3 Models — 4.0 pts — about 360 words (largest text block) — *to do*
0. **Design rationale** (one short paragraph): how the set of models maps onto the RQ.
1. **Baseline**: the official weak baseline, as the reference point.
2. **Classical model on the full text**: one side of the RQ.
3. **Domain language model on the short description**: the other side.
4. **Additional variant** (optional, for the creativity mark): only if it adds to the RQ.

Every model uses the same five slots in the same order: input fields, preprocessing, representation, classifier, imbalance handling. Hyperparameter values are not given here; they are referenced in 4.1.

Points: at least three models described (1.0), replicability (1.0), creativity (2.0).

### 4 Experimental Setup — 1.0 pt — about 200 words including 4.1 — *to do*
1. **Data and splits**: how the own test set is built (stratified, seed, size), what is used for tuning, how notes that appear more than once are handled, and that `test_no_labels.csv` is used only for `results.txt`.
2. **Metrics**: accuracy, macro-F1, per-class F1 and the lowest class F1, each with one clause on why.
3. **Environment**: library versions, hardware, seeds.

**4.1 Parameters/Hyperparameters**: one compact table, one row per hyperparameter with its value and a few-word explanation (the template asks that every hyperparameter mentioned is explained).

### 5 Results — 1.5 pts — about 120 words + Table 2, Table 3, Figure 1 — *to do*
1. **Overall comparison**: Table 2, one row per model, with and without augmentation.
2. **Per-label results of the best model**: Table 3.
3. **Confusion matrix** of one model: Figure 1 with abbreviated labels, plus one or two sentences reading it.
4. **Submitted system**: one sentence naming the model used for `results.txt`.

Kept out: every "because". Interpretation belongs to Discussion.

### 6 Discussion — 3.0 pts — about 360 words + Table 4 — *to do*
1. **Answer to the RQ**: direct, with numbers from Table 2.
2. **Most common errors**: the dominant confusions from Figure 1 and their cause, linked back to the observations of 2.1.
3. **Three misclassified examples** from the own test set: Table 4 (text excerpt, correct label, predicted label) with an interpretation of each in the text.
4. **Effect of augmentation**: what changed for the small classes.
5. **Limitations**: of the data (including labels the group disagrees with) and of the system.

Generic statements score zero here; every claim needs a number or a quoted example.

### 7 Future Work — 0.5 pt — about 60 words — *to do*
Two or three sentences, each tied to the RQ or to a limitation from Discussion subpart 5.

### Bibliography — 1.0 pt — outside the page limit — *partly done*
References for every paper, library and pre-trained model used, plus the explicit statement of how LLMs were used. Placing that statement under this heading costs no body space.

### Appendix A — at most one page, outside the limit
Only material the paper can be understood without: the class distribution chart, the text length chart, augmentation before/after examples, a second confusion matrix, further error examples.

---

## 4. Floats in the body

| Float | Section | Content | Status |
|---|---|---|---|
| Table 1 | 2.1, reused by 2.2 | Class counts, %, and count after augmentation | done |
| Hyperparameter table | 4.1 | Value and short explanation per hyperparameter | to do |
| Table 2 | 5 | Model comparison | to do |
| Table 3 | 5 | Per-label results of the best model | to do |
| Figure 1 | 5 | Confusion matrix | to do |
| Table 4 | 6 | Three misclassified examples | to do |

The class distribution chart and the text length chart are in Appendix A, because Table 1 already carries the class numbers.

---

## 5. Page map (target)

- **Page 1**: title and authors; Introduction; 2.1 with Table 1; 2.2.
- **Page 2**: 3 Models; 4 Experimental Setup with the 4.1 table; start of 5 Results with Table 2.
- **Page 3**: Table 3 and Figure 1; 6 Discussion with Table 4; 7 Future Work.
- **After page 3**: Bibliography with the LLM statement; Appendix A.

---

## 6. Content decisions still open

The structure above already has a slot for each of these.

- Whether a fourth model variant is included in Section 3.
- Which model's confusion matrix is shown in Figure 1.
- How the own test set treats notes that appear more than once under different labels (Section 4, subpart 1).

---

## 7. Style rules from the template (apply to every section)

- Formal English, no contractions, no subjective adjectives.
- Every figure and table is labelled and referred to in the text.
- Every acronym is introduced once and used consistently.
- Statements such as "the dataset is unbalanced" are backed by numbers.
- LaTeX quotation marks (` ``like this'' `).
- Sources are cited; computational resources get a footnote with a URL.
