# Phase 1 — Data Analysis Notes
### Group 26 | Generated: 2026-09-29 | Script: `code/02_data_analysis.py`

This file documents everything found during Phase 1 so that any team member can
understand the dataset without running any code. These findings are already
written into the paper (`26.tex`, section: Data → Data Analysis).

---

## What the Dataset Is

- **File:** `StudentsPack/train.csv`
- **Separator:** semicolons (`;`) — not the standard comma
- **Total rows (after cleaning):** 2,613
- **Number of classes:** 12 medical specialties
- **Columns (5 total):**
  | Column | Description |
  |---|---|
  | `medical_specialty` | The label — what we are trying to predict |
  | `description` | Short 1–2 sentence summary of the visit |
  | `sample_name` | The name of the procedure or type of visit |
  | `transcription` | The full clinical note (can be hundreds of words) |
  | `keywords` | Comma-separated list of relevant medical terms |

---

## Class Distribution (exact numbers from the script)

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

**Most common class:** Surgery (876 samples, 33.5%)
**Least common class:** Dermatology (25 samples, 1.0%)
**Imbalance ratio:** 35.0:1

**→ What this means for the project:** Any model that just guesses "Surgery" every
time will get ~33% accuracy automatically. Our baseline model (01_baseline_model.py)
confirmed exactly this: it scored 37% but showed 0.00 F1 for Dermatology, Neurology,
Radiology and other minority classes.

---

## Text Length Per Class (word counts from the script)

| Specialty | Avg Words (description) | Avg Words (transcription) |
|---|---|---|
| Psychiatry-Psychology | 15.6 | 823.7 |
| Neurosurgery | 84.3 | 572.4 |
| Orthopedic | 21.3 | 558.3 |
| Neurology | 16.6 | 499.2 |
| Surgery | 31.1 | 465.5 |
| General Medicine | 37.6 | 453.8 |
| Obstetrics-Gynecology | 51.4 | 440.0 |
| Cardiovascular-Pulmonary | 26.0 | 427.4 |
| Dermatology | 17.5 | 423.5 |
| Ophthalmology | 17.6 | 356.0 |
| Gastroenterology | 17.8 | 343.2 |
| Radiology | 13.4 | 275.8 |

**Overall average — description:** 28.4 words
**Overall average — transcription:** 452.2 words

**→ Observation 1 (written in the paper):** The `description` field is short and
consistent (~28 words on average). The `transcription` field is long and varies
enormously between classes (275 to 823 words). A classical model trained on the full
transcription will process up to 3x more tokens for some classes than others.
This creates a length bias risk.

---

## Top Words Per Class (from `description` field)

### Surgery (876 samples)
`left (484), right (464), patient (218), bilateral (116), anterior (110), procedure (71), coronary (70), placed (70), pain (69), posterior (68), artery (67), mass (67), placement (67), upper (64), using (64)`

### Neurosurgery (78 samples)
`right (64), anterior (60), cervical (54), left (51), patient (51), fusion (30), bilateral (29), discectomy (29), decompression (23), bone (21), laceration (21), using (21), distal (19), repair (17), surgery (17)`

### Dermatology (25 samples)
`skin (11), right (9), biopsy (8), left (5), lesion (5), acne (4), cheek (4), layer (4), neck (4), neoplasm (4), patient (4), evaluation (3), facial (3), plastic (3), scalp (3)`

### Radiology (217 samples)
`contrast (42), left (41), pain (38), mri (37), right (28), patient (27), spine (26), brain (24), ultrasound (22), abdomen (20), chest (20), stress (19), pelvis (17), old (16), year (15)`

**→ Observation 2 (written in the paper):** Surgery and Neurosurgery share most of their
top words (`right`, `left`, `anterior`, `bilateral`, `patient`). A simple word-counting
model will have great difficulty separating them. However, Neurosurgery has distinctive
terms (`discectomy`, `cervical`, `fusion`, `decompression`) that a smarter model (BERT)
can pick up. Radiology's vocabulary (`MRI`, `contrast`, `ultrasound`) is highly distinctive —
which may explain why even simple models can classify it reasonably well.

---

## Noisy / Incomplete Rows

| Issue | Count |
|---|---|
| Empty or missing transcription | 30 |
| Description under 5 words | 158 |
| Total potentially noisy rows | 188 |

**→ What this means:** About 7% of the dataset has data quality issues. These rows
were kept in training but may hurt models that rely heavily on the transcription field
(30 rows have nothing there at all). Models using only `description` are safer for
these rows.

---

## Figures Generated

| File | What it shows |
|---|---|
| `figures/fig1_class_distribution.png` | Bar chart of class counts (used as Figure 1 in paper) |
| `figures/fig2_text_lengths.png` | Grouped bar chart comparing desc vs. transcription word counts |

---

## What Was Written in the Paper

The following has already been added to `26.tex` based on this analysis:
- ✅ Introduction section (research question stated explicitly)
- ✅ Data Analysis subsection (class table, Figure 1 reference, Observation 1, Observation 2)
- ⏳ Data Augmentation subsection — left as TODO (Phase 2)

---

## What to Do Next (Phase 2)

The next step is **Data Augmentation**. Based on what we found here, the priority
classes to augment (they have too few samples) are:

| Priority | Class | Current Count | Target |
|---|---|---|---|
| 🔴 High | Dermatology | 25 | ~100 |
| 🔴 High | Psychiatry-Psychology | 40 | ~100 |
| 🟡 Medium | Ophthalmology | 71 | ~100 |
| 🟡 Medium | Neurosurgery | 78 | ~100 |
