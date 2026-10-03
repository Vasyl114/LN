# Project Storyline — Group 26

## What we were given

The professors handed us a folder with a spreadsheet of 2,613 clinical records. Each record is a real medical note — the kind a doctor would write after seeing a patient. The records belong to 12 different medical departments (Surgery, Radiology, Neurology, etc.) and each one is already labelled with the correct department.

They also gave us 409 unlabelled records and said: *"Use the 2,613 labelled ones to teach a computer how to read a clinical note and figure out which department it belongs to. Then use what you built to predict the department for these 409 records and give us the answers."*

---

## The problem we noticed immediately

The 2,613 records are not evenly distributed. Surgery alone accounts for a third of all records, while Dermatology has only 25. This means that a computer program that just guesses "Surgery" for everything will already be right 33% of the time — without understanding a single word of medical text. That is the core challenge: we need a model that can correctly identify the rare specialties, not just the dominant one.

---

## The question driving the project

To build something genuinely better, we needed a clear scientific question to answer — otherwise the project is just "we tried things until the number went up", which is not science.

Our question is: **does a state-of-the-art AI trained on medical texts, given only a short sentence of input, outperform a simpler algorithm that reads the entire clinical note word by word?**

This is interesting because the answer is not obvious. More information is not always better — sometimes a smarter tool beats raw volume. We will find out which wins on this specific dataset and explain why.

---

## How we will answer it (the process, step by step)

**First**, we fix the data. Because Dermatology has only 25 examples and Surgery has 876, any model we train will be biased. We generate synthetic clinical records for the rare departments to bring the counts up to a reasonable level. This is called data augmentation.

**Second**, we build the simplest possible model first. This is our baseline — a basic algorithm that counts words and makes predictions. The professors tell us it scores around 49% accuracy. We need to beat that.

**Third**, we build two competing models:
- A **classical algorithm** (Logistic Regression) that reads everything — the short description, the full clinical note, and the keywords — all concatenated together into one massive block of text.
- A **smart language model** (ClinicalBERT) that was pre-trained on millions of real medical documents and already understands medical language. But we only feed it the short two-sentence description — nothing else.

**Fourth**, we compare the results. The classical model has more information. The smart model has better understanding. Whichever wins answers our research question.

**Finally**, we take the winning model, run it on the 409 unlabelled records, and produce a file with 409 predicted department names. That file goes to the professors.

---

## What we hand in

A ZIP file containing three things:
- A 3-page scientific paper (PDF) telling this exact story with numbers and analysis
- The file with the 409 predictions
- All the code we wrote

---

## What the conclusion will be about

The conclusion is not just "our best model got X% accuracy." It is an honest answer to the research question with evidence: which model won, why it won, which departments it still got wrong, and what those specific errors tell us about the limits of current approaches. The most important part of the paper is explaining the mistakes — not just celebrating the successes.
