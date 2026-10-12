"""
M4_error_analysis.py
====================
Looks at the mistakes of Model 4 on the held-out records, for the Results and Discussion
sections of the paper. No training: it reads results/m4_heldout_predictions.csv.

WHAT IT DOES:
    1. draws the confusion matrix of Model 4 (abbreviated labels) -> ../figures/fig2_confusion_matrix.png
    2. prints the per-class precision / recall / F1 of Model 4
    3. prints the most frequent confusions, and how the errors split by kind of record
       (with / without a first keyword, with / without a copy of the note in the training portion)
    4. prints the text of the most frequent confusions (one example per pair) for the Discussion

HOW TO RUN (from 26/code/, after M4_final_system.py):
    python3 M4_error_analysis.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from data_utils import load_train, split_train_heldout, first_keyword_label
from evaluation import RESULTS_DIR

ABBREVIATION = {
    'Cardiovascular-Pulmonary': 'CV-Pulm', 'Dermatology': 'Derm', 'Gastroenterology': 'Gastro',
    'General Medicine': 'GenMed', 'Neurology': 'Neuro', 'Neurosurgery': 'NeuroSurg',
    'Obstetrics-Gynecology': 'Ob-Gyn', 'Ophthalmology': 'Ophth', 'Orthopedic': 'Ortho',
    'Psychiatry-Psychology': 'Psych', 'Radiology': 'Radio', 'Surgery': 'Surg',
}


def main():
    original = load_train()
    train_df, heldout_df = split_train_heldout(original)
    pred_df = pd.read_csv(os.path.join(RESULTS_DIR, 'm4_heldout_predictions.csv')).set_index('record_id')
    held = heldout_df.join(pred_df['pred'])
    assert held['pred'].notna().all() and (held['medical_specialty'] == pred_df.loc[held.index, 'true']).all()
    classes = sorted(held['medical_specialty'].unique())
    names = [ABBREVIATION[c] for c in classes]

    # ── 1. Confusion matrix ──
    cm = confusion_matrix(held['medical_specialty'], held['pred'], labels=classes)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Predicted', fontsize=8); ax.set_ylabel('True', fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=6,
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
    fig.tight_layout()
    out = os.path.join('..', 'figures', 'fig2_confusion_matrix.png')
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"Saved {out}\n")

    # ── 2. Per class ──
    print(classification_report(held['medical_specialty'], held['pred'], zero_division=0, digits=3))

    # ── 3. Errors ──
    wrong = held[held['medical_specialty'] != held['pred']].copy()
    wrong['has_keyword'] = (first_keyword_label(wrong) != 'none')
    train_notes = set(train_df['transcription'].dropna())
    wrong['copy_in_train'] = wrong['transcription'].isin(train_notes)
    print(f"Errors: {len(wrong)} of {len(held)}")
    print(f"  with a first keyword   : {int(wrong['has_keyword'].sum())}")
    print(f"  without a first keyword: {int((~wrong['has_keyword']).sum())}")
    print(f"  whose note also exists in the training portion: {int(wrong['copy_in_train'].sum())}")
    print("\nMost frequent confusions (true -> predicted):")
    pairs = wrong.groupby(['medical_specialty', 'pred']).size().sort_values(ascending=False)
    print(pairs.head(8).to_string())

    # ── 4. One example for each of the most frequent confusions ──
    print("\nExamples (description, sample name, start of the transcription):")
    for (true, pred), n in pairs.head(5).items():
        row = wrong[(wrong['medical_specialty'] == true) & (wrong['pred'] == pred)].iloc[0]
        print(f"\n[{n}x {true} -> {pred}] record {row.name} | copy in train: {row['copy_in_train']}")
        print(f"  description : {row['description']}")
        print(f"  sample name : {row['sample_name']}")
        print(f"  transcript  : {str(row['transcription'])[:300]}")


if __name__ == '__main__':
    main()
