"""
M4_final_system.py
==================
Model 4 for Group 26 NLP Project: the final system, and the table that compares everything.

RESEARCH QUESTION (short): does a highly tuned language model that reads little text
beat a classical classifier that reads all the text?

WHAT MODEL 4 IS:
    Model 3 (Bio_ClinicalBERT fine-tuned on the description only) plus the first keyword.
    The first keyword of a record is the specialty itself (100% of the training records
    that have keywords), so it is added to the standardised scores of Model 3 with a weight
    that no score gap can outvote (model_utils.add_first_keyword). The keyword decides when
    it exists; Model 3 decides for the records without keywords. The weight is fixed in
    advance, not tuned. No copy-aware decoding and no other post-processing is used.

WHAT THIS SCRIPT DOES (no training: it reads the saved held-out scores):
    - measures Model 3 and Model 4 on the held-out records (results/m3.json, m4.json)
    - measures Model 4 by kind of record (with / without keywords)
    - measures, for information, the same keyword rule applied to Model 2 (ablation, not a
      model): it shows how much of the gap between Models 2 and 4 is the keyword itself
    - writes results/summary_table.csv with Models 1 to 4

HOW TO RUN (from 26/code/, after M1_baseline_model.py, M2_classical_model.py and M3_bert_model.py):
    python3 M4_final_system.py
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from data_utils import load_train, split_train_heldout, first_keyword_label
from evaluation import summarize, RESULTS_DIR
from model_utils import add_first_keyword, KEYWORD_WEIGHT


def main():
    print("=" * 60)
    print("MODEL 4 — Model 3 + first keyword, and the comparison of all models")
    print("=" * 60)

    _, heldout_df = split_train_heldout(load_train())
    with open(os.path.join(RESULTS_DIR, 'm2_config.json'), encoding='utf-8') as f:
        classes = np.array(json.load(f)['classes'])
    m2_scores = np.load(os.path.join(RESULTS_DIR, 'm2_heldout_scores.npy'))
    m3_path = os.path.join(RESULTS_DIR, 'm3_heldout_scores.npy')
    assert os.path.exists(m3_path), "run M3_bert_model.py first (about 50 minutes)"
    m3_scores = np.load(m3_path)
    assert m2_scores.shape == m3_scores.shape == (len(heldout_df), len(classes))
    print(f"  Held-out records: {len(heldout_df)} | first-keyword weight: {KEYWORD_WEIGHT} (fixed in advance)\n")

    # ── Model 3 (language model alone) and Model 4 (+ first keyword) ──
    m3 = summarize('m3', heldout_df, classes[m3_scores.argmax(1)], verbose=False,
                   config={'model': 'Bio_ClinicalBERT, description only', 'scores': 'm3_heldout_scores.npy'})
    m4_scores = add_first_keyword(m3_scores, classes, heldout_df)
    m4_pred = classes[m4_scores.argmax(1)]
    m4 = summarize('m4', heldout_df, m4_pred, per_class_table=True, verbose=True, predictions=True,
                   config={'model': 'Model 3 + first keyword', 'keyword_weight': KEYWORD_WEIGHT})

    # ── Ablation: the same keyword rule on Model 2 (information only, not saved) ──
    m2_kw = summarize('m2_first_keyword', heldout_df, classes[add_first_keyword(m2_scores, classes, heldout_df).argmax(1)],
                      verbose=False, save=False)

    # ── Model 4 by kind of record ──
    has_kw = (first_keyword_label(heldout_df) != 'none').values
    truth = heldout_df['medical_specialty'].values
    print("  Model 4 by kind of record (held-out):")
    for flag, label in [(True, 'with a specialty as first keyword'), (False, 'without keywords')]:
        sel = has_kw == flag
        print(f"    {label:<36} {sel.sum():>4} records  accuracy {accuracy_score(truth[sel], m4_pred[sel]) * 100:5.1f}%")

    # ── One table with every model, for the paper ──
    rows = []
    for label, name in [('M1 baseline', 'm1_baseline_original'), ('M2 classical, all fields', 'm2'),
                        ('M3 language model, description', 'm3'), ('M4 = M3 + first keyword', 'm4')]:
        path = os.path.join(RESULTS_DIR, f'{name}.json')
        assert os.path.exists(path), f"missing {path}: run the script of that model first"
        with open(path, encoding='utf-8') as f:
            r = json.load(f)
        rows.append({'model': label, 'accuracy_%': round(r['accuracy'] * 100, 2), 'macro_f1_%': round(r['macro_f1'] * 100, 2),
                     'lowest_class_f1_%': round(r['lowest_class_f1'] * 100, 2), 'lowest_class': r['lowest_class']})
    rows.append({'model': 'ablation: M2 + first keyword', 'accuracy_%': round(m2_kw['accuracy'] * 100, 2),
                 'macro_f1_%': round(m2_kw['macro_f1'] * 100, 2), 'lowest_class_f1_%': round(m2_kw['lowest_class_f1'] * 100, 2),
                 'lowest_class': m2_kw['lowest_class']})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS_DIR, 'summary_table.csv'), index=False)
    print("\n" + table.to_string(index=False))
    print("\n  Saved results/summary_table.csv")


if __name__ == '__main__':
    main()
