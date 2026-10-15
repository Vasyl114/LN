"""
M4_final_system.py: final system, Model 3 plus the first keyword.

Model:   the first keyword of a record is the specialty itself (in 100% of the training records that have
         keywords). That specialty is added to M3's standardised scores with a fixed weight that no score
         gap can outvote, so the keyword decides when it exists and M3 decides otherwise. There is no
         training: M4 reads the scores saved by M3_bert_model.py.
Modes:   (no flag)  evaluate on the 534 held-out records
                    reads   results/m3_heldout_scores.npy, m2_heldout_scores.npy (comparison row only)
                    writes  results/m4.json, m4_heldout_predictions.csv, m2_first_keyword.json
         --test     predict the 398 test records
                    reads   results/m3_test_scores.npy
                    writes  ../results.txt
Run:     python3 M4_final_system.py [--test] [--output PATH]

The same rule applied to M2 (m2_first_keyword.json) is a comparison, not a model: it shows how much of the
gap between M2 and M4 is the keyword itself. The test file is only read to predict; it has no labels.
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from data_utils import (load_train, load_test, split_train_heldout, first_keyword_label,
                        VALID_LABELS, TEST_PATH)
from model_utils import summarize, load_scores, write_results_txt, RESULTS_DIR

# Fixed in advance, not tuned: standardised scores of 12 classes differ by at most about 4.9,
# so a weight of 5 cannot be outvoted by the language model.
KEYWORD_WEIGHT = 5.0
CLASSES = np.array(sorted(VALID_LABELS))   # column order of every score table


def standardize_rows(scores):
    """Rescale every record's scores to mean 0 and standard deviation 1 across the classes,
    so that scores of different models can be added."""
    scores = np.asarray(scores, dtype=float)
    return (scores - scores.mean(axis=1, keepdims=True)) / (scores.std(axis=1, keepdims=True) + 1e-9)


def add_first_keyword(scores, classes, df, weight=KEYWORD_WEIGHT):
    """
    Model 4: standardised scores plus `weight` on the specialty named by the first
    keyword of each record (nothing is added for records without keywords).
    Returns the new score array (rows of df x classes).
    """
    out = standardize_rows(scores)
    keyword = first_keyword_label(df).values
    for j, label in enumerate(classes):
        out[:, j] += weight * (keyword == label)
    return out


def evaluate_heldout():
    print("=" * 60)
    print("MODEL 4 — Model 3 + first keyword, measured on the held-out records")
    print("=" * 60)

    _, heldout_df = split_train_heldout(load_train())
    with open(os.path.join(RESULTS_DIR, 'm2_config.json'), encoding='utf-8') as f:
        assert json.load(f)['classes'] == CLASSES.tolist(), "class order of Model 2 differs"
    m2_scores = load_scores('m2_heldout_scores', 'M2_classical_model.py')
    m3_scores = load_scores('m3_heldout_scores', 'M3_bert_model.py (about 50 minutes)')
    assert m2_scores.shape == m3_scores.shape == (len(heldout_df), len(CLASSES))
    print(f"  Held-out records: {len(heldout_df)} | first-keyword weight: {KEYWORD_WEIGHT} (fixed in advance)\n")

    # ── Model 4 = Model 3 + first keyword ──
    m4_pred = CLASSES[add_first_keyword(m3_scores, CLASSES, heldout_df).argmax(1)]
    summarize('m4', heldout_df, m4_pred, per_class_table=True, verbose=True, predictions=True,
              config={'model': 'Model 3 + first keyword', 'keyword_weight': KEYWORD_WEIGHT})

    # ── Comparison row: the same rule on Model 2 (information only, not a model) ──
    m2_pred = CLASSES[add_first_keyword(m2_scores, CLASSES, heldout_df).argmax(1)]
    summarize('m2_first_keyword', heldout_df, m2_pred, verbose=False,
              config={'model': 'Model 2 + first keyword (comparison row)', 'keyword_weight': KEYWORD_WEIGHT})

    # ── Model 4 by kind of record ──
    has_kw = (first_keyword_label(heldout_df) != 'none').values
    truth = heldout_df['medical_specialty'].values
    print("  Model 4 by kind of record (held-out):")
    for flag, label in [(True, 'with a specialty as first keyword'), (False, 'without keywords')]:
        sel = has_kw == flag
        accuracy = accuracy_score(truth[sel], m4_pred[sel]) * 100
        print(f"    {label:<36} {sel.sum():>4} records  accuracy {accuracy:5.1f}%")
    print("\n  Saved results/m4.json, m4_heldout_predictions.csv, m2_first_keyword.json")


def write_results(output):
    """M4 on the test file (M3 trained on all records + first keyword) -> results.txt."""
    print("=" * 60)
    print("RESULTS — Model 4 (Model 3 + first keyword), trained on the entire training data")
    print("=" * 60)

    scores = load_scores('m3_test_scores', 'M3_bert_model.py --test (about 55 minutes)')
    test_df = load_test()
    assert scores.shape == (len(test_df), len(CLASSES)), "the scores do not match the test file"
    print(f"  Test records : {len(test_df)} (read from {TEST_PATH})")
    print(f"  Keyword weight: {KEYWORD_WEIGHT} (fixed in advance)")

    predictions = CLASSES[add_first_keyword(scores, CLASSES, test_df).argmax(axis=1)]
    by_model_3 = CLASSES[scores.argmax(axis=1)]
    output = write_results_txt(predictions, output, len(test_df))
    print(f"\n  Written: {output}")
    print(f"  {len(predictions)} lines, every line a valid label, no header, no blank line")
    print("\n  Predicted labels:")
    for label, n in pd.Series(predictions).value_counts().items():
        print(f"    {label:<26} {n:>4} ({n / len(predictions) * 100:4.1f}%)")

    # ── Information: how much each part decides ──
    has_kw = (first_keyword_label(test_df) != 'none').values
    print(f"\n  Test records decided by the first keyword      : {has_kw.sum()} of {len(test_df)}")
    print(f"  Test records decided by Model 3 (no keyword)   : {(~has_kw).sum()} of {len(test_df)}")
    print(f"  Records where the keyword changed Model 3's answer: {(predictions != by_model_3).sum()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true',
                        help='predict the test file and write results.txt (default: evaluate on the held-out records)')
    parser.add_argument('--output', default='../results.txt', help='where --test writes the labels')
    args = parser.parse_args()
    write_results(args.output) if args.test else evaluate_heldout()


if __name__ == '__main__':
    main()
