"""
07_generate_results.py
======================
Writes results.txt: the predicted specialty for every record of the clean test file
(../../StudentsPack/test_no_labels.csv), one label per line, no header, line N = record N.

THE SYSTEM: Model 4 = Model 3 + first keyword (see M4_final_system.py).
    Model 3 is Bio_ClinicalBERT fine-tuned on the description only, here trained on the
    ENTIRE training data (all 2,669 original records and the generated ones in
    train_augmented.csv): no part is held out any more, as the teachers allow.
    The first keyword of a test record, when it names a specialty, is added to the
    standardised scores of Model 3 with a weight that cannot be outvoted; records without
    keywords are decided by Model 3 alone.

HOW TO RUN (from 26/code/):
    python3 M3_bert_model.py --full        # about 55 minutes, saves results/m3_test_scores.npy
    python3 07_generate_results.py         # seconds, writes ../results.txt

The test file is only read to produce predictions. No label of it exists or is used.
"""

import argparse
import json
import os
import numpy as np
import pandas as pd

from data_utils import load_test, first_keyword_label, VALID_LABELS, TEST_PATH
from model_utils import add_first_keyword, KEYWORD_WEIGHT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='../results.txt')
    args = parser.parse_args()

    print("=" * 60)
    print("RESULTS — Model 4 (Model 3 + first keyword), trained on the entire training data")
    print("=" * 60)

    path = 'results/m3_test_scores.npy'
    assert os.path.exists(path), "run M3_bert_model.py --full first (about 55 minutes)"
    scores = np.load(path)
    with open('results/m2_config.json', encoding='utf-8') as f:
        classes = np.array(json.load(f)['classes'])       # alphabetical, the column order of Model 3
    assert classes.tolist() == sorted(VALID_LABELS)
    test_df = load_test()
    assert scores.shape == (len(test_df), len(classes)), "the scores do not match the test file"
    print(f"  Test records : {len(test_df)} (read from {TEST_PATH})")
    print(f"  Keyword weight: {KEYWORD_WEIGHT} (fixed in advance)")

    # ── Predict ──
    final = add_first_keyword(scores, classes, test_df)
    predictions = classes[final.argmax(axis=1)]
    by_model_3 = classes[scores.argmax(axis=1)]

    # ── Write: one label per line, no header ──
    output = os.path.abspath(args.output)
    with open(output, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(predictions) + '\n')

    # ── Check the file ──
    with open(output, encoding='utf-8') as f:
        lines = f.read().split('\n')
    assert lines[-1] == '' and len(lines) - 1 == len(test_df), "wrong number of lines"
    assert all(label in VALID_LABELS for label in lines[:-1]), "a line is not a valid label"
    print(f"\n  Written: {output}")
    print(f"  {len(lines) - 1} lines, every line a valid label, no header, no blank line")
    print("\n  Predicted labels:")
    for label, n in pd.Series(predictions).value_counts().items():
        print(f"    {label:<26} {n:>4} ({n / len(predictions) * 100:4.1f}%)")

    # ── Information: how much each part decides ──
    has_kw = (first_keyword_label(test_df) != 'none').values
    print(f"\n  Test records decided by the first keyword      : {has_kw.sum()} of {len(test_df)}")
    print(f"  Test records decided by Model 3 (no keyword)   : {(~has_kw).sum()} of {len(test_df)}")
    print(f"  Records where the keyword changed Model 3's answer: {(predictions != by_model_3).sum()}")


if __name__ == '__main__':
    main()
