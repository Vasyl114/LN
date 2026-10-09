"""
08_generate_results.py
======================
Writes results.txt: the predicted specialty for every record of the clean test file
(../../StudentsPack/test_no_labels.csv), one label per line, no header, line N = record N.

THE SYSTEM (default): Model 2 at layer L2, see 07_final_system.py
    TF-IDF + linear SVM on description + sample name + transcription, plus the
    first-keyword feature, plus copy-aware decoding.
    It is trained on the ENTIRE training data (all 2,669 original records and the
    generated ones in train_augmented.csv): no part is held out any more, as the
    teachers allow. Hyperparameters come from results/m2_config.json (chosen by
    cross-validation in 04_model_classical.py) and results/m5_m2_L2.json (keyword weight).
    It needs no neural network and runs in about 30 seconds on any machine.
    The fusion with Bio_ClinicalBERT (94.8% on held-out instead of 93.3%) is not
    used here because reproducing it needs the 50-minute fine-tuning of 05_model_bert.py.

LAYERS (--layer):
    L2  text + first-keyword feature + copy-aware decoding   (default)
    L1  text + copy-aware decoding, the keywords are not used
    L0  text only
    Use L1 or L0 if the teachers do not accept the use of the keywords field or of
    the labels of notes that are also in the training data.

The test file is only read to produce predictions. No label of it exists or is used.

HOW TO RUN (from 26/code/):
    python3 08_generate_results.py                   # writes ../results.txt
    python3 08_generate_results.py --layer L1 --output ../results_L1.txt
"""

import argparse
import json
import os
import warnings
import numpy as np
import pandas as pd

from data_utils import (load_train, load_test, seen_labels, copy_aware_scores, first_keyword_label,
                        note_key, VALID_LABELS, TEST_PATH)
from model_utils import fit_scores, CLASSICAL_CONFIG

warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', choices=['L0', 'L1', 'L2'], default='L2')
    parser.add_argument('--output', default='../results.txt')
    args = parser.parse_args()

    print("=" * 60)
    print(f"RESULTS — Model 2 at layer {args.layer}, trained on the entire training data")
    print("=" * 60)

    # ── Configuration chosen earlier by cross-validation (never by the test file) ──
    cfg = dict(CLASSICAL_CONFIG)
    with open('results/m2_config.json', encoding='utf-8') as f:
        m2 = json.load(f)
    cfg.update({'ngram_range': tuple(m2['ngram_range']), 'C': m2['C']})
    if os.path.exists('results/m5_m2_L2.json'):
        with open('results/m5_m2_L2.json', encoding='utf-8') as f:
            cfg['keyword_weight'] = json.load(f)['config']['keyword_weight']
    print(f"  Configuration: n-grams {cfg['ngram_range']}, C {cfg['C']}, keyword weight {cfg['keyword_weight']}")

    # ── Data ──
    original = load_train()
    train_df = load_train('train_augmented.csv')      # all originals + generated records
    test_df = load_test()
    assert len(train_df) == 5648 and len(original) == 2669, "unexpected training data"
    print(f"  Training records : {len(train_df)} ({len(original)} original + {len(train_df) - len(original)} generated)")
    print(f"  Test records     : {len(test_df)} (read from {TEST_PATH})")

    # ── Train and predict ──
    classes, scores = fit_scores(train_df, test_df, 'DNT', cfg, use_keyword=(args.layer == 'L2'))
    seen = seen_labels(original)                      # labels of notes in the original training data
    if args.layer in ('L1', 'L2'):
        scores = copy_aware_scores(scores, classes, test_df, seen)
    predictions = classes[scores.argmax(axis=1)]

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

    # ── How much each layer decides (information only) ──
    has_kw = (first_keyword_label(test_df) != 'none').sum()
    has_copy = note_key(test_df).isin(seen.keys()).sum()
    print(f"\n  Test records with a specialty as first keyword: {has_kw} of {len(test_df)}")
    print(f"  Test records whose note is also in the training data: {has_copy} of {len(test_df)}")


if __name__ == '__main__':
    main()
