"""
04_model_classical.py
=====================
Model 2 for Group 26 NLP Project: the classical side of the research question.

RESEARCH QUESTION (short): does a clinical language model that reads only the short
description beat a classical classifier that reads all the text?

WHAT THIS IS:
    TF-IDF + linear SVM (see model_utils.py for why this model), reading the
    description, the sample name and the transcription. The keywords are NOT part of
    the text input on purpose: their first word is the label itself, and a model that
    reads it would win the comparison for a reason unrelated to the text. The keyword
    feature is added separately in 07_final_system.py (layer L2).

WHAT THIS DOES:
    1. Chooses the n-gram range and the SVM regularisation C by 5-fold cross-validation
       INSIDE the training portion (the held-out records are never used to choose).
       Criterion: macro-F1 with copy-aware decoding (the mode the final system runs in).
       Only original records are used for this: the augmented records are variants of
       the same notes, so they would leak between folds.
    2. Trains on the training portion and measures the held-out records, for three
       inputs (description / + sample name / + transcription) and two training files
       (original / augmented), at two layers:
         L0  the model alone
         L1  + copy-aware decoding: a note listed in the training data under label A
             is not the same note listed again under A (the dataset has no repeated
             (label, note) pair), so A is removed from the candidates of its copy.
    3. Saves the held-out scores of the main Model 2 (description + name + transcription,
       augmented training file) for the fusion in 07_final_system.py.

HOW TO RUN (from 26/code/, takes about 2 minutes):
    python3 04_model_classical.py
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

from data_utils import (load_train, split_train_heldout, remove_heldout,
                        seen_labels, copy_aware_scores, RANDOM_SEED)
from evaluation import summarize, RESULTS_DIR
from model_utils import fit_scores, CLASSICAL_CONFIG

warnings.filterwarnings('ignore')

CV_FOLDS = 5
GRID = [(ngram, C) for ngram in [(1, 1), (1, 2)] for C in [0.1, 0.5, 2.0]]
MAIN_FIELDS = 'DNT'
MAIN_TRAIN = 'augmented'


def decode(scores, classes, test_df, train_df, layer):
    """Predicted labels from scores; at layer L1 the labels of training copies are removed first."""
    if layer == 'L1':
        scores = copy_aware_scores(scores, classes, test_df, seen_labels(train_df))
    return classes[np.asarray(scores).argmax(axis=1)]


def main():
    print("=" * 60)
    print("MODEL 2 — TF-IDF + linear SVM (classical side of the research question)")
    print("=" * 60)

    original = load_train()
    train_df, heldout_df = split_train_heldout(original)
    augmented = remove_heldout(load_train('train_augmented.csv'), heldout_df)
    train_sets = {'original': train_df, 'augmented': augmented}
    print(f"  Training portion (original) : {len(train_df)} records")
    print(f"  Training portion (augmented): {len(augmented)} records")
    print(f"  Held-out                    : {len(heldout_df)} records (never used to choose anything)\n")

    # ── 1. Hyperparameters by cross-validation inside the training portion ──
    print(f"STEP 1: {CV_FOLDS}-fold cross-validation inside the training portion, input = {MAIN_FIELDS}")
    print(f"  {'n-grams':<9} {'C':<5} {'L0 acc':>7} {'L0 F1':>7}   {'L1 acc':>7} {'L1 F1':>7}")
    folds = list(StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
                 .split(train_df, train_df['medical_specialty']))
    best, best_f1 = None, -1
    for ngram, C in GRID:
        cfg = {'ngram_range': ngram, 'C': C}
        result = {'L0': [], 'L1': []}
        for a, b in folds:
            fold_train, fold_val = train_df.iloc[a], train_df.iloc[b]
            classes, scores = fit_scores(fold_train, fold_val, MAIN_FIELDS, cfg)
            for layer in result:
                pred = decode(scores, classes, fold_val, fold_train, layer)
                result[layer].append((accuracy_score(fold_val['medical_specialty'], pred),
                                      f1_score(fold_val['medical_specialty'], pred, average='macro', zero_division=0)))
        m = {layer: np.mean(result[layer], axis=0) * 100 for layer in result}
        print(f"  {str(ngram):<9} {C:<5} {m['L0'][0]:7.1f} {m['L0'][1]:7.1f}   {m['L1'][0]:7.1f} {m['L1'][1]:7.1f}")
        if m['L1'][1] > best_f1:
            best, best_f1 = cfg, m['L1'][1]
    chosen = {**CLASSICAL_CONFIG, **best}
    print(f"\n  Selected (highest L1 macro-F1): n-grams {best['ngram_range']}, C = {best['C']}\n")

    # ── 2. Held-out: input fields x training file x layer ──
    print("STEP 2: held-out results (accuracy / macro-F1 / lowest class F1, in %)")
    print(f"  {'input':<6} {'training file':<10} {'L0 (model alone)':>26}   {'L1 (+ copy-aware)':>26}")
    for fields in ['D', 'DN', 'DNT']:
        for tname, tdf in train_sets.items():
            classes, scores = fit_scores(tdf, heldout_df, fields, chosen)
            row = []
            for layer in ['L0', 'L1']:
                # copy labels come from the ORIGINAL training portion only
                pred = decode(scores, classes, heldout_df, train_df, layer)
                m = summarize(f"m2_{fields.lower()}_{tname}_{layer}", heldout_df, pred, verbose=False,
                              config={'model': 'TF-IDF + LinearSVC', 'fields': fields, 'training_file': tname,
                                      'layer': layer, **{k: str(v) for k, v in chosen.items()}})
                row.append(f"{m['accuracy']*100:6.1f} {m['macro_f1']*100:6.1f} {m['lowest_class_f1']*100:6.1f}")
            print(f"  {fields:<6} {tname:<10} {row[0]:>26}   {row[1]:>26}")
            if fields == MAIN_FIELDS and tname == MAIN_TRAIN:
                main_classes, main_scores = classes, scores

    # ── 3. Save the main model's held-out scores ──
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(os.path.join(RESULTS_DIR, 'm2_heldout_scores.npy'), main_scores)
    with open(os.path.join(RESULTS_DIR, 'm2_config.json'), 'w', encoding='utf-8') as f:
        json.dump({'classes': list(main_classes), 'fields': MAIN_FIELDS, 'training_file': MAIN_TRAIN,
                   'ngram_range': list(best['ngram_range']), 'C': best['C']}, f, indent=2)

    print(f"\nMain Model 2 = {MAIN_FIELDS}, {MAIN_TRAIN} training file. Full report:")
    for layer in ['L0', 'L1']:
        pred = decode(main_scores, main_classes, heldout_df, train_df, layer)
        summarize(f"m2_main_{layer}", heldout_df, pred, per_class_table=(layer == 'L1'),
                  train_labels=augmented['medical_specialty'],
                  config={'model': 'TF-IDF + LinearSVC', 'fields': MAIN_FIELDS, 'training_file': MAIN_TRAIN,
                          'layer': layer, **{k: str(v) for k, v in chosen.items()}})


if __name__ == '__main__':
    main()
