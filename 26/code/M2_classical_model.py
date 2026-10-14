"""
M2_classical_model.py
=====================
Model 2 for Group 26 NLP Project: the classical side of the research question.

RESEARCH QUESTION (short): does a highly tuned language model that reads little text
beat a classical classifier that reads all the text?

WHAT THIS IS:
    TF-IDF + linear SVM (see model_utils.py for why this model), reading ALL the fields
    of a record: description, sample name, transcription and keywords. It is the plain
    classical model: it gets no extra feature and no post-processing.

WHAT THIS DOES:
    1. Chooses the n-gram range and the SVM regularisation C by 5-fold cross-validation
       INSIDE the training portion (the held-out records are never used to choose).
       Criterion: macro-F1. Only original records are used for this: the augmented
       records are variants of the same notes, so they would leak between folds.
    2. Trains on the training portion and measures the held-out records for different
       inputs (which fields are read) and two training files (original / augmented).
       This table is information for the paper; only the main model is saved.
    3. Saves the held-out scores of the main Model 2 (all fields, augmented training
       file) in results/ and its metrics as results/m2.json.

HOW TO RUN (from 26/code/, takes about 10 minutes):
    python3 M2_classical_model.py
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

from data_utils import load_train, split_train_heldout, remove_heldout, RANDOM_SEED
from evaluation import summarize, RESULTS_DIR
from model_utils import fit_scores, CLASSICAL_CONFIG

warnings.filterwarnings('ignore')

CV_FOLDS = 5
GRID = [(ngram, C) for ngram in [(1, 1), (1, 2)] for C in [0.1, 0.5, 2.0, 10.0, 50.0, 250.0]]
MAIN_FIELDS = 'DNTK'
MAIN_TRAIN = 'augmented'


def main():
    print("=" * 60)
    print("MODEL 2 — TF-IDF + linear SVM on all the fields (classical side)")
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
    print(f"  {'n-grams':<9} {'C':<5} {'accuracy':>9} {'macro-F1':>9}")
    folds = list(StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
                 .split(train_df, train_df['medical_specialty']))
    best, best_f1 = None, -1
    for ngram, C in GRID:
        cfg = {'ngram_range': ngram, 'C': C}
        result = []
        for a, b in folds:
            fold_train, fold_val = train_df.iloc[a], train_df.iloc[b]
            classes, scores = fit_scores(fold_train, fold_val, MAIN_FIELDS, cfg)
            pred = classes[np.asarray(scores).argmax(axis=1)]
            result.append((accuracy_score(fold_val['medical_specialty'], pred),
                           f1_score(fold_val['medical_specialty'], pred, average='macro', zero_division=0)))
        m = np.mean(result, axis=0) * 100
        print(f"  {str(ngram):<9} {C:<5} {m[0]:9.1f} {m[1]:9.1f}")
        if m[1] > best_f1:
            best, best_f1 = cfg, m[1]
    chosen = {**CLASSICAL_CONFIG, **best}
    print(f"\n  Selected (highest macro-F1): n-grams {best['ngram_range']}, C = {best['C']}\n")

    # ── 2. Held-out: which fields are read x training file ──
    print("STEP 2: held-out results (accuracy / macro-F1 / lowest class F1, in %)")
    print(f"  {'fields':<7} {'training file':<10} {'accuracy':>9} {'macro-F1':>9} {'min F1':>7}")
    rows = []
    for fields in ['D', 'DN', 'DNK', 'DNT', 'DNTK']:
        for tname, tdf in train_sets.items():
            classes, scores = fit_scores(tdf, heldout_df, fields, chosen)
            pred = classes[np.asarray(scores).argmax(axis=1)]
            m = summarize('tmp', heldout_df, pred, verbose=False, save=False)
            print(f"  {fields:<7} {tname:<10} {m['accuracy']*100:9.1f} {m['macro_f1']*100:9.1f} {m['lowest_class_f1']*100:7.1f}")
            rows.append({'fields': fields, 'training_file': tname, 'accuracy_%': round(m['accuracy']*100, 2),
                         'macro_f1_%': round(m['macro_f1']*100, 2), 'lowest_class_f1_%': round(m['lowest_class_f1']*100, 2)})
            if fields == MAIN_FIELDS and tname == MAIN_TRAIN:
                main_classes, main_scores = classes, scores
    print("  (D = description, N = sample name, T = transcription, K = keywords)")

    # ── 3. Save the main model ──
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, 'm2_input_ablation.csv'), index=False)
    np.save(os.path.join(RESULTS_DIR, 'm2_heldout_scores.npy'), main_scores)
    with open(os.path.join(RESULTS_DIR, 'm2_config.json'), 'w', encoding='utf-8') as f:
        json.dump({'classes': list(main_classes), 'fields': MAIN_FIELDS, 'training_file': MAIN_TRAIN,
                   'ngram_range': list(best['ngram_range']), 'C': best['C']}, f, indent=2)

    print(f"\nMain Model 2 = fields {MAIN_FIELDS}, {MAIN_TRAIN} training file. Full report:")
    pred = main_classes[np.asarray(main_scores).argmax(axis=1)]
    summarize('m2', heldout_df, pred, per_class_table=True, train_labels=augmented['medical_specialty'],
              config={'model': 'TF-IDF + LinearSVC', 'fields': MAIN_FIELDS, 'training_file': MAIN_TRAIN,
                      **{k: str(v) for k, v in chosen.items()}})


if __name__ == '__main__':
    main()
