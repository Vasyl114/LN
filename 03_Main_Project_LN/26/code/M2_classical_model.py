"""
M2_classical_model.py: classical model, TF-IDF + linear SVM reading all the fields.

Model:   description, sample name, transcription and keywords are joined into one text; TF-IDF features
         (sublinear tf) feed a class-balanced linear SVM. No extra feature, no post-processing.
         Linear models on TF-IDF are the standard for labelling long texts by topic; the linear SVM matched
         logistic regression in cross-validation in a tenth of the time; balanced class weights counter the
         imbalance without changing the data.
Steps:   1. 5-fold cross-validation inside the training portion chooses the n-gram range and C by macro-F1
            (original records only, so that augmented copies cannot leak between folds).
         2. Held-out results for 5 field combinations x 2 training files (information for the paper).
         3. Saves the main model: all fields, augmented training file.
Reads:   train.csv, train_augmented.csv.
Writes:  results/m2.json, m2_config.json, m2_heldout_scores.npy, m2_input_ablation.csv.
Run:     python3 M2_classical_model.py        (about 10 min)
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from data_utils import load_train, split_train_heldout, remove_heldout, RANDOM_SEED
from model_utils import summarize, save_scores, RESULTS_DIR

warnings.filterwarnings('ignore')

# Fields joined into the text the model reads (D description, N sample name, T transcription, K keywords)
FIELDS = {
    'D':    ['description'],
    'DN':   ['description', 'sample_name'],
    'DNK':  ['description', 'sample_name', 'keywords'],
    'DNT':  ['description', 'sample_name', 'transcription'],
    'DNTK': ['description', 'sample_name', 'transcription', 'keywords'],
}

CLASSICAL_CONFIG = {
    'ngram_range': (1, 2),   # words and pairs of consecutive words
    'min_df': 2,             # ignore terms found in fewer than 2 records
    'max_features': 150000,  # vocabulary limit
    'sublinear_tf': True,    # use 1 + log(tf)
    'C': 0.5,                # SVM regularisation: larger = fits the training data more closely
}

CV_FOLDS = 5
GRID = [(ngram, C) for ngram in [(1, 1), (1, 2)] for C in [0.1, 0.5, 2.0, 10.0, 50.0, 250.0]]
MAIN_FIELDS = 'DNTK'        # fields read by the main model
MAIN_TRAIN = 'augmented'    # training file of the main model


def join_fields(df, fields):
    """Join the chosen fields of every record into one text (empty fields are skipped)."""
    parts = [df[f].fillna('') for f in FIELDS[fields]]
    return parts[0].str.cat(parts[1:], sep=' ') if len(parts) > 1 else parts[0]


def fit_scores(train_df, test_df, fields='DNTK', config=None):
    """
    THE MODEL: train on train_df, return (classes, scores for test_df).
    scores has one row per record of test_df and one column per class (order of `classes`);
    a higher score means a more likely class.
    """
    cfg = {**CLASSICAL_CONFIG, **(config or {})}
    vectorizer = TfidfVectorizer(ngram_range=cfg['ngram_range'], min_df=cfg['min_df'],
                                 max_features=cfg['max_features'], sublinear_tf=cfg['sublinear_tf'])
    X_train = vectorizer.fit_transform(join_fields(train_df, fields))
    X_test = vectorizer.transform(join_fields(test_df, fields))
    model = LinearSVC(C=cfg['C'], class_weight='balanced', random_state=RANDOM_SEED)
    model.fit(X_train, train_df['medical_specialty'])
    return model.classes_, model.decision_function(X_test)


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

    # ── 2. Held-out results: fields read x training file ──
    print("STEP 2: held-out results (accuracy / macro-F1 / lowest class F1, in %)")
    print(f"  {'fields':<7} {'training file':<10} {'accuracy':>9} {'macro-F1':>9} {'min F1':>7}")
    rows = []
    for fields in ['D', 'DN', 'DNK', 'DNT', 'DNTK']:
        for tname, tdf in train_sets.items():
            classes, scores = fit_scores(tdf, heldout_df, fields, chosen)
            pred = classes[np.asarray(scores).argmax(axis=1)]
            m = summarize('tmp', heldout_df, pred, verbose=False, save=False)
            acc, f1, low = m['accuracy'] * 100, m['macro_f1'] * 100, m['lowest_class_f1'] * 100
            print(f"  {fields:<7} {tname:<10} {acc:9.1f} {f1:9.1f} {low:7.1f}")
            rows.append({'fields': fields, 'training_file': tname, 'accuracy_%': round(acc, 2),
                         'macro_f1_%': round(f1, 2), 'lowest_class_f1_%': round(low, 2)})
            if fields == MAIN_FIELDS and tname == MAIN_TRAIN:
                main_classes, main_scores = classes, scores
    print("  (D = description, N = sample name, T = transcription, K = keywords)")

    # ── 3. Save the main model (all fields, augmented training file) ──
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, 'm2_input_ablation.csv'), index=False)
    save_scores('m2_heldout_scores', main_scores)
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
