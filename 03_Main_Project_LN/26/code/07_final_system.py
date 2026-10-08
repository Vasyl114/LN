"""
07_final_system.py
==================
Model 5 for Group 26 NLP Project: the final system, and the table that compares everything.

THE IDEA (layers):
    The dataset has two properties that a text model cannot see in the text, and the
    system uses them in layers, so each one can be measured separately:
      L0  the model alone (text only)
      L1  + copy-aware decoding: the dataset lists a note once per specialty, never twice
          under the same one, so the labels that the training data already carries for
          the same note are removed from the candidates of its copy
      L2  + first-keyword feature: the first keyword of a record is the specialty itself
          (every record that has keywords). It gets its own feature columns, so the
          classifier learns to trust it; no hand-written rule is used.
    The fair answer to the research question (language model on the description vs
    classical model on all the text) is read at L0 and L1. L2 is the submitted system.

THE CANDIDATES:
    Model 2   TF-IDF + linear SVM on description + sample name + transcription
    Model 3   Bio_ClinicalBERT on the description (needs 05_model_bert.py to have run)
    Fusion    Model 2 + Model 3: each record's scores are rescaled to mean 0 / std 1 over
              the 12 classes and added with EQUAL weight. The weight is fixed in advance,
              not tuned, because the held-out records are not used to choose anything.

HOW THE FINAL SYSTEM IS CHOSEN:
    Among the L2 candidates, the one with the highest held-out macro-F1 (accuracy breaks
    ties). This is the only time the held-out records are used to choose between models,
    and only between these two candidates.

HOW TO RUN (from 26/code/, after 04_model_classical.py and 05_model_bert.py):
    python3 07_final_system.py
"""

import glob
import json
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

import evaluation
from data_utils import (load_train, split_train_heldout, remove_heldout, seen_labels,
                        copy_aware_scores, note_key, first_keyword_label, RANDOM_SEED)
from evaluation import summarize, RESULTS_DIR
from model_utils import fit_scores, standardize_rows

warnings.filterwarnings('ignore')

KEYWORD_WEIGHTS = [1.0, 3.0, 10.0]   # scale of the first-keyword feature, chosen by cross-validation
CV_FOLDS = 5


def decide(scores, classes, df, seen, copy_aware):
    scores = copy_aware_scores(scores, classes, df, seen) if copy_aware else scores
    return np.asarray(classes)[np.asarray(scores).argmax(axis=1)]


def main():
    print("=" * 60)
    print("MODEL 5 — final system: layers L0 / L1 / L2 and fusion")
    print("=" * 60)

    original = load_train()
    train_df, heldout_df = split_train_heldout(original)
    augmented = remove_heldout(load_train('train_augmented.csv'), heldout_df)
    seen = seen_labels(train_df)      # copy labels: ORIGINAL training portion only
    assert not set(train_df.index) & set(heldout_df.index), "held-out records found in training"

    with open(os.path.join(RESULTS_DIR, 'm2_config.json'), encoding='utf-8') as f:
        m2 = json.load(f)
    cfg = {'ngram_range': tuple(m2['ngram_range']), 'C': m2['C']}
    classes = np.array(m2['classes'])
    m2_scores = np.load(os.path.join(RESULTS_DIR, 'm2_heldout_scores.npy'))
    m3_path = os.path.join(RESULTS_DIR, 'm3_heldout_scores.npy')
    m3_scores = np.load(m3_path) if os.path.exists(m3_path) else None
    if m3_scores is None:
        print("  (no Model 3 scores found: run 05_model_bert.py first for the language model and the fusion)\n")

    # ── Keyword feature weight, chosen by cross-validation inside the training portion ──
    print(f"STEP 1: weight of the first-keyword feature, {CV_FOLDS}-fold cross-validation inside the training portion")
    folds = list(StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
                 .split(train_df, train_df['medical_specialty']))
    best_w, best_f1 = None, -1
    for w in KEYWORD_WEIGHTS:
        f1s = []
        for a, b in folds:
            fold_train, fold_val = train_df.iloc[a], train_df.iloc[b]
            cls, sc = fit_scores(fold_train, fold_val, 'DNT', {**cfg, 'keyword_weight': w}, use_keyword=True)
            pred = decide(sc, cls, fold_val, seen_labels(fold_train), True)
            f1s.append(f1_score(fold_val['medical_specialty'], pred, average='macro', zero_division=0))
        print(f"  weight {w:<5}: L2 macro-F1 {np.mean(f1s) * 100:.1f}")
        if np.mean(f1s) > best_f1:
            best_w, best_f1 = w, np.mean(f1s)
    print(f"  Selected weight: {best_w}\n")

    # ── Model 2 with the keyword feature ──
    cls_kw, m2_kw_scores = fit_scores(augmented, heldout_df, 'DNT', {**cfg, 'keyword_weight': best_w}, use_keyword=True)
    assert list(cls_kw) == list(classes)
    if m3_scores is not None:
        assert m3_scores.shape == m2_scores.shape

    # ── All candidates, all layers ──
    candidates = {'M2': (m2_scores, m2_kw_scores)}
    if m3_scores is not None:
        candidates['M3'] = (m3_scores, None)
        candidates['Fusion'] = (standardize_rows(m2_scores) + standardize_rows(m3_scores),
                                standardize_rows(m2_kw_scores) + standardize_rows(m3_scores))
    results, preds = {}, {}
    print("STEP 2: held-out results (accuracy / macro-F1 / lowest class F1, in %)")
    print(f"  {'model':<8} {'layer':<28} {'acc':>6} {'macro-F1':>9} {'min F1':>7}")
    for name, (plain, with_kw) in candidates.items():
        for layer, scores, copy_aware, text in [('L0', plain, False, 'L0 model alone'),
                                                ('L1', plain, True, 'L1 + copy-aware decoding'),
                                                ('L2', with_kw, True, 'L2 + first-keyword feature')]:
            if scores is None:
                continue
            pred = decide(scores, classes, heldout_df, seen, copy_aware)
            m = summarize(f"m5_{name.lower()}_{layer}", heldout_df, pred, verbose=False,
                          config={'model': name, 'layer': layer, 'keyword_weight': best_w, **{k: str(v) for k, v in cfg.items()}})
            results[(name, layer)], preds[(name, layer)] = m, pred
            print(f"  {name:<8} {text:<28} {m['accuracy'] * 100:6.1f} {m['macro_f1'] * 100:9.1f} {m['lowest_class_f1'] * 100:7.1f}")

    # ── Final system: best L2 candidate by held-out macro-F1 ──
    l2 = [(n, results[(n, 'L2')]) for n in candidates if (n, 'L2') in results]
    final_name, final = max(l2, key=lambda t: (round(t[1]['macro_f1'], 6), t[1]['accuracy']))
    print(f"\nFINAL SYSTEM = {final_name} at layer L2 (highest held-out macro-F1 among {[n for n, _ in l2]})\n")
    final_pred = preds[(final_name, 'L2')]
    summarize('m5_final', heldout_df, final_pred, per_class_table=True, train_labels=augmented['medical_specialty'],
              config={'candidate': final_name, 'layer': 'L2', 'keyword_weight': best_w,
                      'candidates_compared': {n: {'accuracy': r['accuracy'], 'macro_f1': r['macro_f1']} for n, r in l2},
                      **{k: str(v) for k, v in cfg.items()}})

    # ── Where the final system is right or wrong: with / without keywords, with / without a copy in train ──
    has_kw = (first_keyword_label(heldout_df) != 'none').values
    has_copy = note_key(heldout_df).isin(seen.keys()).values
    truth = heldout_df['medical_specialty'].values
    print("  Final system by kind of record (held-out):")
    for kw_flag, copy_flag, label in [(True, True, 'keywords, copy in train'), (True, False, 'keywords, no copy'),
                                      (False, True, 'no keywords, copy in train'), (False, False, 'no keywords, no copy')]:
        sel = (has_kw == kw_flag) & (has_copy == copy_flag)
        if sel.sum():
            print(f"    {label:<28} {sel.sum():>4} records  accuracy {accuracy_score(truth[sel], final_pred[sel]) * 100:5.1f}%")

    # ── One table with every model, for the paper ──
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, 'm*.json'))):
        name = os.path.basename(path)[:-5]
        if name.endswith('_config') or name.endswith('_final'):
            continue
        with open(path, encoding='utf-8') as f:
            r = json.load(f)
        rows.append({'run': name, 'accuracy_%': round(r['accuracy'] * 100, 2), 'macro_f1_%': round(r['macro_f1'] * 100, 2),
                     'lowest_class_f1_%': round(r['lowest_class_f1'] * 100, 2), 'lowest_class': r['lowest_class']})
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, 'summary_table.csv'), index=False)
    print(f"\n  Saved results/summary_table.csv ({len(rows)} runs)")


if __name__ == '__main__':
    main()
