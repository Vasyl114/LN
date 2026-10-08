"""
evaluation.py
=============
Shared evaluation for Group 26 NLP Project: every model reports its held-out
results through summarize(), so all models are measured in exactly the same way.

WHAT IT REPORTS (on the held-out records of data_utils.split_train_heldout):
    - accuracy
    - macro-F1: the mean of the 12 per-class F1 scores. Unlike accuracy it does not
      reward ignoring the small classes, so it is the main metric for this imbalanced data
    - per-class F1 and the lowest class F1
    - the two conditions of the automatic evaluation of the project statement

WHAT IT SAVES (folder results/, next to the scripts):
    <name>.json                       metrics and the configuration of the run
    <name>_heldout_predictions.csv    record_id, true label, predicted label
                                      (the source for the confusion matrix and the
                                       misclassified examples of the paper)
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

RESULTS_DIR  = 'results'
MIN_ACCURACY = 0.49   # accuracy must be above this value (project statement, Section 4)
MIN_CLASS_F1 = 0.25   # no per-class F1 may fall below this value


def summarize(name, heldout_df, y_pred, config=None, train_labels=None,
              verbose=True, per_class_table=False, save=True):
    """
    Measure y_pred against the labels of heldout_df, print a summary and save it.
    train_labels (optional): labels of the training records, used only to print
    the accuracy of always answering the most frequent training class.
    Returns the metrics as a dictionary.
    """
    y_true = heldout_df['medical_specialty'].reset_index(drop=True)
    y_pred = pd.Series(np.asarray(y_pred)).reset_index(drop=True)

    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    classes = sorted(y_true.unique())
    class_f1 = {c: report[c]['f1-score'] for c in classes}
    worst = min(class_f1, key=class_f1.get)

    metrics = {
        'name': name,
        'n_heldout': int(len(y_true)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'lowest_class_f1': float(class_f1[worst]),
        'lowest_class': worst,
        'classes_below_min_f1': int(sum(f < MIN_CLASS_F1 for f in class_f1.values())),
        'classes_never_predicted': int(sum(report[c]['recall'] == 0 for c in classes)),
        'per_class_f1': {c: float(f) for c, f in class_f1.items()},
        'automatic_evaluation': {
            'accuracy_above_49': bool(accuracy_score(y_true, y_pred) > MIN_ACCURACY),
            'no_class_f1_below_25': bool(class_f1[worst] >= MIN_CLASS_F1),
        },
        'config': config or {},
    }

    if verbose:
        if per_class_table:
            print(classification_report(y_true, y_pred, zero_division=0, digits=3))
        print(f"  [{name}]")
        print(f"    Accuracy              : {metrics['accuracy'] * 100:.2f}%")
        print(f"    Macro F1              : {metrics['macro_f1'] * 100:.2f}%")
        print(f"    Lowest class F1       : {metrics['lowest_class_f1'] * 100:.2f}% ({worst})")
        print(f"    Classes with F1 < {MIN_CLASS_F1 * 100:.0f}%  : "
              f"{metrics['classes_below_min_f1']} of {len(classes)}")
        print(f"    Classes never predicted: {metrics['classes_never_predicted']} of {len(classes)}")
        if train_labels is not None:
            majority = pd.Series(train_labels).value_counts().idxmax()
            print(f"    Always answering '{majority}': {(y_true == majority).mean() * 100:.2f}% (reference)")
        ok = metrics['automatic_evaluation']
        print(f"    Automatic evaluation (measured on held-out): accuracy > 49% "
              f"{'YES' if ok['accuracy_above_49'] else 'NO'} | no class F1 < 25% "
              f"{'YES' if ok['no_class_f1_below_25'] else 'NO'}")

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, f'{name}.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        pd.DataFrame({'record_id': heldout_df.index, 'true': y_true.values, 'pred': y_pred.values}) \
            .to_csv(os.path.join(RESULTS_DIR, f'{name}_heldout_predictions.csv'), index=False)

    return metrics
