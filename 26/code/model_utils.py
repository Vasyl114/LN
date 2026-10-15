"""
model_utils.py: code shared by the model scripts that is not a model itself.

Functions:
    summarize                  the one evaluation of every model: accuracy, macro-F1, per-class F1, lowest
                               class F1 and the two conditions of the automatic evaluation (accuracy > 49%,
                               no class F1 < 25%); prints them and saves results/<name>.json
    save_scores, load_scores   score tables (one row per record, one column per specialty in alphabetical
                               order) that link the scripts
    write_results_txt          write and check the submission file
    set_seed, batches, bucketed_batches, predict_scores
                               helpers of M3 (torch is imported inside them, so M1 and M2 do not need it)

Macro-F1 is the mean of the 12 per-class F1 scores; unlike accuracy it does not reward ignoring small classes.
"""

import json
import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

from data_utils import VALID_LABELS

RESULTS_DIR  = 'results'
MIN_ACCURACY = 0.49   # accuracy must be above this value (project statement, Section 4)
MIN_CLASS_F1 = 0.25   # no per-class F1 may fall below this value


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────
def summarize(name, heldout_df, y_pred, config=None, train_labels=None,
              verbose=True, per_class_table=False, save=True, predictions=False):
    """
    Measure y_pred against the labels of heldout_df, print a summary and save it.
    Saves results/<name>.json and, with predictions=True, results/<name>_heldout_predictions.csv
    (record id, true label, predicted label). train_labels (optional) is only used to print the
    accuracy of always answering the most frequent training class. Returns the metrics as a dict.
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
        if predictions:
            pd.DataFrame({'record_id': heldout_df.index, 'true': y_true.values, 'pred': y_pred.values}) \
                .to_csv(os.path.join(RESULTS_DIR, f'{name}_heldout_predictions.csv'), index=False)

    return metrics


# ─────────────────────────────────────────────────────────────
# Score tables (the link between the scripts)
# ─────────────────────────────────────────────────────────────
def save_scores(name, scores):
    """Save a score table as results/<name>.npy."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(os.path.join(RESULTS_DIR, f'{name}.npy'), scores)


def load_scores(name, produced_by):
    """Load results/<name>.npy; if it is missing, say which script produces it."""
    path = os.path.join(RESULTS_DIR, f'{name}.npy')
    assert os.path.exists(path), f"missing {path}: run {produced_by} first"
    return np.load(path)


# ─────────────────────────────────────────────────────────────
# The submission file
# ─────────────────────────────────────────────────────────────
def write_results_txt(predictions, output, n_records):
    """Write one label per line (no header) and check the file. Returns the absolute path."""
    output = os.path.abspath(output)
    with open(output, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(predictions) + '\n')
    with open(output, encoding='utf-8') as f:
        lines = f.read().split('\n')
    assert lines[-1] == '' and len(lines) - 1 == n_records, "wrong number of lines"
    assert all(label in VALID_LABELS for label in lines[:-1]), "a line is not a valid label"
    return output


# ─────────────────────────────────────────────────────────────
# Helpers of M3 (torch is imported only when they run)
# ─────────────────────────────────────────────────────────────
def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def batches(texts, size, order=None):
    order = np.arange(len(texts)) if order is None else order
    for start in range(0, len(order), size):
        idx = order[start:start + size]
        yield idx, [texts[i] for i in idx]


def bucketed_batches(lengths, batch_size, bucket, seed):
    """Index batches of similar-length records: shuffle, cut into groups of `bucket`
    batches, sort each group by length, cut into batches, shuffle the batch order."""
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(lengths))
    group = batch_size * bucket
    out = []
    for start in range(0, len(order), group):
        chunk = order[start:start + group]
        chunk = chunk[np.argsort([lengths[i] for i in chunk], kind='stable')]
        out += [chunk[i:i + batch_size] for i in range(0, len(chunk), batch_size)]
    rng.shuffle(out)
    return out


def predict_scores(model, tokenizer, texts, max_length):
    """Log-probabilities (records x 12 classes) in alphabetical order of the specialties."""
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for _, chunk in batches(texts, 64):
            enc = tokenizer(chunk, truncation=True, max_length=max_length, padding=True, return_tensors='pt')
            out.append(torch.log_softmax(model(**enc).logits, dim=-1).numpy())
    return np.concatenate(out)
