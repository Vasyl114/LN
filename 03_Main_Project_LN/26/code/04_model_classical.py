"""
04_model_classical.py
=====================
Model 2 — Improved classical ML for Group 26 NLP Project.

WHAT THIS IS:
    The "full-text classical" side of the research question:
      - input         : combined clinical record fields (ablation A/B/C below)
      - preprocessing : lowercase only (no stemming, no stop-word removal;
                        the project PDF warns against blindly removing stop words,
                        and medical function words carry signal, e.g. "no", "not")
      - features      : TF-IDF, word n-grams (1,2)
      - classifier    : Logistic Regression with balanced class weights
                        (+ one LinearSVC reference run on the best input)
    This directly tests whether reading more fields beats the
    description-only baseline (01_baseline_model.py).

INPUT VARIANTS (ablation, all measured on the SAME held-out set):
    A : description only                    (baseline input, stronger classifier)
    B : description + sample_name + keywords (full short-text record)
    C : B + transcription (first 400 words; the full note is ~457 words on
        average and 58% of notes occur under more than one label, so this
        variant tests whether the long note helps or adds noise)

HOW IT IS EVALUATED (same protocol as the baseline):
    1. The original records are divided once into training (80%) and held-out
       (20%), stratified, seed 42 (see split_train_heldout in data_utils.py).
    2. The model trains on the given file MINUS the held-out records
       (so training on train_augmented.csv never sees the evaluation rows).
    3. Accuracy, macro-F1 and per-class F1 are measured on the held-out
       original records. test_no_labels.csv is not used here.

HOW TO RUN (CPU only, a few minutes):
    From the 26/code/ directory:
        python3 04_model_classical.py                    # train_augmented.csv (default)
        python3 04_model_classical.py train_augmented.csv
        python3 04_model_classical.py ../../StudentsPack/train.csv

OUTPUT (printed + saved next to this script, all NEW files):
    04_classical_results.csv  — one row per variant (accuracy, macro-F1, ...)

REQUIREMENTS: scikit-learn, pandas, nltk (only for data_utils loading)
"""

import os
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

from data_utils import (
    load_train,
    split_train_heldout,
    remove_heldout,
    TRAIN_PATH,
    RANDOM_SEED,
)

# Same automatic-evaluation gates as the baseline (project statement, Sec. 4)
MIN_ACCURACY = 0.49
MIN_CLASS_F1 = 0.25

# TF-IDF settings (reported in the paper, Sec. 4.1)
TFIDF_PARAMS = dict(
    max_features=30000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    lowercase=True,
    stop_words=None,  # justified: PDF warns against blind stop-word removal
)

# Logistic Regression settings (reported in the paper, Sec. 4.1)
LOGREG_PARAMS = dict(
    C=1.0,
    max_iter=1000,
    class_weight="balanced",
    random_state=RANDOM_SEED,
)

# Transcription is truncated to this many words in variant C
TRANS_MAX_WORDS = 400

DEFAULT_TRAIN = "train_augmented.csv"


def truncate_words(text, max_words):
    """Keep the first max_words whitespace-separated tokens."""
    if not isinstance(text, str):
        return ""
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def build_input(df, variant):
    """Combine record fields into one text string per row."""
    desc = df["description"].fillna("")
    name = df["sample_name"].fillna("")
    kw = df["keywords"].fillna("")
    if variant == "A":
        return desc
    if variant == "B":
        return desc + " " + name + " " + kw
    if variant == "C":
        trans = df["transcription"].fillna("").apply(
            lambda t: truncate_words(t, TRANS_MAX_WORDS)
        )
        return desc + " " + name + " " + kw + " " + trans
    raise ValueError(f"Unknown variant: {variant}")


def run_variant(X_train_raw, y_train, X_held_raw, y_held, variant, classifier="logreg"):
    """Train one variant and return (accuracy, macro_f1, report_dict, y_pred)."""
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_held = vectorizer.transform(X_held_raw)

    if classifier == "logreg":
        clf = LogisticRegression(**LOGREG_PARAMS)
    elif classifier == "svm":
        clf = LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_SEED)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_held)
    acc = accuracy_score(y_held, y_pred)
    macro_f1 = f1_score(y_held, y_pred, average="macro", zero_division=0)
    report = classification_report(y_held, y_pred, zero_division=0, output_dict=True)
    return acc, macro_f1, report, y_pred


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TRAIN
    if not os.path.exists(data_path):
        # Fall back to the original training file (e.g. fresh checkout)
        data_path = TRAIN_PATH
    if not os.path.exists(data_path):
        print(f"Error: could not find {DEFAULT_TRAIN} nor {TRAIN_PATH}.")
        sys.exit(1)

    # ── Data: fixed held-out portion, training on everything else ──
    original_df = load_train()
    _, heldout_df = split_train_heldout(original_df)
    dataset_df = load_train(data_path)
    train_df = remove_heldout(dataset_df, heldout_df)
    y_held = heldout_df["medical_specialty"]

    print("=" * 60)
    print("MODEL 2 — Logistic Regression + TF-IDF on combined fields")
    print("=" * 60)
    print(f"  Training file              : {data_path}")
    print(f"  Records in the file        : {len(dataset_df)}")
    print(f"  Training on                : {len(train_df)} records")
    print(f"  Evaluating on (held-out)   : {len(heldout_df)} original records")
    print(f"  TF-IDF                     : {TFIDF_PARAMS}")
    print(f"  LogisticRegression         : C={LOGREG_PARAMS['C']}, "
          f"max_iter={LOGREG_PARAMS['max_iter']}, "
          f"class_weight={LOGREG_PARAMS['low_weight'] if False else LOGREG_PARAMS['class_weight']}, "
          f"random_state={RANDOM_SEED}")
    print(f"  Transcription truncation   : first {TRANS_MAX_WORDS} words (variant C only)")
    print()

    y_train = train_df["medical_specialty"]
    majority_acc = (y_held == y_train.value_counts().idxmax()).mean()
    print(f"  Reference (always majority class): {majority_acc * 100:.2f}% accuracy\n")

    # ── Ablation: same classifier, three inputs ──
    rows = []
    reports = {}
    for variant in ["A", "B", "C"]:
        X_train_raw = build_input(train_df, variant)
        X_held_raw = build_input(heldout_df, variant)
        acc, macro_f1, report, _ = run_variant(
            X_train_raw, y_train, X_held_raw, y_held, variant
        )
        class_f1 = {
            cls: report[cls]["f1-score"]
            for cls in y_held.unique()
            if cls in report
        }
        worst = min(class_f1, key=class_f1.get)
        n_below = sum(f < MIN_CLASS_F1 for f in class_f1.values())
        rows.append(
            dict(
                variant=variant,
                classifier="logreg",
                accuracy=round(acc, 4),
                macro_f1=round(macro_f1, 4),
                worst_class=worst,
                worst_f1=round(class_f1[worst], 4),
                classes_below_25pct=n_below,
            )
        )
        reports[variant] = report
        print(f"--- Variant {variant} (logreg) ---")
        print(f"  Accuracy            : {acc * 100:.2f}% "
              f"({'PASS' if acc > MIN_ACCURACY else 'FAIL'} vs {MIN_ACCURACY * 100:.0f}%)")
        print(f"  Macro F1            : {macro_f1 * 100:.2f}%")
        print(f"  Worst class         : {worst} ({class_f1[worst] * 100:.1f}%)")
        print(f"  Classes F1 < 25%    : {n_below} of {len(class_f1)}")
        print()

    # ── Reference: LinearSVC on the best LogReg input ──
    best = max(rows, key=lambda r: r["accuracy"])["variant"]
    X_train_raw = build_input(train_df, best)
    X_held_raw = build_input(heldout_df, best)
    acc, macro_f1, report, _ = run_variant(
        X_train_raw, y_train, X_held_raw, y_held, best, classifier="svm"
    )
    class_f1 = {cls: report[cls]["f1-score"] for cls in y_held.unique() if cls in report}
    worst = min(class_f1, key=class_f1.get)
    rows.append(
        dict(
            variant=best + "+svm",
            classifier="svm",
            accuracy=round(acc, 4),
            macro_f1=round(macro_f1, 4),
            worst_class=worst,
            worst_f1=round(class_f1[worst], 4),
            classes_below_25pct=sum(f < MIN_CLASS_F1 for f in class_f1.values()),
        )
    )
    print(f"--- Variant {best}+svm (LinearSVC reference) ---")
    print(f"  Accuracy            : {acc * 100:.2f}%")
    print(f"  Macro F1            : {macro_f1 * 100:.2f}%")
    print()

    # ── Full per-class report of the best run (paper Table 3 material) ──
    best_row = max(rows, key=lambda r: r["accuracy"])
    print("=" * 60)
    print(f"BEST: {best_row['variant']} ({best_row['classifier']}) "
          f"acc={best_row['accuracy'] * 100:.2f}% macro-F1={best_row['macro_f1'] * 100:.2f}%")
    print("=" * 60)
    best_report = reports.get(best_row["variant"].replace("+svm", ""), report)
    print(classification_report(
        y_held,
        run_variant(
            build_input(train_df, best_row["variant"].replace("+svm", "")),
            y_train,
            build_input(heldout_df, best_row["variant"].replace("+svm", "")),
            y_held,
            best_row["variant"].replace("+svm", ""),
            classifier="logreg" if best_row["classifier"] == "logreg" else "svm",
        )[3],
        zero_division=0,
        digits=3,
    ))

    # ── Save comparison table (paper Table 2 material) ──
    out = pd.DataFrame(rows)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "04_classical_results.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print("Done. Use the best variant above for results.txt and paper Table 2.")


if __name__ == "__main__":
    main()
