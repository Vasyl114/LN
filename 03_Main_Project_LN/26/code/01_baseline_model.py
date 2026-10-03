"""
01_baseline_model.py
====================
Baseline model for Group 26 NLP Project.

WHAT THIS IS:
    The weak baseline described in the project statement, and nothing else:
      - input         : the 'description' field only
      - preprocessing : lowercasing and stemming (Porter stemmer)
      - features      : TF-IDF
      - classifier    : Multinomial Naive Bayes
    Everything the statement does not mention is left at the library default
    (no stop-word removal, no n-grams, no class weights, no tuning).

HOW IT IS EVALUATED:
    1. The original records are divided once into a training portion (80%)
       and a held-out portion (20%), keeping the proportion of each specialty.
       The division is always the same (see split_train_heldout() in data_utils.py).
    2. The model is trained on the given dataset WITHOUT the held-out records.
    3. The model is measured on the held-out records, which are always
       original records (never augmented ones).
    test_no_labels.csv is not used here.

HOW TO RUN:
    From the 26/code/ directory:
        python3 01_baseline_model.py                        # original training data
        python3 01_baseline_model.py train_augmented.csv    # any other training file
    The training file must have the same five columns as train.csv.
"""

import os
import sys
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report

from data_utils import load_train, split_train_heldout, remove_heldout, TRAIN_PATH

# Download NLTK data for tokenization (only needed the first time)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# The two conditions of the automatic evaluation (project statement, Section 4)
MIN_ACCURACY = 0.49   # accuracy must be above this value
MIN_CLASS_F1 = 0.25   # no per-class F1 may fall below this value

stemmer = PorterStemmer()


def preprocess_text(text):
    """
    Applies lowercasing and stemming to a given text.
    """
    if not isinstance(text, str):
        return ""
    tokens = word_tokenize(text.lower())
    return " ".join(stemmer.stem(token) for token in tokens)


def main():
    # Allow passing a different training file via terminal
    data_path = sys.argv[1] if len(sys.argv) > 1 else TRAIN_PATH
    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}.")
        sys.exit(1)

    # ── Data: fixed held-out portion, training on everything else ──
    original_df = load_train()
    _, heldout_df = split_train_heldout(original_df)

    dataset_df = original_df if data_path == TRAIN_PATH else load_train(data_path)
    train_df = remove_heldout(dataset_df, heldout_df)

    print("=" * 60)
    print("BASELINE — Multinomial Naive Bayes + TF-IDF on 'description'")
    print("=" * 60)
    print(f"  Training file              : {data_path}")
    print(f"  Records in the file        : {len(dataset_df)}")
    print(f"  Held-out records removed   : {len(dataset_df) - len(train_df)}")
    print(f"  Training on                : {len(train_df)} records")
    print(f"  Evaluating on (held-out)   : {len(heldout_df)} original records\n")

    # ── Preprocessing: lowercasing and stemming ──
    X_train = train_df['description'].apply(preprocess_text)
    y_train = train_df['medical_specialty']
    X_heldout = heldout_df['description'].apply(preprocess_text)
    y_heldout = heldout_df['medical_specialty']

    # ── Features: TF-IDF (vocabulary learned from the training records only) ──
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_heldout_tfidf = vectorizer.transform(X_heldout)

    # ── Classifier: Multinomial Naive Bayes ──
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_heldout_tfidf)

    # ── Results ──
    accuracy = accuracy_score(y_heldout, y_pred)
    macro_f1 = f1_score(y_heldout, y_pred, average='macro', zero_division=0)
    report = classification_report(y_heldout, y_pred, zero_division=0, output_dict=True)
    class_f1 = {cls: report[cls]['f1-score'] for cls in sorted(y_heldout.unique())}
    worst_class = min(class_f1, key=class_f1.get)

    # Reference point: always answering with the most frequent training class
    majority_class = y_train.value_counts().idxmax()
    majority_acc = (y_heldout == majority_class).mean()

    print("Results per specialty:")
    print(classification_report(y_heldout, y_pred, zero_division=0, digits=3))

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Accuracy                   : {accuracy * 100:.2f}%")
    print(f"  Macro F1                   : {macro_f1 * 100:.2f}%")
    print(f"  Lowest class F1            : {class_f1[worst_class] * 100:.2f}% ({worst_class})")
    print(f"  Classes with F1 below {MIN_CLASS_F1 * 100:.0f}%  : "
          f"{sum(f1 < MIN_CLASS_F1 for f1 in class_f1.values())} of {len(class_f1)}")
    print(f"  Classes never predicted    : "
          f"{sum(report[cls]['recall'] == 0 for cls in class_f1)} of {len(class_f1)}")
    print(f"  Always answering '{majority_class}' : {majority_acc * 100:.2f}% accuracy (reference)")
    print()
    print("  Conditions of the automatic evaluation, measured here on the")
    print("  held-out records (the official ones use the hidden test labels):")
    print(f"    Accuracy above {MIN_ACCURACY * 100:.0f}%          : "
          f"{'YES' if accuracy > MIN_ACCURACY else 'NO'}")
    print(f"    No class F1 below {MIN_CLASS_F1 * 100:.0f}%       : "
          f"{'YES' if class_f1[worst_class] >= MIN_CLASS_F1 else 'NO'}")


if __name__ == "__main__":
    main()
