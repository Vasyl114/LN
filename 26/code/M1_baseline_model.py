"""
M1_baseline_model.py: baseline, the weak reference model of the project statement.

Model:   description only; lowercase + Porter stemming; TF-IDF; Multinomial Naive Bayes (library defaults).
Reads:   train.csv, without the 534 held-out records.
Writes:  results/m1_baseline_original.json (metrics on the held-out records).
Run:     python3 M1_baseline_model.py [training_file.csv]        (about 20 s)
"""

import os
import sys
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from data_utils import load_train, split_train_heldout, remove_heldout, TRAIN_PATH
from model_utils import summarize

# NLTK tokenizer data (downloaded on first use)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

stemmer = PorterStemmer()


def preprocess_text(text):
    """Lowercase, tokenize and stem a text (empty for missing values)."""
    if not isinstance(text, str):
        return ""
    tokens = word_tokenize(text.lower())
    return " ".join(stemmer.stem(token) for token in tokens)


def main():
    # An optional argument replaces the training file
    data_path = sys.argv[1] if len(sys.argv) > 1 else TRAIN_PATH
    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}.")
        sys.exit(1)

    # ── Data: the fixed held-out portion is removed from the training file ──
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

    # ── Preprocessing ──
    X_train = train_df['description'].apply(preprocess_text)
    y_train = train_df['medical_specialty']
    X_heldout = heldout_df['description'].apply(preprocess_text)

    # ── Features: TF-IDF, vocabulary learned from the training records only ──
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_heldout_tfidf = vectorizer.transform(X_heldout)

    # ── Classifier: Multinomial Naive Bayes ──
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_heldout_tfidf)

    # ── Results (same evaluation as every model, see model_utils.summarize) ──
    print("Results per specialty:")
    source = 'original' if data_path == TRAIN_PATH else os.path.splitext(os.path.basename(data_path))[0]
    name = f'm1_baseline_{source}'
    summarize(name, heldout_df, y_pred, per_class_table=True, train_labels=y_train,
              config={'model': 'MultinomialNB + TfidfVectorizer (defaults)', 'input': 'description',
                      'preprocessing': 'lowercase + PorterStemmer', 'training_file': os.path.basename(data_path),
                      'training_records': int(len(train_df))})


if __name__ == "__main__":
    main()
