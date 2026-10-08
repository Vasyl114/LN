"""
model_utils.py
==============
The classical model pipeline of Group 26 NLP Project, shared by
04_model_classical.py (Model 2) and 07_final_system.py (final system).

THE MODEL (Model 2):
    TF-IDF features -> linear Support Vector Machine, class-balanced.

WHY THIS AND NOT THE ALTERNATIVES:
    - Linear models on TF-IDF are the standard for labelling long documents by topic.
    - A linear SVM and logistic regression reached the same accuracy in cross-validation
      (74.8% vs 74.8% with copy-aware decoding), the SVM in one tenth of the time.
    - class_weight='balanced' makes an error on a small class cost more, which
      counteracts the imbalance without changing the data.
    - sublinear TF-IDF (1 + log tf) stops very long notes from dominating by repetition.
    - An optional extra feature block holds the first keyword (the specialty itself).
      It gets its own columns because inside the bag of words it is diluted
      (47% accuracy) while as a separate feature it is used fully.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

from data_utils import first_keyword_label, RANDOM_SEED

# Which fields are joined into the text the model reads
FIELDS = {
    'D':   ['description'],
    'DN':  ['description', 'sample_name'],
    'DNT': ['description', 'sample_name', 'transcription'],
}

CLASSICAL_CONFIG = {
    'ngram_range': (1, 2),   # words and pairs of consecutive words
    'min_df': 2,             # ignore terms found in fewer than 2 records
    'max_features': 150000,  # vocabulary limit
    'sublinear_tf': True,    # use 1 + log(tf)
    'C': 0.5,                # SVM regularisation: larger = fits the training data more closely
    'keyword_weight': 3.0,   # scale of the first-keyword feature block
}


def join_fields(df, fields):
    """Join the chosen fields of every record into one text (empty fields are skipped)."""
    parts = [df[f].fillna('') for f in FIELDS[fields]]
    return parts[0].str.cat(parts[1:], sep=' ') if len(parts) > 1 else parts[0]


def fit_scores(train_df, test_df, fields='DNT', config=None, use_keyword=False):
    """
    Train on train_df and return (classes, scores for test_df).
    scores has one row per record of test_df and one column per class (in the order
    of `classes`); a higher score means a more likely class.
    use_keyword=True adds the first-keyword feature block.
    """
    cfg = {**CLASSICAL_CONFIG, **(config or {})}
    vectorizer = TfidfVectorizer(ngram_range=cfg['ngram_range'], min_df=cfg['min_df'],
                                 max_features=cfg['max_features'], sublinear_tf=cfg['sublinear_tf'])
    X_train = vectorizer.fit_transform(join_fields(train_df, fields))
    X_test = vectorizer.transform(join_fields(test_df, fields))

    if use_keyword:
        encoder = OneHotEncoder(handle_unknown='ignore')
        K_train = encoder.fit_transform(first_keyword_label(train_df).to_frame('k'))
        K_test = encoder.transform(first_keyword_label(test_df).to_frame('k'))
        X_train = sp.hstack([X_train, cfg['keyword_weight'] * K_train]).tocsr()
        X_test = sp.hstack([X_test, cfg['keyword_weight'] * K_test]).tocsr()

    model = LinearSVC(C=cfg['C'], class_weight='balanced', random_state=RANDOM_SEED)
    model.fit(X_train, train_df['medical_specialty'])
    return model.classes_, model.decision_function(X_test)


def standardize_rows(scores):
    """Rescale every record's scores to mean 0 and standard deviation 1 across the classes,
    so that scores of different models can be added without choosing a scale."""
    scores = np.asarray(scores, dtype=float)
    return (scores - scores.mean(axis=1, keepdims=True)) / (scores.std(axis=1, keepdims=True) + 1e-9)
