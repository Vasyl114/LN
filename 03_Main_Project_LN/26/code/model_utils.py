"""
model_utils.py
==============
The classical model pipeline and the first-keyword rule of Group 26 NLP Project, shared by
M2_classical_model.py (Model 2) and M4_final_system.py / generate_results.py (Model 4).

THE MODEL (Model 2):
    TF-IDF features -> linear Support Vector Machine, class-balanced, reading ALL the
    fields of a record (description, sample name, transcription and keywords).

WHY THIS AND NOT THE ALTERNATIVES:
    - Linear models on TF-IDF are the standard for labelling long documents by topic.
    - A linear SVM and logistic regression reached the same accuracy in cross-validation,
      the SVM in one tenth of the time.
    - class_weight='balanced' makes an error on a small class cost more, which
      counteracts the imbalance without changing the data.
    - sublinear TF-IDF (1 + log tf) stops very long notes from dominating by repetition.

THE FIRST-KEYWORD RULE (Model 4, add_first_keyword):
    The first keyword of a record is the specialty itself. Model 4 adds that specialty
    to the (standardised) scores of the language model with a weight that no score
    gap can outvote, so the keyword decides whenever it exists and the language model
    decides for the records without keywords.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from data_utils import first_keyword_label, RANDOM_SEED

# Which fields are joined into the text the model reads
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

# Fixed in advance, not tuned. Standardised scores of 12 classes differ by at most about
# 4.9, so a weight of 5 means that the keyword cannot be outvoted by the language model.
KEYWORD_WEIGHT = 5.0


def join_fields(df, fields):
    """Join the chosen fields of every record into one text (empty fields are skipped)."""
    parts = [df[f].fillna('') for f in FIELDS[fields]]
    return parts[0].str.cat(parts[1:], sep=' ') if len(parts) > 1 else parts[0]


def fit_scores(train_df, test_df, fields='DNTK', config=None):
    """
    Train on train_df and return (classes, scores for test_df).
    scores has one row per record of test_df and one column per class (in the order
    of `classes`); a higher score means a more likely class.
    """
    cfg = {**CLASSICAL_CONFIG, **(config or {})}
    vectorizer = TfidfVectorizer(ngram_range=cfg['ngram_range'], min_df=cfg['min_df'],
                                 max_features=cfg['max_features'], sublinear_tf=cfg['sublinear_tf'])
    X_train = vectorizer.fit_transform(join_fields(train_df, fields))
    X_test = vectorizer.transform(join_fields(test_df, fields))
    model = LinearSVC(C=cfg['C'], class_weight='balanced', random_state=RANDOM_SEED)
    model.fit(X_train, train_df['medical_specialty'])
    return model.classes_, model.decision_function(X_test)


def standardize_rows(scores):
    """Rescale every record's scores to mean 0 and standard deviation 1 across the classes,
    so that scores of different models can be added."""
    scores = np.asarray(scores, dtype=float)
    return (scores - scores.mean(axis=1, keepdims=True)) / (scores.std(axis=1, keepdims=True) + 1e-9)


def add_first_keyword(scores, classes, df, weight=KEYWORD_WEIGHT):
    """
    Model 4: standardised scores plus `weight` on the specialty named by the first
    keyword of each record (nothing is added for records without keywords).
    Returns the new score array (rows of df x classes).
    """
    out = standardize_rows(scores)
    keyword = first_keyword_label(df).values
    for j, label in enumerate(classes):
        out[:, j] += weight * (keyword == label)
    return out
