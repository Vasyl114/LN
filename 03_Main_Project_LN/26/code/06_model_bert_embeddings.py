"""
06_model_bert_embeddings.py
===========================
Model 4 for Group 26 NLP Project: separates "type of model" from "amount of text".

THE PROBLEM IT SOLVES:
    The research question compares Model 3 (language model, description only) with
    Model 2 (classical model, all the text). They differ in TWO things at once: the
    model family and the input. If the language model lost, was it the model or the
    missing text? Model 4 gives the language model the same text as Model 2, which
    completes a 2 x 2 comparison:

                          description only      description + name + note
        classical         Model 2 (input D)     Model 2 (input DNT)
        language model    Model 3               Model 4

WHAT THIS IS:
    The same Bio_ClinicalBERT as Model 3, but FROZEN: every record is turned into one
    768-number vector (the mean of the encoder's output over its tokens) and a
    logistic regression is trained on those vectors.

WHY FROZEN AND NOT FINE-TUNED:
    Fine-tuning on 500-token notes costs about 10 times more than on 30-token
    descriptions, which is not feasible on this 4-core CPU. A frozen encoder answers the
    question that matters here: how much does a pre-trained clinical encoder get out of
    the full text without any adaptation?

LIMITS (stated in the paper):
    - the encoder reads at most 512 tokens, so long notes are cut (coverage printed below)
    - it is trained on the original records only (embedding the generated records
      would double the cost), so it is compared with Model 2 trained on the original file

HOW TO RUN (from 26/code/; about 1 hour on CPU, the embeddings are cached in
results/embeddings_dnt_512.npy so that a second run takes seconds):
    python3 06_model_bert_embeddings.py
    python3 06_model_bert_embeddings.py --smoke      # 24 records, saves nothing
"""

import argparse
import json
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import time
import warnings
import numpy as np
import torch
import transformers
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

import evaluation
from data_utils import (load_train, split_train_heldout, seen_labels, copy_aware_scores, RANDOM_SEED)
from evaluation import summarize
from model_utils import join_fields

warnings.filterwarnings('ignore')

CONFIG = {
    'model_name': 'emilyalsentzer/Bio_ClinicalBERT',
    'revision': 'd5892b39a4adaed74b92212a44081509db72f87b',   # pinned: exact weights
    'input': 'description + sample_name + transcription',
    'max_length': 512,       # tokens read per record (the model's maximum)
    'batch_size': 8,
    'pooling': 'mean of the last layer over the non-padding tokens',
    'threads': 4,
    'C_grid': [0.01, 0.1, 1.0],   # logistic regression regularisation, chosen by 5-fold CV inside the training portion
    'seed': RANDOM_SEED,
}
CACHE = 'embeddings_dnt_512.npy'


@torch.no_grad()
def embed(texts, tokenizer, model, cfg):
    """One vector per text. Texts are processed in order of length to avoid padding."""
    lengths = [len(t) for t in texts]
    order = np.argsort(lengths, kind='stable')
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    started = time.time()
    for n, start in enumerate(range(0, len(order), cfg['batch_size'])):
        idx = order[start:start + cfg['batch_size']]
        enc = tokenizer([texts[i] for i in idx], truncation=True, max_length=cfg['max_length'],
                        padding=True, return_tensors='pt')
        hidden = model(**enc).last_hidden_state
        mask = enc['attention_mask'].unsqueeze(-1).float()
        out[idx] = ((hidden * mask).sum(1) / mask.sum(1)).numpy()
        if n % 25 == 0:
            print(f"  embedded {min(start + len(idx), len(texts))}/{len(texts)} ({time.time() - started:.0f}s)", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='tiny run to check the setup; saves nothing')
    args = parser.parse_args()
    cfg = dict(CONFIG)
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(cfg['threads'])

    print("=" * 60)
    print("MODEL 4 — frozen Bio_ClinicalBERT on description + name + note, then logistic regression")
    print("=" * 60)

    original = load_train()
    train_df, heldout_df = split_train_heldout(original)
    texts = join_fields(original, 'DNT').tolist()

    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'], revision=cfg['revision'])
    n_tokens = [len(ids) for ids in tokenizer(texts[:300] if args.smoke else texts)['input_ids']]
    print(f"  Records whose text fits in {cfg['max_length']} tokens: "
          f"{np.mean(np.array(n_tokens) <= cfg['max_length']) * 100:.0f}% | median tokens {int(np.median(n_tokens))}\n")

    if args.smoke:
        model = AutoModel.from_pretrained(cfg['model_name'], revision=cfg['revision']).eval()
        t = time.time(); embed(texts[:24], tokenizer, model, cfg)
        print(f"  Smoke run: 24 records embedded in {time.time() - t:.0f}s -> estimate for all "
              f"{len(texts)}: {(time.time() - t) / 24 * len(texts) / 60:.0f} minutes. Nothing saved.")
        return

    cache = os.path.join(evaluation.RESULTS_DIR, CACHE)
    if os.path.exists(cache):
        vectors = np.load(cache)
        assert len(vectors) == len(original)
        print(f"  Loaded cached embeddings from {cache}")
    else:
        model = AutoModel.from_pretrained(cfg['model_name'], revision=cfg['revision']).eval()
        vectors = embed(texts, tokenizer, model, cfg)
        os.makedirs(evaluation.RESULTS_DIR, exist_ok=True)
        np.save(cache, vectors)

    # record ids index the rows of `vectors` (load_train gives a 0..n-1 index)
    X_train, X_held = vectors[train_df.index], vectors[heldout_df.index]
    y_train = train_df['medical_specialty'].values
    scaler = StandardScaler().fit(X_train)
    X_train, X_held = scaler.transform(X_train), scaler.transform(X_held)

    # ── C by cross-validation inside the training portion (macro-F1 with copy-aware decoding) ──
    print("\nSTEP 1: regularisation C, 5-fold cross-validation inside the training portion")
    folds = list(StratifiedKFold(5, shuffle=True, random_state=cfg['seed']).split(X_train, y_train))
    best_C, best_f1 = None, -1
    for C in cfg['C_grid']:
        f1s = []
        for a, b in folds:
            clf = LogisticRegression(C=C, max_iter=500, class_weight='balanced').fit(X_train[a], y_train[a])
            sc = copy_aware_scores(clf.decision_function(X_train[b]), clf.classes_, train_df.iloc[b],
                                   seen_labels(train_df.iloc[a]))
            f1s.append(f1_score(y_train[b], clf.classes_[sc.argmax(1)], average='macro', zero_division=0))
        print(f"  C = {C:<5}: L1 macro-F1 {np.mean(f1s) * 100:.1f}")
        if np.mean(f1s) > best_f1:
            best_C, best_f1 = C, np.mean(f1s)
    print(f"  Selected C = {best_C}\n")

    clf = LogisticRegression(C=best_C, max_iter=500, class_weight='balanced').fit(X_train, y_train)
    scores = clf.decision_function(X_held)
    seen = seen_labels(train_df)
    np.save(os.path.join(evaluation.RESULTS_DIR, 'm4_heldout_scores.npy'), scores)
    for layer, sc in [('L0', scores), ('L1', copy_aware_scores(scores, clf.classes_, heldout_df, seen))]:
        summarize(f"m4_{layer}", heldout_df, clf.classes_[sc.argmax(1)], per_class_table=(layer == 'L1'),
                  train_labels=train_df['medical_specialty'],
                  config={**cfg, 'layer': layer, 'selected_C': best_C, 'torch': torch.__version__,
                          'transformers': transformers.__version__, 'training_records': int(len(train_df))})


if __name__ == '__main__':
    main()
