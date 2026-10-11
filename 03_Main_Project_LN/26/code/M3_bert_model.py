"""
M3_bert_model.py
================
Model 3 for Group 26 NLP Project: the language model that Model 4 is built on.

RESEARCH QUESTION (short): does a highly tuned language model that reads little text
beat a classical classifier that reads all the text?

WHAT THIS IS:
    Bio_ClinicalBERT (Alsentzer et al., 2019) with a linear classification layer,
    fine-tuned on the description field ONLY (little text). Model 4 adds the first
    keyword to its scores (see model_utils.add_first_keyword).

WHY THESE CHOICES:
    - Bio_ClinicalBERT, not BioBERT or a general BERT: it was pre-trained on clinical
      notes, the same kind of text as these dictated transcriptions. BioBERT was
      pre-trained on PubMed abstracts, a different genre.
    - Fine-tuned, not frozen: the 12 labels depend on specialty wording that the
      pre-trained encoder was never trained to separate.
    - Description only: it is the field the research question gives the language model.
      It is short (median 33 BERT tokens, only 0.1% of the descriptions are longer than
      128 tokens), which keeps CPU training feasible. The classical model reads everything.
    - Hyperparameters FIXED IN ADVANCE from the usual range for BERT fine-tuning
      (learning rate 2e-5..5e-5, 2..4 epochs, Devlin et al., 2019). They are not tuned:
      the held-out set is not used to choose anything, and the language model is too
      slow on this CPU for a search. The held-out accuracy after every epoch is only
      logged; the model after the LAST epoch is always the one reported.
    - Trained on the augmented file with plain cross-entropy: that file is already
      balanced (about 400 records per class), so adding class weights would
      compensate for the imbalance twice.
    - The pre-trained model is pinned to one exact revision, so that the same weights
      are downloaded every time.
    - Batches are built from records of similar length (the order is shuffled every epoch,
      then cut into groups of 50 batches that are sorted by length). A random batch is
      padded to its longest description (60..107 tokens) although the median is about
      30 tokens, which made a step 2-3 times slower on this 4-core CPU for no benefit.

HOW TO RUN (from 26/code/):
    python3 M3_bert_model.py                  # evaluation run: trains without the held-out records and
                                              # saves their scores (results/m3_heldout_scores.npy)
    python3 M3_bert_model.py --full           # final run for results.txt: trains on ALL the records of
                                              # train_augmented.csv and saves the scores of the 398 test
                                              # records (results/m3_test_scores.npy), about 55 minutes
    python3 M3_bert_model.py --smoke          # 1 minute check that everything works, saves nothing
    python3 M3_bert_model.py --train ../../StudentsPack/train.csv   # original data only
"""

import argparse
import json
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import random
import time
import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from data_utils import (load_train, load_test, split_train_heldout, remove_heldout,
                        TRAIN_PATH, VALID_LABELS, RANDOM_SEED)
import evaluation
from evaluation import summarize

CONFIG = {
    'model_name': 'emilyalsentzer/Bio_ClinicalBERT',
    'revision': 'd5892b39a4adaed74b92212a44081509db72f87b',   # pinned: exact weights
    'input': 'description',
    'max_length': 128,       # tokens kept per description (longer ones are cut)
    'learning_rate': 3e-5,   # step size of the optimiser (AdamW)
    'batch_size': 16,        # records per update
    'epochs': 3,             # passes over the training file
    'warmup_ratio': 0.1,     # share of the updates during which the learning rate grows from 0
    'weight_decay': 0.01,    # penalty on large weights
    'bucket_batches': 50,    # batches per length-sorted group (see above)
    'threads': 4,            # CPU threads (the 4 physical cores of the machine used)
    'seed': RANDOM_SEED,
}
CLASSES = sorted(VALID_LABELS)            # same column order as scikit-learn (alphabetical)


def set_seed(seed):
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


@torch.no_grad()
def predict_scores(model, tokenizer, texts, max_length):
    """Log-probabilities (records x 12 classes) in the order of CLASSES."""
    model.eval()
    out = []
    for _, chunk in batches(texts, 64):
        enc = tokenizer(chunk, truncation=True, max_length=max_length, padding=True, return_tensors='pt')
        out.append(torch.log_softmax(model(**enc).logits, dim=-1).numpy())
    return np.concatenate(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train_augmented.csv', help='training file (default: train_augmented.csv)')
    parser.add_argument('--smoke', action='store_true', help='tiny run to check the setup; saves nothing')
    parser.add_argument('--full', action='store_true',
                        help='train on ALL the records and score the test file (for results.txt)')
    args = parser.parse_args()

    cfg = dict(CONFIG)
    set_seed(cfg['seed'])
    torch.set_num_threads(cfg['threads'])

    print("=" * 60)
    print("MODEL 3 — Bio_ClinicalBERT fine-tuned on the description" + (" (--full: all records, scores the test file)" if args.full else ""))
    print("=" * 60)
    print(f"  torch {torch.__version__} | transformers {transformers.__version__} | threads {torch.get_num_threads()}")

    # ── Data ──
    original = load_train()
    _, heldout_df = split_train_heldout(original)
    if args.full:       # final run: every record is used for training, the test file has no labels
        train_df, eval_df = load_train(args.train), load_test()
    else:               # evaluation run: the held-out records are removed from the training file
        train_df, eval_df = remove_heldout(load_train(args.train), heldout_df), heldout_df
    if args.smoke:
        train_df, eval_df = train_df.sample(320, random_state=cfg['seed']), eval_df.head(64)
        cfg['epochs'] = 1
    print(f"  Training file : {args.train} -> {len(train_df)} records"
          + ("" if args.full else " (held-out records removed)"))
    print(f"  Scoring       : {len(eval_df)} " + ("test records" if args.full else "held-out original records"))
    print(f"  Config        : {json.dumps({k: v for k, v in cfg.items() if k != 'model_name'})}\n")

    label_id = {label: i for i, label in enumerate(CLASSES)}
    x_train = train_df['description'].fillna('').tolist()
    y_train = torch.tensor([label_id[l] for l in train_df['medical_specialty']])
    x_eval = eval_df['description'].fillna('').tolist()

    # ── Model ──
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'], revision=cfg['revision'])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg['model_name'], revision=cfg['revision'], num_labels=len(CLASSES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    steps_per_epoch = int(np.ceil(len(x_train) / cfg['batch_size']))
    total_steps = steps_per_epoch * cfg['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg['warmup_ratio'] * total_steps), total_steps)

    # Tokenise once; batches are padded to their own longest record
    encoded = tokenizer(x_train, truncation=True, max_length=cfg['max_length'])
    lengths = [len(ids) for ids in encoded['input_ids']]
    print(f"  Tokens per description: median {int(np.median(lengths))}, max {max(lengths)}\n")

    # ── Training ──
    log = []
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        started, running = time.time(), 0.0
        for step, idx in enumerate(bucketed_batches(lengths, cfg['batch_size'], cfg['bucket_batches'],
                                                    cfg['seed'] + epoch), start=1):
            enc = tokenizer.pad({k: [encoded[k][i] for i in idx] for k in encoded.keys()}, return_tensors='pt')
            loss = model(**enc, labels=y_train[idx]).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            running += loss.item()
            if step % 50 == 0 or step == steps_per_epoch:
                print(f"  epoch {epoch} step {step}/{steps_per_epoch} loss {running / step:.4f} "
                      f"({time.time() - started:.0f}s)", flush=True)
        train_time = time.time() - started

        entry = {'epoch': epoch, 'train_loss': running / steps_per_epoch, 'seconds': round(train_time)}
        if args.full:
            print(f"  >> epoch {epoch}: {train_time:.0f}s\n", flush=True)
            if epoch < cfg['epochs']:
                log.append(entry)
                continue
        scores = predict_scores(model, tokenizer, x_eval, cfg['max_length'])
        if not args.full:   # held-out accuracy after every epoch: logged only, never used to choose the epoch
            pred = np.array(CLASSES)[scores.argmax(1)]
            y_eval = eval_df['medical_specialty'].tolist()
            entry['heldout_accuracy'] = accuracy_score(y_eval, pred)
            entry['heldout_macro_f1'] = f1_score(y_eval, pred, average='macro', zero_division=0)
            print(f"  >> epoch {epoch}: {train_time:.0f}s | held-out accuracy {entry['heldout_accuracy']*100:.1f}% "
                  f"macro-F1 {entry['heldout_macro_f1']*100:.1f}% (logged only)\n", flush=True)
        log.append(entry)

    # ── Report the model after the last epoch ──
    if args.smoke:      # same code path as a real run, but written to a scratch folder
        evaluation.RESULTS_DIR = os.path.join('/tmp', 'm3_smoke_results')
    os.makedirs(evaluation.RESULTS_DIR, exist_ok=True)
    if args.full:
        np.save(os.path.join(evaluation.RESULTS_DIR, 'm3_test_scores.npy'), scores)
        print(f"Saved {evaluation.RESULTS_DIR}/m3_test_scores.npy: scores of the {len(eval_df)} test records.")
    else:
        np.save(os.path.join(evaluation.RESULTS_DIR, 'm3_heldout_scores.npy'), scores)
        run_config = {**cfg, 'training_file': args.train, 'training_records': int(len(train_df)),
                      'torch': torch.__version__, 'transformers': transformers.__version__,
                      'epoch_log': log, 'classes': CLASSES}
        summarize("m3", eval_df, np.array(CLASSES)[scores.argmax(1)], per_class_table=True,
                  train_labels=train_df['medical_specialty'], config=run_config)
    if args.smoke:
        print(f"Smoke run finished; results written to {evaluation.RESULTS_DIR} only.")


if __name__ == '__main__':
    main()
