"""
M3_bert_model.py: language model, Bio_ClinicalBERT fine-tuned on the description.

Model:   Bio_ClinicalBERT (pre-trained on clinical notes, pinned release) plus a linear classification layer;
         all weights are fine-tuned for 3 epochs on the description only, cut at 128 tokens (about 0.1% of
         the descriptions are longer). It is the only script that trains a neural network; M4 builds on it.
Choices: - Bio_ClinicalBERT, not BioBERT: its pre-training text (clinical notes) is the genre of this data.
         - Fine-tuned, not frozen: the encoder was never trained to separate these 12 specialties.
         - Hyperparameters fixed in advance from the usual BERT range (Devlin et al., 2019), none tuned on
           the held-out set; the model after the last epoch is always the one reported.
         - Plain cross-entropy: the augmented file is already balanced, so class weights would count the
           imbalance twice.
         - Batches of similar length (shuffle, cut into groups of 50 batches, sort each group by length):
           padding to the longest of a random batch made a step 2-3 times slower on the 4-core CPU.
Reads:   train_augmented.csv (and the test file with --test). Downloads the pre-trained model on first use.
Writes:  (no flag)  results/m3_heldout_scores.npy, m3.json: trains without the 534 held-out records and
                    scores them (about 50 min)
         --test     results/m3_test_scores.npy: trains on all 5,648 records and scores the 398 test
                    records (about 55 min)
         --smoke    one-minute check (320 records, 1 epoch, 64 scored); writes only to a temporary folder
Run:     python3 M3_bert_model.py [--test] [--smoke] [--train FILE.csv]
"""

import argparse
import json
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import tempfile
import time
import numpy as np
import torch
import transformers
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from data_utils import load_train, load_test, split_train_heldout, remove_heldout, VALID_LABELS, RANDOM_SEED
import model_utils
from model_utils import (summarize, save_scores, set_seed, bucketed_batches, predict_scores)

CONFIG = {
    'model_name': 'emilyalsentzer/Bio_ClinicalBERT',
    'revision': 'd5892b39a4adaed74b92212a44081509db72f87b',   # pinned release of the pre-trained weights
    'input': 'description',
    'max_length': 128,       # tokens kept per description (longer ones are cut)
    'learning_rate': 3e-5,   # step size of the optimiser (AdamW)
    'batch_size': 16,        # records per update
    'epochs': 3,             # passes over the training file
    'warmup_ratio': 0.1,     # share of the updates during which the learning rate grows from 0
    'weight_decay': 0.01,    # penalty on large weights
    'bucket_batches': 50,    # batches per length-sorted group (see the header)
    'threads': 4,            # CPU threads
    'seed': RANDOM_SEED,
}
CLASSES = sorted(VALID_LABELS)   # column order of every score table (alphabetical)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train_augmented.csv', help='training file (default: train_augmented.csv)')
    parser.add_argument('--smoke', action='store_true', help='one-minute check; writes only to a temporary folder')
    parser.add_argument('--test', action='store_true',
                        help='train on ALL the records and score the test file (input of results.txt)')
    args = parser.parse_args()

    cfg = dict(CONFIG)
    set_seed(cfg['seed'])
    torch.set_num_threads(cfg['threads'])

    print("=" * 60)
    title = "MODEL 3 — Bio_ClinicalBERT fine-tuned on the description"
    print(title + (" (--test: all records, scores the test file)" if args.test else ""))
    print("=" * 60)
    print(f"  torch {torch.__version__} | transformers {transformers.__version__} | threads {torch.get_num_threads()}")

    # ── Data ──
    original = load_train()
    _, heldout_df = split_train_heldout(original)
    if args.test:       # final run: all records train the model; the test file has no labels
        train_df, eval_df = load_train(args.train), load_test()
    else:               # evaluation run: the held-out records are removed from the training file
        train_df, eval_df = remove_heldout(load_train(args.train), heldout_df), heldout_df
    if args.smoke:
        train_df, eval_df = train_df.sample(320, random_state=cfg['seed']), eval_df.head(64)
        cfg['epochs'] = 1
    print(f"  Training file : {args.train} -> {len(train_df)} records"
          + ("" if args.test else " (held-out records removed)"))
    print(f"  Scoring       : {len(eval_df)} " + ("test records" if args.test else "held-out original records"))
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

    # Tokenise once; each batch is padded to its own longest record
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
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            if step % 50 == 0 or step == steps_per_epoch:
                print(f"  epoch {epoch} step {step}/{steps_per_epoch} loss {running / step:.4f} "
                      f"({time.time() - started:.0f}s)", flush=True)
        train_time = time.time() - started

        entry = {'epoch': epoch, 'train_loss': running / steps_per_epoch, 'seconds': round(train_time)}
        if args.test:
            print(f"  >> epoch {epoch}: {train_time:.0f}s\n", flush=True)
            if epoch < cfg['epochs']:
                log.append(entry)
                continue
        scores = predict_scores(model, tokenizer, x_eval, cfg['max_length'])
        if not args.test:   # held-out accuracy per epoch: logged only, never used to choose the epoch
            pred = np.array(CLASSES)[scores.argmax(1)]
            y_eval = eval_df['medical_specialty'].tolist()
            entry['heldout_accuracy'] = accuracy_score(y_eval, pred)
            entry['heldout_macro_f1'] = f1_score(y_eval, pred, average='macro', zero_division=0)
            print(f"  >> epoch {epoch}: {train_time:.0f}s | held-out accuracy {entry['heldout_accuracy']*100:.1f}% "
                  f"macro-F1 {entry['heldout_macro_f1']*100:.1f}% (logged only)\n", flush=True)
        log.append(entry)

    # ── Report the model after the last epoch ──
    if args.smoke:      # same code path as a real run, written to a temporary folder
        model_utils.RESULTS_DIR = os.path.join(tempfile.gettempdir(), 'm3_smoke_results')
    os.makedirs(model_utils.RESULTS_DIR, exist_ok=True)
    if args.test:
        save_scores('m3_test_scores', scores)
        print(f"Saved {model_utils.RESULTS_DIR}/m3_test_scores.npy: scores of the {len(eval_df)} test records.")
    else:
        save_scores('m3_heldout_scores', scores)
        run_config = {**cfg, 'training_file': args.train, 'training_records': int(len(train_df)),
                      'torch': torch.__version__, 'transformers': transformers.__version__,
                      'epoch_log': log, 'classes': CLASSES}
        summarize("m3", eval_df, np.array(CLASSES)[scores.argmax(1)], per_class_table=True,
                  train_labels=train_df['medical_specialty'], config=run_config)
    if args.smoke:
        print(f"Smoke run finished; results written to {model_utils.RESULTS_DIR} only.")


if __name__ == '__main__':
    main()
