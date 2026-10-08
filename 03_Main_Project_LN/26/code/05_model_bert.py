"""
05_model_bert.py
================
Model 3 — Domain language model for Group 26 NLP Project.

WHAT THIS IS:
    The "language model on the short description" side of the research
    question, to compare against the classical full-text model
    (04_model_classical.py):
      - input         : (a) description only, (b) description + keywords
      - preprocessing : model tokenizer only (lowercasing/casing follows the
                        checkpoint; no stemming, no stop-word removal)
      - representation: Bio_ClinicalBERT contextual embeddings, [CLS] head
      - classifier    : linear classification head fine-tuned end-to-end,
                        weighted loss for class imbalance

MODEL: emilyalsentzer/Bio_ClinicalBERT (fallback: dmis-lab/biobert-base-cased-v1.2).
Both are free on HuggingFace and pre-trained on clinical/biomedical text.

HOW IT IS EVALUATED (same protocol as Models 1-2):
    1. Fixed 80/20 stratified split, seed 42 (data_utils.split_train_heldout).
    2. Train on the given file MINUS the held-out records; evaluate accuracy,
       macro-F1 and per-class F1 on the held-out originals.
    test_no_labels.csv is not used here.

HOW TO RUN — Google Colab with GPU (recommended, ~30-60 min):
    1. Runtime > Change runtime type > GPU (T4).
    2. Upload to the Colab working directory: data_utils.py,
       train_augmented.csv (or train.csv) and this script.
    3. !pip install -q transformers scikit-learn pandas torch
    4. Run one variant at a time:
           !python3 05_model_bert.py --input desc
           !python3 05_model_bert.py --input desc_kw
       Optional flags: --model <checkpoint> --epochs 4 --lr 2e-5
                       --batch 16 --maxlen 128 --train_file train_augmented.csv
    CPU-only runs work but take hours; prefer --input desc on CPU.

OUTPUT (printed + saved next to this script, all NEW files):
    05_bert_desc_results.json / 05_bert_desc_kw_results.json — metrics + config
    (per-epoch accuracy/macro-F1, final classification report, hyperparams)

HYPERPARAMETER DEFAULTS (reported in the paper, Sec. 4.1):
    learning_rate=2e-5, batch_size=16, epochs=4, weight_decay=0.01,
    max_length=128 (desc) / 256 (desc+keywords), seed 42.
"""

import argparse
import json
import os

DEFAULT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
FALLBACK_MODEL = "dmis-lab/biobert-base-cased-v1.2"
DEFAULT_TRAIN = "train_augmented.csv"
SEED = 42

MIN_ACCURACY = 0.49
MIN_CLASS_F1 = 0.25


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune ClinicalBERT (Model 3).")
    p.add_argument("--input", choices=["desc", "desc_kw"], default="desc",
                   help="desc: description only; desc_kw: description + keywords")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="HuggingFace checkpoint")
    p.add_argument("--train_file", default=DEFAULT_TRAIN)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--maxlen", type=int, default=None,
                   help="tokenizer max length (default 128 for desc, 256 for desc_kw)")
    p.add_argument("--output", default=None, help="output JSON path")
    return p.parse_args()


def build_text(df, mode):
    desc = df["description"].fillna("")
    if mode == "desc":
        return desc
    kw = df["keywords"].fillna("")
    return desc + " [SEP] " + kw


def main():
    args = parse_args()
    maxlen = args.maxlen or (128 if args.input == "desc" else 256)

    # Heavy imports live here so --help works without GPU libraries installed.
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from sklearn.utils.class_weight import compute_class_weight
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        Trainer, TrainingArguments, set_seed,
    )

    from data_utils import (load_train, split_train_heldout, remove_heldout,
                            TRAIN_PATH, VALID_LABELS)

    set_seed(SEED)
    label2id = {label: i for i, label in enumerate(VALID_LABELS)}
    id2label = {i: label for label, i in label2id.items()}

    train_file = args.train_file
    if not os.path.exists(train_file):
        train_file = TRAIN_PATH
    original_df = load_train()
    _, heldout_df = split_train_heldout(original_df)
    dataset_df = load_train(train_file)
    train_df = remove_heldout(dataset_df, heldout_df).reset_index(drop=True)
    held_df = heldout_df.reset_index(drop=True)

    print("=" * 60)
    print("MODEL 3 — Bio_ClinicalBERT fine-tuning")
    print("=" * 60)
    print(f"  Checkpoint   : {args.model} (fallback: {FALLBACK_MODEL})")
    print(f"  Input mode   : {args.input} (max_length={maxlen})")
    print(f"  Training file: {train_file} -> {len(train_df)} records")
    print(f"  Held-out     : {len(held_df)} original records")
    print(f"  Hyperparams  : epochs={args.epochs} lr={args.lr} "
          f"batch={args.batch} weight_decay=0.01 seed={SEED}")
    print(f"  Device       : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print()

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=len(VALID_LABELS),
            id2label=id2label, label2id=label2id)
    except Exception as e:
        print(f"  Could not load {args.model} ({e}); trying {FALLBACK_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(
            FALLBACK_MODEL, num_labels=len(VALID_LABELS),
            id2label=id2label, label2id=label2id)

    def tokenize(texts):
        return tokenizer(list(texts), truncation=True, padding=True,
                         max_length=maxlen)

    class CSVDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[i])
            return item

    y_train = [label2id[l] for l in train_df["medical_specialty"]]
    y_held = [label2id[l] for l in held_df["medical_specialty"]]
    train_ds = CSVDataset(tokenize(build_text(train_df, args.input)), y_train)
    held_ds = CSVDataset(tokenize(build_text(held_df, args.input)), y_held)

    # Weighted loss: mistakes on rare classes (e.g. Dermatology) cost more.
    weights = compute_class_weight(class_weight="balanced",
                                   classes=np.arange(len(VALID_LABELS)),
                                   y=np.array(y_train))
    weight_tensor = torch.tensor(weights, dtype=torch.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits, labels, weight=weight_tensor.to(model.device))
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro",
                                 zero_division=0),
        }

    history = {"per_epoch": []}

    def log_callback(state, logs=None, **kwargs):
        if state.epoch is not None and logs and "eval_accuracy" in logs:
            history["per_epoch"].append({
                "epoch": round(state.epoch, 2),
                "eval_accuracy": logs.get("eval_accuracy"),
                "eval_macro_f1": logs.get("eval_macro_f1"),
            })

    from transformers import TrainerCallback

    class EpochLogger(TrainerCallback):
        def on_log(self, args_, state, control, logs=None, **kwargs):
            log_callback(state, logs)

    training_args = TrainingArguments(
        output_dir=f"./bert_{args.input}_checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        weight_decay=0.01,
        seed=SEED,
        logging_steps=50,
        report_to="none",
    )

    trainer = WeightedTrainer(model=model, args=training_args,
                              train_dataset=train_ds, eval_dataset=held_ds,
                              compute_metrics=compute_metrics,
                              callbacks=[EpochLogger()])
    trainer.train()

    preds = trainer.predict(held_ds)
    y_pred = np.argmax(preds.predictions, axis=1)
    acc = accuracy_score(y_held, y_pred)
    macro_f1 = f1_score(y_held, y_pred, average="macro", zero_division=0)
    true_names = [id2label[i] for i in y_held]
    pred_names = [id2label[i] for i in y_pred]
    report = classification_report(true_names, pred_names,
                                   labels=VALID_LABELS,
                                   zero_division=0, digits=3,
                                   output_dict=True)
    print(classification_report(true_names, pred_names, labels=VALID_LABELS,
                                zero_division=0, digits=3))
    worst = min(VALID_LABELS, key=lambda c: report[c]["f1-score"])
    print("=" * 60)
    print(f"FINAL  accuracy={acc * 100:.2f}% "
          f"({'PASS' if acc > MIN_ACCURACY else 'FAIL'} vs {MIN_ACCURACY * 100:.0f}%)  "
          f"macro-F1={macro_f1 * 100:.2f}%  "
          f"worst={worst} ({report[worst]['f1-score'] * 100:.1f}%)")
    print("Per-epoch:", history["per_epoch"])

    out = {
        "model": args.model,
        "input": args.input,
        "max_length": maxlen,
        "train_file": train_file,
        "n_train": len(train_df),
        "n_heldout": len(held_df),
        "hyperparameters": {
            "epochs": args.epochs, "learning_rate": args.lr,
            "batch_size": args.batch, "weight_decay": 0.01,
            "seed": SEED, "class_weight": "balanced (weighted CE loss)",
        },
        "accuracy": acc,
        "macro_f1": macro_f1,
        "worst_class": worst,
        "worst_f1": report[worst]["f1-score"],
        "per_class_f1": {c: report[c]["f1-score"] for c in VALID_LABELS},
        "per_epoch": history["per_epoch"],
    }
    out_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"05_bert_{args.input}_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
