# nlu_engine/train_intent.py
# (entire file) — robust to small datasets / older transformers versions

import os
import json
import argparse
import traceback
from collections import Counter

def load_intents(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    labels = []
    label2id = {}
    id2label = {}

    for i, intent in enumerate(data["intents"]):
        label2id[intent["name"]] = i
        id2label[i] = intent["name"]

    for intent in data["intents"]:
        for ex in intent["examples"]:
            texts.append(ex)
            labels.append(label2id[intent["name"]])

    return texts, labels, label2id, id2label

def encode_data(tokenizer, texts, labels):
    import torch
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": torch.tensor(labels)
    }

class SimpleDataset(object):
    def __init__(self, enc):
        self.enc = enc

    def __len__(self):
        return len(self.enc["labels"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.enc["input_ids"][idx],
            "attention_mask": self.enc["attention_mask"][idx],
            "labels": self.enc["labels"][idx]
        }

def build_training_args(TrainingArgumentsClass, out_dir, epochs, batch_size, lr):
    modern_kwargs = dict(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=os.path.join(out_dir, "logs"),
    )
    try:
        return TrainingArgumentsClass(**modern_kwargs)
    except TypeError:
        fallback_kwargs = dict(
            output_dir=out_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            logging_dir=os.path.join(out_dir, "logs"),
            do_eval=True,
            save_steps=1000,
            load_best_model_at_end=True
        )
        try:
            return TrainingArgumentsClass(**fallback_kwargs)
        except Exception:
            minimal_kwargs = dict(
                output_dir=out_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                logging_dir=os.path.join(out_dir, "logs")
            )
            return TrainingArgumentsClass(**minimal_kwargs)

def choose_train_test_split(texts, labels, default_frac=0.2):
    """
    Return (train_texts, val_texts, train_labels, val_labels)
    Robust handling for small datasets and few examples per class.
    """
    from sklearn.model_selection import train_test_split

    n = len(texts)
    if n < 2:
        # nothing to split
        return texts, [], labels, []

    counts = Counter(labels)
    n_classes = len(counts)
    # desired test count based on fraction
    desired_test = int(max(1, round(n * default_frac)))
    # ensure test size at least number of classes if possible
    test_size_count = max(desired_test, n_classes)
    if test_size_count >= n:
        test_size_count = max(1, n - 1)  # ensure at least one train sample

    # decide stratify: only if every class has >=2 examples AND test_size >= n_classes
    can_stratify = (min(counts.values()) >= 2) and (test_size_count >= n_classes)

    if can_stratify:
        stratify = labels
        # sklearn accepts integer test_size as count
        train_txt, val_txt, train_lbl, val_lbl = train_test_split(
            texts, labels, test_size=test_size_count, random_state=42, stratify=stratify
        )
    else:
        # fallback: random split without stratify using integer count
        train_txt, val_txt, train_lbl, val_lbl = train_test_split(
            texts, labels, test_size=test_size_count, random_state=42
        )

    return train_txt, val_txt, train_lbl, val_lbl

def train(args):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    except Exception as e:
        print("ERROR: could not import training dependencies:", e)
        traceback.print_exc()
        raise

    print("Loading intents from:", args.intents)
    texts, labels, label2id, id2label = load_intents(args.intents)
    n = len(texts)
    print(f"Loaded {n} examples across {len(set(labels))} classes.")

    if n < 5:
        print("Warning: very small dataset — consider adding more examples per intent.")

    train_txt, val_txt, train_lbl, val_lbl = choose_train_test_split(texts, labels, default_frac=0.2)
    print(f"Train size: {len(train_txt)}  Val size: {len(val_txt)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # if val set empty, still create small dummy val to avoid Trainer complaining
    if len(val_txt) == 0:
        val_txt = train_txt[:1]
        val_lbl = train_lbl[:1]

    train_enc = encode_data(tokenizer, train_txt, train_lbl)
    val_enc = encode_data(tokenizer, val_txt, val_lbl)

    train_ds = SimpleDataset(train_enc)
    val_ds = SimpleDataset(val_enc)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id)
    )

    TrainingArgumentsClass = TrainingArguments
    training_args = build_training_args(TrainingArgumentsClass, args.output_dir, args.epochs, args.batch_size, args.lr)

    from sklearn.model_selection import train_test_split  # keep mt imports local
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer
    )

    print("Starting training...")
    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f)

    with open(os.path.join(args.output_dir, "id2label.json"), "w", encoding="utf-8") as f:
        json.dump(id2label, f)

    print("Model training complete! Saved to:", args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--intents", default="nlu_engine/intents.json")
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--output_dir", default="models/intent_model")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    train(args)
