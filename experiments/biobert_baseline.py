import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABEL_NAMES = ["inconclusive", "evidence", "NEI"]


class CliniFactDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.labels = data["label"].astype(int).tolist()
        self.encodings = []

        for _, item in data.iterrows():
            claim = item["claim"]
            abstract = f"{item['article_title']} {item['article_abstract']}"

            encoded_length = len(
                tokenizer.encode(claim, abstract, add_special_tokens=True)
            )
            while encoded_length > 512:
                sentences = abstract.split(". ")
                if len(sentences) > 1:
                    abstract = ". ".join(sentences[1:])
                else:
                    words = abstract.split()
                    if len(words) > 15:
                        abstract = " ".join(words[1:])
                    else:
                        break
                encoded_length = len(
                    tokenizer.encode(claim, abstract, add_special_tokens=True)
                )

            self.encodings.append(
                tokenizer.encode_plus(
                    text=claim,
                    text_pair=abstract,
                    add_special_tokens=True,
                    max_length=512,
                    padding=False,
                    truncation=True,
                )
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        encoding = self.encodings[index]
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": self.labels[index],
        }


def collate_batch(features, tokenizer):
    labels = torch.tensor([feature["labels"] for feature in features], dtype=torch.long)
    model_inputs = tokenizer.pad(
        [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ],
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    model_inputs["labels"] = labels
    return model_inputs


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, dataloader, device):
    model.eval()
    labels = []
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions.extend(outputs.logits.argmax(dim=1).cpu().tolist())
            labels.extend(batch["labels"].tolist())
    return labels, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/primary_outcome_publication_dataset"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/results/biobert_baseline")
    )
    parser.add_argument(
        "--model-name", default="dmis-lab/biobert-base-cased-v1.1"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 37, 42])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.train_batch_size % args.micro_batch_size:
        parser.error("--train-batch-size must be divisible by --micro-batch-size")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    train_df = pd.read_csv(args.data_dir / "train_set.csv")
    val_df = pd.read_csv(args.data_dir / "validation_set.csv")
    test_df = pd.read_csv(args.data_dir / "test_set.csv")

    config = {
        "model_name": args.model_name,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "patience": args.patience,
        "train_batch_size": args.train_batch_size,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.train_batch_size // args.micro_batch_size,
        "cpu_threads": args.cpu_threads,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "data_dir": str(args.data_dir.resolve()),
        "rows": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
        "seed_policy": "Python, NumPy, PyTorch, and DataLoader generator seeded per run",
        "checkpoint_selection": "highest validation macro-F1",
    }
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = CliniFactDataset(train_df, tokenizer)
    val_dataset = CliniFactDataset(val_df, tokenizer)
    test_dataset = CliniFactDataset(test_df, tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_metrics = []
    for seed in args.seeds:
        print(f"\nTraining seed {seed}")
        seed_everything(seed)
        generator = torch.Generator()
        generator.manual_seed(seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.micro_batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=lambda features: collate_batch(features, tokenizer),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=lambda features: collate_batch(features, tokenizer),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=lambda features: collate_batch(features, tokenizer),
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=3
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        best_val_f1 = -1.0
        best_epoch = 0
        epochs_without_improvement = 0
        checkpoint_dir = args.output_dir / f"seed_{seed}" / "best_model"
        checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
        epoch_history = []

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            optimizer.zero_grad()
            accumulation_steps = args.train_batch_size // args.micro_batch_size
            for batch_index, batch in enumerate(train_loader):
                window_start = (
                    batch_index // accumulation_steps
                ) * accumulation_steps
                window_stop = min(
                    window_start + accumulation_steps, len(train_loader)
                )
                window_size = min(
                    (window_stop - window_start) * args.micro_batch_size,
                    len(train_dataset) - window_start * args.micro_batch_size,
                )
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(
                    input_ids, attention_mask=attention_mask, labels=labels
                )
                scaled_loss = outputs.loss * (labels.size(0) / window_size)
                scaled_loss.backward()
                total_loss += outputs.loss.item()
                if (batch_index + 1) == window_stop:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            val_labels, val_predictions = evaluate(model, val_loader, device)
            val_f1 = f1_score(val_labels, val_predictions, average="macro")
            epoch_metrics = {
                "epoch": epoch + 1,
                "average_train_loss": total_loss / len(train_loader),
                "validation_accuracy": accuracy_score(val_labels, val_predictions),
                "validation_f1_macro": val_f1,
            }
            epoch_history.append(epoch_metrics)
            print(
                f"seed={seed} epoch={epoch + 1} "
                f"train_loss={epoch_metrics['average_train_loss']:.4f} "
                f"val_accuracy={epoch_metrics['validation_accuracy']:.4f} "
                f"val_f1_macro={val_f1:.4f}"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        pd.DataFrame(epoch_history).to_csv(
            checkpoint_dir.parent / "training_metrics.csv", index=False
        )
        del model
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_dir, num_labels=3
        ).to(device)
        test_labels, test_predictions = evaluate(model, test_loader, device)
        metrics = {
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_f1_macro": best_val_f1,
            "accuracy": accuracy_score(test_labels, test_predictions),
            "f1_macro": f1_score(test_labels, test_predictions, average="macro"),
            "classification_report": classification_report(
                test_labels,
                test_predictions,
                labels=[0, 1, 2],
                target_names=LABEL_NAMES,
                output_dict=True,
                zero_division=0,
            ),
        }
        run_metrics.append(metrics)
        predictions_df = test_df[
            ["index", "nctId", "PMID", "claim", "label"]
        ].copy()
        predictions_df["Predicted_Label"] = test_predictions
        predictions_df.to_csv(
            checkpoint_dir.parent / "test_predictions.csv", index=False
        )
        with (checkpoint_dir.parent / "metrics.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(metrics, file, indent=2)
        print(
            f"seed={seed} test_accuracy={metrics['accuracy']:.4f} "
            f"test_f1_macro={metrics['f1_macro']:.4f}"
        )
        del model

    summary = {
        "runs": run_metrics,
        "accuracy_mean": float(np.mean([run["accuracy"] for run in run_metrics])),
        "accuracy_std": float(np.std([run["accuracy"] for run in run_metrics], ddof=1)),
        "f1_macro_mean": float(np.mean([run["f1_macro"] for run in run_metrics])),
        "f1_macro_std": float(
            np.std([run["f1_macro"] for run in run_metrics], ddof=1)
        ),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(
        f"\nThree-seed test summary: accuracy={summary['accuracy_mean']:.4f} "
        f"+/- {summary['accuracy_std']:.4f}; "
        f"macro-F1={summary['f1_macro_mean']:.4f} +/- {summary['f1_macro_std']:.4f}"
    )


if __name__ == "__main__":
    main()
