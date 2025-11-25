import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler, autocast

from dataset import PIIDataset, collate_batch
from labels import LABELS
from model import create_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # -------------------------
    # Tokenizer + Dataset
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_ds = PIIDataset(
        args.train,
        tokenizer,
        LABELS,
        max_length=args.max_length,
        is_train=True
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )

    # -------------------------
    # Model
    # -------------------------
    model = create_model(args.model_name)
    model.to(args.device)
    model.train()

    # -------------------------
    # Optimizer + Scheduler
    # -------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    scaler = GradScaler()

    # -------------------------
    # Training Loop
    # -------------------------
    for epoch in range(args.epochs):
        running_loss = 0.0

        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}"):

            input_ids = torch.tensor(batch["input_ids"], device=args.device)
            attention_mask = torch.tensor(batch["attention_mask"], device=args.device)
            labels = torch.tensor(batch["labels"], device=args.device)

            optimizer.zero_grad()

            # FP16 automatic mixed precision
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

            # Backward pass
            scaler.scale(loss).backward()

            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}: avg loss = {running_loss / len(train_dl):.4f}")

   
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved model + tokenizer to {args.out_dir}")


if __name__ == "__main__":
    main()
