from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
from argparse import Namespace
import time
from torch.amp import autocast

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import PreTrainedTokenizerBase, get_linear_schedule_with_warmup

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

from transformers import set_seed

use_bf16 = (
    torch.cuda.is_available()
    and torch.cuda.is_bf16_supported()
)
print("="*20)
print(f'Using device: {device}')
print(f'Using bf16: {use_bf16}')
print("="*20)

def get_optimizer(model: Module,
                  args: Namespace) -> Optimizer:
    # Same weight decay implmentation as in transformers
    # https://github.com/huggingface/transformers/blob/a75c64d80c76c3dc71f735d9197a4a601847e0cd/examples/contrib/run_openai_gpt.py#L230-L237

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    "weight_decay": args.weight_decay,
                },
                {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    return optimizer

def train(args: Namespace,
          model: Module, 
          train_dataloader: DataLoader,
          dev_train_dataloader: DataLoader,
          dev_test_dataloader: DataLoader,
          test_dataloader: DataLoader,
          tokenizer_test: PreTrainedTokenizerBase,
    ):

    set_seed(args.seed)
    
    start = time.time()
    model.to(device)
    optimizer = get_optimizer(args=args, model=model)
    num_training_steps = args.epochs * len(train_dataloader)
    warmup_steps = int(0.03 * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(total=args.epochs * len(train_dataloader))

    train_loss = []
    dev_loss = []
    dev_metrics = []
    test_metrics = []

    for epoch in range(1, args.epochs + 1):
        model.train() 
        total_loss = 0.0
        total_batches = len(train_dataloader)

        for batch_idx, batch in enumerate(train_dataloader):

            batch = {k: v.to(device) for k, v in batch.items()}
            
            with autocast(
                device_type='cuda',
                dtype=torch.bfloat16,
                enabled=use_bf16
            ):
                outputs = model(**batch)
                loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)

            optimizer.step()
            scheduler.step() 
            optimizer.zero_grad(set_to_none=True)

            # Train loss
            if batch_idx % args.train_log_step == 0:
                print(f"Epoch {epoch} | "
                      f"Batch {batch_idx+1}/{total_batches} | "
                      f"Training Loss = {loss.item():.4f}"
                )
                train_loss.append(
                    {"epoch": epoch, "batch": batch_idx, "loss": float(loss.item())}
                )

            # Dev loss
            if batch_idx % (args.train_log_step * 3) == 0:
                
                eval_loss = evaluate_dev_loss(model, dev_train_dataloader)
                print(f"Epoch {epoch} | "
                      f"Batch {batch_idx+1}/{total_batches} | "
                      f"Dev Loss = {eval_loss:.4f}"
                )
                dev_loss.append(
                    {"epoch": epoch, "batch": batch_idx, "loss": float(eval_loss)}
                )
                model.train() 

            total_loss += loss.item()

            progress_bar.update(1)

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} - train loss: {avg_loss:.4f}")
        
        print("\n", "="*20)
        print("DEV SET EVALUATION ...")

        metrics = evaluate_wrapper(
                    model_type=args.model_type,
                    model=model,
                    tokenizer_test=tokenizer_test,
                    dataloader=dev_test_dataloader
                    )

        dev_metrics.append({'epoch': epoch, "metrics": metrics})

        print("\n", "="*20)
        print("TEST SET EVALUATION ...")
        metrics = evaluate_wrapper(
                    model_type=args.model_type,
                    model=model,
                    tokenizer_test=tokenizer_test,
                    dataloader=test_dataloader
                    )
        test_metrics.append({'epoch': epoch, "metrics": metrics})

    end = time.time()
    duration = round((end-start) / 60)
    return train_loss, dev_loss, dev_metrics, test_metrics, duration

def evaluate_slm(model: Module, 
                 tokenizer_test,
                 dataloader: torch.utils.data.DataLoader
                 ) -> dict[str, float]:
    """Test set evluation for SLM models"""
    progress_bar = tqdm(range(len(dataloader)))

    model.eval()

    predictions = []
    references = []
    for batch_idx, batch in enumerate(dataloader):
        
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        cd_labels = batch["label"]
        # batch: Batch x Batch input token sequence
        # print("Batch shape", input_ids.shape)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=24,
                )
        
        # Get only newly generated ids
        input_seq_length = input_ids.shape[1]
        output_ids = generated_ids[:, input_seq_length:]
        decoded_preds = tokenizer_test.batch_decode(output_ids, skip_special_tokens=True)
        
        # find the label in the decoded_preds
        for i in range(len(decoded_preds)):
            pred_text = decoded_preds[i]
            if '"label": 1' in pred_text:
                decoded_preds[i] = 1
            elif '"label": 0' in pred_text:
                decoded_preds[i] = 0
            else:
                decoded_preds[i] = -1  # unknown

        predictions.extend(decoded_preds)
        references.extend(cd_labels.tolist())

        progress_bar.update(1)
    
    # get accuract and f1
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p != -1]
    preds_valid = [p for p, _ in valid_pairs]
    refs_valid  = [r for _, r in valid_pairs]

    accuracy = accuracy_score(refs_valid, preds_valid)
    f1 = f1_score(refs_valid, preds_valid)

    cm = confusion_matrix(refs_valid, preds_valid, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "="*20)
    print(f"\tAccuracy: {accuracy:.4f}")
    print(f"\tF1: {f1:.4f}")
    print(f"\tValid predictions: {sum(1 for p in predictions if p != -1)} out of {len(predictions)}")
    print("="*20, "\n")

    return {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n_valid": int(len(valid_pairs)),
        "n_total": int(len(predictions)),
    }

def evaluate_dev_loss(model: Module,
                      dataloader: DataLoader) -> float:
    """
    Computes the dev loss in trainin.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    progress_bar = tqdm(range(len(dataloader)), desc="Evaluating (dev set)")

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                outputs = model(**batch)
                loss = outputs.loss
            
            total_loss += loss.item()
            n_batches += 1

            progress_bar.update(1)

    return round(total_loss / n_batches, 4)

def evaluate_classification(model: torch.nn.Module,
                            dataloader: torch.utils.data.DataLoader
                            ) -> dict[str, float]:
    """"Evaluation for classification models"""
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating (classification) ..."):
            labels = batch["labels"]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
            logits = outputs.logits

            # 2 classes
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 20)
    print(f"\tAccuracy: {accuracy:.4f}")
    print(f"\tF1:       {f1:.4f}")
    print("=" * 20 + "\n")

    return {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n_total": len(all_labels),
    }

def evaluate_wrapper(model_type: str,
                        model: Module,
                        tokenizer_test,
                        dataloader: DataLoader) -> float:
    """Wrapper for test set evaluation"""
    if model_type == "clf" or model_type == "plm":
        metrics = evaluate_classification(model=model, 
                                          dataloader=dataloader)
        return metrics
    if model_type == "slm":
        metrics = evaluate_slm(model=model, 
                              tokenizer_test=tokenizer_test,
                              dataloader=dataloader)
        return metrics