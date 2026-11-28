import os
import sys
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any
import argparse

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from sklearn.metrics import mean_absolute_error, cohen_kappa_score


# Ensure project root is accessible in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.scoring.dataset import IELTSScoringDataset, TARGET_COLUMNS
from src.scoring.model import DebertaIELTSScorer

CRITERIA_NAMES = ["TR", "CC", "LR", "GRA", "Overall"]


def round_to_ielts_band(scores: np.ndarray)-> np.ndarray:
    """
    Rounds continuous predicted scores to standard IELTS half-bands (e.g. 6.5, 7.0, 7.5).
    Formula: round(score * 2) / 2, clipped between 0.0 and 9.0.
    """
    return np.clip(np.round(scores * 2.0) / 2.0, 0.0, 9.0)


def compute_scoring_metrics(eval_pred)-> Dict[str, float]:
    """
    Evaluates multi-output predictions against ground truth labels.
    
    Inputs (eval_pred tuple from Trainer):
        - predictions: np.ndarray of shape [N, 5]
            Continuous model outputs: [TR, CC, LR, GRA, Overall]
        - label_ids: np.ndarray of shape [N, 5]
            Ground truth examiner scores
            
    Returns:
        dict containing:
            - 'eval_mae': Mean Absolute Error averaged over all 5 targets
            - 'eval_qwk': Quadratic Weighted Kappa on overall band
            - Per-criterion MAE for TR, CC, LR, GRA, and Overall
    """

    predictions, labels=eval_pred

    metrics={}
    total_mae=0.0

    # Calculate MAE for each individual criterion
    for i, name in enumerate(CRITERIA_NAMES):
        pred_i=predictions[:, i]
        label_i=labels[:, i]

        mae_i=mean_absolute_error(label_i, pred_i)
        metrics[f"mae_{name}"]=round(float(mae_i), 4)
        total_mae+=mae_i

    # Average MAE across all 5 criteria
    avg_mae=total_mae/len(CRITERIA_NAMES)
    metrics["eval_mae"]=round(float(avg_mae), 4)

    # Compute Quadratic Weighted Kappa (QWK) on discrete half-bands (Standard AES metric)
    pred_bands = (round_to_ielts_band(predictions[:, 4]) * 2).astype(int)
    true_bands = (round_to_ielts_band(labels[:, 4]) * 2).astype(int)

    try:
        qwk = cohen_kappa_score(true_bands, pred_bands, weights="quadratic")
        metrics["qwk_overall"] = round(float(qwk), 4)
    except Exception:
        metrics["qwk_overall"] = 0.0

    return metrics



def train(
    train_csv: str = "data/processed/train_essays.csv",
    val_csv: str = "data/processed/val_essays.csv",
    model_name: str = "microsoft/deberta-v3-small",
    output_dir: str = "checkpoints/deberta_ielts_scorer",
    batch_size: int = 8,
    num_epochs: int = 4,
    learning_rate: float = 2e-5,
    max_length: int = 512,
):
    
    """
    Fine-tunes DeBERTa-v3 on multi-criteria IELTS essay scoring.
    """
    is_cuda = torch.cuda.is_available()
    device_name = "CUDA GPU" if is_cuda else "CPU (Mac)"

    print("=" * 60)
    print(f"🚀 Starting DeBERTa Multi-Criteria Scoring Fine-Tuning ({device_name})")
    print(f"• Model Backbone: {model_name}")
    print(f"• Batch Size:     {batch_size}")
    print(f"• Learning Rate:  {learning_rate}")
    print(f"• Max Sequence:   {max_length} tokens")
    print("=" * 60)

    # Step 1: Initialize Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Step 2: Initialize Datasets
    print(f"Loading datasets: {train_csv} and {val_csv}...")
    train_dataset = IELTSScoringDataset(train_csv, tokenizer, max_length=max_length)
    val_dataset = IELTSScoringDataset(val_csv, tokenizer, max_length=max_length)
    print(f"• Train size: {len(train_dataset)} essays")
    print(f"• Val size:   {len(val_dataset)} essays")


    # Step 3: Instantiate Model
    print("Initializing DebertaIELTSScorer architecture...")
    model = DebertaIELTSScorer(model_name=model_name, num_targets=5)

    # Step 4: Configure Hugging Face TrainingArguments
    training_args=TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,                # 10% warmup steps prevents early gradient explosion
        evaluation_strategy="epoch",      # Evaluate at the end of each epoch
        save_strategy="epoch",            # Save checkpoint at the end of each epoch
        save_total_limit=2,               # Keep only top 2 checkpoints
        load_best_model_at_end=True,      # Automatically restore weights with lowest validation MAE
        metric_for_best_model="mae",
        greater_is_better=False,          # Lower MAE is better
        fp16=is_cuda,                     # Mixed precision acceleration if GPU is available
        logging_steps=25,
        report_to="none",                 # Disable wandb/external telemetry
    )

    # Step 5: Initialize Trainer
    rainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_scoring_metrics,
    )

    # Step 6: Execute Training Loop
    print("\n⚡ Training underway...")
    train_result = trainer.train()

    # Step 7: Final Validation Evaluation
    print("\n🔍 Running final evaluation on validation set...")
    eval_metrics = trainer.evaluate()

    print("\n" + "=" * 60)
    print("🏆 Training Complete! Validation Results:")
    print("=" * 60)
    for k, v in eval_metrics.items():
        if k.startswith("eval_"):
            print(f"  {k[5:]:15}: {v}")



    # Step 8: Save Final Model and Tokenizer
    final_save_dir = Path(output_dir) / "final_model"
    final_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save custom PyTorch state dict and tokenizer
    torch.save(model.state_dict(), final_save_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(str(final_save_dir))
    print(f"\n💾 Model weights & tokenizer saved to: {final_save_dir}")

    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa-v3 for IELTS Multi-Criteria Scoring")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-small")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)

    args = parser.parse_args()

    train(
        model_name=args.model_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
    )



