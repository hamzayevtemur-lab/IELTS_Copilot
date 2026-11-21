import os
import argparse
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from src.data.dataset import IELTSAutocompleteDataset, get_data_collator

def train(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    train_file: str = "data/autocomplete/train.jsonl",
    val_file: str = "data/autocomplete/val.jsonl",
    output_dir: str = "checkpoints/qwen_ielts_copilot",
    epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_length: int = 768,
    lora_r: int = 16,
    lora_alpha: int = 32,
    use_4bit: bool = True,
):
    print("=" * 60)
    print(f"Starting QLoRA Fine-Tuning for: {model_name}")
    print(f"Effective Batch Size: {batch_size * gradient_accumulation_steps}")
    print("=" * 60)

    is_cuda=torch.cuda.is_available()
    device_map="auto" if is_cuda else None

    # 1. Load Tokenizer
    tokenizer=AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token

    # 2. Configure 4-bit Quantization (NF4)
    bnb_config=None
    compute_dtype = (
        torch.bfloat16 if (is_cuda and torch.cuda.is_bf16_supported()) else torch.float16
    )

    if use_4bit and is_cuda:
        print(" Configuring 4-bit NormalFloat (NF4) with Double Quantization...")
        bnb_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        print(" Running without 4-bit quantization (Standard precision)...")

    # 3. Load base Model
    print(f"Loading base model weights from {model_name}...")
    model=AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=compute_dtype if is_cuda else torch.float32,
        trust_remote_code=True,
    )

    if use_4bit and is_cuda:
        model = prepare_model_for_kbit_training(model)

    # 4. Configure LoRA (PEFT) across all 7 linear projections
    lora_config=LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )

    model=get_peft_model(model, lora_config)

    print("\n--- Trainable Parameters Summary ---")
    model.print_trainable_parameters()


    # 5. Load Datasets using our custom response-masked Dataset
    train_dataset = IELTSAutocompleteDataset(train_file, tokenizer, max_length=max_length)
    val_dataset = IELTSAutocompleteDataset(val_file, tokenizer, max_length=max_length)
    data_collator = get_data_collator(tokenizer)

    
    # 6. Training Arguments
    training_args=TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=(is_cuda and compute_dtype == torch.float16),
        bf16=(is_cuda and compute_dtype == torch.bfloat16),
        optim="paged_adamw_8bit" if is_cuda else "adamw_torch",
        report_to="none",
        dataloader_pin_memory=False if not is_cuda else True,
    )

    # 7. Initialize Hugging Face Trainer
    trainer=Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # 8. Run Training
    print("\n🚀 Commencing training loop...")
    trainer.train()

    # 9. Save Best Adapter Weights & Tokenizer
    best_adapter_dir = os.path.join(output_dir, "best_adapter")
    print(f"\n Saving best LoRA adapter to: {best_adapter_dir}")

    model.save_pretrained(best_adapter_dir)
    tokenizer.save_pretrained(best_adapter_dir)
    print(" Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning for IELTS Writing Copilot")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--use_4bit", action="store_true", default=True)
    args = parser.parse_args()

    train(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        use_4bit=args.use_4bit,
    )
    
