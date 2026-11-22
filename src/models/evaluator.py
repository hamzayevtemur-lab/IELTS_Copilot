import os
import sys
import json
import math
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import IELTSAutocompleteDataset, get_data_collator


def load_evaluation_model(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    adapter_path: str = None,
):
    """Loads base model or fine-tuned PeftModel."""
    is_cuda = torch.cuda.is_available()
    device_map = "auto" if is_cuda else None
    compute_dtype = torch.float16 if is_cuda else torch.float32

    print(f"Loading base model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        dtype=compute_dtype,
        attn_implementation="sdpa" if is_cuda else None,
        trust_remote_code=True,
    )

    if adapter_path and os.path.exists(adapter_path):
        print(f" Loading LoRA adapter weights from: {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
    elif adapter_path:
        print(f"⚠️ Adapter path '{adapter_path}' not found! Evaluating untuned base model.")

    model.eval()
    return model, tokenizer


def compute_test_perplexity(
    model,
    tokenizer,
    test_jsonl: str = "data/autocomplete/test.jsonl",
    batch_size: int = 4,
    max_length: int = 512,
) -> tuple[float, float]:
    """
    Computes Cross-Entropy Loss and Perplexity on test set completions.
    Prompt tokens (-100) are excluded from the loss calculation.
    """
    dataset = IELTSAutocompleteDataset(test_jsonl, tokenizer, max_length=max_length)
    collator = get_data_collator(tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    total_loss = 0.0
    total_tokens = 0
    device = next(model.parameters()).device

    print(f"\nEvaluating Loss & Perplexity over {len(dataset)} test samples...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits

            # Shift logits and labels by 1 for causal LM loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Cross entropy with reduction='none' to sum over completion tokens only
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # Count valid tokens (where label != -100)
            valid_tokens = (shift_labels != -100).sum().item()
            total_loss += loss.item()
            total_tokens += valid_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, perplexity


def generate_qualitative_samples(
    model,
    tokenizer,
    test_jsonl: str = "data/autocomplete/test.jsonl",
    num_samples: int = 10,
    max_new_tokens: int = 35,
):
    """
    Generates side-by-side completions on unseen test prompts for human evaluation.
    """
    print(f"\n{'='*70}")
    print(f" QUALITATIVE HUMAN AUDIT ({num_samples} Unseen Test Samples)")
    print(f"{'='*70}")

    samples = []
    with open(test_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
            if len(samples) >= num_samples:
                break

    device = next(model.parameters()).device

    for idx, sample in enumerate(samples, 1):
        messages = sample["messages"]
        prompt_messages = messages[:-1]  # System + User
        reference_completion = sample["completion"]

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Slice out generated completion tokens
        gen_tokens = output_ids[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        print(f"\n[Sample #{idx}] Progress: {int(sample.get('prefix_ratio', 0)*100)}% through essay")
        print(f"Question:  {sample['question'][:90]}...")
        print(f"Draft:     ...{sample['prefix'][-120:]}")
        print(f"Human Ref: '{reference_completion}'")
        print(f"Model Gen: '{generated_text}'")
        print("-" * 70)


def run_evaluation(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    adapter_path: str = None,
    test_file: str = "data/autocomplete/test.jsonl",
    batch_size: int = 4,
    num_samples: int = 10,
):
    label = "Fine-Tuned Model" if adapter_path else "Untuned Base Model (Baseline)"
    print(f"\n{'#'*70}")
    print(f" BENCHMARKING: {label}")
    print(f"{'#'*70}")

    model, tokenizer = load_evaluation_model(model_name, adapter_path)
    test_loss, perplexity = compute_test_perplexity(
        model, tokenizer, test_jsonl=test_file, batch_size=batch_size
    )

    print("\n" + "=" * 50)
    print(f" RESULTS FOR: {label}")
    print(f"  ├── Test Cross-Entropy Loss: {test_loss:.4f}")
    print(f"  └── Test Perplexity (PPL):    {perplexity:.2f}")
    print("=" * 50)

    # Human inspection samples
    generate_qualitative_samples(model, tokenizer, test_jsonl=test_file, num_samples=num_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate IELTS Autocomplete Model on Unseen Test Set")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to best_adapter directory")
    parser.add_argument("--test_file", type=str, default="data/autocomplete/test.jsonl")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--samples", type=int, default=10, help="Number of qualitative samples to inspect")
    args = parser.parse_args()

    run_evaluation(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        test_file=args.test_file,
        batch_size=args.batch_size,
        num_samples=args.samples,
    )
