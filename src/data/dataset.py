import json
import torch
from pathlib import Path
from typing import Dict, List, Any
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, DataCollatorForSeq2Seq


class IELTSAutocompleteDataset(Dataset):
    """
    PyTorch Dataset for IELTS Writing Copilot.
    Loads prefix-completion pairs from JSONL and applies response-only loss masking (-100).
    """
    def __init__(
        self, 
        jsonl_path: str | Path,
        tokenizer: PreTrainedTokenizer,
        max_length: int=768
    ):
        self.jsonl_path=Path(jsonl_path)
        self.tokenizer=tokenizer
        self.max_length=max_length

        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found at: {self.jsonl_path}")

        print(f"Loading autocomplete examples from: {self.jsonl_path.name}...")
        self.examples: List[Dict[str, Any]]=[]

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))

        print(f" Loaded {len(self.examples)} samples from {self.jsonl_path.name}")
    

    def __len__(self)->int:
        return len(self.examples)

    
    def __getitem__(self, idx: int)-> Dict[str, torch.Tensor]:
        example=self.examples[idx]
        messages=example["messages"] # [system, user, assistant]

        # 1. Format the prompt (system + user) with generation prompt
        prompt_text = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True
        )


        # 2. Format the entire sequence including assistant completion
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # 3. Tokenize both texts
        prompt_ids = self.tokenizer.encode(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length
        )
        full_ids = self.tokenizer.encode(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length
        )
        

        # Append EOS token if not already included
        #if len(full_ids)< self.max_length and full_ids[-1] !=self.tokenizer.eos_token_id:
        #    full_ids.append(self.tokenizer.eos_token_id)

        
        # 4. Construct labels with -100 masking for prompt tokens
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }


def get_data_collator(tokenizer: PreTrainedTokenizer)-> DataCollatorForSeq2Seq:
    """
    Returns a dynamic padding collator that pads input_ids with pad_token_id
    and labels with -100 so padded positions do not contribute to loss.
    """
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        label_pad_token_id=-100,
    )


if __name__=="__main__":
    # Smoke test with Qwen tokenizer
    from transformers import AutoTokenizer

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Testing dataset collation with tokenizer: {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    sample_dataset = IELTSAutocompleteDataset(
        jsonl_path="data/autocomplete/val.jsonl",
        tokenizer=tokenizer,
        max_length=512,
    )

    collator = get_data_collator(tokenizer)
    batch = collator([sample_dataset[0], sample_dataset[1]])

    print("\n--- Smoke Test Verification (Collator Output) ---")
    print(f"Batch Input IDs shape:  {batch['input_ids'].shape}")
    print(f"Batch Labels shape:     {batch['labels'].shape}")
    
    # Verify masking on first item in batch
    labels_0 = batch['labels'][0]
    masked_count = (labels_0 == -100).sum().item()
    target_count = (labels_0 != -100).sum().item()
    print(f"Prompt + Pad tokens masked (-100): {masked_count}")
    print(f"Target completion tokens:          {target_count}")

    # Decode target tokens to verify correctness
    target_ids = labels_0[labels_0 != -100]
    print(f"Decoded target text: '{tokenizer.decode(target_ids).strip()}'")




