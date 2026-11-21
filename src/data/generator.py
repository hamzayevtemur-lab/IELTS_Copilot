import os
import re
import json
import random
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "autocomplete"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# System prompt used during Qwen fine-tuning / inference
SYSTEM_PROMPT = (
    "You are an expert IELTS Writing Task 2 assistant. Given the essay prompt "
    "and the draft written so far, continue the essay with a fluent, grammatically "
    "accurate, and contextually relevant continuation."
)

def extract_prefix_completion_pairs(
    essay_text: str, 
    question: str, 
    essay_id: str, 
    cuts_per_essay: int=5,
    min_completion_words: int=8,
    max_completion_words: int=30
)-> list:
    """
    Slices an essay into multiple (prefix, completion) examples 
    sampled across early, middle, and late stages.
    """
    words=essay_text.split()
    total_words=len(words)

    if total_words<100:
        return []


    # 5 progression stages covering Intro, Body 1, Transition, Body 2, and Conclusion
    cut_intervals = [
        (0.12, 0.25),  # 1. Introduction / Thesis statement
        (0.28, 0.42),  # 2. Body Paragraph 1: Main claim
        (0.45, 0.58),  # 3. Body Paragraph 1: Evidence / Transition to Body 2
        (0.62, 0.75),  # 4. Body Paragraph 2: Development / Counter-claim
        (0.78, 0.90),  # 5. Conclusion / Final summary
    ]
    pairs=[]

    for low_ration, high_ratio in cut_intervals:
        cut_idx=int(total_words * random.uniform(low_ration, high_ratio))

        prefix_words=words[:cut_idx]
        remaining_words=words[cut_idx:]

        if len(remaining_words)<min_completion_words:
            continue

        # Completion length: 15 to 25 words
        completion_len=min(len(remaining_words), random.randint(12, max_completion_words))
        completion_words=remaining_words[:completion_len]

        prefix_str = " ".join(prefix_words)
        completion_str = " ".join(completion_words)

        # Standard ChatML formatted conversation for Qwen
        chat_format = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"[Question]: {question}\n\n[Draft]:\n{prefix_str}"
                },
                {"role": "assistant", "content": completion_str}
            ]
        }

        # Also store raw fields for inspection / evaluation
        example = {
            "essay_id": essay_id,
            "question": question,
            "prefix": prefix_str,
            "completion": completion_str,
            "prefix_ratio": round(cut_idx / total_words, 2),
            "messages": chat_format["messages"]
        }

        pairs.append(example)

    return pairs




def generate_jsonl(input_csv: Path, output_jsonl: Path, cuts_per_essay: int = 5):
    """Processes a CSV split and exports prefix-completion pairs to JSONL."""

    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)

    total_samples=0
    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for _, row in df.iterrows():
            pairs = extract_prefix_completion_pairs(
                essay_text=row["essay_text"],
                question=row["question"],
                essay_id=row["essay_id"],
                cuts_per_essay=cuts_per_essay
            )
            for pair in pairs:
                f_out.write(json.dumps(pair, ensure_ascii=False)+"\n")
                total_samples += 1

    print(f"Generated {total_samples} pairs ➔ {output_jsonl.name}")


if __name__ == "__main__":
    random.seed(42)
    
    print("=== Generating Autocomplete Datasets (Phase 3) ===")
    generate_jsonl(PROCESSED_DIR / "train_essays.csv", OUTPUT_DIR / "train.jsonl")
    generate_jsonl(PROCESSED_DIR / "val_essays.csv",   OUTPUT_DIR / "val.jsonl")
    generate_jsonl(PROCESSED_DIR / "test_essays.csv",  OUTPUT_DIR / "test.jsonl")
    print("\n All JSONL datasets generated successfully!")



