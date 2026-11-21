import os
import re
from pathlib import Path
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit

# 1. Project paths
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Load the raw dataset
RAW_DATA_PATH = PROJECT_ROOT / "data" / "ielts_task2_train.parquet"
df = pd.read_parquet(RAW_DATA_PATH)

print(f"Raw essays count: {len(df)}")

# 2. Drop GPT-related advice & highlights columns
gpt_cols = ["gpt_advices", "gpt_positive_highlights"]
clean_df = df.drop(columns=[col for col in gpt_cols if col in df.columns]).copy()
print(f"Dropped columns: {gpt_cols}")

# 3. Text cleaning function (removes hashtags, @mentions, trailing word count tags, excess spaces)
def clean_essay_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Strip telegram handles, hashtags, and word-count footers
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'⚡️.*?⚡️', '', text)
    text = re.sub(r'(?i)\b(?:word\s*count[:\s]*)?\d+\s*words\b', '', text)
    # Standardize horizontal spaces and paragraph newlines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()




# Apply cleaning
clean_df["question"] = clean_df["question"].apply(lambda q: re.sub(r'\s+', ' ', str(q)).strip())
clean_df["essay_text"] = clean_df["essay_text"].apply(clean_essay_text)


# 4. Compute cleaned word counts
clean_df["essay_words"] = clean_df["essay_text"].apply(lambda x: len(x.split()))


# 5. Apply filters: Band >= 6.0 AND Words >= 220 AND Proper sentence ending
valid_mask = (
    (clean_df["overall_band"] >= 6.0) &
    (clean_df["essay_words"] >= 220) &
    clean_df["essay_text"].apply(lambda x: x.endswith(('.', '!', '?', '"', "'")))
)

clean_df = clean_df[valid_mask].reset_index(drop=True)

# 6. Summary Stats & Verification
print(f"\n Cleaned essays count: {len(clean_df)} / {len(df)} ({len(clean_df)/len(df)*100:.1f}% retained)")
print(f" Unique IELTS questions: {clean_df['question'].nunique()}")
print(f" Remaining columns: {list(clean_df.columns)}")
print(f" Min band: {clean_df['overall_band'].min()}, Max band: {clean_df['overall_band'].max()}")
print(f" Min words: {clean_df['essay_words'].min()}, Max words: {clean_df['essay_words'].max()}")


# 7. Save to data/processed/
output_path = PROCESSED_DIR / "cleaned_essays.parquet"
clean_df.to_parquet(output_path, index=False)
print(f"\n Saved cleaned dataset to: {output_path}")



print("\n--- Splitting Dataset (Grouped by Question to Prevent Leakage) ---")

# Step 1: Split clean_df into (Train + Val) and Test (90% / 10%)
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
train_val_idx, test_idx = next(gss_test.split(clean_df, groups=clean_df["question"]))

train_val_df = clean_df.iloc[train_val_idx].reset_index(drop=True)
test_df = clean_df.iloc[test_idx].reset_index(drop=True)


# Step 2: Split (Train + Val) into Train (80% total) and Val (10% total)
# 0.10 / 0.90 ≈ 0.1111 relative validation size
val_relative_size = 0.10 / 0.90
gss_val = GroupShuffleSplit(n_splits=1, test_size=val_relative_size, random_state=42)
train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df["question"]))

train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
val_df = train_val_df.iloc[val_idx].reset_index(drop=True)


# Step 3: Zero-Leakage Assertion Check
train_q = set(train_df["question"])
val_q = set(val_df["question"])
test_q = set(test_df["question"])

assert len(train_q.intersection(val_q)) == 0, "Leakage detected: Train and Val share questions!"
assert len(train_q.intersection(test_q)) == 0, "Leakage detected: Train and Test share questions!"
assert len(val_q.intersection(test_q)) == 0, "Leakage detected: Val and Test share questions!"
print(" Zero-Leakage Verification PASSED: All splits contain mutually exclusive questions.")

# Step 4: Print Split Statistics
print(f" Train set: {len(train_df)} essays ({train_df['question'].nunique()} unique questions)")
print(f" Val set:   {len(val_df)} essays ({val_df['question'].nunique()} unique questions)")
print(f" Test set:  {len(test_df)} essays ({test_df['question'].nunique()} unique questions)")


# Step 5: Save as CSV files
train_path = PROCESSED_DIR / "train_essays.csv"
val_path = PROCESSED_DIR / "val_essays.csv"
test_path = PROCESSED_DIR / "test_essays.csv"

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"\n Successfully saved CSV splits to {PROCESSED_DIR}:")
print(f"  ├── {train_path.name}")
print(f"  ├── {val_path.name}")
print(f"  └── {test_path.name}")