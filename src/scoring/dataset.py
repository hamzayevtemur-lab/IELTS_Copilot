import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# The 5 criteria scores we train the model to predict
TARGET_COLUMNS = [
    "task_achievement_band",     # Index 0: Task Response (TR)
    "coherence_cohesion_band",   # Index 1: Coherence & Cohesion (CC)
    "lexical_resource_band",     # Index 2: Lexical Resource (LR)
    "grammatical_range_band",    # Index 3: Grammatical Range & Accuracy (GRA)
    "overall_band",              # Index 4: Overall Band Score
]


class IELTSScoringDataset(Dataset):
    """
    PyTorch Dataset for multi-criteria IELTS essay scoring.
    Tokenizes pair: [CLS] Question [SEP] Essay Text [SEP].
    
    Tensor Shape Lifecycle:
      - Raw text: (str, str)
      - Tokenizer output: [1, 512] (contains batch dim of 1)
      - Squeezed item: [512] (1D tensor for single sample)
      - Labels: [5] (1D float32 tensor of target scores)
      - DataLoader batch: [batch_size, 512] for inputs, [batch_size, 5] for labels
    """

    def __init__(
        self, 
        csv_file: str,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        """
        Inputs:
            csv_file: Path to train/val/test CSV with columns ['question', 'essay_text', ...]
            tokenizer: Pretrained DeBERTa/BERT tokenizer
            max_length: Maximum sequence length in tokens (default 512)
        """
        # Load CSV into DataFrame: shape is (num_essays, num_columns)
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Ensure all 5 target score columns exist and contain valid numeric floats
        for col in TARGET_COLUMNS:
            if col not in self.df.columns:
                raise ValueError(f"Missing required target column '{col}' in {csv_file}")

            # Replace missing/corrupt values with Band 6.0 as a safe fallback
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(6.0)

    def __len__(self) -> int:
        """
        Returns:
            int: Total number of essay examples in this dataset (e.g. 3321 for train)
        """
        return len(self.df)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves and tokenizes the idx-th essay example.
        
        Input:
            idx: Integer index of the sample (0 <= idx < len(self))
            
        Returns:
            dict containing:
                - 'input_ids': torch.LongTensor of shape [512]
                    Token IDs encoding: [CLS] question [SEP] essay [SEP] [PAD] ...
                - 'attention_mask': torch.LongTensor of shape [512]
                    1 for real tokens, 0 for padded [PAD] tokens
                - 'token_type_ids' (if DeBERTa): torch.LongTensor of shape [512]
                    0 for Question tokens, 1 for Essay tokens
                - 'labels': torch.FloatTensor of shape [5]
                    [TR_score, CC_score, LR_score, GRA_score, Overall_score]
        """
        row = self.df.iloc[idx]
        question = str(row["question"]).strip()
        essay = str(row["essay_text"]).strip()

        # Tokenize pair: [CLS] question [SEP] essay [SEP]
        # inputs contains tensors with an initial batch dimension: shape [1, 512]
        inputs = self.tokenizer(
            question,
            essay,
            max_length=self.max_length,
            padding="max_length",     # Pad shorter sequences with [PAD] (0) up to 512
            truncation=True,          # Cut off sequences longer than 512 tokens
            return_tensors="pt",      # Return PyTorch tensors instead of python lists
        )

        # Build 5-dimensional target vector: shape [5], dtype=float32
        # Example: tensor([7.0, 6.5, 8.0, 7.0, 7.0])
        labels = torch.tensor(
            [float(row[col]) for col in TARGET_COLUMNS],
            dtype=torch.float32,
        )

        # Squeeze out the leading dimension [1, 512] -> [512]
        # This allows DataLoader to automatically stack N samples into [batch_size, 512]
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item["labels"] = labels

        return item


def get_scoring_dataloaders(
    train_csv: str = "data/processed/train_essays.csv",
    val_csv: str = "data/processed/val_essays.csv",
    model_name: str = "microsoft/deberta-v3-small",
    batch_size: int = 8,
    max_length: int = 512,
):
    """
    Constructs PyTorch DataLoaders for model training and validation.
    
    Inputs:
        train_csv: Path to training essays CSV
        val_csv: Path to validation essays CSV
        model_name: Hugging Face model identifier for tokenizer
        batch_size: Number of essays per mini-batch (e.g. 8 or 16)
        max_length: Sequence length (512 tokens)
        
    Returns:
        tuple (train_loader, val_loader, tokenizer):
            - train_loader: DataLoader yielding batches of shape:
                input_ids:      [batch_size, 512]
                attention_mask: [batch_size, 512]
                labels:         [batch_size, 5]
            - val_loader: DataLoader yielding same batch shapes (shuffle=False)
            - tokenizer: Instantiated AutoTokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = IELTSScoringDataset(train_csv, tokenizer, max_length=max_length)
    val_ds = IELTSScoringDataset(val_csv, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, tokenizer


if __name__ == "__main__":
    # Test dataset pipeline locally
    test_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
    ds = IELTSScoringDataset("data/processed/train_essays.csv", test_tokenizer)
    sample = ds[0]

    print("Dataset verification:")
    print(f"Total essays loaded: {len(ds)}")
    print(f"input_ids shape:      {sample['input_ids'].shape}      (expected: torch.Size([512]))")
    print(f"attention_mask shape: {sample['attention_mask'].shape} (expected: torch.Size([512]))")
    print(f"labels shape:         {sample['labels'].shape}         (expected: torch.Size([5]))")
    print(f"Sample labels values: {sample['labels'].tolist()}")
