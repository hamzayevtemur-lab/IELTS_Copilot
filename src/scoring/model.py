import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class MeanPooling(nn.Module):
    """
    Mean Pooling Layer:
    Averages the transformer hidden states across the sequence length dimension,
    while masking out padding tokens so they do not dilute the sentence embeddings.
    
    Tensor Transformations:
      Input last_hidden_state: [batch_size, seq_len, hidden_size]  (e.g. [8, 512, 768])
      Input attention_mask:    [batch_size, seq_len]               (e.g. [8, 512])
      Step 1 (unsqueeze):      [batch_size, seq_len, 1]
      Step 2 (expand):         [batch_size, seq_len, 768]
      Step 3 (sum over dim 1): [batch_size, 768]
      Step 4 (divide by mask): [batch_size, 768]
      Output:                  [batch_size, hidden_size]           (e.g. [8, 768])
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, 
        last_hidden_state: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Inputs:
            last_hidden_state: Tensor of shape [batch_size, seq_len, hidden_size]
                Full contextual representations output by DeBERTa encoder for all tokens.
            attention_mask: Tensor of shape [batch_size, seq_len]
                Binary mask: 1 for real word tokens, 0 for padding tokens.
                
        Returns:
            torch.Tensor of shape [batch_size, hidden_size]
                Single dense vector representing the semantics of the entire essay.
        """
        # Step 1 & 2: Broadcast attention_mask from [batch, seq_len] to [batch, seq_len, hidden_size]
        # Example: [8, 512] -> [8, 512, 1] -> [8, 512, 768]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

        # Step 3: Zero out padded token embeddings and sum real token embeddings along seq_len (dim 1)
        # Result shape: [batch_size, hidden_size]  (e.g. [8, 768])
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)

        # Step 4: Count number of real tokens per essay along seq_len (dim 1)
        # Result shape: [batch_size, hidden_size]
        sum_mask = input_mask_expanded.sum(dim=1)

        # Step 5: Prevent division by zero if an essay has all zeroes (clamp min to 1e-9)
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        # Step 6: Compute average embedding = sum of embeddings / count of tokens
        # Return shape: [batch_size, hidden_size]  (e.g. [8, 768])
        return sum_embeddings / sum_mask



class DebertaIELTSScorer(nn.Module):
    """
    Encoder-only multi-criteria IELTS essay scoring model.
    Backbone: Pretrained DeBERTa-v3 (Bidirectional Attention).
    Head: 2-Layer Multilayer Perceptron (MLP) for continuous multi-output regression.
    
    Complete End-to-End Tensor Flow:
      1. input_ids:          [batch_size, 512]
      2. attention_mask:      [batch_size, 512]
      3. encoder output:      [batch_size, 512, 768]   (DeBERTa token representations)
      4. mean pooler output:  [batch_size, 768]        (Document-level embedding)
      5. regressor layer 1:   [batch_size, 256]        (Non-linear projection + GELU)
      6. regressor dropout:   [batch_size, 256]        (Random zeroing during train)
      7. regressor layer 2:   [batch_size, 5]          (Final predicted band scores)
      8. loss_fn (if labels): scalar float tensor      (Smooth L1 distance between logits and labels)
    """

    def __init__(
        self, 
        model_name: str = "microsoft/deberta-v3-small",
        num_targets: int = 5,
        dropout_rate: float = 0.2,
    ):
        """
        Inputs:
            model_name: Hugging Face model repository (e.g. 'microsoft/deberta-v3-small' or 'base')
            num_targets: Number of continuous scores to predict (5: TR, CC, LR, GRA, Overall)
            dropout_rate: Probability of dropout regularization (0.2 = 20%)
        """
        super().__init__()

        # Load transformer architecture configuration
        self.config = AutoConfig.from_pretrained(model_name)
        # Load pretrained bidirectional encoder backbone (768-dim for base/small)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        # Initialize mean pooling layer
        self.pooler = MeanPooling()

        hidden_size = self.config.hidden_size  # Typically 768 for small/base

        # Multi-output regression head: projects [batch, 768] -> [batch, 256] -> [batch, 5]
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 256),     # Compress 768 features to 256
            nn.GELU(),                       # Gaussian Error Linear Unit activation
            nn.Dropout(dropout_rate),        # Prevents overfitting to training essays
            nn.Linear(256, num_targets),     # Output 5 continuous scores
        )

        # Loss function: Smooth L1 Loss (Huber Loss)
        # - Behavers like MSE (quadratic) for small errors (smooth gradient updates)
        # - Behavers like L1 (linear) for large errors (robust against outlier scores in dataset)
        self.loss_fn = nn.SmoothL1Loss()

    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        **kwargs,
    ) -> dict:
        """
        Executes forward pass of the scoring model.
        
        Inputs:
            input_ids: torch.LongTensor of shape [batch_size, seq_len] (e.g. [8, 512])
                Token indices encoding [CLS] question [SEP] essay [SEP]
            attention_mask: torch.LongTensor of shape [batch_size, seq_len] (e.g. [8, 512])
                1 for real tokens, 0 for padded tokens
            labels: Optional torch.FloatTensor of shape [batch_size, 5] (e.g. [8, 5])
                Ground-truth scores: [TR, CC, LR, GRA, Overall]
                
        Returns:
            dict containing:
                - 'logits': torch.FloatTensor of shape [batch_size, 5]
                    Raw continuous score predictions:
                    logits[:, 0] = Task Response (TR)
                    logits[:, 1] = Coherence & Cohesion (CC)
                    logits[:, 2] = Lexical Resource (LR)
                    logits[:, 3] = Grammatical Range & Accuracy (GRA)
                    logits[:, 4] = Overall Band Score
                - 'loss': torch.FloatTensor (scalar, shape []) if labels is provided, else None
        """
        # Step 1: Forward pass through DeBERTa encoder
        # outputs.last_hidden_state shape: [batch_size, seq_len, 768]
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden_state = outputs.last_hidden_state

        # Step 2: Pool sequence tokens into a single document vector
        # pooled shape: [batch_size, 768]
        pooled = self.pooler(last_hidden_state, attention_mask)

        # Step 3: Pass document vector through regression head
        # logits shape: [batch_size, 5]
        logits = self.regressor(pooled)

        # Step 4: Compute loss if ground-truth labels are provided
        loss = None
        if labels is not None:
            # Compares logits [batch_size, 5] against labels [batch_size, 5]
            # Returns a single scalar loss tensor
            loss = self.loss_fn(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
        }


if __name__ == "__main__":
    print("Running Model Tensor Shape Verification:")
    
    # 1. Instantiate small model
    model = DebertaIELTSScorer("microsoft/deberta-v3-small")

    # 2. Simulate a mini-batch of 2 essays with sequence length 128
    batch_size = 2
    seq_len = 128
    
    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len))   # Shape: [2, 128]
    dummy_mask = torch.ones((batch_size, seq_len))                     # Shape: [2, 128]
    dummy_labels = torch.tensor([                                     # Shape: [2, 5]
        [7.0, 6.5, 8.0, 7.0, 7.0],  # Essay 1
        [6.0, 6.0, 6.5, 6.0, 6.0],  # Essay 2
    ])

    print(f"Input IDs shape:      {dummy_input_ids.shape}")
    print(f"Attention mask shape: {dummy_mask.shape}")
    print(f"Labels shape:         {dummy_labels.shape}")

    # 3. Execute forward pass
    out = model(dummy_input_ids, dummy_mask, labels=dummy_labels)

    print("\nForward pass output:")
    print(f"Logits shape:         {out['logits'].shape}  (Expected: [2, 5])")
    print(f"Computed loss:        {out['loss'].item():.4f} (Scalar Huber Loss)")
    print(f"Essay 1 predictions:  {[round(x, 2) for x in out['logits'][0].tolist()]}")
    print(f"Targets correspond to: [TR, CC, LR, GRA, Overall]")
