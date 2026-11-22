# ✍️ IELTS Writing Copilot — Real-Time Autocomplete Assistant

An AI-powered IELTS Writing Task 2 assistant built to provide real-time, context-aware autocomplete suggestions as students write essays. Built with **Qwen2.5-1.5B-Instruct**, fine-tuned via **4-bit NormalFloat (NF4) QLoRA**, and engineered with **leak-free prompt grouping** and **response-only loss masking**.

---

## 📌 Architectural Overview

```
[IELTS Task 2 Prompt] + [Draft Typed So Far]
                     │
                     ▼
       ┌───────────────────────────┐
       │   Qwen2.5-1.5B (4-bit NF4)│
       │   + LoRA Adapter (r=16)   │
       └─────────────┬─────────────┘
                     │  (sub-100ms latency)
                     ▼
  "provides affordable mobility for lower-income citizens."
```

### Key Technical Highlights:
* **Zero Prompt Leakage**: Grouped train/val/test splitting on `question` via `GroupShuffleSplit`. Guarantees that evaluation tests the model on 100% unseen IELTS prompts.
* **Response-Only Loss Masking**: Prompts (System prompt + IELTS question + student draft) are masked with target `-100`. Gradients only update on the assistant completion tokens.
* **Rhetorical 5-Stage Slicing**: Slices essays cumulatively across 5 progression bands (Intro, Body 1 Claim, Body 1 Evidence, Body 2 Development, Conclusion).
* **Quantized Training**: 4-bit NormalFloat (NF4) with Double Quantization, PyTorch SDPA, and gradient checkpointing. Fits in <6 GB VRAM.

---

## 📊 Dataset & Quality Pipeline

1. **Raw Source**: `chillies/ielts-writing-task2-essays` (8,049 essays).
2. **Quality Filtering**:
   * Removed truncated essays (<220 words) and samples lacking sentence punctuation.
   * Filtered for `overall_band >= 6.0` to eliminate low-quality language priors.
   * Stripped forum/Telegram watermarks, hashtags, and word-count footers.
   * Retained **4,719 gold-standard essays** across 1,320 unique prompts.
3. **Generated Prefix-Completion Sets**:
   * **`data/autocomplete/train.jsonl`**: 16,605 pairs (1,056 questions)
   * **`data/autocomplete/val.jsonl`**: 3,975 pairs (132 questions)
   * **`data/autocomplete/test.jsonl`**: 3,015 pairs (132 questions)

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/hamzayevtemur-lab/IELTS_Copilot.git
cd IELTS_Copilot
pip install -r requirements.txt
```

### 2. Fine-Tuning with QLoRA
```bash
PYTHONPATH=. python3 src/models/qlora_trainer.py \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --epochs 1 \
    --batch_size 8 \
    --grad_accum 2 \
    --lr 2e-4 \
    --use_4bit
```

### 3. Benchmarking Baseline vs. Fine-Tuned Model
```bash
# Evaluate untuned base model (Baseline)
PYTHONPATH=. python3 src/models/evaluator.py

# Evaluate fine-tuned model
PYTHONPATH=. python3 src/models/evaluator.py \
    --adapter_path checkpoints/qwen_ielts_copilot/best_adapter
```

---

## 📂 Repository Structure

```
IELTS_Copilot/
├── configs/                  # Modular YAML configuration files
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── training_config.yaml
├── data/
│   ├── autocomplete/         # JSONL prefix-completion pairs (train, val, test)
│   └── processed/            # Cleaned parquet & CSV splits
├── src/
│   ├── data/
│   │   ├── cleaner.py        # Normalization & grouped leak-free split
│   │   ├── generator.py      # 5-stage cumulative prefix-completion generator
│   │   └── dataset.py        # PyTorch Dataset with -100 response loss masking
│   ├── models/
│   │   ├── qlora_trainer.py  # NF4 QLoRA fine-tuning loop
│   │   └── evaluator.py      # Perplexity, cross-entropy loss & human audit
│   └── inference/
│       ├── engine.py         # Model loading, KV-caching & generation
│       └── streamer.py       # Real-time token streaming generator
├── backend/                  # FastAPI real-time serving layer
└── frontend/                 # React inline ghost-text writing editor
```
