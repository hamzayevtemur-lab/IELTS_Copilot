import os
import sys
import argparse
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, login

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



def merge_lora_weights(
    base_model_name: str="Qwen/Qwen2.5-1.5B-Instruct",
    adapter_path: str="checkpoints/qwen_ielts_copilot/checkpoint-200",
    output_dir: str = "checkpoints/qwen_ielts_merged",
):
    """
    Loads base model in 16-bit, attaches LoRA adapter, fuses weights (W = W0 + Delta W),
    and saves a standalone, unquantized model to disk.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(" FUSING LORA ADAPTER INTO BASE MODEL")
    print(f"  ├── Base Model:    {base_model_name}")
    print(f"  ├── LoRA Adapter:  {adapter_path}")
    print(f"  └── Output Target: {output_dir}")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n1. Loading base model in 16-bit ({device})...")
    base_model=AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto" if device=="cuda" else None,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    print(f"2. Attaching fine-tuned LoRA adapter from: {adapter_path}...")
    lora_model=PeftModel.from_pretrained(
        base_model, adapter_path
    )

    print("3. Executing merge_and_unload() mathematical fusion...")
    merged_model=lora_model.merge_and_unload()

    print(f"4. Saving fused standalone model to: {output_dir}...")
    merged_model.save_pretrained(str(output_path), safe_serialization=True)
    tokenizer.save_pretrained(str(output_path))

    print("\n Standalone merged model successfully saved locally!")
    return str(output_path)




def push_to_huggingface(
    local_dir: str,
    repo_id: str,
    token: str = None,
    private: bool = False,
):
    """
    Creates repository if needed and uploads the merged model folder to Hugging Face Hub.
    """
    hf_token = token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN must be provided as argument or set as environment variable.")

    print("\n" + "=" * 65)
    print(" PUBLISHING TO HUGGING FACE HUB")
    print(f"  ├── Local Directory: {local_dir}")
    print(f"  └── Target Repo ID:  {repo_id}")
    print("=" * 65)

    login(token=hf_token)
    api = HfApi()

    print(f"1. Ensuring repository '{repo_id}' exists...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )

    print(f"2. Uploading files from '{local_dir}' to Hugging Face...")
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model",
    )


    print("\n" + "=" * 65)
    print(f"🎉 Model is live on Hugging Face Hub!")
    print(f"👉 https://huggingface.co/{repo_id}")
    print("=" * 65)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA weights and push to Hugging Face Hub.")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="checkpoints/qwen_ielts_copilot/checkpoint-200")
    parser.add_argument("--output_dir", type=str, default="checkpoints/qwen_ielts_merged")
    parser.add_argument("--repo_id", type=str, default="TemurbekHamzaev/qwen2.5-1.5b-ielts-copilot")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face write token")
    parser.add_argument("--push", action="store_true", help="Push to Hugging Face Hub after merging")
    args = parser.parse_args()


    # Step 1: Merge
    merged_path = merge_lora_weights(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
    )


    # Step 2: Optional Push to Hub
    if args.push:
        push_to_huggingface(
            local_dir=merged_path,
            repo_id=args.repo_id,
            token=args.token,
        )



