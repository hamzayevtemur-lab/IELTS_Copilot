import os
import sys
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert IELTS Writing Task 2 assistant. Given the essay prompt "
    "and the draft written so far, continue the essay with a fluent, grammatically "
    "accurate, and contextually relevant continuation."
)


class IELTSInferenceEngine:
    """
    Inference Engine for IELTS Writing Copilot.
    Handles model loading, KV-caching, and prompt formatting.
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path: str = None,
        use_4bit: bool = True,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
    
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.system_prompt = system_prompt
        self.is_cuda = torch.cuda.is_available()

        print(f"Initializing Inference Engine ({'CUDA' if self.is_cuda else 'CPU/MPS'})...")
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model(use_4bit)

    
    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    
    def _load_model(self, use_4bit: bool):
        device_map = "auto" if self.is_cuda else None
        compute_dtype = torch.float16 if self.is_cuda else torch.float32

        bnb_config=None
        if use_4bit and self.is_cuda:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )


        print(f"Loading weights for {self.model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            dtype=compute_dtype,
            attn_implementation="sdpa" if self.is_cuda else None,
            trust_remote_code=True,
        )

        if self.adapter_path and os.path.exists(self.adapter_path):
            print(f" Attaching fine-tuned LoRA adapter from: {self.adapter_path}...")
            model = PeftModel.from_pretrained(model, self.adapter_path)
        else:
            print(" Running on untuned foundation weights.")

        model.eval()

        return model

    def build_prompt_ids(self, question: str, draft: str)-> torch.Tensor:
        """Formats the student prompt into ChatML token IDs."""
        messages=[
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"[Question]: {question.strip()}\n\n[Draft]:\n{draft.strip()}"
            }
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        device = next(self.model.parameters()).device
        return self.tokenizer.encode(prompt_text, return_tensors="pt").to(device)


    def generate(
        self,
        question: str,
        draft: str,
        max_new_tokens: int = 30,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generates a complete continuation synchronously."""
        input_ids = self.build_prompt_ids(question, draft)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.1,
                use_cache=True,  # KV-Cache for fast generation
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        
        gen_tokens = output_ids[0][prompt_len:]

        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()



if __name__ == "__main__":
    # Test local generation
    engine = IELTSInferenceEngine(use_4bit=False)
    q = "Some people believe that university education should be free for everyone. Discuss."
    d = "Over the past few decades, the accessibility of tertiary education has sparked intense debate. Proponents argue that"
    
    print("\n[Test Autocomplete Generation]")
    print(f"Question: {q}")
    print(f"Draft:    {d}")
    print("Generating...")
    
    completion = engine.generate(q, d, max_new_tokens=25)
    print(f"\n➔ Suggested Completion: '{completion}'")
