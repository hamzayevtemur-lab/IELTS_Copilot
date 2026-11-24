import sys
import threading
from typing import Generator
from pathlib import Path
from transformers import TextIteratorStreamer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.inference.engine import IELTSInferenceEngine



def stream_autocomplete(
    engine: IELTSInferenceEngine,
    question: str,
    draft: str,
    max_new_tokens: int = 30,
    temperature: float = 0.7,
    top_p: float = 0.9,
)-> Generator[str, None, None]:
    """
    Yields newly generated tokens in real time using a background thread and TextIteratorStreamer.
    """

    input_ids=engine.build_prompt_ids(question, draft)

    # Initialize streamer
    streamer=TextIteratorStreamer(
        engine.tokenizer,
        skip_prompt=True,  # Only stream completion tokens
        skip_special_tokens=True,  # Strip <|im_end|>
    )

    # Keyword arguments for model.generate
    generation_kwargs=dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.1,
        use_cache=True,
        eos_token_id=engine.tokenizer.eos_token_id,
        pad_token_id=engine.tokenizer.pad_token_id,
    )

    
    # Run generation in background thread
    thread = threading.Thread(target=engine.model.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream tokens word-by-word as they emerge from the model
    for new_text in streamer:
        if new_text:
            yield new_text

    thread.join()



if __name__ == "__main__":
    # Test streaming in terminal using your Hugging Face model
    engine = IELTSInferenceEngine(use_4bit=False)

    q = "In many countries, traditional customs are disappearing. Is this positive or negative?"
    d = "Globalization has accelerated the decline of cultural traditions. In my view, this development"

    print("\n[Test Real-Time Streaming]")
    print(f"Draft: {d} ...")
    print("Streaming: ", end="", flush=True)

    for chunk in stream_autocomplete(engine, q, d, max_new_tokens=25):
        print(chunk, end="", flush=True)
        
    print("\n\n Done streaming!")
