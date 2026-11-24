import sys
import json
from pathlib import Path
from typing import Generator, Optional

# Ensure project root is accessible
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.engine import IELTSInferenceEngine
from src.inference.streamer import stream_autocomplete


class ModelService:
    """
    Singleton service managing the lifecycle of the IELTS Autocomplete model.
    Loads once on server startup and handles synchronous and streaming inference.
    """

    _instance: Optional["ModelService"] = None

    def __init__(
        self, 
        model_name: str="TemurbekHamzaev/qwen2.5-1.5b-ielts-copilot",
        use_4bit: bool=False
    ):
        self.model_name=model_name
        self.use_4bit=use_4bit
        self.engine: Optional[IELTSInferenceEngine]=None

    
    @classmethod
    def get_instance(cls)-> "ModelService":
        """Returns the global singleton instance."""
        if cls._instance is None:
            cls._instance = ModelService()

        return cls._instance


    def initialize(self):
        """Loads model weights into memory (called during FastAPI startup lifespan)."""
        if self.engine is None:
            print(f"[ModelService] Loading model: {self.model_name}...")
            self.engine=IELTSInferenceEngine(
                model_id=self.model_name,
                use_4bit=self.use_4bit
            )
            print("[ModelService] Model successfully loaded and ready for requests.")


    def generate_completion(
        self, 
        question: str, 
        draft: str,
        max_tokens: int=30,
        temperature: float=0.7,
    )->str:
        """Synchronously generates an essay continuation."""
        if self.engine is None:
            raise RuntimeError("ModelService is not initialized. Call initialize() first.")

        return self.engine.generate(
            question=question, 
            draft=draft,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )


    def stream_completion(
        self, 
        question: str,
        draft: str, 
        max_tokens: int=30,
        temperature: float=0.7,
    )-> Generator[str, None, None]:
        """
        Streams newly generated tokens in real time, formatted as Server-Sent Events (SSE).
        Yields: 'data: {"token": "..."}\n\n'
        """

        if self.engine is None:
            raise RuntimeError("ModelService is not initialized. Call initialize() first.")

        token_generator=stream_autocomplete(
            engine=self.engine,
            question=question,
            draft=draft, 
            max_new_tokens=max_tokens,
            temperature=temperature,
        )

        for token in token_generator:
            # Format as standard Server-Sent Event (SSE) JSON payload
            payload = json.dumps({"token": token})
            yield f"data: {payload}\n\n"


        # Signal completion to frontend
        yield f"data: {json.dumps({'done': True})}\n\n"


    
