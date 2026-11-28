import sys
import json
import asyncio
import threading
from pathlib import Path
from typing import Generator, Optional, AsyncGenerator

# Ensure project root is accessible
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.engine import IELTSInferenceEngine
from src.inference.streamer import stream_autocomplete

LOCAL_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "qwen_ielts_merged"
DEFAULT_MODEL = str(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH.exists() else "TemurbekHamzaev/qwen2.5-1.5b-ielts-copilot"


class ModelService:
    """
    Singleton service managing the lifecycle of the IELTS Autocomplete model.
    Loads once on server startup and handles synchronous and streaming inference.
    """

    _instance: Optional["ModelService"] = None

    def __init__(
        self, 
        model_name: str = DEFAULT_MODEL,
        use_4bit: bool = False
    ):
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.engine: Optional[IELTSInferenceEngine] = None
        self.lock = threading.Lock()

    
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
            self.engine = IELTSInferenceEngine(
                model_name=self.model_name,
                use_4bit=self.use_4bit,
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


    async def stream_completion(
        self, 
        question: str,
        draft: str, 
        max_tokens: int = 30,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """
        Streams newly generated tokens in real time, formatted as Server-Sent Events (SSE).
        Yields: 'data: {"token": "..."}\n\n'
        """
        if self.engine is None:
            raise RuntimeError("ModelService is not initialized. Call initialize() first.")

        # Timeout acquisition prevents deadlocks if previous request was aborted by client
        acquired = self.lock.acquire(timeout=2.0)
        if not acquired:
            print("[ModelService] Previous generation busy, skipping stale request")
            yield f"data: {json.dumps({'done': True})}\n\n"
            return

        try:
            token_generator = stream_autocomplete(
                engine=self.engine,
                question=question,
                draft=draft, 
                max_new_tokens=max_tokens,
                temperature=temperature,
            )

            for token in token_generator:
                payload = json.dumps({"token": token})
                yield f"data: {payload}\n\n"
                await asyncio.sleep(0.001)  # Flush network buffer and yield to event loop

            yield f"data: {json.dumps({'done': True})}\n\n"
        except GeneratorExit:
            # Client aborted or disconnected - exit cleanly
            pass
        except Exception as e:
            print(f"[ModelService] Stream exception: {e}")
            yield f"data: {json.dumps({'done': True})}\n\n"
        finally:
            if self.lock.locked():
                self.lock.release()


    
