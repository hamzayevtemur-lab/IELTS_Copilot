from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse

from backend.schemas.autocomplete import AutocompleteRequest, AutocompleteResponse
from backend.services.model_service import ModelService

router = APIRouter(prefix="/api/autocomplete", tags=["Autocomplete"])


def get_model_service()-> ModelService:
    """Dependency injection helper to access the singleton model service."""

    return ModelService.get_instance()


@router.post(
    "",
    response_model=AutocompleteResponse,
    summary="Generate full autocomplete suggestion (Synchronous)",
    description="Accepts an IELTS prompt and the draft written so far, returning a complete continuation."
)
async def generate_autocomplete(
    request: AutocompleteRequest,
    service: ModelService = Depends(get_model_service),
):  

    try: 
        completion=service.generate_completion(
            question=request.question,
            draft=request.draft,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        return AutocompleteResponse(
            completion=completion,
            question=request.question,
            tokens_generated=len(completion.split()),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")



@router.post(
    "/stream",
    summary="Stream autocomplete suggestions word-by-word (Server-Sent Events)",
    description="Streams newly generated tokens in real time using Server-Sent Events (SSE)."
)
async def stream_autocomplete_endpoint(
    request: AutocompleteRequest,
    service: ModelService = Depends(get_model_service),
):
    try:
        event_stream=service.stream_completion(
            question=request.question,
            draft=request.draft,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        return StreamingResponse(
            event_stream,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disables proxy buffering (e.g. Nginx)
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming error: {str(e)}")
