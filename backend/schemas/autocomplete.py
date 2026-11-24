from pydantic import BaseModel, Field


class AutocompleteRequest(BaseModel):
    """Payload sent by the frontend editor when requesting an essay completion."""

    question: str = Field(
        ...,
        min_length=5,
        description="The IELTS Writing Task 2 prompt or question.",
        example="Some people believe university education should be free. Discuss."
    )

    draft: str = Field(
        ...,
        min_length=1,
        description="The essay draft written so far up to the user's cursor.",
        example="In recent years, higher education accessibility has sparked intense debate. Proponents argue that"
    )


    max_tokens: int = Field(
        default=30,
        ge=5,
        le=100,
        description="Maximum new tokens to generate."
    )

    temperature: float = Field(
        default=0.7,
        ge=0.1,
        le=1.5,
        description="Sampling temperature for creativity vs predictability."
    )



class AutocompleteResponse(BaseModel):
    """Synchronous JSON response format."""
    completion: str = Field(..., description="The suggested continuation text.")
    question: str
    tokens_generated: int


