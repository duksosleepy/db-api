from typing import List

from pydantic import BaseModel


class RetrievalSetting(BaseModel):
    """Settings for retrieval operations."""

    top_k: int = 2
    score_threshold: float = 0.5


class RetrievalRequest(BaseModel):
    """Request model for document retrieval."""

    knowledge_id: str
    query: str
    retrieval_setting: RetrievalSetting


class Metadata(BaseModel):
    """Metadata for retrieved documents."""

    path: str = ""
    description: str = ""


class RecordResponse(BaseModel):
    """Response model for a single retrieved document."""

    metadata: Metadata
    score: float
    title: str
    content: str


class RetrievalResponse(BaseModel):
    """Response model for document retrieval."""

    records: List[RecordResponse]
