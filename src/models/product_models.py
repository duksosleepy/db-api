from typing import List, Optional

from pydantic import BaseModel


class ProductCreate(BaseModel):
    name: str
    brand: str
    color: str
    material: str


class ProductResponse(BaseModel):
    id: int
    name: str
    brand: str
    color: str
    material: str
    embedding: Optional[List[float]] = None


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    name: str
    brand: str
    color: str
    material: str
    similarity: float
