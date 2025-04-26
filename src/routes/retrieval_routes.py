import asyncpg
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from database.postgres import get_postgres
from models.retrieval_models import (
    Metadata,
    RecordResponse,
    RetrievalRequest,
    RetrievalResponse,
)
from routes.product_routes import generate_embedding
from utils.auth import validate_api_key

router = APIRouter()


@router.post("/retrieval", response_model=RetrievalResponse)
async def retrieval(
    request: RetrievalRequest,
    api_key: str = Depends(validate_api_key),
    pool: asyncpg.Pool = Depends(get_postgres),
):
    """
    Retrieve documents from product_embeddings based on similarity to query.

    Parameters
    ----------
    request : RetrievalRequest
        Request payload with knowledge_id, query, and retrieval settings
    api_key : str
        API key from Bearer token (validated by dependency)
    pool : asyncpg.Pool
        Database connection pool

    Returns
    -------
    RetrievalResponse
        Response with matching documents

    Raises
    ------
    HTTPException
        If retrieval fails
    """
    try:
        # Extract parameters
        knowledge_id = request.knowledge_id
        query = request.query
        top_k = request.retrieval_setting.top_k
        score_threshold = request.retrieval_setting.score_threshold

        logger.info(
            f"Processing retrieval request for knowledge_id={knowledge_id}, query='{query}'"
        )

        # Generate embedding for query
        query_embedding = await generate_embedding(query)
        query_embedding_np = np.array(query_embedding)

        # Get database connection
        async with pool.acquire() as conn:
            # Retrieve similar products
            # Note: 1 - (embedding <=> $1) converts distance to similarity score
            rows = await conn.fetch(
                """
                SELECT id, name, brand, color, material
                FROM public.product_embeddings
                ORDER BY embedding <-> $1 DESC
                """,
                query_embedding_np,
            )

            # Filter by score threshold and limit results
            filtered_rows = [
                row for row in rows if row["similarity"] >= score_threshold
            ]
            limited_rows = filtered_rows[:top_k]

            # Format results into response
            records = []
            for row in limited_rows:
                # Combine product attributes for content field
                content = f"{row['name']} is a {row['color']} product made of {row['material']} by {row['brand']}."

                record = RecordResponse(
                    metadata=Metadata(
                        path=f"s3://products/{row['id']}.txt",
                        description=f"{row['brand']} product information",
                    ),
                    score=float(row["similarity"]),
                    title=row["name"],
                    content=content,
                )
                records.append(record)

            # Create response
            response = RetrievalResponse(records=records)
            logger.info(f"Returning {len(records)} matching products")
            return response

    except Exception as e:
        logger.error(f"Error processing retrieval request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process retrieval request: {e}",
        )
