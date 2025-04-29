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


# Updated retrieval_routes.py with SQL query logging


@router.post("/retrieval", response_model=RetrievalResponse)
async def retrieval(
    request: RetrievalRequest,
    api_key: str = Depends(validate_api_key),
    pool: asyncpg.Pool = Depends(get_postgres),
):
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

        # Log planned SQL query (without embedding values for brevity)
        sql_query = """
        SELECT id, name, brand, color, material,
               1 - (embedding <=> $1) AS similarity
        FROM public.product_embeddings
        WHERE 1 - (embedding <=> $1) >= $2
        ORDER BY similarity DESC
        LIMIT $3
        """

        logger.info(f"Executing SQL query: {sql_query}")
        logger.info(
            f"Query parameters: score_threshold={score_threshold}, top_k={top_k}"
        )

        # Get database connection
        async with pool.acquire() as conn:
            # Execute the query with improved ordering and similarity calculation
            rows = await conn.fetch(
                sql_query, query_embedding_np, score_threshold, top_k
            )

            logger.info(f"Query returned {len(rows)} rows")

            # Process results as before
            records = []
            for row in rows:
                # Log individual row data for debugging
                logger.debug(
                    f"Processing row: id={row['id']}, name={row['name']}, similarity={row['similarity']}"
                )

                content = f"{row['name']}, {row['color']}, {row['material']}, {row['brand']}."

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

            response = RetrievalResponse(records=records)
            logger.info(f"Returning {len(records)} matching products")
            return response

    except Exception as e:
        logger.error(f"Error processing retrieval request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process retrieval request: {e}",
        )
