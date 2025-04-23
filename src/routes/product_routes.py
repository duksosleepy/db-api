from typing import List

import asyncpg
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastembed import TextEmbedding

from database.postgres import get_postgres
from models.product_models import (
    ProductCreate,
    ProductResponse,
    QueryRequest,
    QueryResponse,
)

router = APIRouter()

# Initialize the embedding model once
embedding_model = TextEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


async def generate_embedding(text: str) -> List[float]:
    """
    Generate an embedding for the given text using FastEmbed with sentence-transformers/all-MiniLM-L6-v2.
    """  # noqa: E501
    try:
        text = text.replace("\n", " ")

        # FastEmbed's embed function returns an iterator of embeddings
        embeddings = list(embedding_model.embed([text]))

        # Return the first (and only) embedding as a list
        return embeddings[0].tolist()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating embedding: {e}"
        )


@router.post("/add-product/", response_model=ProductResponse)
async def add_product(
    product: ProductCreate, pool: asyncpg.Pool = Depends(get_postgres)
):
    """
    Add a new product and store its embedding in PostgreSQL.
    """
    try:
        # Combine all text fields for embedding generation
        combined_text = (
            f"{product.name} {product.brand} {product.color} {product.material}"
        )
        embedding = await generate_embedding(combined_text)

        embedding_np = np.array(embedding)

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO product_embeddings (name, brand, color, material, embedding)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id, name, brand, color, material, embedding
                """,
                product.name,
                product.brand,
                product.color,
                product.material,
                embedding_np,
            )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to add product: {e}"
        )


@router.post("/query/", response_model=List[QueryResponse])
async def query_products(
    query: QueryRequest, pool: asyncpg.Pool = Depends(get_postgres)
):
    """
    Query products based on similarity to the input query.
    """
    try:
        query_embedding = await generate_embedding(query.query)

        query_embedding_np = np.array(query_embedding)

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT name, brand, color, material, embedding <=> $1 AS similarity
                FROM products
                ORDER BY similarity
                LIMIT 5
                """,
                query_embedding_np,
            )

            results = [
                QueryResponse(
                    name=row["name"],
                    brand=row["brand"],
                    color=row["color"],
                    material=row["material"],
                    similarity=row["similarity"],
                )
                for row in rows
            ]

            return results

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to process query request: {e}"
        )
