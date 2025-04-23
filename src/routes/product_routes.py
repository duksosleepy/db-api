import os
from typing import List

import asyncpg
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from openai import OpenAI

from database.postgres import get_postgres
from models.product_models import (
    PortfolioEntryCreate,
    PortfolioEntryResponse,
    QueryRequest,
)

router = APIRouter()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID"),
    project=os.getenv("OPENAI_PROJECT_ID"),
)


async def generate_embedding(content: str) -> List[float]:
    """
    Generate an embedding for the given content using OpenAI API.
    """
    try:
        content = content.replace("\n", " ")
        response = client.embeddings.create(
            input=[content],
            model="text-embedding-3-small",
        )
        return response.data[0].embedding
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating embedding: {e}"
        )


@router.post("/add-entry/", response_model=PortfolioEntryResponse)
async def add_portfolio_entry(
    entry: PortfolioEntryCreate, pool: asyncpg.Pool = Depends(get_postgres)
):
    """
    Add a new portfolio entry and store its embedding in PostgreSQL.
    """
    try:
        embedding = await generate_embedding(entry.content)

        embedding_np = np.array(embedding)

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO portfolio_embeddings (content, embedding)
                VALUES ($1, $2)
                RETURNING id, content, embedding
                """,
                entry.content,
                embedding_np,
            )
            if row:
                return PortfolioEntryResponse(
                    id=row["id"],
                    content=row["content"],
                    embedding=row["embedding"],
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to insert entry into the database.",
                )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add entry: {e}")


@router.post("/chat/")
async def chat(query: QueryRequest, pool: asyncpg.Pool = Depends(get_postgres)):
    """
    Chat with the portfolio chatbot by retrieving relevant information from stored embeddings
    and using it as context to generate a response.
    """
    try:
        query_embedding = await generate_embedding(query.query)

        query_embedding_np = np.array(query_embedding)

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT content, embedding <=> $1 AS similarity
                FROM portfolio_embeddings
                ORDER BY similarity
                LIMIT 5
                """,
                query_embedding_np,
            )

            context = "\n".join([row["content"] for row in rows])

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are YOUR NAME, and you are answering questions as yourself, using 'I' and 'my' in your responses. "
                        "You should respond to questions about your portfolio, skills, experience, and education. For example, you should answer questions about specific technologies you've worked with, such as Java, React, or other tools. "
                        "If you have relevant experience with a technology, describe it concisely. For example, if asked about Java, describe your experience using it. "
                        "You should also answer questions about your education, including your experience at school, and your work in relevant industries. "
                        "However, if a question is completely unrelated to your professional experience, such as questions about recipes, trivia, or non-technical personal matters, respond with: 'That question isn't relevant to my experience or skills.' "
                        "Focus on answering technical and career-related questions, but only reject questions that are clearly off-topic."
                        "If they ask you about a technology you havent used, you can say: 'I haven't worked with that technology yet, but I'm always eager to learn new things.'"
                        "Answer any personal questions that are related to technology, like 'What are our favorite languages?' or 'What technology/language/anything tech are you most excited about?'"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {query.query}",
                },
            ],
            max_tokens=200,
            stream=False,
        )

        return {"response": response.choices[0].message.content}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to process chat request: {e}"
        )
