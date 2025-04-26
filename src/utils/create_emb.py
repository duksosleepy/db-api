import asyncio
import os
from typing import List, Tuple

import asyncpg
import dotenv
import numpy as np
from fastembed import TextEmbedding
from loguru import logger
from pgvector.asyncpg import register_vector

# Tune this batch size to optimize throughput
BATCH_SIZE = 30

# Initialize the embedding model once
embedding_model = TextEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


async def generate_batch_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using FastEmbed.

    Parameters
    ----------
    texts : List[str]
        List of text strings to generate embeddings for

    Returns
    -------
    List[List[float]]
        List of embedding vectors
    """
    try:
        # Clean text by replacing newlines with spaces
        cleaned_texts = [text.replace("\n", " ") for text in texts if text]

        # Generate embeddings for all texts in the batch
        embeddings = list(embedding_model.embed(cleaned_texts))

        # Convert embeddings to lists
        return [embedding.tolist() for embedding in embeddings]
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise


async def update_table_embeddings(
    conn: asyncpg.Connection, table_name: str, id_column: str, text_column: str
) -> None:
    """
    Update the embedding column for all rows in a given table using batched processing.

    Parameters
    ----------
    conn : asyncpg.Connection
        Database connection
    table_name : str
        Name of the table to update
    id_column : str
        Name of the ID column
    text_column : str
        Name of the text column to generate embeddings from
    """
    try:
        # Fetch all rows that need embeddings
        rows = await conn.fetch(
            f"SELECT {id_column}, {text_column} FROM {table_name};"
        )

        if not rows:
            logger.info(f"No rows found in table {table_name}.")
            return

        # Prepare batches
        batches: List[Tuple[List[int], List[str]]] = []
        current_ids: List[int] = []
        current_texts: List[str] = []

        for row in rows:
            row_id, text = row[id_column], row[text_column]
            if not text:
                logger.warning(
                    f"{table_name.capitalize()} {id_column} {row_id} has no {text_column}; skipping."
                )
                continue

            current_ids.append(row_id)
            current_texts.append(text)

            if len(current_texts) >= BATCH_SIZE:
                batches.append((current_ids, current_texts))
                current_ids = []
                current_texts = []

        if current_texts:
            batches.append((current_ids, current_texts))

        total = sum(len(batch_ids) for batch_ids, _ in batches)
        logger.info(
            f"Processing {total} rows from {table_name} in {len(batches)} batches (batch size = {BATCH_SIZE})..."
        )

        # Process each batch
        for batch_ids, batch_texts in batches:
            try:
                # Generate embeddings for this batch
                embeddings = await generate_batch_embeddings(batch_texts)

                # Update database with embeddings
                for row_id, embedding in zip(batch_ids, embeddings):
                    # Convert embedding to numpy array for pgvector
                    embedding_np = np.array(embedding)
                    await conn.execute(
                        f"UPDATE {table_name} SET embedding = $1 WHERE {id_column} = $2;",
                        embedding_np,
                        row_id,
                    )

                logger.info(
                    f"Updated batch for {table_name}: {len(batch_ids)} rows"
                )
            except Exception as e:
                logger.error(
                    f"Error processing batch for {table_name} with {len(batch_ids)} rows: {e}"
                )

    except Exception as e:
        logger.error(f"Error updating embeddings for table {table_name}: {e}")
        raise


async def update_product_embeddings(conn: asyncpg.Connection) -> None:
    """
    Update embeddings for products by combining text fields and generating embeddings.

    Parameters
    ----------
    conn : asyncpg.Connection
        Database connection
    """
    try:
        # Get products that need embeddings
        rows = await conn.fetch(
            """
            SELECT id, name, brand, color, material
            FROM product_embeddings
            WHERE embedding IS NULL OR vector_dims(embedding) = 0
            """
        )

        if not rows:
            logger.info("No products found that need embeddings.")
            return

        logger.info(f"Found {len(rows)} products that need embeddings.")

        # Process in batches
        batch_size = BATCH_SIZE
        total_products = len(rows)

        for i in range(0, total_products, batch_size):
            batch = rows[i : i + batch_size]
            product_ids = []
            combined_texts = []

            # Combine text fields for each product
            for row in batch:
                product_id = row["id"]
                # Combine fields just like in product_routes.py
                combined_text = f"{row['name']} {row['brand']} {row['color']} {row['material']}"

                product_ids.append(product_id)
                combined_texts.append(combined_text)

            # Generate embeddings for the batch
            embeddings = await generate_batch_embeddings(combined_texts)

            # Update each product with its embedding
            for product_id, embedding in zip(product_ids, embeddings):
                embedding_np = np.array(embedding)
                await conn.execute(
                    """
                    UPDATE product_embeddings
                    SET embedding = $1
                    WHERE id = $2
                    """,
                    embedding_np,
                    product_id,
                )

            logger.info(
                f"Updated embeddings for batch of {len(batch)} products ({i + len(batch)}/{total_products})"
            )

    except Exception as e:
        logger.error(f"Error updating product embeddings: {e}")
        raise


async def main() -> None:
    """
    Main function to update embeddings in the product_embeddings table in the lug_rag database.
    """
    # Load environment variables
    dotenv.load_dotenv()

    # Connect to PostgreSQL database
    DATABASE_URL = os.environ.get("DATABASE_URL")
    if not DATABASE_URL:
        logger.error("DATABASE_URL environment variable not set.")
        return

    try:
        # Initialize connection pool with pgvector support
        logger.info("Initializing PostgreSQL connection pool...")

        async def initialize_vector(conn):
            await register_vector(conn)

        conn_pool = await asyncpg.create_pool(
            dsn=os.getenv("DATABASE_URL"), init=initialize_vector
        )
        logger.info("PostgreSQL connection pool created successfully.")

        # Update product embeddings - use a connection from the pool
        logger.info(
            "Starting embedding update process for product_embeddings table..."
        )

        async with conn_pool.acquire() as conn:
            await update_product_embeddings(conn)

        await conn_pool.close()
        logger.info("Embedding update process completed successfully.")

    except Exception as e:
        logger.error(f"Error in main function: {e}")


if __name__ == "__main__":
    asyncio.run(main())
