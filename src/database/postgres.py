import os
from typing import Optional

import asyncpg
from loguru import logger
from pgvector.asyncpg import register_vector

conn_pool: Optional[asyncpg.Pool] = None


async def init_postgres() -> None:
    """
    Initialize the PostgreSQL connection pool
    """
    global conn_pool
    try:
        logger.info("Initializing PostgreSQL connection pool...")

        async def initialize_vector(conn):
            await register_vector(conn)

        conn_pool = await asyncpg.create_pool(
            dsn=os.getenv("DATABASE_URL"), init=initialize_vector
        )
        logger.info("PostgreSQL connection pool created successfully.")

    except Exception as e:
        logger.error(f"Error initializing PostgreSQL connection pool: {e}")
        raise


async def get_postgres() -> asyncpg.Pool:
    """
    Get a reference to the PostgreSQL connection pool.

    Returns
    -------
    asyncpg.Pool
        The connection pool object to the PostgreSQL database.
    """
    global conn_pool
    if conn_pool is None:
        logger.error("Connection pool is not initialized.")
        raise ConnectionError("PostgreSQL connection pool is not initialized.")
    try:
        return conn_pool
    except Exception as e:
        logger.error(f"Failed to return PostgreSQL connection pool: {e}")
        raise


async def close_postgres() -> None:
    """
    Close the PostgreSQL connection pool.
    """
    global conn_pool
    if conn_pool is not None:
        try:
            logger.info("Closing PostgreSQL connection pool...")
            await conn_pool.close()
            logger.info("PostgreSQL connection pool closed successfully.")
        except Exception as e:
            logger.error(f"Error closing PostgreSQL connection pool: {e}")
            raise
    else:
        logger.warning("PostgreSQL connection pool was not initialized.")
