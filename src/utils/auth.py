import os

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger

security = HTTPBearer()


async def validate_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """
    Validate the API key from the Authorization header.

    Parameters
    ----------
    credentials : HTTPAuthorizationCredentials
        The credentials from the Authorization header

    Returns
    -------
    str
        The validated API key

    Raises
    ------
    HTTPException
        If the API key is missing, invalid, or the environment variable is not set
    """
    api_key = os.getenv("API_KEY")

    if not api_key:
        logger.error("API_KEY environment variable not set")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: API key not configured",
        )

    if credentials.scheme.lower() != "bearer":
        logger.warning(f"Invalid authentication scheme: {credentials.scheme}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme. Expected 'Bearer'",
        )

    if credentials.credentials != api_key:
        logger.warning("Invalid API key provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return credentials.credentials
