from fastapi import APIRouter

from routes.product_routes import router as product_router
from routes.retrieval_routes import router as retrieval_router

# Create main router
router = APIRouter()

# Include product routes
router.include_router(product_router, tags=["products"])

# Include retrieval routes
router.include_router(retrieval_router, tags=["retrieval"])
