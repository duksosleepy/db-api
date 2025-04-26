import os
from contextlib import asynccontextmanager

import dotenv
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database.postgres import close_postgres, init_postgres
from routes.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    dotenv.load_dotenv()
    await init_postgres()
    yield
    await close_postgres()


app: FastAPI = FastAPI(
    lifespan=lifespan, title="FastAPI Portfolio RAG ChatBot API"
)
app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Run the application directly if this script is executed
if __name__ == "__main__":
    # Get port from environment variable or use 80 as default
    port = int(os.getenv("PORT", 80))

    # Run with uvicorn server
    uvicorn.run(
        "main:app", host="127.0.0.1", port=port, reload=False, log_level="info"
    )
