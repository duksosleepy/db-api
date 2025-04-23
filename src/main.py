from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from database.postgres import init_postgres, close_postgres
from routes.routes import router
import dotenv
import uvicorn


@asynccontextmanager
async def lifespan(app: FastAPI):
    dotenv.load_dotenv()
    await init_postgres()
    yield
    await close_postgres()


app: FastAPI = FastAPI(lifespan=lifespan, title="FastAPI Portfolio RAG ChatBot API")
app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
