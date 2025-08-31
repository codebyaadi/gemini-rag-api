import os
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from dotenv import load_dotenv
from rag_service import RAGService

load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("GEMINI_API_KEY not found in environment variables.")

app = FastAPI(
    title="Gemini RAG API",
    description="A simple API for a Retrieval-Augmented Generation service using Google Gemini.",
    version="1.0.0",
)

try:
    rag_service = RAGService()
except Exception as e:
    raise RuntimeError(f"Failed to initialize RAGService: {e}")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_rag(request: QueryRequest):
    if not request.question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question field cannot be empty.",
        )
    try:
        answer = rag_service.answer_question(request.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred: {e}",
        )
