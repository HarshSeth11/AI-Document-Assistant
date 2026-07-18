from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import engine, Base
from app.routers import documents, search, chat

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="DocuMind API",
    description="Agentic RAG system for document Q&A, summarization, and comparison",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(documents.router)
app.include_router(search.router)
app.include_router(chat.router)

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "message": "DocuMind API running", "version": "1.0.0"}