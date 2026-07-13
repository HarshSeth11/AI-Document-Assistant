from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from app.database import get_db, engine, Base
from app.models import Document
import app.document_service as document_service
from app.vector_service import search_chunks, get_collection_stats
from app.retrieval_service import retrieve_with_confidence
from app.agent_service import ask_agent
from pydantic import BaseModel

Base.metadata.create_all(bind=engine)

app = FastAPI(title="DocuMind API", version="1.0.0")

ALLOWED_EXTENSIONS = {"pdf", "txt"}

@app.get("/health")
def health():
    return {"status": "ok", "message": "DocuMind API running"}

@app.post("/documents/upload")
def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Validate file type
    print("Validating File extension...")
    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}"
        )
    print("File extension validated...")
    
    # Read file content
    print("Reading content...")
    content = file.file.read()
    print("Content reading completed...")
    
    # Process it
    try:
        document = document_service.process_document(
            db=db,
            file_content=content,
            filename=file.filename,
            file_type=file_ext
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {
        "id": document.id,
        "filename": document.filename,
        "status": document.status,
        "chunk_count": document.chunk_count,
        "upload_date": document.upload_date
    }

@app.get("/documents")
def list_documents(db: Session = Depends(get_db)):
    documents = document_service.get_all_documents(db)
    return [
        {
            "id": d.id,
            "filename": d.filename,
            "status": d.status,
            "chunk_count": d.chunk_count,
            "upload_date": d.upload_date
        }
        for d in documents
    ]

@app.get("/documents/{document_id}/chunks")
def get_chunks(document_id: str, db: Session = Depends(get_db)):
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    chunks = document_service.get_document_chunks(db, document_id)
    return {
        "document_id": document_id,
        "filename": document.filename,
        "total_chunks": len(chunks),
        "chunks": [
            {"index": c.chunk_index, "content": c.content[:200] + "..."}
            for c in chunks
        ]
    }

@app.delete("/documents/{document_id}")
def delete_document_route(document_id: str, db: Session = Depends(get_db)):
    success = document_service.delete_document(db, document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "success", "message": f"Document {document_id} deleted"}

@app.get("/search")
def search(query: str, document_id: str = None, n_results: int = 4):
    results = search_chunks(query, n_results=n_results, document_id=document_id)
    return {
        "query": query,
        "results_count": len(results),
        "results": results
    }

@app.get("/debug/chroma-stats")
def chroma_stats():
    return get_collection_stats()


@app.get("/search/hybrid")
def hybrid_search_endpoint(
    query: str, 
    document_id: str = None, 
    confidence_threshold: float = 0.7,
    db: Session = Depends(get_db)
):
    result = retrieve_with_confidence(
        db=db, 
        query=query, 
        document_id=document_id,
        confidence_threshold=confidence_threshold
    )
    
    return {
        "query": query,
        "confident": result["confident"],
        "reason": result["reason"],
        "chunks_found": len(result["chunks"]),
        "chunks": [
            {
                "content": c["content"][:200] + "...",
                "source_type": c["source"],
                "filename": c["metadata"]["filename"],
                "chunk_index": c["metadata"]["chunk_index"]
            }
            for c in result["chunks"]
        ]
    }

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    result = ask_agent(db, request.question)
    return {
        "question": request.question,
        "answer": result["answer"],
        "tools_used": result["tools_used"]
    }