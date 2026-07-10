from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from database import get_db, engine, Base
from models import Document
import document_service

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