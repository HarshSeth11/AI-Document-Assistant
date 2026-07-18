from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Document
from app import document_service

router = APIRouter(prefix="/documents", tags=["Documents"])

ALLOWED_EXTENSIONS = {"pdf", "txt"}

@router.post("/upload")
def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_ext}'. Allowed: {ALLOWED_EXTENSIONS}"
        )

    content = file.file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        document = document_service.process_document(
            db=db, file_content=content, filename=file.filename, file_type=file_ext
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

    return {
        "id": document.id,
        "filename": document.filename,
        "status": document.status,
        "chunk_count": document.chunk_count,
        "upload_date": document.upload_date
    }

@router.get("")
def list_documents(db: Session = Depends(get_db)):
    documents = document_service.get_all_documents(db)
    return [
        {
            "id": d.id, "filename": d.filename, "status": d.status,
            "chunk_count": d.chunk_count, "upload_date": d.upload_date
        }
        for d in documents
    ]

@router.get("/{document_id}/chunks")
def get_chunks(document_id: str, db: Session = Depends(get_db)):
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")

    chunks = document_service.get_document_chunks(db, document_id)
    return {
        "document_id": document_id,
        "filename": document.filename,
        "total_chunks": len(chunks),
        "chunks": [{"index": c.chunk_index, "content": c.content[:200] + "..."} for c in chunks]
    }

@router.delete("/{document_id}")
def delete_document_route(document_id: str, db: Session = Depends(get_db)):
    success = document_service.delete_document(db, document_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")
    return {"status": "success", "message": f"Document {document_id} deleted"}