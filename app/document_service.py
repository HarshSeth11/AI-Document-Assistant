import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session
from app.models import Document, Chunk

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_uploaded_file(file_content: bytes, filename: str) -> str:
    """Save uploaded file to disk, return the file path"""
    filepath = os.path.join(UPLOAD_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(file_content)
    return filepath

def load_document(filepath: str, file_type: str) -> list:
    """Load document content based on file type"""
    if file_type == "pdf":
        loader = PyPDFLoader(filepath)
    elif file_type == "txt":
        loader = TextLoader(filepath, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    return loader.load()

def chunk_document(documents: list, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """Split documents into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_documents(documents)

def process_document(
    db: Session,
    file_content: bytes,
    filename: str,
    file_type: str
) -> Document:
    
    print("STEP 1: Creating document record...")
    document = Document(
        filename=filename,
        file_type=file_type,
        status="processing",
        chunk_count=0
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    print(f"STEP 1 DONE: {document.id}")
    
    try:
        print("STEP 2: Saving file to disk...")
        filepath = save_uploaded_file(file_content, f"{document.id}_{filename}")
        print(f"STEP 2 DONE: {filepath}")
        
        print("STEP 3: Loading document...")
        loaded_docs = load_document(filepath, file_type)
        print(f"STEP 3 DONE: {len(loaded_docs)} docs loaded")
        
        print("STEP 4: Chunking...")
        chunks = chunk_document(loaded_docs)
        print(f"STEP 4 DONE: {len(chunks)} chunks")
        
        print("STEP 5: Creating chunk records...")
        for i, chunk in enumerate(chunks):
            chunk_record = Chunk(
                document_id=document.id,
                chunk_index=i,
                content=chunk.page_content,
                chroma_id=None
            )
            db.add(chunk_record)
        print("STEP 5 DONE")
        
        print("STEP 6: Updating document status...")
        document.status = "ready"
        document.chunk_count = len(chunks)
        db.commit()
        db.refresh(document)
        print("STEP 6 DONE")
        
        print(f"✅ Processed {filename}: {len(chunks)} chunks created")
        
    except Exception as e:
        document.status = "failed"
        db.commit()
        print(f"❌ Failed to process {filename}: {str(e)}")
        raise
    
    return document

def get_document_chunks(db: Session, document_id: str) -> list:
    """Get all chunks for a document"""
    return db.query(Chunk).filter(Chunk.document_id == document_id).order_by(Chunk.chunk_index).all()

def get_all_documents(db: Session) -> list:
    """Get all documents"""
    return db.query(Document).order_by(Document.upload_date.desc()).all()

def delete_document(db: Session, document_id: str) -> bool:
    """Delete a document and its chunks (cascade)"""
    document = db.query(Document).filter(Document.id == document_id).first()
    if document:
        # Delete file from disk
        for f in os.listdir(UPLOAD_DIR):
            if f.startswith(document.id):
                os.remove(os.path.join(UPLOAD_DIR, f))
        
        db.delete(document)  # cascade deletes chunks automatically
        db.commit()
        return True
    return False