import chromadb
from chromadb.utils import embedding_functions
from sqlalchemy.orm import Session
from app.models import Chunk, Document

CHROMA_PATH = "./chroma_store"

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = chroma_client.get_or_create_collection(
    name="document_chunks",
    embedding_function=embedding_fn
)

def embed_and_store_chunks(db: Session, document_id: str):
    """
    Take all chunks for a document from PostgreSQL,
    embed them in ChromaDB, and link them back.
    """
    chunks = db.query(Chunk).filter(Chunk.document_id == document_id).all()
    
    if not chunks:
        print(f"⚠️ No chunks found for document {document_id}")
        return 0
    
    document = db.query(Document).filter(Document.id == document_id).first()
    
    ids = []
    documents_text = []
    metadatas = []
    
    for chunk in chunks:
        chroma_chunk_id = f"chunk_{chunk.id}"
        
        ids.append(chroma_chunk_id)
        documents_text.append(chunk.content)
        metadatas.append({
            "document_id": document_id,
            "filename": document.filename,
            "chunk_index": chunk.chunk_index,
            "postgres_chunk_id": chunk.id
        })
        
        # Link back to PostgreSQL
        chunk.chroma_id = chroma_chunk_id
    
    # Add to ChromaDB in one batch call — efficient
    collection.add(
        ids=ids,
        documents=documents_text,
        metadatas=metadatas
    )
    
    db.commit()
    
    print(f"✅ Embedded {len(chunks)} chunks for document {document_id}")
    return len(chunks)

def search_chunks(query: str, n_results: int = 4, document_id: str = None) -> list:
    """
    Search ChromaDB for relevant chunks.
    Optionally filter by document_id.
    """
    where_filter = {"document_id": document_id} if document_id else None
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter
    )
    
    if not results["documents"] or not results["documents"][0]:
        return []
    
    chunks_found = []
    for i in range(len(results["documents"][0])):
        chunks_found.append({
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })
    
    return chunks_found

def delete_document_vectors(document_id: str):
    """Delete all ChromaDB vectors for a document"""
    collection.delete(where={"document_id": document_id})
    print(f"✅ Deleted vectors for document {document_id}")

def get_collection_stats():
    """Debug helper — see what's in ChromaDB"""
    return {
        "total_chunks": collection.count()
    }