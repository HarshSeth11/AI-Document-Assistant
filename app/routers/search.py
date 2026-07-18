from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.retrieval_service import retrieve_with_confidence
from app.vector_service import get_collection_stats

router = APIRouter(prefix="/search", tags=["Search"])

@router.get("/hybrid")
def hybrid_search_endpoint(
    query: str,
    document_id: str = None,
    confidence_threshold: float = 0.7,
    db: Session = Depends(get_db)
):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    result = retrieve_with_confidence(
        db=db, query=query, document_id=document_id, confidence_threshold=confidence_threshold
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

@router.get("/debug/chroma-stats")
def chroma_stats():
    return get_collection_stats()