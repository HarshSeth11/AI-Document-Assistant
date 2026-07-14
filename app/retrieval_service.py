import os
import json
from sqlalchemy.orm import Session
from sqlalchemy import or_
from groq import Groq
from app.models import Chunk, Document
from app.vector_service import search_chunks as semantic_search
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def keyword_search_postgres(db: Session, query: str, document_id: str = None, k: int = 4) -> list:
    """
    Simple keyword search using PostgreSQL ILIKE (case-insensitive match).
    For production, you'd use PostgreSQL full-text search (tsvector),
    but ILIKE is enough to demonstrate the concept.
    """
    query_words = query.lower().split()
    
    db_query = db.query(Chunk).join(Document)
    
    if document_id:
        db_query = db_query.filter(Chunk.document_id == document_id)
    
    # Build OR conditions for each significant word (skip short words)
    conditions = [
        Chunk.content.ilike(f"%{word}%") 
        for word in query_words 
        if len(word) > 3  # skip "the", "is", "a" etc.
    ]
    
    if not conditions:
        return []
    
    results = db_query.filter(or_(*conditions)).limit(k).all()
    
    return [
        {
            "content": chunk.content,
            "metadata": {
                "document_id": chunk.document_id,
                "filename": chunk.document.filename,
                "chunk_index": chunk.chunk_index,
                "postgres_chunk_id": chunk.id
            },
            "source": "keyword"
        }
        for chunk in results
    ]

def hybrid_search(db: Session, query: str, document_id: str = None, k: int = 4) -> list:
    semantic_results = semantic_search(query, n_results=k, document_id=document_id)
    for r in semantic_results:
        r["source"] = "semantic"
    print(f"DEBUG: semantic found {len(semantic_results)} results")

    print(f"DEBUG hybrid_search: query='{query}', document_id={document_id}")
    keyword_results = keyword_search_postgres(db, query, document_id=document_id, k=k)
    print(f"DEBUG hybrid_search: keyword_results type={type(keyword_results)}, len={len(keyword_results)}")
    print(f"DEBUG: keyword found {len(keyword_results)} results")
    for r in keyword_results:
        print(f"  keyword match: {r['content'][:60]}")

    seen = set()
    combined = []
    for r in semantic_results + keyword_results:
        chunk_id = r["metadata"]["postgres_chunk_id"]
        if chunk_id not in seen:
            seen.add(chunk_id)
            combined.append(r)

    print(f"DEBUG: combined total: {len(combined)}")
    return combined

def rerank_chunks(query: str, candidates: list, top_k: int = 3) -> list:
    """Use LLM to rerank candidates by relevance"""
    if not candidates:
        return []
    
    # Anchor: always keep top semantic result
    top_semantic = next((c for c in candidates if c["source"] == "semantic"), None)
    
    try:
        candidate_text = ""
        for i, c in enumerate(candidates):
            candidate_text += f"[{i}] ({c['source']}): {c['content'][:200]}\n\n"
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=100,
            temperature=0,
            messages=[{
                "role": "user",
                "content": f"""Given this query: "{query}"
Rank these chunks by relevance. Return ONLY a JSON array of indices.
Example: [2, 0, 1]

Chunks:
{candidate_text}

Return JSON array only:"""
            }]
        )
        
        raw = response.choices[0].message.content.strip()
        indices = json.loads(raw)
        reranked = [candidates[i] for i in indices[:top_k] if i < len(candidates)]
        
        if top_semantic and top_semantic not in reranked:
            reranked = [top_semantic] + reranked[:top_k-1]
        
        return reranked
        
    except Exception as e:
        print(f"⚠️ Reranking failed: {e}, falling back to top candidates")
        return candidates[:top_k]

def retrieve_with_confidence(db: Session, query: str, document_id: str = None, 
                                confidence_threshold: float = 0.7, k: int = 4) -> dict:
    """
    Full retrieval pipeline: hybrid search + rerank + confidence check.
    Returns chunks + whether confidence is sufficient to answer.
    """
    candidates = hybrid_search(db, query, document_id=document_id, k=k)
    
    if not candidates:
        return {
            "chunks": [],
            "confident": False,
            "reason": "No matching chunks found"
        }
    
    # Check top semantic distance for confidence
    semantic_only = [c for c in candidates if c["source"] == "semantic"]
    if semantic_only:
        top_distance = semantic_only[0].get("distance", 1.0)
        if top_distance > confidence_threshold:
            return {
                "chunks": [],
                "confident": False,
                "reason": f"Best match distance {top_distance:.2f} exceeds threshold {confidence_threshold}"
            }
    
    reranked = rerank_chunks(query, candidates, top_k=3)
    
    return {
        "chunks": reranked,
        "confident": True,
        "reason": "OK"
    }

def get_all_chunks_for_document(db: Session, document_id: str) -> list:
    """Get every chunk for a document, in order — used for summarization"""
    chunks = db.query(Chunk).filter(
        Chunk.document_id == document_id
    ).order_by(Chunk.chunk_index).all()
    
    return chunks

def find_document_by_name(db: Session, filename_hint: str) -> Document:
    """Fuzzy match a document by partial filename — handles spaces/underscores"""
    
    # Normalize: replace spaces with wildcards for flexible matching
    normalized_hint = filename_hint.replace(" ", "%")
    
    documents = db.query(Document).filter(
        Document.filename.ilike(f"%{normalized_hint}%"),
        Document.status == "ready"
    ).all()
    
    return documents[0] if documents else None