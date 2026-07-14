import os
from groq import Groq
from sqlalchemy.orm import Session
from app.retrieval_service import get_all_chunks_for_document, find_document_by_name
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def summarize_document(db: Session, document_name_hint: str) -> str:
    """Summarize an entire document by name"""
    
    document = find_document_by_name(db, document_name_hint)
    if not document:
        return f"No document found matching '{document_name_hint}'"
    
    chunks = get_all_chunks_for_document(db, document.id)
    if not chunks:
        return f"Document '{document.filename}' has no content to summarize."
    
    full_text = "\n\n".join([c.content for c in chunks])
    
    # Guard against very large documents exceeding context window
    max_chars = 6000
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n\n[Content truncated for length]"
    
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=400,
        temperature=0,
        messages=[{
            "role": "user",
            "content": f"""Summarize this document in 4-6 clear sentences.
Cover the main topics and key points.

Document: {document.filename}

Content:
{full_text}

Summary:"""
        }]
    )
    
    summary = response.choices[0].message.content.strip()
    return f"[Summary of {document.filename}]\n\n{summary}"

def compare_documents(db: Session, doc1_hint: str, doc2_hint: str) -> str:
    """Compare two documents by name"""
    
    doc1 = find_document_by_name(db, doc1_hint)
    doc2 = find_document_by_name(db, doc2_hint)
    
    if not doc1:
        return f"No document found matching '{doc1_hint}'"
    if not doc2:
        return f"No document found matching '{doc2_hint}'"
    
    chunks1 = get_all_chunks_for_document(db, doc1.id)
    chunks2 = get_all_chunks_for_document(db, doc2.id)
    
    text1 = "\n\n".join([c.content for c in chunks1])[:3000]
    text2 = "\n\n".join([c.content for c in chunks2])[:3000]
    
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=400,
        temperature=0,
        messages=[{
            "role": "user",
            "content": f"""Compare these two documents. Identify:
1. Key similarities
2. Key differences
Be concise — 4-5 bullet points total.

Document A: {doc1.filename}
{text1}

Document B: {doc2.filename}
{text2}

Comparison:"""
        }]
    )
    
    comparison = response.choices[0].message.content.strip()
    return f"[Comparison: {doc1.filename} vs {doc2.filename}]\n\n{comparison}"