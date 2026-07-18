from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from app.database import get_db
from app.agent_service import ask_agent
from app.session_service import (
    get_or_create_session, load_conversation_history, save_message,
    get_session_history, generate_session_title
)

router = APIRouter(prefix="/chat", tags=["Chat"])

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None

@router.post("")
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        session = get_or_create_session(db, request.session_id, request.user_id)
        history = load_conversation_history(db, session.id, max_messages=10)
        result = ask_agent(db, request.question, conversation_history=history)

        save_message(db, session.id, "user", request.question)
        save_message(db, session.id, "assistant", result["answer"], tools_used=result["tools_used"])
        generate_session_title(db, session.id, request.question)

        return {
            "session_id": session.id,
            "question": request.question,
            "answer": result["answer"],
            "tools_used": result["tools_used"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@router.get("/history/{session_id}")
def chat_history(session_id: str, db: Session = Depends(get_db)):
    history = get_session_history(db, session_id)
    if not history:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return history

@router.get("/sessions")
def list_sessions(user_id: Optional[str] = None, db: Session = Depends(get_db)):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id query parameter is required")

    from models import ChatSession
    sessions = db.query(ChatSession).filter(
        ChatSession.user_id == user_id
    ).order_by(ChatSession.last_active.desc()).all()

    return [
        {
            "session_id": s.id, "title": s.title,
            "created_at": s.created_at, "last_active": s.last_active
        }
        for s in sessions
    ]