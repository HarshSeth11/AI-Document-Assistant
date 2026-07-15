from sqlalchemy.orm import Session
from app.models import ChatSession, Message
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
import uuid

def get_or_create_session(db: Session, session_id: str = None, user_id: str = None) -> ChatSession:
    """Get existing session or create a new one"""
    
    if session_id:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session:
            return session
    
    # If no user_id provided, generate a unique anonymous identity
    if not user_id:
        user_id = f"anon_{uuid.uuid4()}"
    
    session = ChatSession(user_id=user_id, title="New Conversation")
    db.add(session)
    db.commit()
    db.refresh(session)
    return session

def load_conversation_history(db: Session, session_id: str, max_messages: int = 10) -> list:
    """Load recent messages for a session, convert to LangChain message format"""
    messages = db.query(Message).filter(
        Message.session_id == session_id
    ).order_by(Message.created_at.asc()).limit(max_messages).all()
    
    langchain_messages = []
    for msg in messages:
        if msg.role == "user":
            langchain_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            langchain_messages.append(AIMessage(content=msg.content))
    
    return langchain_messages

def save_message(db: Session, session_id: str, role: str, content: str, 
                  tools_used: list = None, sources: list = None):
    """Save a message to a session"""
    message = Message(
        session_id=session_id,
        role=role,
        content=content,
        tools_used=",".join(tools_used) if tools_used else None,
        sources=json.dumps(sources) if sources else None
    )
    db.add(message)
    db.commit()  # ← commit FIRST so message.created_at gets populated by PostgreSQL
    db.refresh(message)  # ← refresh to pull the generated timestamp into our Python object
    
    # Update session last_active
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session:
        session.last_active = message.created_at
        db.commit()
    
    return message

def get_session_history(db: Session, session_id: str) -> dict:
    """Get full readable history for a session — for a /history endpoint"""
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        return None
    
    messages = db.query(Message).filter(
        Message.session_id == session_id
    ).order_by(Message.created_at.asc()).all()
    
    return {
        "session_id": session.id,
        "user_id": session.user_id,
        "created_at": session.created_at,
        "message_count": len(messages),
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "tools_used": m.tools_used.split(",") if m.tools_used else [],
                "created_at": m.created_at
            }
            for m in messages
        ]
    }

def generate_session_title(db: Session, session_id: str, first_message: str):
    """Set a readable title from the first user message"""
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session and session.title == "New Conversation":
        title = first_message[:50] + ("..." if len(first_message) > 50 else "")
        session.title = title
        db.commit()