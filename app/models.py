from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
import uuid

def generate_uuid():
    return str(uuid.uuid4())

class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=generate_uuid)
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)  # pdf, txt
    upload_date = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String, default="processing")  # processing, ready, failed
    chunk_count = Column(Integer, default=0)
    
    # Relationship — one document has many chunks
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Document {self.filename}>"

class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(String, primary_key=True, default=generate_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    chroma_id = Column(String, nullable=True)  # links to ChromaDB vector

    # Relationship — many chunks belong to one document
    document = relationship("Document", back_populates="chunks")

    def __repr__(self):
        return f"<Chunk {self.chunk_index} of {self.document_id}>"

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_active = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    title = Column(String, default="New Conversation")

    # Relationship — one session has many messages
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<ChatSession {self.id} for {self.user_id}>"

class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=generate_uuid)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String, nullable=False)  # user, assistant, tool
    content = Column(Text, nullable=False)
    tools_used = Column(String, nullable=True)  # comma-separated tool names
    sources = Column(Text, nullable=True)  # JSON string of source documents
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship — many messages belong to one session
    session = relationship("ChatSession", back_populates="messages")

    def __repr__(self):
        return f"<Message {self.role} in {self.session_id}>"