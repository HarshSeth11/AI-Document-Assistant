from database import SessionLocal
from models import Document, ChatSession, Message
import uuid

db = SessionLocal()

# Create a test document
doc = Document(
    filename="test_policy.txt",
    file_type="txt",
    status="ready",
    chunk_count=5
)
db.add(doc)
db.commit()
db.refresh(doc)
print(f"✅ Created document: {doc.id} - {doc.filename}")

# Create a test session
session = ChatSession(user_id="harsh")
db.add(session)
db.commit()
db.refresh(session)
print(f"✅ Created session: {session.id}")

# Create a test message
message = Message(
    session_id=session.id,
    role="user",
    content="What's in this document?"
)
db.add(message)
db.commit()
print(f"✅ Created message: {message.content}")

# Query it back
all_docs = db.query(Document).all()
print(f"\n📄 All documents: {[d.filename for d in all_docs]}")

all_sessions = db.query(ChatSession).all()
print(f"💬 All sessions: {[s.id for s in all_sessions]}")

db.close()