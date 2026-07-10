from database import SessionLocal
from models import Document

db = SessionLocal()

print("Creating document record...")
doc = Document(
    filename="test.txt",
    file_type="txt",
    status="processing",
    chunk_count=0
)
db.add(doc)
print("Committing...")
db.commit()
print("Committed successfully")
db.refresh(doc)
print(f"Document created: {doc.id}")

db.close()
print("Done")