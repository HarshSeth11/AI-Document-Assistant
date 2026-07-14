from app.database import SessionLocal
from app.analysis_service import summarize_document

db = SessionLocal()
result = summarize_document(db, "company handbook")
print(result)
db.close()