from app.database import SessionLocal
from app.retrieval_service import keyword_search_postgres

db = SessionLocal()

results = keyword_search_postgres(db, "401k matching", document_id=None, k=4)
print(f"Function returned: {len(results)} results")
for r in results:
    print(f"  - {r['content'][:80]}")

db.close()