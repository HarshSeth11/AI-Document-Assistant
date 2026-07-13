from app.database import SessionLocal
from app.models import Chunk

db = SessionLocal()

# Direct check: does ANY chunk contain "401k"?
results = db.query(Chunk).filter(Chunk.content.ilike("%401k%")).all()
print(f"Chunks containing '401k': {len(results)}")
for r in results:
    print(f"  - {r.content[:80]}")

# Check exact word from your query
results2 = db.query(Chunk).filter(Chunk.content.ilike("%matching%")).all()
print(f"\nChunks containing 'matching': {len(results2)}")
for r in results2:
    print(f"  - {r.content[:80]}")

db.close()