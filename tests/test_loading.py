from app.document_service import load_document, chunk_document

print("Loading...")
docs = load_document("sample_policy.txt", "txt")
print(f"Loaded {len(docs)} docs")

print("Chunking...")
chunks = chunk_document(docs)
print(f"Created {len(chunks)} chunks")
print(chunks[0].page_content[:100])