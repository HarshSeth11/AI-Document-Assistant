from app.document_service import load_document, chunk_document

docs = load_document("company_handbook.txt", "txt")
print(f"Loaded {len(docs)} docs")
print(f"Full content length: {len(docs[0].page_content)}")
print(f"Full content:\n{docs[0].page_content}")

chunks = chunk_document(docs)
print(f"\n{len(chunks)} chunks created:")
for i, c in enumerate(chunks):
    print(f"\n--- Chunk {i} ---")
    print(c.page_content)