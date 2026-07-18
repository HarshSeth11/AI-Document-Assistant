# Edge Case Test Results

| # | Test | Expected | Actual | Status |
|---|------|----------|--------|--------|
| 1 | Upload empty file | 400 error | | |
| 2 | Upload unsupported extension (.docx) | 400 error | | |
| 3 | Upload same file twice | Both succeed, separate IDs | | |
| 4 | Search with empty query | 400 error | | |
| 5 | Search with very long query (500+ words) | Handles gracefully | | |
| 6 | Chat with empty question | 400 error | | |
| 7 | Get history for non-existent session_id | 404 error | | |
| 8 | Get chunks for non-existent document_id | 404 error | | |
| 9 | Delete non-existent document | 404 error | | |
| 10 | Summarize a document that doesn't exist | Honest "not found" message | | |
| 11 | Compare with only one valid document name | Honest "not found" message | | |
| 12 | Ask a question before any documents uploaded | Honest "no documents" response | | |
| 13 | SQL-injection-style query string | No crash, treated as plain text | | |
| 14 | Very short query (1 character) | Handles gracefully, no crash | | |