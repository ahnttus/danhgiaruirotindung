# CreditRisk AI — RAG Phase 1: Structured History Retrieval

## What this phase implements

This first phase grounds the chatbot in the existing `credit_risk_history` PostgreSQL/Supabase table. It is intentionally SQL-first: exact numeric values (PD, grade, amounts, IDs, timestamps) must come from the database, not vector similarity or model memory.

Supported queries:

- Exact record IDs: `hồ sơ số 125`, including comparisons such as `hồ sơ 125 và 128`.
- Current in-session assessment: `hồ sơ hiện tại`, `hồ sơ này`.
- Latest/previous saved record.
- Grade filters A/B/C/D.
- Highest/lowest PD.
- Today/yesterday and aggregate counts.
- Not-found and ambiguous requests stop before Gemini, so it cannot silently substitute another record.

The history table is not deleted or recreated. New metadata (`_history_id`, `_db_created_at`) is attached to the UI object only; existing JSONB payloads remain compatible.

## Apply to the project

1. Back up the repository or commit current changes.
2. Copy this package's `src/rag_service.py` into the project's `src/` folder.
3. Copy `scripts/apply_rag_phase1.py` into the project's `scripts/` folder (or keep it in this package).
4. From the project root, run:

   ```bash
   python scripts/apply_rag_phase1.py
   ```

   If you keep the script in a different folder, run it from its package path; it locates the project root relative to itself. Therefore the simplest method is to copy `src/` and `scripts/` folders into the project while preserving their structure.

5. In Supabase SQL Editor, run `sql/phase1_history_retrieval.sql`. The indexes are additive and do not alter existing records. The application can run without this optional index script; indexes improve query performance as history grows.
6. Run tests:

   ```bash
   python -m unittest discover -s tests -v
   python -m py_compile app.py src/rag_service.py
   ```

7. Deploy the changes to Streamlit Cloud.

## Integration details

The patch script modifies only the current `app.py` integration points:

- Imports the retrieval service.
- Adds retrieved evidence to Gemini's context.
- Runs retrieval before calling Gemini.
- Stops on missing/ambiguous records or retrieval failures.
- Returns database IDs on the history page for later lookup.
- Saves a backup copy of `app.py` before editing.

The patch script uses exact anchors and stops without changing `app.py` if the expected source structure is not found. It is safe to run again after a successful patch.

## Important limitations / next phase

This is Phase 1: structured history retrieval, not yet the document-vector RAG. Queries such as "theo báo cáo/notebook/tài liệu" intentionally return a message that the document index has not yet been ingested. Phase 2 will add document parsing, chunking, embeddings, pgvector, hybrid search, citations, and re-indexing.

The website currently uses a shared history store. Do not put real personal/customer financial information into it until authentication and per-user/per-tenant database policies are implemented. Continue using synthetic/demo data.
