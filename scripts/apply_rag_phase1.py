from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
IMPORT_LINE = "from src.rag_service import retrieve_creditrisk_context, format_grounding_context"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count == 0:
        if new in text:
            return text
        raise RuntimeError(f"Không tìm thấy điểm chèn '{label}'. File app.py có thể khác phiên bản dự kiến; chưa chỉnh sửa file.")
    if count != 1:
        raise RuntimeError(f"Điểm chèn '{label}' xuất hiện {count} lần; dừng để tránh sửa nhầm.")
    return text.replace(old, new, 1)


def patch_history_functions(text: str) -> str:
    pattern = re.compile(r"def save_history_item\(item\):[\s\S]*?(?=\n# -{5,}\n# 2\.)")
    match = pattern.search(text)
    if not match:
        # Already patched or different layout: do not guess.
        if "RETURNING id, created_at" in text and '"_history_id"' in text:
            return text
        raise RuntimeError("Không tìm thấy cụm hàm save_history_item/load_history trước phần Gemini.")

    replacement = '''def save_history_item(item):
    """Save a prediction and attach its database ID to the in-session result."""
    conn = db_connect()
    if conn is None:
        return False, "Chưa cấu hình DATABASE_URL trong Streamlit Secrets."
    try:
        payload = json.dumps(item, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, "item") else str(x))
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO credit_risk_history (payload) VALUES (%s::jsonb) RETURNING id, created_at",
                    (payload,),
                )
                row = cur.fetchone()
        if row:
            item["_history_id"] = int(row[0])
            item["_db_created_at"] = row[1].isoformat() if hasattr(row[1], "isoformat") else str(row[1])
        return True, ""
    except Exception as exc:
        return False, str(exc)
    finally:
        conn.close()


def load_history(limit=500):
    """Return recent predictions with stable database IDs for user lookup."""
    conn = db_connect()
    if conn is None:
        return None, "Chưa cấu hình DATABASE_URL trong Streamlit Secrets."
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, created_at, payload FROM credit_risk_history ORDER BY created_at DESC, id DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
        items = []
        for record_id, created_at, payload in rows:
            if isinstance(payload, str):
                payload = json.loads(payload)
            payload = dict(payload or {})
            payload["_history_id"] = int(record_id)
            payload["_db_created_at"] = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)
            items.append(payload)
        return items, ""
    except Exception as exc:
        return None, str(exc)
    finally:
        conn.close()
'''
    return text[:match.start()] + replacement + text[match.end():]



def patch_pre_llm_retrieval(text: str) -> str:
    pattern = re.compile(r"^(?P<indent>[ \t]*)contents = _build_chat_contents\(question, current_result, conversation\)\s*$", re.M)
    matches = list(pattern.finditer(text))
    if not matches:
        if "retrieval = retrieve_creditrisk_context(" in text:
            return text
        raise RuntimeError("Không tìm thấy lời gọi _build_chat_contents trong stream_gemini_answer.")
    if len(matches) != 1:
        raise RuntimeError(f"Lời gọi _build_chat_contents xuất hiện {len(matches)} lần; dừng để tránh sửa nhầm.")
    match = matches[0]
    indent = match.group("indent")
    lines = [
        "retrieval = retrieve_creditrisk_context(",
        "    question=question,",
        "    current_result=current_result,",
        "    database_url=get_database_url(),",
        '    current_record_id=(current_result or {}).get("_history_id") if isinstance(current_result, dict) else None,',
        ")",
        'if retrieval and retrieval.get("status") in {"not_found", "clarify", "retrieval_error", "documents_pending"}:',
        '    yield retrieval.get("message", "Không có đủ dữ liệu truy xuất để trả lời chính xác.")',
        "    return",
        "contents = _build_chat_contents(",
        "    question, current_result, conversation, retrieved_context=retrieval",
        ")",
    ]
    replacement = "\n".join(indent + line if line else "" for line in lines)
    return text[:match.start()] + replacement + text[match.end():]

def main() -> None:
    require(APP.exists(), f"Không tìm thấy {APP}. Hãy chạy script tại bản project đầy đủ.")
    original = APP.read_text(encoding="utf-8")
    updated = original

    # 1) Import the retrieval layer.
    if IMPORT_LINE not in updated:
        model_import = re.search(r"from src\.model_service import \([\s\S]*?\n\)\n", updated)
        if model_import:
            updated = updated[:model_import.end()] + IMPORT_LINE + "\n" + updated[model_import.end():]
        else:
            single_import = "from src.model_service import"
            require(single_import in updated, "Không tìm thấy import src.model_service trong app.py.")
            updated = updated.replace(single_import, IMPORT_LINE + "\n" + single_import, 1)

    # 2) Extend the prompt builder with evidence retrieved from PostgreSQL.
    updated = replace_once(
        updated,
        "def _build_chat_contents(question, current_result, conversation):",
        "def _build_chat_contents(question, current_result, conversation, retrieved_context=None):",
        "chat contents signature",
    )
    if 'context["retrieved_evidence"]' not in updated:
        anchor = "    user_text = str(question).strip()\n"
        insertion = '''    if retrieved_context:
        context["retrieved_evidence"] = format_grounding_context(retrieved_context)
    user_text = str(question).strip()
'''
        updated = replace_once(updated, anchor, insertion, "grounding context insertion")

    # 3) Retrieve before the LLM call. Missing/ambiguous records stop here so Gemini cannot guess.
    updated = patch_pre_llm_retrieval(updated)

    # 4) Attach stable IDs and timestamps to rows shown on the history page.
    updated = patch_history_functions(updated)
    if '"Mã hồ sơ": item.get("_history_id"' not in updated:
        row_pattern = re.compile(r"^(?P<indent>[ \t]*)\"STT\": i,\s*$", re.M)
        row_matches = list(row_pattern.finditer(updated))
        if len(row_matches) != 1:
            raise RuntimeError(f"Không xác định được duy nhất cột STT trong bảng lịch sử (tìm thấy {len(row_matches)}).")
        match = row_matches[0]
        indent = match.group("indent")
        replacement = indent + '"STT": i,\n' + indent + '"Mã hồ sơ": item.get("_history_id", "N/A"),' 
        updated = updated[:match.start()] + replacement + updated[match.end():]

    if updated == original:
        print("app.py đã có tích hợp RAG Phase 1; không cần thay đổi thêm.")
        return

    backup = APP.with_name(f"app.py.backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(APP, backup)
    APP.write_text(updated, encoding="utf-8")
    print("Đã tích hợp RAG Phase 1 vào app.py.")
    print(f"Bản sao lưu: {backup.name}")
    print("Đã thêm: truy xuất theo ID/current/latest/previous/grade/PD/date/statistics; mã hồ sơ hiển thị trong lịch sử; chặn trả lời khi không tìm thấy hồ sơ.")


if __name__ == "__main__":
    main()
