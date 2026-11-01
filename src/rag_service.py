"""Grounded retrieval for CreditRisk AI's existing PostgreSQL history table.

Phase 1 deliberately uses exact SQL/JSONB retrieval for structured credit records.
It does not use embeddings for numeric facts such as PD, grade, or application fields.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo


VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

_RECORD_ID_PATTERNS = (
    re.compile(r"\b(?:hồ\s*sơ|ma\s*hồ\s*sơ|mã\s*hồ\s*sơ|record|id)\s*(?:số|#|:)?\s*(\d+)\b", re.I),
    re.compile(r"\b(?:#|ID\s*)(\d+)\b", re.I),
)
_GRADE_RE = re.compile(r"\b(?:hạng|grade)\s*([ABCD])\b", re.I)
_CURRENT_TERMS = (
    "hồ sơ này", "hồ sơ hiện tại", "hồ sơ vừa thẩm định", "hồ sơ vừa đánh giá",
    "kết quả hiện tại", "kết quả vừa dự báo", "dự báo vừa rồi", "case này",
)
_LATEST_TERMS = ("mới nhất", "gần nhất", "vừa lưu", "vừa được lưu", "hồ sơ mới", "latest")
_PREVIOUS_TERMS = ("hồ sơ trước", "lần trước", "bản ghi trước", "previous")
_HISTORY_TERMS = (
    "hồ sơ đã lưu", "hồ sơ trong lịch sử", "lịch sử thẩm định", "hồ sơ cũ",
    "đã được ghi", "đã lưu", "hồ sơ trước", "hôm qua", "ngày hôm qua",
    "hôm nay", "mã hồ sơ", "hồ sơ số", "record id",
)
_DOC_TERMS = ("theo báo cáo", "trong báo cáo", "theo notebook", "trong notebook", "tài liệu", "trích dẫn nguồn")


def classify_history_query(question: str) -> dict[str, Any]:
    """Classify only requests that need the persisted record store.

    General conceptual questions such as "hồ sơ thế nào là tốt?" are not treated
    as requests to look up a specific saved record.
    """
    q = str(question or "").strip().lower()
    ids: list[int] = []
    for pattern in _RECORD_ID_PATTERNS:
        ids.extend(int(x) for x in pattern.findall(q))
    ids = list(dict.fromkeys(ids))
    if ids:
        return {"intent": "by_ids", "record_ids": ids[:5]}

    if any(term in q for term in _CURRENT_TERMS):
        return {"intent": "current"}

    if any(term in q for term in _DOC_TERMS):
        return {"intent": "documents_pending"}

    grade_match = _GRADE_RE.search(q)
    count_words = ("bao nhiêu", "số lượng", "đếm", "count")
    if grade_match and any(w in q for w in ("hồ sơ", "lịch sử", "khách hàng", "record")):
        if any(w in q for w in count_words):
            return {"intent": "grade_count", "grade": grade_match.group(1).upper()}
        return {"intent": "by_grade", "grade": grade_match.group(1).upper()}

    if any(w in q for w in ("bao nhiêu hồ sơ", "có bao nhiêu", "thống kê", "tổng số hồ sơ", "phân bố hạng")):
        return {"intent": "stats"}

    if any(w in q for w in ("pd cao nhất", "rủi ro cao nhất", "nguy cơ cao nhất", "pd lớn nhất")):
        return {"intent": "highest_pd"}
    if any(w in q for w in ("pd thấp nhất", "rủi ro thấp nhất", "nguy cơ thấp nhất", "pd nhỏ nhất")):
        return {"intent": "lowest_pd"}

    if any(term in q for term in _PREVIOUS_TERMS):
        return {"intent": "previous"}
    if any(term in q for term in _LATEST_TERMS):
        return {"intent": "latest"}

    if "hôm qua" in q or "ngày hôm qua" in q:
        return {"intent": "yesterday"}
    if "hôm nay" in q:
        return {"intent": "today"}

    if any(term in q for term in _HISTORY_TERMS):
        return {"intent": "clarify"}

    return {"intent": "none"}


def _connect(database_url: str):
    if not database_url:
        raise RuntimeError("DATABASE_URL chưa được cấu hình.")
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError("Thiếu psycopg[binary]. Hãy cài dependencies trong requirements.txt.") from exc
    return psycopg.connect(database_url, connect_timeout=8, sslmode="require", row_factory=dict_row)


def _pd_sql(alias: str = "payload") -> str:
    """Safe numeric expression for PD in either pd_value or pd_display."""
    return f"""CASE
      WHEN COALESCE({alias}->>'pd_value','') ~ '^[0-9]+([.][0-9]+)?$'
        THEN ({alias}->>'pd_value')::numeric
      WHEN COALESCE({alias}->>'pd_display','') ~ '^[0-9]+([.][0-9]+)?%$'
        THEN (replace({alias}->>'pd_display','%',''))::numeric / 100
      ELSE NULL END"""


def _fetch_by_ids(conn, ids: list[int]) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """SELECT id, created_at, payload
               FROM credit_risk_history WHERE id = ANY(%s::bigint[])
               ORDER BY created_at DESC, id DESC""",
            (ids,),
        )
        rows = cur.fetchall()
    by_id = {int(row["id"]): row for row in rows}
    return [by_id[i] for i in ids if i in by_id]


def _fetch_latest(conn, limit: int = 1, exclude_application: dict | None = None) -> list[dict[str, Any]]:
    from psycopg.types.json import Jsonb
    with conn.cursor() as cur:
        if exclude_application is not None:
            cur.execute(
                """SELECT id, created_at, payload FROM credit_risk_history
                   WHERE payload->'application' IS DISTINCT FROM %s
                   ORDER BY created_at DESC, id DESC LIMIT %s""",
                (Jsonb(exclude_application), limit),
            )
        else:
            cur.execute(
                """SELECT id, created_at, payload FROM credit_risk_history
                   ORDER BY created_at DESC, id DESC LIMIT %s""",
                (limit,),
            )
        return cur.fetchall()


def _fetch_grade(conn, grade: str, limit: int = 10) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """SELECT id, created_at, payload FROM credit_risk_history
               WHERE upper(coalesce(payload->>'grade','')) = %s
               ORDER BY created_at DESC, id DESC LIMIT %s""",
            (grade, limit),
        )
        return cur.fetchall()


def _fetch_extreme_pd(conn, highest: bool) -> list[dict[str, Any]]:
    direction = "DESC" if highest else "ASC"
    expr = _pd_sql("payload")
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT id, created_at, payload FROM credit_risk_history
                WHERE ({expr}) IS NOT NULL
                ORDER BY ({expr}) {direction}, created_at DESC LIMIT 1"""
        )
        row = cur.fetchone()
    return [row] if row else []


def _fetch_date(conn, target_date, limit: int = 10) -> list[dict[str, Any]]:
    start = datetime.combine(target_date, time.min, tzinfo=VN_TZ)
    end = start + timedelta(days=1)
    with conn.cursor() as cur:
        cur.execute(
            """SELECT id, created_at, payload FROM credit_risk_history
               WHERE created_at >= %s AND created_at < %s
               ORDER BY created_at DESC, id DESC LIMIT %s""",
            (start, end, limit),
        )
        return cur.fetchall()


def _fetch_grade_count(conn, grade: str) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*)::bigint AS total, upper(%s) AS grade FROM credit_risk_history WHERE upper(coalesce(payload->>'grade','')) = upper(%s)",
            (grade, grade),
        )
        return cur.fetchone()


def _fetch_stats(conn) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            """SELECT COUNT(*)::bigint AS total,
                 COUNT(*) FILTER (WHERE upper(coalesce(payload->>'grade',''))='A')::bigint AS grade_a,
                 COUNT(*) FILTER (WHERE upper(coalesce(payload->>'grade',''))='B')::bigint AS grade_b,
                 COUNT(*) FILTER (WHERE upper(coalesce(payload->>'grade',''))='C')::bigint AS grade_c,
                 COUNT(*) FILTER (WHERE upper(coalesce(payload->>'grade',''))='D')::bigint AS grade_d,
                 COUNT(*) FILTER (WHERE coalesce(payload->>'mode','')='model')::bigint AS model_count
               FROM credit_risk_history"""
        )
        return cur.fetchone()


def _safe_record(row: dict[str, Any]) -> dict[str, Any]:
    """Whitelist fields sent to Gemini; avoid passing arbitrary payload fields."""
    payload = row.get("payload") or {}
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            payload = {}
    if not isinstance(payload, dict):
        payload = {}
    app = payload.get("application") if isinstance(payload, dict) else {}
    if not isinstance(app, dict):
        app = {}
    allowed = (
        "person_age", "person_income", "person_home_ownership", "person_emp_length",
        "loan_intent", "loan_amnt", "loan_int_rate", "loan_percent_income",
        "cb_person_default_on_file",
    )
    return {
        "record_id": int(row["id"]),
        "created_at": row["created_at"].isoformat() if hasattr(row.get("created_at"), "isoformat") else str(row.get("created_at")),
        "mode": payload.get("mode"),
        "application": {k: app[k] for k in allowed if k in app},
        "pd_value": payload.get("pd_value"),
        "pd_display": payload.get("pd_display"),
        "grade": payload.get("grade"),
        "risk_level": payload.get("risk_level"),
        "cluster": payload.get("cluster"),
        "cluster_description": payload.get("cluster_description"),
        "summary": str(payload.get("summary") or "")[:1200],
        "recommendation": str(payload.get("recommendation") or "")[:800],
        "factors": (payload.get("factors") or [])[:8] if isinstance(payload.get("factors"), list) else [],
    }


def retrieve_creditrisk_context(
    question: str,
    current_result: dict | None,
    database_url: str,
    current_record_id: int | None = None,
) -> dict[str, Any] | None:
    """Retrieve authoritative history evidence or return a clarification state.

    Return None for general/domain questions which do not ask about stored records.
    """
    plan = classify_history_query(question)
    intent = plan["intent"]
    if intent == "none":
        return None
    if intent == "documents_pending":
        return {
            "status": "documents_pending",
            "message": "Kho tài liệu RAG chưa được nạp trong giai đoạn hiện tại. Mình chưa thể trích dẫn báo cáo/notebook; hãy hỏi về dữ liệu hồ sơ đã lưu hoặc kiến thức chung.",
        }
    if intent == "current":
        if isinstance(current_result, dict) and current_result:
            return {
                "status": "grounded",
                "source": "Kết quả thẩm định hiện tại trong phiên làm việc",
                "records": [{"record_id": current_record_id, **_safe_session_result(current_result)}],
            }
        return {"status": "not_found", "message": "Phiên này chưa có hồ sơ thẩm định hiện tại. Hãy thẩm định một hồ sơ hoặc cung cấp mã hồ sơ đã lưu."}
    if intent == "clarify":
        return {"status": "clarify", "message": "Mình cần xác định đúng hồ sơ trong lịch sử. Bạn hãy cung cấp mã hồ sơ hiển thị ở trang Lịch sử thẩm định, hoặc thời điểm/hạng tín dụng cần tìm."}

    try:
        with _connect(database_url) as conn:
            stats = None
            if intent == "by_ids":
                ids = plan["record_ids"]
                rows = _fetch_by_ids(conn, ids)
                found = {int(row["id"]) for row in rows}
                missing = [i for i in ids if i not in found]
                if missing:
                    return {"status": "not_found", "message": "Không tìm thấy mã hồ sơ " + ", ".join(map(str, missing)) + " trong lịch sử đã lưu. Mình sẽ không tự chọn hồ sơ khác thay thế."}
            elif intent == "latest":
                rows = _fetch_latest(conn, 1)
            elif intent == "previous":
                current_app = current_result.get("application") if isinstance(current_result, dict) else None
                rows = _fetch_latest(conn, 1, exclude_application=current_app if isinstance(current_app, dict) else None)
            elif intent == "by_grade":
                rows = _fetch_grade(conn, plan["grade"], 10)
            elif intent == "grade_count":
                stats = _fetch_grade_count(conn, plan["grade"])
                rows = []
            elif intent == "highest_pd":
                rows = _fetch_extreme_pd(conn, True)
            elif intent == "lowest_pd":
                rows = _fetch_extreme_pd(conn, False)
            elif intent in ("today", "yesterday"):
                today = datetime.now(VN_TZ).date()
                day = today if intent == "today" else today - timedelta(days=1)
                rows = _fetch_date(conn, day, 10)
            elif intent == "stats":
                stats = _fetch_stats(conn)
                rows = []
            else:
                rows = []

        if intent in ("stats", "grade_count"):
            return {"status": "grounded", "source": "Truy vấn thống kê trực tiếp PostgreSQL", "statistics": dict(stats or {})}
        if not rows:
            if intent == "by_grade":
                return {"status": "not_found", "message": f"Không tìm thấy hồ sơ hạng {plan['grade']} trong dữ liệu đã lưu."}
            return {"status": "not_found", "message": "Không tìm thấy hồ sơ phù hợp trong lịch sử PostgreSQL. Mình sẽ không suy đoán hoặc thay bằng hồ sơ khác."}
        return {
            "status": "grounded",
            "source": "PostgreSQL credit_risk_history",
            "intent": intent,
            "records": [_safe_record(row) for row in rows],
        }
    except Exception as exc:
        # Do not leak a database URL, credentials, or raw SQL in user-facing text.
        return {
            "status": "retrieval_error",
            "message": f"Chưa truy xuất được lịch sử PostgreSQL ({type(exc).__name__}). Mình sẽ không trả lời bằng cách đoán dữ liệu hồ sơ.",
        }


def _safe_session_result(result: dict[str, Any]) -> dict[str, Any]:
    app = result.get("application") if isinstance(result.get("application"), dict) else {}
    allowed = (
        "person_age", "person_income", "person_home_ownership", "person_emp_length",
        "loan_intent", "loan_amnt", "loan_int_rate", "loan_percent_income",
        "cb_person_default_on_file",
    )
    return {
        "created_at": result.get("created_at"),
        "mode": result.get("mode"),
        "application": {k: app[k] for k in allowed if k in app},
        "pd_value": result.get("pd_value"),
        "pd_display": result.get("pd_display"),
        "grade": result.get("grade"),
        "risk_level": result.get("risk_level"),
        "cluster": result.get("cluster"),
        "cluster_description": result.get("cluster_description"),
        "summary": str(result.get("summary") or "")[:1200],
        "recommendation": str(result.get("recommendation") or "")[:800],
        "factors": result.get("factors", [])[:8] if isinstance(result.get("factors"), list) else [],
    }


def format_grounding_context(retrieval: dict[str, Any] | None) -> str:
    """Convert retrieved evidence to a compact, explicit evidence block."""
    if not retrieval:
        return ""
    return (
        "BẰNG CHỨNG TRUY XUẤT TỪ HỆ THỐNG (ưu tiên cao hơn suy luận nội bộ):\n"
        + json.dumps(retrieval, ensure_ascii=False, default=str)
        + "\nQUY TẮC: Chỉ khẳng định số liệu được nêu trong bằng chứng. Khi có record_id, phải dẫn mã hồ sơ. "
          "Không suy đoán hồ sơ không có trong kết quả. Nếu thiếu trường dữ liệu, nói rõ là không có dữ liệu."
    )
