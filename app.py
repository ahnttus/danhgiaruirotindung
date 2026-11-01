"""
CreditRisk AI — Intelligent Credit Risk Assessment & Analytics Platform
Thiết kế kết hợp kiến trúc trải nghiệm & bố cục tinh tế của VietTrip.online
và hệ thống thiết kế chuẩn mực GetDesign Al Jazeera (Editorial rigor, hairline grid,
section eyebrow lockup, Roboto/Inter typography scale, Indigo #5266EB, Deep Navy #173B68,
Live-Red #C31833 và Canvas #F3F7FF).

Bảo toàn 100% logic mô hình Random Forest, ngưỡng phân vị Q1/Q2/Q3, SHAP TreeExplainer,
K-Means 4 cụm, Supabase/PostgreSQL schema và Gemini AI Assistant.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import time

import altair as alt
import numpy as np
import pandas as pd
import psycopg
import streamlit as st
from google import genai
from google.genai import types

from src.model_service import (
    CLUSTER_NAMES,
    DEFAULT_RATE_MAP,
    RECOMMENDATIONS,
    evaluate_application,
    model_status,
)

# -----------------------------------------------------------------------------
# 1. DATABASE CONNECTIVITY & SHARED STORAGE (100% BẢO TOÀN NGUYÊN BẢN)
# -----------------------------------------------------------------------------
def get_database_url():
    """Read the shared PostgreSQL URL from Streamlit secrets."""
    try:
        return st.secrets.get("DATABASE_URL", "")
    except Exception:
        return ""


def db_connect():
    url = get_database_url()
    if not url:
        return None
    return psycopg.connect(url, connect_timeout=10, sslmode="require")


def init_history_table():
    """Create the shared history table if it does not exist."""
    conn = db_connect()
    if conn is None:
        return False, "Chưa cấu hình DATABASE_URL trong Streamlit Secrets."
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """CREATE TABLE IF NOT EXISTS credit_risk_history (
                        id BIGSERIAL PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        payload JSONB NOT NULL
                    )"""
                )
        return True, ""
    except Exception as exc:
        return False, str(exc)
    finally:
        conn.close()


def save_history_item(item):
    """Save one prediction to the shared PostgreSQL database."""
    conn = db_connect()
    if conn is None:
        return False, "Chưa cấu hình DATABASE_URL trong Streamlit Secrets."
    try:
        payload = json.dumps(item, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, "item") else str(x))
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO credit_risk_history (payload) VALUES (%s::jsonb)",
                    (payload,),
                )
        return True, ""
    except Exception as exc:
        return False, str(exc)
    finally:
        conn.close()


def load_history(limit=500):
    """Return recent shared predictions, newest first."""
    conn = db_connect()
    if conn is None:
        return None, "Chưa cấu hình DATABASE_URL trong Streamlit Secrets."
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT payload FROM credit_risk_history ORDER BY created_at DESC, id DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
        return [row[0] for row in rows], ""
    except Exception as exc:
        return None, str(exc)
    finally:
        conn.close()


# -----------------------------------------------------------------------------
# 2. GEMINI AI ASSISTANT MULTITURN ENGINE (100% BẢO TOÀN NGUYÊN BẢN)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _build_gemini_client(api_key: str):
    """Reuse one configured client per API key; avoid rebuilding it per turn."""
    http_options = types.HttpOptions(
        timeout=20_000,
        retry_options=types.HttpRetryOptions(attempts=1),
    )
    return genai.Client(api_key=api_key, http_options=http_options)


def get_gemini_client():
    """Read the API key and return a cached Gemini client."""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        api_key = ""
    api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key or not str(api_key).strip():
        return None
    return _build_gemini_client(str(api_key).strip())


def _friendly_gemini_error(exc):
    """Translate provider errors into actionable, secret-safe messages."""
    message = str(exc).lower()
    error_type = type(exc).__name__
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    status_text = str(status) if status else ""

    if any(term in message for term in ("api_key_invalid", "invalid api key", "api key not valid", "unauthorized", "unauthenticated")) or status_text in {"401", "UNAUTHENTICATED"}:
        return "Gemini từ chối API key. Hãy kiểm tra GEMINI_API_KEY trong Streamlit Secrets."
    if any(term in message for term in ("permission_denied", "permission denied", "forbidden")) or status_text in {"403", "PERMISSION_DENIED"}:
        return "Gemini API không cấp quyền cho yêu cầu này. Kiểm tra API key và quyền Gemini API của project."
    if any(term in message for term in ("resource_exhausted", "quota", "rate limit", "too many requests")) or status_text in {"429", "RESOURCE_EXHAUSTED"}:
        return "Gemini API đang chạm hạn mức hoặc giới hạn tốc độ. Kiểm tra quota/billing trong Google AI Studio."
    if any(term in message for term in ("not_found", "model not found", "not supported for generatecontent", "unknown model")) or status_text == "404":
        return "Model Gemini không khả dụng với API key/project hiện tại. Kiểm tra model access trong Google AI Studio."
    if status_text in {"503", "UNAVAILABLE"} or "service unavailable" in message or "overloaded" in message:
        return "Máy chủ Gemini đang quá tải (503). Thử lại sau và kiểm tra trạng thái/quota API."
    if any(term in message for term in ("deadline_exceeded", "timed out", "timeout", "connection reset", "temporarily unavailable")):
        return "Kết nối tới Gemini vượt thời gian chờ. Hãy thử gửi lại câu hỏi."
    if "billing" in message or "billing_not_enabled" in message:
        return "Project Gemini chưa đáp ứng điều kiện thanh toán/quota hiện tại. Kiểm tra trạng thái project trong Google AI Studio."
    return f"Gemini gặp lỗi {error_type}" + (f" (mã {status_text})" if status_text else "") + ". Vui lòng thử lại sau."


_GEMINI_SYSTEM_INSTRUCTION = """Bạn là CreditRisk AI Assistant — trợ lý AI phân tích rủi ro tín dụng chuyên nghiệp của nền tảng CreditRisk AI (Nhóm 14).

NHIỆM VỤ CỐT LÕI:
- Trả lời bằng tiếng Việt chuyên nghiệp, sắc sảo, tự nhiên, rành mạch và có cấu trúc logic biên tập cao cấp.
- Giải thích chuyên sâu về xác suất vỡ nợ (Probability of Default - PD), phân hạng rủi ro tín dụng A–D, phân khúc khách hàng K-Means (4 cụm), và giải thích mô hình bằng SHAP (Shapley Additive exPlanations).
- Khi có kết quả hồ sơ đang thẩm định, sử dụng chính xác các chỉ số định lượng được cung cấp trong ngữ cảnh để phân tích. Làm rõ yếu tố nào kéo rủi ro lên (SHAP dương), yếu tố nào giúp giảm thiểu rủi ro (SHAP âm).
- Đưa ra khuyến nghị tín dụng thực tế (tài sản bảo đảm, điều chỉnh hạn mức hoặc lãi suất) dựa trên hồ sơ rủi ro.
- Giữ phong thái khách quan, chuẩn xác. Tuyệt đối không bịa đặt số liệu hay tự ý phê duyệt hồ sơ; ứng dụng là hệ thống hỗ trợ ra quyết định.
"""


def _build_chat_contents(question, current_result, conversation):
    """Send a compact, structured multi-turn conversation instead of one giant prompt."""
    contents = []
    recent = conversation[-8:] if conversation else []
    while recent and recent[0].get("role") == "assistant":
        recent = recent[1:]
    for item in recent:
        role = "model" if item.get("role") == "assistant" else "user"
        text_value = str(item.get("content", ""))[:3500]
        if text_value.strip():
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text_value)]))

    context = {}
    if isinstance(current_result, dict):
        app = current_result.get("application", {})
        allowed_application_fields = (
            "person_age", "person_income", "person_home_ownership", "person_emp_length",
            "loan_intent", "loan_amnt", "loan_int_rate", "loan_percent_income",
            "cb_person_default_on_file",
        )
        if isinstance(app, dict):
            context["application"] = {k: app[k] for k in allowed_application_fields if k in app}
        for key in ("mode", "pd_display", "grade", "risk_level", "cluster", "summary", "recommendation"):
            if current_result.get(key) is not None:
                context[key] = str(current_result.get(key))[:1800] if key in ("summary", "recommendation") else current_result.get(key)
        factors = current_result.get("factors", [])
        if isinstance(factors, list):
            context["factors"] = factors[:8]

    user_text = str(question).strip()
    if context:
        user_text = (
            "NGỮ CẢNH HỒ SƠ TÍN DỤNG HIỆN TẠI (dùng để trả lời nếu câu hỏi liên quan):\n"
            + json.dumps(context, ensure_ascii=False, default=str)
            + "\n\nCÂU HỎI MỚI CỦA CHUYÊN VIÊN:\n"
            + user_text
        )
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_text)]))
    return contents


def _select_gemini_models(question, current_result):
    """Route routine turns to a low-latency model and complex analysis to the stronger model."""
    q = str(question).lower()
    complex_markers = (
        "phân tích sâu", "phân tích chi tiết", "đánh giá toàn diện", "so sánh", "tính toán chi tiết",
        "lập kế hoạch", "nhiều bước", "phân tích định lượng", "giải thích chuyên sâu", "báo cáo chi tiết",
        "shap", "tại sao", "vì sao", "nguyên nhân", "đánh giá hồ sơ này", "giải thích kết quả",
        "dự đoán này", "phân tích hồ sơ", "rủi ro cụ thể", "khuyến nghị",
    )
    if len(q) > 450 or any(marker in q for marker in complex_markers):
        primary = "gemini-3.8-flash"
    else:
        primary = "gemini-3.5-flash-lite"
    fallback = "gemini-3.5-flash-lite" if primary == "gemini-3.8-flash" else "gemini-3.8-flash"
    return [primary, fallback]


def _is_fallback_worthy(exc):
    """Use the alternate model only for availability/model-specific failures, not bad credentials."""
    message = str(exc).lower()
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    status_text = str(status).upper() if status is not None else ""
    if status_text in {"401", "UNAUTHENTICATED", "403", "PERMISSION_DENIED"}:
        return False
    if status_text in {"404", "429", "500", "502", "503", "504", "NOT_FOUND", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED"}:
        return True
    return any(term in message for term in (
        "overloaded", "service unavailable", "temporarily unavailable", "timed out",
        "deadline exceeded", "connection reset", "model not found", "not_found", "resource_exhausted",
    ))


def stream_gemini_answer(question, current_result, conversation):
    """Stream real model output; route by task and fall back once if a model is unavailable."""
    started = time.perf_counter()
    first_token_at = None
    got_text = False
    selected_model = "unknown"
    last_exc = None
    attempted_models = []
    thinking_level = "low"
    try:
        client = get_gemini_client()
        if client is None:
            yield "⚠️ Chưa cấu hình GEMINI_API_KEY trong Streamlit Secrets."
            return

        q = str(question).lower()
        complex_markers = (
            "phân tích sâu", "phân tích chi tiết", "đánh giá toàn diện", "so sánh", "tính toán chi tiết",
            "lập kế hoạch", "nhiều bước", "phân tích định lượng", "giải thích chuyên sâu", "báo cáo chi tiết",
            "shap", "tại sao", "vì sao", "nguyên nhân", "đánh giá hồ sơ này", "giải thích kết quả",
            "dự đoán này", "phân tích hồ sơ", "rủi ro cụ thể", "khuyến nghị",
        )
        thinking_level = "medium" if len(q) > 450 or any(marker in q for marker in complex_markers) else "low"
        contents = _build_chat_contents(question, current_result, conversation)
        config = types.GenerateContentConfig(
            system_instruction=_GEMINI_SYSTEM_INSTRUCTION,
            max_output_tokens=1400,
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
        )
        model_candidates = _select_gemini_models(question, current_result)

        for model_index, candidate in enumerate(model_candidates):
            selected_model = candidate
            attempted_models.append(candidate)
            model_got_text = False
            try:
                response_stream = client.models.generate_content_stream(
                    model=candidate,
                    contents=contents,
                    config=config,
                )
                for chunk in response_stream:
                    piece = getattr(chunk, "text", None)
                    if piece:
                        if first_token_at is None:
                            first_token_at = time.perf_counter() - started
                        got_text = True
                        model_got_text = True
                        yield piece
                if model_got_text:
                    last_exc = None
                    break
                last_exc = RuntimeError(f"Model {candidate} returned no text")
                if model_index < len(model_candidates) - 1:
                    continue
            except Exception as exc:
                last_exc = exc
                if got_text:
                    yield "\n\n⚠️ Kết nối bị gián đoạn giữa câu trả lời. " + _friendly_gemini_error(exc)
                    return
                if model_index < len(model_candidates) - 1 and _is_fallback_worthy(exc):
                    continue
                break

        if not got_text:
            if last_exc is not None:
                error_text = _friendly_gemini_error(last_exc)
                if len(attempted_models) > 1:
                    yield "⚠️ Cả hai model Gemini đều chưa xử lý được yêu cầu. " + error_text
                else:
                    yield "⚠️ " + error_text
            else:
                yield "Mình chưa nhận được nội dung trả lời từ Gemini. Bạn thử gửi lại câu hỏi nhé."
    finally:
        elapsed = time.perf_counter() - started
        st.session_state.ai_last_metrics = {
            "model": selected_model,
            "models_attempted": attempted_models,
            "thinking_level": thinking_level,
            "first_token_seconds": first_token_at,
            "total_seconds": elapsed,
        }


# -----------------------------------------------------------------------------
# 3. DATA & VISUALIZATION HELPERS (AL JAZEERA FINTECH VISUAL TOKENS)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_cluster_sample():
    """Load sample data for the K-Means 2D scatter chart."""
    csv_path = Path(__file__).resolve().parent / "credit_risk_cleaned.csv"
    bundle_path = Path(__file__).resolve().parent / "models" / "group14_model_bundle.joblib"
    if not csv_path.exists() or not bundle_path.exists():
        return None
    try:
        import joblib
        bundle = joblib.load(bundle_path)
        df = pd.read_csv(csv_path)
        cluster_feats = ["person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income"]
        sample_df = df.sample(n=min(320, len(df)), random_state=42).copy()
        scaled = bundle["cluster_scaler"].transform(sample_df[cluster_feats])
        sample_df["cluster_id"] = bundle["kmeans"].predict(scaled)
        sample_df["cluster_label"] = sample_df["cluster_id"].map(lambda x: f"Cụm {x}: {CLUSTER_NAMES.get(x, ('Cluster', ''))[0]}")
        return sample_df
    except Exception:
        return None


def build_shap_bar_chart(factors):
    """Build an intuitive diverging horizontal bar chart using Altair (Al Jazeera Fintech Style)."""
    valid_factors = [f for f in factors if isinstance(f.get("Giá trị SHAP"), (int, float))]
    if not valid_factors:
        return None
    df = pd.DataFrame(valid_factors).copy()
    df["Tác động"] = df["Giá trị SHAP"].apply(lambda v: "Tăng rủi ro (+)" if v > 0 else "Giảm rủi ro (-)")
    df["abs_val"] = df["Giá trị SHAP"].abs()
    df = df.sort_values(by="abs_val", ascending=True)

    chart = alt.Chart(df).mark_bar(cornerRadius=5, height=22).encode(
        x=alt.X(
            "Giá trị SHAP:Q",
            title="Đóng góp vào xác suất rủi ro (SHAP Value)",
            axis=alt.Axis(grid=True, gridColor="#dce6f5", tickColor="#bccfe8", titleColor="#173b68", titleFontWeight="bold"),
        ),
        y=alt.Y(
            "Yếu tố:N",
            title=None,
            sort=None,
            axis=alt.Axis(tickSize=0, labelColor="#173b68", labelFontSize=12, labelFontWeight="bold"),
        ),
        color=alt.Color(
            "Tác động:N",
            scale=alt.Scale(
                domain=["Tăng rủi ro (+)", "Giảm rủi ro (-)"],
                range=["#c31833", "#059669"],  # Al Jazeera Live-Red for risk increase, Emerald for risk decrease
            ),
            legend=alt.Legend(
                title=None,
                orient="top",
                labelFontSize=12,
                labelColor="#173b68",
                labelFontWeight="bold",
            ),
        ),
        tooltip=[
            alt.Tooltip("Yếu tố:N", title="Đặc trưng"),
            alt.Tooltip("Giá trị SHAP:Q", format="+.4f", title="Giá trị SHAP"),
            alt.Tooltip("Hướng tác động:N", title="Xu hướng"),
        ],
    ).properties(
        height=min(320, len(df) * 44 + 40),
        background="transparent",
    ).configure_view(
        strokeWidth=0
    )
    return chart


def build_cluster_scatter_chart(sample_df, current_record=None):
    """Build a 2D scatter plot of customer financial profile with current applicant pin."""
    if sample_df is None or sample_df.empty:
        return None

    # Palette: Amber, Emerald, Royal Indigo, Al Jazeera Live-Red
    cluster_colors = ["#d97706", "#059669", "#5266eb", "#c31833"]

    base = alt.Chart(sample_df).mark_circle(size=64, opacity=0.75).encode(
        x=alt.X(
            "person_income:Q",
            title="Thu nhập hằng năm ($)",
            axis=alt.Axis(grid=True, gridColor="#dce6f5", tickColor="#bccfe8", format="$,.0f", titleColor="#173b68", titleFontWeight="bold"),
        ),
        y=alt.Y(
            "loan_amnt:Q",
            title="Số tiền đề nghị vay ($)",
            axis=alt.Axis(grid=True, gridColor="#dce6f5", tickColor="#bccfe8", format="$,.0f", titleColor="#173b68", titleFontWeight="bold"),
        ),
        color=alt.Color(
            "cluster_label:N",
            scale=alt.Scale(range=cluster_colors),
            legend=alt.Legend(
                title="Phân khúc khách hàng (K-Means)",
                orient="bottom",
                columns=2,
                labelFontSize=11,
                labelColor="#173b68",
                symbolSize=80,
            ),
        ),
        tooltip=[
            alt.Tooltip("cluster_label:N", title="Phân khúc"),
            alt.Tooltip("person_income:Q", format="$,.0f", title="Thu nhập"),
            alt.Tooltip("loan_amnt:Q", format="$,.0f", title="Số tiền vay"),
            alt.Tooltip("loan_int_rate:Q", format=".1f", title="Lãi suất (%)"),
            alt.Tooltip("loan_percent_income:Q", format=".2f", title="Tỷ lệ vay/thu nhập"),
        ],
    )

    if current_record:
        app_info = current_record.get("application", current_record) if isinstance(current_record, dict) else {}
        inc = app_info.get("person_income")
        amt = app_info.get("loan_amnt")
        if inc is not None and amt is not None:
            applicant_df = pd.DataFrame([{
                "person_income": float(inc),
                "loan_amnt": float(amt),
                "label": "★ Hồ sơ đang thẩm định",
            }])
            beacon = alt.Chart(applicant_df).mark_point(
                shape="diamond",
                size=400,
                color="#173b68",
                fill="#fa9000",  # Al Jazeera Gold accent beacon
                stroke="#ffffff",
                strokeWidth=3.0,
                opacity=1.0,
            ).encode(
                x="person_income:Q",
                y="loan_amnt:Q",
                tooltip=[
                    alt.Tooltip("label:N", title="Vị trí"),
                    alt.Tooltip("person_income:Q", format="$,.0f", title="Thu nhập"),
                    alt.Tooltip("loan_amnt:Q", format="$,.0f", title="Số tiền vay"),
                ],
            )
            final_chart = (base + beacon)
        else:
            final_chart = base
    else:
        final_chart = base

    return final_chart.properties(height=350, background="transparent").configure_view(strokeWidth=0)


# -----------------------------------------------------------------------------
# 4. STREAMLIT APP CONFIG & GETDESIGN AL JAZEERA STYLESHEET
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CreditRisk AI — Intelligent Credit Risk Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Roboto:wght@300;400;500;700;900&display=swap');

:root {
  --primary-indigo: #5266eb;
  --primary-hover: #4353ca;
  --deep-navy: #173b68;
  --indigo-royal: #272967;
  --accent-gold: #fa9000;
  --live-red: #c31833;
  --status-green: #059669;
  --status-amber: #d97706;
  --status-teal: #0d9488;
  --text-main: #243b5a;
  --text-muted: #5f7594;
  --bg-canvas: #f3f7ff;
  --surface: #ffffff;
  --surface-alt: #f8fbff;
  --hairline: #dce6f5;
  --hairline-soft: #eaf1fb;
  --shadow-subtle: 0 2px 8px rgba(23, 59, 104, 0.04);
  --shadow-card: 0 6px 20px rgba(23, 59, 104, 0.06);
  --shadow-hover: 0 10px 28px rgba(82, 102, 235, 0.12);
}

/* Global Canvas Styling */
.stApp {
  background: radial-gradient(ellipse at 12% 0%, rgba(82, 102, 235, 0.09), transparent 38%),
              radial-gradient(ellipse at 94% 8%, rgba(250, 144, 0, 0.06), transparent 28%),
              linear-gradient(145deg, #f8faff 0%, #f3f7ff 50%, #f6f9ff 100%) !important;
  color: var(--text-main);
  font-family: 'Inter', 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

[data-testid="stHeader"] {
  background: rgba(243, 247, 255, 0.92) !important;
  backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--hairline-soft);
}
[data-testid="stToolbar"] { right: 1.2rem; }

.block-container {
  max-width: 1420px !important;
  padding-top: 1.25rem !important;
  padding-bottom: 3.5rem !important;
}

/* Sidebar Editorial Styling */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #ffffff 0%, #f7faff 60%, #eef5ff 100%) !important;
  border-right: 1px solid var(--hairline) !important;
  box-shadow: 4px 0 24px rgba(23, 59, 104, 0.035) !important;
}
[data-testid="stSidebar"] .block-container { padding-top: 1.35rem !important; }

/* Brand Mark & Navigation */
.brand-mark {
  width: 48px;
  height: 48px;
  border-radius: 14px;
  background: linear-gradient(135deg, #3679f6 0%, #5266eb 60%, #272967 100%);
  color: #fff;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  font-weight: 800;
  margin-right: 12px;
  vertical-align: middle;
  box-shadow: 0 8px 20px rgba(82, 102, 235, 0.28);
  border: 1px solid rgba(255, 255, 255, 0.85);
}
.brand-title {
  display: inline-block;
  vertical-align: middle;
  font-size: 1.35rem;
  font-weight: 800;
  color: var(--deep-navy);
  line-height: 1.15;
  letter-spacing: -0.03em;
}
.brand-sub {
  font-size: 0.82rem;
  color: var(--text-muted);
  margin-top: 8px;
  line-height: 1.6;
}
.brand-sub b { color: var(--primary-indigo); }

/* Al Jazeera Eyebrow Lockup: 4px vertical bar + tracked uppercase label */
.aljazeera-eyebrow-lockup {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 1.25rem 0 0.35rem;
}
.eyebrow-accent-bar {
  width: 4px;
  height: 16px;
  border-radius: 2px;
  background: var(--primary-indigo);
  flex-shrink: 0;
}
.eyebrow-accent-bar-gold {
  background: var(--accent-gold);
}
.eyebrow-text {
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--primary-indigo);
  line-height: 1;
}
.section-headline {
  color: var(--deep-navy);
  font-size: 1.45rem;
  font-weight: 780;
  letter-spacing: -0.02em;
  margin: 0.15rem 0 0.25rem;
  line-height: 1.25;
}
.section-description {
  color: var(--text-muted);
  font-size: 0.88rem;
  margin: 0 0 1.15rem 0;
  line-height: 1.6;
}

/* VietTrip Inspired Hero Section */
.viettrip-hero {
  position: relative;
  overflow: hidden;
  background: linear-gradient(125deg, #ffffff 0%, #f5f9ff 55%, #edf4ff 100%);
  border: 1px solid var(--hairline);
  border-radius: 20px;
  padding: 2.2rem 2.4rem;
  margin: 0.2rem 0 1.45rem;
  box-shadow: var(--shadow-card);
}
.viettrip-hero::before {
  content: "";
  position: absolute;
  width: 320px;
  height: 320px;
  right: -60px;
  top: -120px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(82, 102, 235, 0.14), rgba(82, 102, 235, 0) 70%);
  pointer-events: none;
}
.viettrip-hero::after {
  content: "";
  position: absolute;
  width: 200px;
  height: 200px;
  right: 140px;
  bottom: -90px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(250, 144, 0, 0.08), rgba(250, 144, 0, 0) 70%);
  pointer-events: none;
}
.hero-tag {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--primary-indigo);
  font-weight: 800;
  font-size: 0.72rem;
  margin-bottom: 0.85rem;
  padding: 0.35rem 0.85rem;
  border-radius: 999px;
  background: #eef3ff;
  border: 1px solid #dce6f5;
}
.hero-tag::before { content: "✦"; font-size: 0.75rem; color: var(--accent-gold); }
.viettrip-hero h1 {
  color: var(--deep-navy);
  font-size: 2.35rem;
  line-height: 1.18;
  margin: 0;
  font-weight: 820;
  letter-spacing: -0.035em;
}
.viettrip-hero p {
  color: #4a5e78;
  font-size: 1.02rem;
  margin: 0.85rem 0 1.25rem;
  max-width: 900px;
  line-height: 1.7;
}
.hero-chips-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 1.1rem;
}
.hero-chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 0.28rem 0.75rem;
  border-radius: 8px;
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--deep-navy);
  background: rgba(255, 255, 255, 0.85);
  border: 1px solid var(--hairline);
  box-shadow: 0 1px 3px rgba(23, 59, 104, 0.03);
}

/* Al Jazeera Stat Metric Cards */
div[data-testid="stMetric"] {
  background: #ffffff !important;
  border: 1px solid var(--hairline) !important;
  border-top: 3px solid var(--primary-indigo) !important;
  padding: 1.15rem 1.25rem !important;
  border-radius: 16px !important;
  box-shadow: var(--shadow-subtle) !important;
  transition: transform 0.18s ease, box-shadow 0.18s ease !important;
}
div[data-testid="stMetric"]:hover {
  transform: translateY(-2px) !important;
  box-shadow: var(--shadow-hover) !important;
}
div[data-testid="stMetricLabel"] {
  color: var(--text-muted) !important;
  font-weight: 650 !important;
  font-size: 0.82rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.04em !important;
}
div[data-testid="stMetricValue"] {
  color: var(--deep-navy) !important;
  font-weight: 800 !important;
  font-size: 1.95rem !important;
  letter-spacing: -0.03em !important;
}

/* VietTrip Planning Console / Forms */
[data-testid="stForm"] {
  background: #ffffff !important;
  border: 1px solid var(--hairline) !important;
  border-radius: 18px !important;
  padding: 1.45rem 1.6rem !important;
  box-shadow: var(--shadow-card) !important;
}
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
  border-radius: 10px !important;
  border-color: #d6e2f2 !important;
  background: #ffffff !important;
  color: var(--text-main) !important;
  font-weight: 500 !important;
  transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
[data-testid="stTextInput"] input:focus, [data-testid="stNumberInput"] input:focus {
  border-color: var(--primary-indigo) !important;
  box-shadow: 0 0 0 3px rgba(82, 102, 235, 0.15) !important;
}

/* Nút Bấm Chính & Phụ (CTA) */
.stButton > button[kind="primary"], button[kind="primary"] {
  background: linear-gradient(105deg, var(--primary-indigo) 0%, #3679f6 100%) !important;
  border: 1px solid var(--primary-hover) !important;
  color: #ffffff !important;
  border-radius: 12px !important;
  font-weight: 750 !important;
  padding: 0.65rem 1.4rem !important;
  box-shadow: 0 6px 18px rgba(82, 102, 235, 0.28) !important;
  transition: all 0.18s ease !important;
}
.stButton > button[kind="primary"]:hover, button[kind="primary"]:hover {
  background: linear-gradient(105deg, var(--primary-hover) 0%, #2960da 100%) !important;
  box-shadow: 0 9px 24px rgba(82, 102, 235, 0.38) !important;
  transform: translateY(-1px) !important;
}
.stDownloadButton > button, .stButton > button:not([kind="primary"]) {
  border-color: #d1dfef !important;
  color: var(--deep-navy) !important;
  background: linear-gradient(180deg, #ffffff, #f7faff) !important;
  border-radius: 11px !important;
  font-weight: 650 !important;
  transition: all 0.16s ease !important;
}
.stDownloadButton > button:hover, .stButton > button:not([kind="primary"]):hover {
  border-color: var(--primary-indigo) !important;
  color: var(--primary-indigo) !important;
  background: #eef4ff !important;
}

/* Scorecard & Panels */
.scorecard-panel {
  background: #ffffff;
  border: 1px solid var(--hairline);
  border-radius: 16px;
  padding: 1.35rem 1.45rem;
  box-shadow: var(--shadow-subtle);
  margin-bottom: 1.15rem;
}
.scorecard-panel b { color: var(--deep-navy); }

/* Thước đo rủi ro trực quan (Visual Risk Meter) */
.risk-meter-wrapper { margin: 1.25rem 0 0.85rem; }
.risk-meter-bar {
  display: flex;
  height: 12px;
  border-radius: 999px;
  overflow: hidden;
  background: #e9f0f8;
  gap: 3px;
}
.risk-meter-seg { height: 100%; flex: 1; transition: opacity 0.2s ease; }
.risk-meter-seg-A { background: var(--status-green); }
.risk-meter-seg-B { background: var(--status-teal); }
.risk-meter-seg-C { background: var(--status-amber); }
.risk-meter-seg-D { background: var(--live-red); }
.risk-meter-labels {
  display: flex;
  justify-content: space-between;
  font-size: 0.74rem;
  color: var(--text-muted);
  margin-top: 0.5rem;
  font-weight: 600;
}

/* Quick Preset Chips Bar */
.preset-section-box {
  background: linear-gradient(180deg, #ffffff 0%, #f9fcff 100%);
  border: 1px solid var(--hairline);
  border-radius: 14px;
  padding: 0.95rem 1.15rem;
  margin-bottom: 1.15rem;
}
.preset-bar-title {
  font-size: 0.82rem;
  font-weight: 750;
  color: var(--deep-navy);
  margin-bottom: 0.55rem;
  display: flex;
  align-items: center;
  gap: 6px;
}

/* Tables & DataFrames */
[data-testid="stDataFrame"] {
  border: 1px solid var(--hairline) !important;
  border-radius: 14px !important;
  overflow: hidden !important;
  box-shadow: var(--shadow-subtle) !important;
}

hr { border-color: var(--hairline-soft) !important; margin: 1.35rem 0 !important; }

/* Floating AI Assistant Chatbot */
.st-key-ai_floating_button {
  position: fixed !important;
  right: 24px !important;
  bottom: 22px !important;
  z-index: 99999 !important;
  width: 64px !important;
}
.st-key-ai_floating_button button {
  width: 58px !important;
  height: 58px !important;
  border-radius: 50% !important;
  border: 0 !important;
  color: #ffffff !important;
  background: linear-gradient(135deg, var(--primary-indigo), #272967) !important;
  box-shadow: 0 8px 28px rgba(82, 102, 235, 0.42) !important;
  font-size: 24px !important;
  transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
.st-key-ai_floating_button button:hover {
  transform: scale(1.08) !important;
  box-shadow: 0 12px 36px rgba(82, 102, 235, 0.52) !important;
}

.st-key-ai_chat_panel {
  position: fixed !important;
  right: 24px !important;
  bottom: 92px !important;
  width: min(440px, calc(100vw - 28px)) !important;
  max-height: min(640px, calc(100vh - 120px)) !important;
  overflow-y: auto !important;
  z-index: 99998 !important;
  background: #ffffff !important;
  border: 1px solid var(--hairline) !important;
  border-radius: 20px !important;
  padding: 16px !important;
  box-shadow: 0 20px 60px rgba(23, 59, 104, 0.22) !important;
}
.st-key-ai_chat_panel [data-testid="stVerticalBlock"] { gap: 0.55rem; }
.st-key-ai_chat_panel [data-testid="stChatMessage"] { padding: 0.65rem 0.8rem; }
.st-key-ai_chat_panel [data-testid="stForm"] { border: 0 !important; padding: 0 !important; background: transparent !important; box-shadow: none !important; }

/* Enterprise FinTech Footer */
.enterprise-footer {
  border-top: 1px solid var(--hairline);
  margin-top: 3.5rem;
  padding: 2.2rem 0.5rem 1rem;
  color: #657894;
  font-size: 0.84rem;
  line-height: 1.7;
}
.footer-grid {
  display: grid;
  grid-template-columns: 2fr 1.5fr 1.5fr;
  gap: 2rem;
  margin-bottom: 1.5rem;
}
.footer-col h4 {
  color: var(--deep-navy);
  font-size: 0.95rem;
  font-weight: 750;
  margin: 0 0 0.75rem;
  letter-spacing: -0.01em;
}
.footer-bottom {
  border-top: 1px solid var(--hairline-soft);
  padding-top: 1.1rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.78rem;
}

@media (max-width: 850px) {
  .footer-grid { grid-template-columns: 1fr; gap: 1.2rem; }
  .footer-bottom { flex-direction: column; gap: 0.5rem; text-align: center; }
  .viettrip-hero { padding: 1.4rem 1.4rem; }
  .viettrip-hero h1 { font-size: 1.7rem; }
  .st-key-ai_chat_panel { right: 8px !important; bottom: 82px !important; width: calc(100vw - 16px) !important; }
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 5. INITIALIZATION & SESSION STATE
# -----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "preset_data" not in st.session_state:
    st.session_state.preset_data = None

if "ai_chat_open" not in st.session_state:
    st.session_state.ai_chat_open = False

if "ai_chat_messages" not in st.session_state:
    st.session_state.ai_chat_messages = []

db_ok, db_message = init_history_table()
status = model_status()
sample_cluster_df = load_cluster_sample()


# -----------------------------------------------------------------------------
# SECTION 01: MAIN NAVIGATION & SIDEBAR SYSTEM
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; margin-bottom: 0.85rem;">
        <span class="brand-mark">🛡️</span>
        <div>
            <span class="brand-title">CreditRisk AI</span>
            <div style="font-size:0.75rem; color:#5266eb; font-weight:700; letter-spacing:0.04em;">INTELLIGENT FINTECH</div>
        </div>
    </div>
    <div class="brand-sub">
        Nền tảng thẩm định tín dụng máy học<br>
        <b>Học thuật · Nhóm 14</b> · Bản phát hành v3.0
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "ĐIỀU HƯỚNG NỀN TẢNG",
        [
            "🧭 Tổng quan & Thẩm định",
            "👥 Phân khúc K-Means",
            "📋 Lịch sử thẩm định",
            "⚙️ Cấu trúc hệ thống",
        ],
        label_visibility="visible",
    )
    st.divider()

    st.markdown('<div class="preset-bar-title">TRẠNG THÁI HỆ THỐNG</div>', unsafe_allow_html=True)
    if status["ready"]:
        st.markdown("""
        <div style="background:#ecfdf5; border:1px solid #a7f3d0; border-radius:10px; padding:0.6rem 0.8rem; margin-bottom:0.5rem; display:flex; align-items:center; gap:8px;">
            <span style="color:#059669; font-size:1rem;">●</span>
            <span style="color:#065f46; font-size:0.8rem; font-weight:700;">Random Forest: Sẵn sàng</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#fffbeb; border:1px solid #fde68a; border-radius:10px; padding:0.6rem 0.8rem; margin-bottom:0.5rem; display:flex; align-items:center; gap:8px;">
            <span style="color:#d97706; font-size:1rem;">●</span>
            <span style="color:#92400e; font-size:0.8rem; font-weight:700;">Chưa tìm thấy model bundle</span>
        </div>
        """, unsafe_allow_html=True)

    if db_ok:
        st.markdown("""
        <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px; padding:0.6rem 0.8rem; margin-bottom:0.5rem; display:flex; align-items:center; gap:8px;">
            <span style="color:#16a34a; font-size:1rem;">●</span>
            <span style="color:#166534; font-size:0.8rem; font-weight:700;">PostgreSQL: Đã kết nối</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:0.6rem 0.8rem; margin-bottom:0.5rem; display:flex; align-items:center; gap:8px;">
            <span style="color:#64748b; font-size:1rem;">○</span>
            <span style="color:#475569; font-size:0.8rem; font-weight:600;">Lưu trữ phiên (In-memory)</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="brand-sub" style="margin-top:14px; font-size:0.77rem; line-height:1.5;">Hệ thống hỗ trợ ra quyết định tín dụng. Không sử dụng để giải ngân tự động ngoài thực tế.</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TRANG 1: TỔNG QUAN & THẨM ĐỊNH (SECTION 02 -> SECTION 06)
# -----------------------------------------------------------------------------
if page == "🧭 Tổng quan & Thẩm định":
    # SECTION 02: HERO / INTRODUCTION (VietTrip Inspired)
    st.markdown("""
    <div class="viettrip-hero">
        <div class="hero-tag">INTELLIGENT CREDIT RISK PLATFORM</div>
        <h1>Đánh giá rủi ro tín dụng thông minh với AI</h1>
        <p>Kết hợp Machine Learning và Explainable AI để phân tích xác suất vỡ nợ, phân loại mức độ rủi ro và hỗ trợ ra quyết định tín dụng nhanh chóng, minh bạch và tin cậy.</p>
        <div class="hero-chips-strip">
            <span class="hero-chip">🌲 Random Forest (100 Trees)</span>
            <span class="hero-chip">📈 ROC-AUC 92.10%</span>
            <span class="hero-chip">🎯 Recall 73.91%</span>
            <span class="hero-chip">🔍 SHAP TreeExplainer</span>
            <span class="hero-chip">👥 K-Means 4 Clusters</span>
            <span class="hero-chip">☁️ Supabase Cloud DB</span>
            <span class="hero-chip">🤖 Gemini Multiturn AI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # SECTION 03: MODEL PERFORMANCE (Al Jazeera Stat Cards)
    st.markdown("""
    <div class="aljazeera-eyebrow-lockup">
        <span class="eyebrow-accent-bar"></span>
        <span class="eyebrow-text">MODEL PERFORMANCE & CORE METRICS</span>
    </div>
    <div class="section-headline">Năng lực mô hình máy học thẩm định</div>
    <div class="section-description">Các thông số đo lường hiệu năng của mô hình Random Forest tốt nhất theo báo cáo thực nghiệm Nhóm 14.</div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(
            "MÔ HÌNH PHÂN LOẠI",
            "Random Forest",
            help="100 cây quyết định, tối ưu hóa tiêu chuẩn Gini với pipeline tiền xử lý Median và One-Hot Encoder",
        )
    with m2:
        st.metric(
            "ĐỘ CHÍNH XÁC PHÂN LOẠI (ROC-AUC)",
            "92.10%",
            help="Khả năng phân biệt vượt trội giữa người vay an toàn và người vay có nguy cơ vỡ nợ trên tập kiểm tra độc lập",
        )
    with m3:
        st.metric(
            "KHẢ NĂNG BẮT RỦI RO (RECALL)",
            "73.91%",
            help="Tỷ lệ nhận diện chuẩn xác các trường hợp thực tế có rủi ro vỡ nợ nhằm bảo vệ an toàn vốn cho ngân hàng",
        )

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    # SECTION 04: CREDIT ASSESSMENT PLANNER (VietTrip Planning Console)
    st.markdown("""
    <div class="aljazeera-eyebrow-lockup">
        <span class="eyebrow-accent-bar"></span>
        <span class="eyebrow-text">CREDIT ASSESSMENT CONSOLE</span>
    </div>
    <div class="section-headline">Phân tích hồ sơ tín dụng</div>
    <div class="section-description">Nhập thông tin khách hàng và khoản vay để hệ thống ước tính xác suất vỡ nợ, xếp hạng tín dụng và đề xuất giải pháp.</div>
    """, unsafe_allow_html=True)

    # 1-Click Quick Presets Bar
    st.markdown("""
    <div class="preset-section-box">
        <div class="preset-bar-title">⚡ Chọn nhanh hồ sơ mẫu để thẩm định tức thì (1-Click Presets):</div>
    </div>
    """, unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        if st.button("⭐ Hồ sơ A (Tối ưu · PD 0.8%)", use_container_width=True, help="Thu nhập $95k, 7 năm KN, Sở hữu nhà, Lãi 7.5%"):
            st.session_state.preset_data = {
                "age": 32, "income": 95000, "emp_length": 7, "home": "OWN",
                "purpose": "PERSONAL", "loan_amount": 8000, "interest_rate": 7.5,
                "loan_percent_income": 0.08, "default_history": "N"
            }
            st.rerun()
    with p2:
        if st.button("🔹 Hồ sơ B (Khá · PD 10.9%)", use_container_width=True, help="Thu nhập $60k, 4 năm KN, Thuê nhà, Lãi 10.5%"):
            st.session_state.preset_data = {
                "age": 28, "income": 60000, "emp_length": 4, "home": "RENT",
                "purpose": "EDUCATION", "loan_amount": 10000, "interest_rate": 10.5,
                "loan_percent_income": 0.17, "default_history": "N"
            }
            st.rerun()
    with p3:
        if st.button("🔸 Hồ sơ C (Chú ý · PD 20.6%)", use_container_width=True, help="Thu nhập $48k, 3 năm KN, Vay y tế, Lãi 12%"):
            st.session_state.preset_data = {
                "age": 26, "income": 48000, "emp_length": 3, "home": "RENT",
                "purpose": "MEDICAL", "loan_amount": 10000, "interest_rate": 12.0,
                "loan_percent_income": 0.21, "default_history": "N"
            }
            st.rerun()
    with p4:
        if st.button("🔻 Hồ sơ D (Rủi ro · PD 98.2%)", use_container_width=True, help="Thu nhập $22k, 1 năm KN, Tỷ lệ vay 64%, Lãi 17.5%, Từng nợ xấu"):
            st.session_state.preset_data = {
                "age": 23, "income": 22000, "emp_length": 1, "home": "RENT",
                "purpose": "MEDICAL", "loan_amount": 14000, "interest_rate": 17.5,
                "loan_percent_income": 0.64, "default_history": "Y"
            }
            st.rerun()

    preset = st.session_state.preset_data or {}

    # Assessment Input Form: 3 Clean Columns (Customer, Loan, Credit History)
    with st.form("application_assessment_form"):
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.markdown("""
            <div style="font-weight:750; color:#173b68; font-size:0.92rem; margin-bottom:0.75rem; border-bottom:2px solid #5266eb; padding-bottom:4px;">
                01 · Hồ sơ khách hàng
            </div>
            """, unsafe_allow_html=True)
            age = st.number_input("Tuổi khách hàng", min_value=18, max_value=100, value=preset.get("age", 28), step=1)
            income = st.number_input("Thu nhập hằng năm ($)", min_value=1, max_value=2000000, value=preset.get("income", 60000), step=1000)
            emp_length = st.number_input("Thâm niên làm việc (năm)", min_value=0, max_value=70, value=preset.get("emp_length", 4), step=1)
            home_options = ["RENT", "MORTGAGE", "OWN", "OTHER"]
            home_index = home_options.index(preset.get("home", "RENT")) if preset.get("home") in home_options else 0
            home = st.selectbox("Tình trạng nhà ở", home_options, index=home_index)

        with col_c2:
            st.markdown("""
            <div style="font-weight:750; color:#173b68; font-size:0.92rem; margin-bottom:0.75rem; border-bottom:2px solid #5266eb; padding-bottom:4px;">
                02 · Thông tin khoản vay
            </div>
            """, unsafe_allow_html=True)
            purpose_options = ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
            purpose_index = purpose_options.index(preset.get("purpose", "PERSONAL")) if preset.get("purpose") in purpose_options else 3
            purpose = st.selectbox("Mục đích vay vốn", purpose_options, index=purpose_index)
            loan_amount = st.number_input("Số tiền đề nghị vay ($)", min_value=500, max_value=50000, value=preset.get("loan_amount", 8000), step=500)
            interest_rate = st.number_input("Lãi suất danh nghĩa (%)", min_value=0.0, max_value=35.0, value=preset.get("interest_rate", 10.5), step=0.25)
            default_calc_ratio = min(round(float(loan_amount) / float(max(income, 1)), 2), 1.0)
            ratio_val = min(float(preset.get("loan_percent_income", default_calc_ratio)), 1.0)
            loan_percent_income = st.number_input("Tỷ lệ khoản vay / Thu nhập (DTI)", min_value=0.0, max_value=1.0, value=ratio_val, step=0.01, format="%.2f")

        with col_c3:
            st.markdown("""
            <div style="font-weight:750; color:#173b68; font-size:0.92rem; margin-bottom:0.75rem; border-bottom:2px solid #5266eb; padding-bottom:4px;">
                03 · Lịch sử tín dụng
            </div>
            """, unsafe_allow_html=True)
            default_options = ["N", "Y"]
            default_index = default_options.index(preset.get("default_history", "N")) if preset.get("default_history") in default_options else 0
            default_history = st.selectbox("Từng có lịch sử nợ xấu", default_options, index=default_index, help="Y: Đã từng quá hạn/vỡ nợ; N: Chưa từng vi phạm hợp đồng")

            st.markdown("""
            <div style="background:#f8faff; border:1px solid #dce6f5; border-radius:10px; padding:0.85rem; margin-top:1.1rem;">
                <div style="font-size:0.78rem; font-weight:700; color:#173b68; margin-bottom:4px;">LƯU Ý NGHIỆP VỤ</div>
                <div style="font-size:0.76rem; color:#5f7594; line-height:1.5;">
                    Mô hình Random Forest áp dụng ngưỡng phân vị chuẩn mực: Q1 (7.65%), Q2 (17.61%), Q3 (40.86%) để phân hạng Grade A đến D.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("✨ Phân tích hồ sơ tín dụng (Analyze Credit Profile)", type="primary", use_container_width=True)

    if submitted:
        record = {
            "person_age": int(age),
            "person_income": float(income),
            "person_home_ownership": home,
            "person_emp_length": int(emp_length),
            "loan_intent": purpose,
            "loan_amnt": float(loan_amount),
            "loan_int_rate": float(interest_rate),
            "loan_percent_income": float(loan_percent_income),
            "cb_person_default_on_file": default_history,
        }
        with st.spinner("Đang tính toán xác suất vỡ nợ, giải thích SHAP và phân cụm K-Means..."):
            result = evaluate_application(record)
            result["application"] = record
            result["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.last_result = result

            if db_ok:
                saved, save_error = save_history_item(result)
                if saved:
                    st.toast("✅ Đã lưu kết quả vào cơ sở dữ liệu PostgreSQL", icon="💾")
                else:
                    st.toast(f"⚠️ Chưa ghi được vào DB: {save_error}", icon="⚠️")

            st.session_state.history.insert(0, result)
            st.session_state.history = st.session_state.history[:100]

    # SECTION 05: RISK ASSESSMENT RESULT (VietTrip Scorecard)
    result = st.session_state.get("last_result")
    if result:
        st.divider()
        st.markdown("""
        <div class="aljazeera-eyebrow-lockup">
            <span class="eyebrow-accent-bar"></span>
            <span class="eyebrow-text">ASSESSMENT SCORECARD</span>
        </div>
        <div class="section-headline">Báo cáo thẩm định rủi ro tín dụng</div>
        <div class="section-description">Kết quả tính toán thời gian thực từ mô hình Random Forest và các ngưỡng phân vị xác suất vỡ nợ.</div>
        """, unsafe_allow_html=True)

        grade_val = result.get("grade", "A")
        grade_color_map = {
            "A": "#059669",
            "B": "#0d9488",
            "C": "#d97706",
            "D": "#c31833",
        }
        badge_color = grade_color_map.get(grade_val, "#5266eb")

        # 3 Scorecard Columns
        res1, res2, res3 = st.columns(3)
        with res1:
            st.metric(
                "XÁC SUẤT VỠ NỢ (PD)",
                result.get("pd_display", "N/A"),
                help="Probability of Default: Tỷ lệ ước tính người vay không có khả năng hoàn trả khoản vay",
            )
        with res2:
            st.metric(
                "XẾP HẠNG TÍN DỤNG",
                f"Hạng {grade_val}",
                delta=result.get("risk_level", ""),
                delta_color="off",
            )
        with res3:
            st.metric(
                "PHÂN KHÚC KHÁCH HÀNG (K-MEANS)",
                result.get("cluster", "Chưa phân cụm"),
                help="Phân loại hành vi tài chính đa chiều theo thuật toán K-Means 4 cụm",
            )

        # Visual Risk Meter
        st.markdown("""
        <div class="risk-meter-wrapper">
            <div class="risk-meter-bar">
                <div class="risk-meter-seg risk-meter-seg-A" title="Hạng A (Rất thấp: PD <= 7.65%)"></div>
                <div class="risk-meter-seg risk-meter-seg-B" title="Hạng B (Thấp: 7.65% < PD <= 17.61%)"></div>
                <div class="risk-meter-seg risk-meter-seg-C" title="Hạng C (Chú ý: 17.61% < PD <= 40.86%)"></div>
                <div class="risk-meter-seg risk-meter-seg-D" title="Hạng D (Rủi ro cao: PD > 40.86%)"></div>
            </div>
            <div class="risk-meter-labels">
                <span>Hạng A: Rất an toàn (≤ 7.65%)</span>
                <span>Hạng B: Rủi ro thấp (7.65% – 17.61%)</span>
                <span>Hạng C: Cảnh giác (17.61% – 40.86%)</span>
                <span>Hạng D: Rủi ro vỡ nợ cao (> 40.86%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Underwriting Action Recommendation Banner
        st.markdown(f"""
        <div class="scorecard-panel" style="border-left: 5px solid {badge_color};">
            <div style="font-size:0.82rem; font-weight:750; text-transform:uppercase; letter-spacing:0.06em; color:{badge_color}; margin-bottom:4px;">
                KHUYẾN NGHỊ THẨM ĐỊNH TÍN DỤNG (UNDERWRITING ACTION)
            </div>
            <div style="font-size:1.05rem; font-weight:700; color:#173b68; margin-bottom:0.5rem;">
                {result.get("summary", "")}
            </div>
            <div style="font-size:0.88rem; color:#4a5e78; line-height:1.65;">
                {result.get("recommendation", "")}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # SECTION 06: EXPLAINABLE AI (SHAP TreeExplainer)
        factors = result.get("factors", [])
        if factors:
            st.markdown("""
            <div class="aljazeera-eyebrow-lockup">
                <span class="eyebrow-accent-bar"></span>
                <span class="eyebrow-text">EXPLAINABLE AI INSIGHTS</span>
            </div>
            <div class="section-headline">Hiểu cách AI đưa ra quyết định đánh giá</div>
            <div class="section-description">Thuật toán SHAP (Shapley Additive Explanations) bóc tách chính xác mức độ đóng góp của từng yếu tố hồ sơ vào xác suất rủi ro.</div>
            """, unsafe_allow_html=True)

            shap_chart = build_shap_bar_chart(factors)
            if shap_chart:
                st.altair_chart(shap_chart, use_container_width=True)

            st.dataframe(pd.DataFrame(factors), hide_index=True, use_container_width=True)
        else:
            st.caption("Chưa có SHAP explainer tương thích hoặc chế độ minh họa. Dự báo phân hạng vẫn hoạt động bình thường.")


# -----------------------------------------------------------------------------
# TRANG 2: PHÂN KHÚC KHÁCH HÀNG K-MEANS (SECTION 07)
# -----------------------------------------------------------------------------
elif page == "👥 Phân khúc K-Means":
    st.markdown("""
    <div class="viettrip-hero">
        <div class="hero-tag">CUSTOMER RISK SEGMENTATION</div>
        <h1>Khám phá các phân khúc khách hàng mục tiêu</h1>
        <p>Thuật toán K-Means phân loại hồ sơ thành 4 cụm dựa trên các đặc trưng tài chính cốt lõi: Độ tuổi, Thu nhập, Thâm niên công tác, Số tiền vay, Lãi suất và Tỷ lệ nợ trên thu nhập.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="aljazeera-eyebrow-lockup">
        <span class="eyebrow-accent-bar"></span>
        <span class="eyebrow-text">CLUSTER PROFILES & STRATEGIES</span>
    </div>
    <div class="section-headline">Đặc trưng & Chiến lược quản trị rủi ro 4 Cụm</div>
    <div class="section-description">Hướng dẫn tiếp cận và chính sách cấp tín dụng cho từng nhóm khách hàng mục tiêu.</div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class="scorecard-panel" style="border-left: 4px solid #d97706; margin-bottom:1rem;">
            <b style="font-size:1rem; color:#173b68;">Cụm 0: {CLUSTER_NAMES[0][0]}</b>
            <p style="margin: 6px 0; color:#536985; font-size:0.88rem;">{CLUSTER_NAMES[0][1]}</p>
            <div style="font-size:0.8rem; color:#657790; margin-top:8px;">
                <b>Chiến lược cấp vốn:</b> Cần kiểm tra dòng tiền phụ, giới hạn hạn mức tín chấp và thẩm định thu nhập thực tế.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="scorecard-panel" style="border-left: 4px solid #059669; margin-bottom:1rem;">
            <b style="font-size:1rem; color:#173b68;">Cụm 1: {CLUSTER_NAMES[1][0]}</b>
            <p style="margin: 6px 0; color:#536985; font-size:0.88rem;">{CLUSTER_NAMES[1][1]}</p>
            <div style="font-size:0.8rem; color:#657790; margin-top:8px;">
                <b>Chiến lược cấp vốn:</b> Nhóm khách hàng hoàn hảo nhất; áp dụng luồng phê duyệt tự động và chính sách lãi suất ưu đãi.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div class="scorecard-panel" style="border-left: 4px solid #5266eb; margin-bottom:1rem;">
            <b style="font-size:1rem; color:#173b68;">Cụm 2: {CLUSTER_NAMES[2][0]}</b>
            <p style="margin: 6px 0; color:#536985; font-size:0.88rem;">{CLUSTER_NAMES[2][1]}</p>
            <div style="font-size:0.8rem; color:#657790; margin-top:8px;">
                <b>Chiến lược cấp vốn:</b> Khách hàng cao cấp (Affluent); tiềm năng cho vay mua nhà, phát triển doanh nghiệp và cross-sale thẻ tín dụng hạng sang.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="scorecard-panel" style="border-left: 4px solid #c31833; margin-bottom:1rem;">
            <b style="font-size:1rem; color:#173b68;">Cụm 3: {CLUSTER_NAMES[3][0]}</b>
            <p style="margin: 6px 0; color:#536985; font-size:0.88rem;">{CLUSTER_NAMES[3][1]}</p>
            <div style="font-size:0.8rem; color:#657790; margin-top:8px;">
                <b>Chiến lược cấp vốn:</b> Nhóm chịu gánh nặng nợ lớn và tỷ lệ vỡ nợ cao; bắt buộc tài sản thế chấp thanh khoản cao hoặc từ chối giải ngân.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="aljazeera-eyebrow-lockup" style="margin-top:1.4rem;">
        <span class="eyebrow-accent-bar eyebrow-accent-bar-gold"></span>
        <span class="eyebrow-text">2D FINANCIAL EXPLORATION SCATTER MAP</span>
    </div>
    <div class="section-headline">Bản đồ phân tán tài chính 2D (Thu nhập vs Khoản vay)</div>
    <div class="section-description">Định vị hồ sơ hiện tại (biểu tượng kim cương vàng) trong tương quan 400 mẫu phân khúc khách hàng.</div>
    """, unsafe_allow_html=True)

    curr_res = st.session_state.get("last_result")
    scatter_chart = build_cluster_scatter_chart(sample_cluster_df, curr_res)
    if scatter_chart:
        st.altair_chart(scatter_chart, use_container_width=True)


# -----------------------------------------------------------------------------
# TRANG 3: LỊCH SỬ THẨM ĐỊNH (SECTION 08)
# -----------------------------------------------------------------------------
elif page == "📋 Lịch sử thẩm định":
    st.markdown("""
    <div class="viettrip-hero">
        <div class="hero-tag">SYSTEM AUDIT TRAIL</div>
        <h1>Lịch sử thẩm định & Lưu vết tín dụng</h1>
        <p>Theo dõi và tra cứu toàn bộ hồ sơ đã đánh giá được lưu trữ đồng bộ trên cơ sở dữ liệu PostgreSQL / Supabase đám mây.</p>
    </div>
    """, unsafe_allow_html=True)

    if db_ok:
        history, history_error = load_history(limit=500)
        if history is None:
            history = []
            st.error("Không đọc được lịch sử từ cơ sở dữ liệu: " + history_error)
    else:
        history = st.session_state.history
        st.info("Đang hiển thị lịch sử tạm thời trong phiên làm việc hiện tại.")

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("TỔNG HỒ SƠ ĐÃ LƯU", len(history))
    with k2:
        st.metric("DỰ BÁO TỪ MODEL THẬT", sum(1 for x in history if x.get("mode") == "model"))
    with k3:
        st.metric("DỰ BÁO MINH HỌA", sum(1 for x in history if x.get("mode") != "model"))

    if not history:
        st.info("Chưa có hồ sơ nào được đánh giá. Hãy thực hiện thẩm định tại trang Tổng quan & Thẩm định.")
    else:
        s1, s2 = st.columns([3, 1])
        with s1:
            q = st.text_input("🔍 Tìm kiếm theo thông tin hồ sơ (tuổi, thu nhập, số tiền vay, mục đích)...", placeholder="Nhập từ khóa tìm kiếm...", label_visibility="collapsed")
        with s2:
            gf = st.selectbox("Lọc phân hạng", ["Tất cả hạng", "Hạng A", "Hạng B", "Hạng C", "Hạng D"], label_visibility="collapsed")

        rows = []
        for i, item in enumerate(history, start=1):
            app = item.get("application", {})
            g = item.get("grade", "N/A")
            if gf != "Tất cả hạng" and f"Hạng {g}" != gf:
                continue

            row_data = {
                "STT": i,
                "Thời gian": item.get("created_at", "N/A"),
                "Tuổi": app.get("person_age"),
                "Thu nhập": f"${app.get('person_income', 0):,.0f}",
                "Khoản vay": f"${app.get('loan_amnt', 0):,.0f}",
                "Mục đích": app.get("loan_intent"),
                "Xác suất PD": item.get("pd_display", "N/A"),
                "Hạng tín dụng": g,
                "Phân khúc K-Means": item.get("cluster", "N/A"),
                "Chế độ": "Mô hình thật" if item.get("mode") == "model" else "Minh họa",
            }
            if q:
                row_str = " ".join(str(v).lower() for v in row_data.values())
                if q.lower() not in row_str:
                    continue
            rows.append(row_data)

        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, hide_index=True, use_container_width=True)
            st.download_button(
                "⬇️ Tải xuống dữ liệu CSV",
                df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"creditrisk_history_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                type="primary",
            )
        else:
            st.info("Không tìm thấy hồ sơ nào khớp với bộ lọc tra cứu.")


# -----------------------------------------------------------------------------
# TRANG 4: CẤU TRÚC HỆ THỐNG & ĐỒ ÁN (SECTION 10 PREVIEW)
# -----------------------------------------------------------------------------
elif page == "⚙️ Cấu trúc hệ thống":
    st.markdown("""
    <div class="viettrip-hero">
        <div class="hero-tag">SYSTEM SPECIFICATION & ARCHITECTURE</div>
        <h1>Cấu trúc hệ thống & Tài liệu kỹ thuật</h1>
        <p>Tổng quan kiến trúc giải pháp Machine Learning, quy trình tiền xử lý, tiêu chuẩn phân hạng rủi ro tín dụng theo đồ án nghiên cứu Nhóm 14.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="aljazeera-eyebrow-lockup">
        <span class="eyebrow-accent-bar"></span>
        <span class="eyebrow-text">RESEARCH FOUNDATION & SPECIFICATIONS</span>
    </div>
    <div class="section-headline">Cơ sở lý thuyết & Báo cáo kỹ thuật Nhóm 14</div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="scorecard-panel">
            <b style="color:#173b68; font-size:1rem;">🎯 Mục tiêu bài toán</b>
            <p style="margin:6px 0 12px; color:#4a5e78; font-size:0.88rem;">
                Dự báo nguy cơ vỡ nợ <code>loan_status</code> (0: Không vỡ nợ; 1: Vỡ nợ thực tế) dựa trên 9 biến định lượng và phân loại của khách hàng.
            </p>
            <b style="color:#173b68; font-size:1rem;">🌲 Mô hình máy học tối ưu</b>
            <p style="margin:6px 0 12px; color:#4a5e78; font-size:0.88rem;">
                Random Forest Classifier với 100 cây quyết định, đạt hiệu năng vượt trội ROC-AUC 92.10% và độ nhạy Recall 73.91%.
            </p>
            <b style="color:#173b68; font-size:1rem;">🏅 Ngưỡng phân hạng tín dụng (PD Cutoffs)</b>
            <p style="margin:6px 0 0; color:#4a5e78; font-size:0.88rem;">
                Hạng A (≤ 7.65%), Hạng B (7.65% – 17.61%), Hạng C (17.61% – 40.86%), Hạng D (> 40.86%). Các ngưỡng được trích xuất từ phân vị Q1, Q2, Q3 của tập dữ liệu huấn luyện.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="scorecard-panel">
            <b style="color:#173b68; font-size:1rem;">🔍 Giải thích mô hình (Explainable AI - XAI)</b>
            <p style="margin:6px 0 12px; color:#4a5e78; font-size:0.88rem;">
                Tích hợp SHAP TreeExplainer nhằm tính toán giá trị Shapley cho từng đặc trưng, làm rõ nguyên nhân AI tăng hoặc giảm điểm rủi ro.
            </p>
            <b style="color:#173b68; font-size:1rem;">👥 Phân khúc hành vi (K-Means Clustering)</b>
            <p style="margin:6px 0 12px; color:#4a5e78; font-size:0.88rem;">
                Phân loại khách hàng thành 4 cụm dựa trên các biến tài chính chuẩn hóa bằng StandardScaler, hỗ trợ chính sách cấp tín dụng tùy biến.
            </p>
            <b style="color:#173b68; font-size:1rem;">☁️ Hạ tầng & Cơ sở dữ liệu</b>
            <p style="margin:6px 0 0; color:#4a5e78; font-size:0.88rem;">
                Lưu trữ lịch sử thẩm định qua PostgreSQL/Supabase, giao diện web Streamlit cao cấp và trợ lý đa lượt hội thoại tích hợp Gemini 2.5/Flash.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="aljazeera-eyebrow-lockup" style="margin-top:1.3rem;">
        <span class="eyebrow-accent-bar"></span>
        <span class="eyebrow-text">RUNTIME ARTIFACT AUDIT</span>
    </div>
    <div class="section-headline">Trạng thái các artifact trong dự án</div>
    """, unsafe_allow_html=True)

    state_rows = [
        {"Thành phần": "Random Forest Pipeline Bundle", "Đường dẫn": status["pipeline_path"], "Trạng thái": "✅ Đã tìm thấy" if status["ready"] else "❌ Chưa tìm thấy"},
        {"Thành phần": "K-Means Cluster Model", "Đường dẫn": status["cluster_path"], "Trạng thái": "✅ Đã tìm thấy" if status["cluster_ready"] else "❌ Chưa tìm thấy"},
        {"Thành phần": "SHAP TreeExplainer Engine", "Đường dẫn": "Tích hợp sẵn trong model bundle", "Trạng thái": "✅ Sẵn sàng" if status["shap_available"] else "❌ Chưa khả dụng"},
        {"Thành phần": "Ngưỡng phân hạng Q1, Q2, Q3", "Đường dẫn": status["thresholds_path"], "Trạng thái": "✅ Khớp chuẩn bundle" if status["ready"] else "❌ Chưa tìm thấy"},
    ]
    st.dataframe(pd.DataFrame(state_rows), hide_index=True, use_container_width=True)


# -----------------------------------------------------------------------------
# SECTION 09: FLOATING GEMINI AI ASSISTANT CHATBOT
# -----------------------------------------------------------------------------
with st.container(key="ai_floating_button"):
    if st.button("✦", key="toggle_ai_chat", help="Mở / Thu gọn Trợ lý AI"):
        st.session_state.ai_chat_open = not st.session_state.ai_chat_open
        st.rerun()

if st.session_state.ai_chat_open:
    with st.container(key="ai_chat_panel", border=True):
        head1, head2 = st.columns([5, 1])
        with head1:
            st.markdown("""
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="color:#5266eb; font-size:1.15rem; font-weight:800;">✦</span>
                <span style="font-weight:750; color:#173b68; font-size:1.02rem;">CreditRisk AI Assistant</span>
            </div>
            """, unsafe_allow_html=True)
            st.caption("Gemini 2.5 · Tự động định tuyến mô hình · Phân tích hồ sơ thời gian thực")
        with head2:
            if st.button("—", key="minimize_ai_chat", help="Thu nhỏ"):
                st.session_state.ai_chat_open = False
                st.rerun()

        # Quick Prompt Chips
        st.markdown('<span style="font-size:0.75rem; color:#5f7594; font-weight:700;">GỢI Ý CÂU HỎI THẨM ĐỊNH:</span>', unsafe_allow_html=True)
        qc1, qc2 = st.columns(2)
        with qc1:
            if st.button("🧭 Giải thích SHAP", key="qc_shap", use_container_width=True):
                st.session_state.ai_chat_injected = "Hãy phân tích chi tiết các yếu tố SHAP của hồ sơ vừa thẩm định và đặc trưng nào tác động mạnh nhất đến xác suất rủi ro."
                st.rerun()
        with qc2:
            if st.button("💡 Khuyến nghị tín dụng", key="qc_rec", use_container_width=True):
                st.session_state.ai_chat_injected = "Dựa trên kết quả xác suất vỡ nợ (PD) và xếp hạng tín dụng hiện tại, hãy đề xuất các biện pháp kiểm soát rủi ro cụ thể."
                st.rerun()

        if not st.session_state.ai_chat_messages:
            st.info("Xin chào! Tôi là Trợ lý AI chuyên trách phân tích rủi ro tín dụng. Bạn có thể hỏi tôi về kết quả PD, phân hạng A-D hoặc giải mã các yếu tố SHAP của hồ sơ vừa thẩm định.")

        for message in st.session_state.ai_chat_messages[-10:]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        injected_text = st.session_state.pop("ai_chat_injected", "")
        with st.form("ai_chat_form", clear_on_submit=True):
            user_question = st.text_input("Câu hỏi", value=injected_text, placeholder="Ví dụ: Vì sao hồ sơ này được xếp hạng rủi ro cao?", label_visibility="collapsed", max_chars=1200)
            send_chat = st.form_submit_button("Gửi câu hỏi", type="primary", use_container_width=True)

        if send_chat and user_question.strip():
            current_result = st.session_state.get("last_result")
            clean_question = user_question.strip()
            previous_conversation = list(st.session_state.ai_chat_messages)
            st.session_state.ai_chat_messages.append({"role": "user", "content": clean_question})

            with st.chat_message("user"):
                st.markdown(clean_question)

            with st.chat_message("assistant"):
                answer = st.write_stream(
                    stream_gemini_answer(clean_question, current_result, previous_conversation)
                )

            if not isinstance(answer, str):
                answer = str(answer or "")
            st.session_state.ai_chat_messages.append({"role": "assistant", "content": answer})

            metrics = st.session_state.get("ai_last_metrics", {})
            if metrics:
                first_token = metrics.get("first_token_seconds")
                first_label = f"{first_token:.1f}s" if isinstance(first_token, (int, float)) else "sẵn sàng"
                st.caption(
                    f"{metrics.get('model', 'Gemini')} · token đầu: {first_label} · "
                    f"thời gian: {metrics.get('total_seconds', 0):.1f}s · "
                    f"suy luận: {metrics.get('thinking_level', 'low')}"
                )
            st.rerun()


# -----------------------------------------------------------------------------
# SECTION 10: ABOUT & ENTERPRISE FINTECH FOOTER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="enterprise-footer">
    <div class="footer-grid">
        <div class="footer-col">
            <h4>CREDITRISK AI PLATFORM</h4>
            <p>
                Nền tảng đánh giá rủi ro tín dụng thông minh, kết hợp mô hình Random Forest, 
                giải thích minh bạch SHAP TreeExplainer, phân cụm tài chính K-Means 
                và cơ sở dữ liệu phân tán PostgreSQL/Supabase.
            </p>
        </div>
        <div class="footer-col">
            <h4>KIẾN TRÚC KỸ THUẬT</h4>
            <p>
                • Scikit-Learn Random Forest (100 Trees)<br>
                • SHAP (Shapley Additive Explanations)<br>
                • K-Means Financial Clustering (4 cụm)<br>
                • Google Gemini Multiturn API
            </p>
        </div>
        <div class="footer-col">
            <h4>ĐỒ ÁN HỌC THUẬT</h4>
            <p>
                Được nghiên cứu và phát triển bởi <b>Nhóm 14</b>.<br>
                Bản quyền giao diện kết hợp phong cách VietTrip.online và Hệ thống thiết kế GetDesign Al Jazeera.
            </p>
        </div>
    </div>
    <div class="footer-bottom">
        <span>© 2026 CreditRisk AI · Nhóm 14. All rights reserved.</span>
        <span>Hệ thống hỗ trợ ra quyết định (Decision Support System) — Tuân thủ nguyên tắc AI có đạo đức.</span>
    </div>
</div>
""", unsafe_allow_html=True)
