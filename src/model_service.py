"""Inference layer matching the Group 14 notebook preprocessing and model choices."""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
BUNDLE_PATH = MODEL_DIR / "group14_model_bundle.joblib"

FEATURES = [
    "person_age", "person_income", "person_home_ownership",
    "person_emp_length", "loan_intent", "loan_amnt", "loan_int_rate",
    "loan_percent_income", "cb_person_default_on_file"
]
CLUSTER_FEATURES = [
    "person_age", "person_income", "person_emp_length",
    "loan_amnt", "loan_int_rate", "loan_percent_income"
]
CLUSTER_NAMES = {
    0: ("Nhóm thu nhập thấp, kinh nghiệm hạn chế", "Thu nhập thấp hơn mức trung bình và kinh nghiệm làm việc còn hạn chế."),
    1: ("Nhóm khoản vay nhỏ, rủi ro thấp", "Khoản vay nhỏ và tỷ lệ vỡ nợ thấp nhất."),
    2: ("Nhóm thu nhập cao, ổn định", "Thu nhập cao và kinh nghiệm làm việc dài."),
    3: ("Nhóm áp lực vay cao", "Khoản vay lớn và tỷ lệ vỡ nợ cao.")
}
DEFAULT_RATE_MAP = {"A": "1.23%", "B": "5.00%", "C": "12.41%", "D": "68.85%"}
RECOMMENDATIONS = {
    "A": "Rủi ro rất thấp. Có thể ưu tiên xử lý hồ sơ.",
    "B": "Rủi ro thấp. Thực hiện quy trình thẩm định thông thường.",
    "C": "Rủi ro trung bình. Cần xem xét thêm các thông tin tài chính.",
    "D": "Rủi ro cao. Cần thẩm định chuyên sâu và xác minh bổ sung."
}
FEATURE_LABELS = {
    "person_age": "Tuổi khách hàng", "person_income": "Thu nhập khách hàng",
    "person_emp_length": "Kinh nghiệm làm việc", "loan_amnt": "Giá trị khoản vay",
    "loan_int_rate": "Lãi suất khoản vay", "loan_percent_income": "Tỷ lệ khoản vay trên thu nhập",
    "person_home_ownership_RENT": "Khách hàng thuộc nhóm thuê nhà",
    "person_home_ownership_OWN": "Khách hàng sở hữu nhà ở",
    "cb_person_default_on_file_Y": "Khách hàng từng có nợ xấu",
    "loan_intent_PERSONAL": "Mục đích vay cá nhân",
    "loan_intent_EDUCATION": "Mục đích vay giáo dục",
    "loan_intent_MEDICAL": "Mục đích vay y tế",
    "loan_intent_VENTURE": "Mục đích vay kinh doanh",
}


def model_status():
    return {
        "pipeline_path": str(BUNDLE_PATH.relative_to(ROOT)),
        "ready": BUNDLE_PATH.exists(),
        "cluster_path": str(BUNDLE_PATH.relative_to(ROOT)),
        "cluster_ready": BUNDLE_PATH.exists(),
        "thresholds_path": str(BUNDLE_PATH.relative_to(ROOT)),
        "shap_available": BUNDLE_PATH.exists(),
    }


def _load_bundle():
    if not BUNDLE_PATH.exists():
        return None
    return joblib.load(BUNDLE_PATH)


def _grade(pd_value, thresholds):
    if pd_value <= thresholds["q1"]:
        return "A"
    if pd_value <= thresholds["q2"]:
        return "B"
    if pd_value <= thresholds["q3"]:
        return "C"
    return "D"


def evaluate_application(record):
    bundle = _load_bundle()
    if bundle is None:
        return {
            "mode": "illustration", "pd_display": "Chưa có mô hình",
            "grade": "Chưa xác định", "cluster": "Chưa có mô hình",
            "summary": "Chưa có mô hình được huấn luyện. Hãy chạy scripts/train_export.py với file credit_risk_cleaned.csv theo hướng dẫn README.",
            "factors": [],
        }

    raw = pd.DataFrame([{k: record[k] for k in FEATURES}])
    categorical = ["person_home_ownership", "loan_intent", "cb_person_default_on_file"]
    numeric = raw.drop(columns=categorical).reset_index(drop=True)
    encoded = bundle["encoder"].transform(raw[categorical])
    encoded_df = pd.DataFrame(encoded, columns=bundle["encoder"].get_feature_names_out(categorical))
    model_frame = pd.concat([numeric, encoded_df], axis=1)
    model_frame = model_frame.reindex(columns=bundle["feature_columns"], fill_value=0)
    scaled = pd.DataFrame(bundle["rf_scaler"].transform(model_frame), columns=model_frame.columns)

    model = bundle["rf"]
    probs = model.predict_proba(scaled)[0]
    classes = list(model.classes_)
    pd_value = float(probs[classes.index(1)])
    grade = _grade(pd_value, bundle["thresholds"])
    risk_level = {"A": "Rất thấp", "B": "Thấp", "C": "Trung bình", "D": "Cao"}[grade]

    cluster_input = raw[CLUSTER_FEATURES]
    cluster_scaled = bundle["cluster_scaler"].transform(cluster_input)
    cluster_id = int(bundle["kmeans"].predict(cluster_scaled)[0])
    cluster_name, cluster_description = CLUSTER_NAMES.get(cluster_id, (f"Cluster {cluster_id}", "Chưa có mô tả cho cụm này."))

    factors = []
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(scaled)
        if isinstance(shap_values, list):
            values = np.asarray(shap_values[1])[0]
        else:
            arr = np.asarray(shap_values)
            values = arr[0, :, 1] if arr.ndim == 3 else arr[0]
        pairs = sorted(zip(model_frame.columns, values), key=lambda x: abs(float(x[1])), reverse=True)
        for feature, value in pairs:
            if feature.startswith(("loan_intent_", "person_home_ownership_", "cb_person_default_on_file_")) and feature in model_frame.columns and model_frame.iloc[0][feature] != 1:
                continue
            factors.append({
                "Yếu tố": FEATURE_LABELS.get(feature, feature),
                "Giá trị SHAP": round(float(value), 5),
                "Hướng tác động": "Tăng rủi ro dự báo" if value > 0 else "Giảm rủi ro dự báo"
            })
            if len(factors) >= 6:
                break
    except Exception as exc:
        factors = [{"Yếu tố": "SHAP", "Giá trị SHAP": "Không khả dụng", "Hướng tác động": f"Lỗi tương thích: {type(exc).__name__}"}]

    return {
        "mode": "model", "pd_display": f"{pd_value:.2%}", "pd_value": pd_value,
        "grade": grade, "risk_level": risk_level,
        "default_rate": DEFAULT_RATE_MAP.get(grade, "Không có dữ liệu"),
        "cluster": f"Cụm {cluster_id}: {cluster_name}", "cluster_id": cluster_id,
        "cluster_description": cluster_description,
        "summary": f"Mô hình Random Forest ước tính PD = {pd_value:.2%}. Hạng {grade} — mức rủi ro {risk_level}. {RECOMMENDATIONS[grade]} Đây là thông tin hỗ trợ thẩm định, không phải quyết định tự động.",
        "recommendation": RECOMMENDATIONS[grade], "factors": factors,
    }
