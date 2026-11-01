"""Train/export artifacts using the preprocessing and model settings in Group 14 notebook.
Run from project root after copying credit_risk_cleaned.csv into the project root.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTENC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "credit_risk_cleaned.csv"
OUT = ROOT / "models" / "group14_model_bundle.joblib"

if not DATA.exists():
    raise FileNotFoundError(
        f"Không tìm thấy {DATA}. Trong notebook nhóm, Cell 12 tạo file credit_risk_cleaned.csv; "
        "hãy sao chép file đó vào thư mục gốc project rồi chạy lại."
    )

df = pd.read_csv(DATA)
required = ["loan_status", "person_age", "person_income", "person_home_ownership", "person_emp_length", "loan_intent", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_default_on_file"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Dataset thiếu các cột cần thiết: {missing}")

# Match notebook: target loan_status; exclude target/leakage and selected unused fields.
X = df.drop(columns=["loan_status"])
y = df["loan_status"].astype(int)
removed = ["loan_grade", "cb_person_cred_hist_length", "age_group", "emp_group"]
X = X.drop(columns=removed, errors="ignore")
selected_features = ["person_age", "person_income", "person_home_ownership", "person_emp_length", "loan_intent", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_default_on_file"]
X = X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
categorical = ["person_home_ownership", "loan_intent", "cb_person_default_on_file"]
encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
train_encoded = encoder.fit_transform(X_train[categorical])
test_encoded = encoder.transform(X_test[categorical])
train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(categorical), index=X_train.index)
test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(categorical), index=X_test.index)
X_train_encoded = pd.concat([X_train.drop(columns=categorical), train_encoded_df], axis=1)
X_test_encoded = pd.concat([X_test.drop(columns=categorical), test_encoded_df], axis=1)
cat_cols = [c for c in X_train_encoded.columns if any(t in c for t in ["person_home_ownership", "loan_intent", "cb_person_default_on_file"])]
cat_idx = [X_train_encoded.columns.get_loc(c) for c in cat_cols]
smotenc = SMOTENC(categorical_features=cat_idx, random_state=42)
X_train_smote, y_train_smote = smotenc.fit_resample(X_train_encoded, y_train)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_smote), columns=X_train_smote.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_encoded), columns=X_test_encoded.columns)
rf = RandomForestClassifier(n_estimators=200, max_depth=16, min_samples_leaf=10, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train_smote)
# Match notebook rating method: quartiles of predicted PD on X_test.
test_pd = rf.predict_proba(X_test_scaled)[:, list(rf.classes_).index(1)]
thresholds = {"q1": float(pd.Series(test_pd).quantile(.25)), "q2": float(pd.Series(test_pd).quantile(.50)), "q3": float(pd.Series(test_pd).quantile(.75))}

cluster_features = ["person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income"]
cluster_scaler = StandardScaler()
cluster_scaled = cluster_scaler.fit_transform(df[cluster_features])
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(cluster_scaled)

bundle = {
    "rf": rf, "encoder": encoder, "rf_scaler": scaler,
    "feature_columns": list(X_train_encoded.columns),
    "thresholds": thresholds,
    "cluster_scaler": cluster_scaler, "kmeans": kmeans,
    "metadata": {"source": "MACHINE LEARNING NHÓM 14.ipynb", "n_estimators": 200, "max_depth": 16, "min_samples_leaf": 10}
}
OUT.parent.mkdir(exist_ok=True)
joblib.dump(bundle, OUT)
print(f"Đã lưu mô hình: {OUT}")
print("Ngưỡng xếp hạng PD:", json.dumps(thresholds, ensure_ascii=False, indent=2))
print("Lưu ý: K-Means có thể gán nhãn cụm khác nếu thứ tự cụm trong lần fit khác notebook; tên cụm cần được đối chiếu hồ sơ cụm trước khi trình bày.")
