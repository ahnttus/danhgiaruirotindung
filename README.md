# CreditRisk AI — Website Streamlit (Nhóm 14)

Prototype website kiểm tra hồ sơ tín dụng dựa theo báo cáo PDF của Nhóm 14.

## Có gì trong project?
- Form nhập thông tin hồ sơ vay.
- Khu vực hiển thị PD, hạng A–D, phân khúc K-Means và yếu tố SHAP khi có artifact tương thích.
- Lịch sử hồ sơ trong phiên Streamlit, có thể tải CSV.
- Không giả lập dự báo khi thiếu mô hình: app hiển thị rõ trạng thái chưa tích hợp.

## 1. Cài đặt
Khuyến nghị Python 3.10 hoặc 3.11.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Cài thư viện:
```bash
pip install -r requirements.txt
```

## 2. Chạy website
Tại thư mục `credit_risk_streamlit`:
```bash
streamlit run app.py
```
Trình duyệt sẽ mở địa chỉ local được Streamlit hiển thị (thường là `http://localhost:8501`).

## 3. Tích hợp code Nhóm 14 để tạo mô hình thật

Đã đối chiếu notebook `MACHINE LEARNING NHÓM 14.ipynb`. Project có script xuất mô hình theo các bước chính trong notebook:
- Chia train/test 80/20, `random_state=42`, phân tầng theo `loan_status`.
- Loại các cột `loan_grade`, `cb_person_cred_hist_length`, `age_group`, `emp_group` nếu có.
- One-Hot Encoding cho `person_home_ownership`, `loan_intent`, `cb_person_default_on_file`.
- SMOTENC chỉ trên tập huấn luyện; StandardScaler.
- Random Forest: `n_estimators=200`, `max_depth=16`, `min_samples_leaf=10`, `random_state=42`.
- Tính ngưỡng A/B/C/D từ các phân vị 25%, 50%, 75% của PD trên tập test, theo logic trong notebook.
- K-Means 4 cụm, dùng 6 đặc trưng tài chính và StandardScaler riêng.

### Bước 1 — chuẩn bị dữ liệu
Trong notebook nhóm, Cell 12 xuất `credit_risk_cleaned.csv`. Hãy sao chép file đó vào thư mục gốc `credit_risk_streamlit`, cùng cấp với `app.py`.

### Bước 2 — tạo artifact mô hình
Chạy ở thư mục project:

```bash
python scripts/train_export.py
```

Lệnh tạo `models/group14_model_bundle.joblib`. Đây là bước huấn luyện lại từ dữ liệu sạch; file notebook tự nó không chứa đủ dữ liệu để website gọi trực tiếp mô hình đã fit.

### Bước 3 — khởi động website

```bash
streamlit run app.py
```

Nếu chưa có `credit_risk_cleaned.csv` hoặc chưa chạy script export, website vẫn mở giao diện nhưng sẽ không giả lập PD. Nếu gặp lỗi phiên bản Python/thư viện, ưu tiên Python 3.10 hoặc 3.11 và cài lại `pip install -r requirements.txt`.

### Lưu ý về tính nhất quán
- Script huấn luyện lại mô hình từ CSV sạch theo các thiết lập chính trong notebook; kết quả có thể không trùng tuyệt đối nếu CSV, phiên bản thư viện hoặc trạng thái notebook khác.
- Notebook định nghĩa tên cụm K-Means thủ công theo ID 0–3. Nhãn cụm có thể hoán đổi sau khi fit lại; cần kiểm tra bảng `cluster_profile`/tỷ lệ vỡ nợ trong notebook trước khi dùng tên cụm như một kết luận nghiệp vụ.
- SHAP được tạo khi người dùng yêu cầu dự báo; nếu phiên bản SHAP không tương thích, phần PD/hạng/cụm vẫn trả về và phần SHAP sẽ báo không khả dụng.
- Các tỷ lệ vỡ nợ theo hạng A–D (1.23%, 5.00%, 12.41%, 68.85%) là các giá trị được khai báo trong notebook, không được tính lại tự động từ tập dữ liệu ở script này.

## 5. Lưu ý
- Lịch sử hiện lưu trong `st.session_state`, không phải database; sẽ mất khi phiên chạy kết thúc.
- Không nhập thông tin định danh cá nhân vào bản demo.
- Website là công cụ học tập/hỗ trợ đánh giá, không phải quyết định cấp tín dụng tự động.
