# Model artifacts

Không có mô hình được tạo sẵn trong project này vì hiện chỉ có báo cáo PDF, chưa có notebook hoặc file mô hình đã huấn luyện.

Để bật dự báo thực:
1. Từ notebook, xuất toàn bộ preprocessing + Random Forest thành một sklearn Pipeline đã fit.
2. Lưu thành `credit_risk_pipeline.joblib` trong thư mục này.
3. Lưu ngưỡng Q1/Q2/Q3 thực tế tính từ phân phối PD trên tập huấn luyện/validation vào `grade_thresholds.json`.
4. Nếu có K-Means pipeline tương thích với đúng đầu vào, lưu `kmeans_pipeline.joblib`.
5. Nếu có SHAP explainer tương thích, lưu `shap_explainer.joblib`.

Không chỉ lưu riêng RandomForest nếu notebook còn dùng OneHotEncoder/StandardScaler; cần đóng gói đúng preprocessing và thứ tự đặc trưng.
