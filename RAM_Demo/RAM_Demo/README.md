# RAM Price Demo — Streamlit

Ứng dụng một trang để chạy 5 mô hình có trong 5 notebook đã cung cấp:

1. Ridge Regression (mô hình hồi quy tuyến tính cải tiến ở D2, log-target)
2. Random Forest Regressor
3. Gradient Boosting Regressor
4. Support Vector Regression (SVR)
5. XGBoost Regressor (log-target)

## 1. Chuẩn bị dữ liệu

Các notebook được gửi chỉ chứa mã nguồn, chưa có các CSV dữ liệu. Đặt các CSV gốc vào thư mục `data/` với đúng tên:

- `ram_pricing_intelligence_2026.csv` — dùng cho Ridge, SVR, XGBoost và nguồn B của RF/GB.
- `ram-price-observations.csv` — nguồn A cho RF/GB.
- `costinflation-ddr5-ram-retail-prices-raw-2026-07-13-to-2026-08-10.csv` — nguồn C cho RF/GB.

Không đổi tên cột trong các CSV. Script sẽ báo lỗi nếu thiếu file/cột cần thiết.

## 2. Cài đặt

Khuyến nghị Python 3.10 hoặc 3.11 trong môi trường ảo:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## 3. Huấn luyện và xuất mô hình

Chạy từ thư mục `RAM_Demo`:

```bash
python train_models.py
```

Script lưu từng mô hình `.joblib` và metadata `.json` vào `models/`. Metadata ghi lại danh sách feature, giá trị mặc định và MAE/RMSE/R² tính trên test split theo cách chia dữ liệu của notebook tương ứng.

Lưu ý: hai mô hình RF/GB thực hiện RandomizedSearchCV như notebook nên có thể mất vài phút hoặc lâu hơn tùy máy. Các model artifacts chỉ được tạo khi CSV đầu vào có đủ và đúng schema.

## 4. Chạy ứng dụng

```bash
streamlit run app.py
```

Mở địa chỉ localhost mà Streamlit hiển thị trên terminal.

## 5. Lưu ý phương pháp luận

- RF và Gradient Boosting học `usd_per_gb`. Demo nhân kết quả với tổng dung lượng để hiển thị giá ước tính cho cả bộ RAM.
- Ridge Regression, SVR và XGBoost học `price` (USD/bộ).
- Ridge D2 sử dụng feature engineering từ `title` như trong notebook. Khi dự đoán trực tiếp, demo dựng các feature tương ứng từ form.
- XGBoost dùng schema đặc trưng động của notebook. Các trường listing không có trong form (nếu có trong CSV) được điền bằng median/mode của tập huấn luyện; do đó dự đoán XGBoost có điều kiện listing mặc định, không phải toàn bộ thông tin listing do người dùng nhập.
- MAE/RMSE của RF/GB có đơn vị USD/GB; của Ridge/SVR/XGBoost có đơn vị USD/bộ. Không nên xếp hạng MAE/RMSE giữa hai nhóm như thể cùng đơn vị. R² cũng được tính trên các tập test khác nhau nên chỉ nên báo cáo đúng bối cảnh notebook.
- Đây là dự đoán theo dữ liệu nghiên cứu, không phải giá thị trường trực tiếp hoặc cam kết giá bán.

## 6. Cấu trúc

```text
RAM_Demo/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── data/       # đặt 3 CSV gốc tại đây
└── models/     # tự sinh .joblib và .json sau khi train
```
