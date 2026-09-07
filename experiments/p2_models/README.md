# Phần 2 — Mô hình dự đoán chữ số theo ngày

## Đơn vị đánh giá

Mỗi ngày có 27 kết quả từ toàn bộ các nhóm giải. Với mỗi vị trí trong số có
5 chữ số, target là vector nhị phân 10 chiều:

```text
target[position, digit] = 1
```

nếu chữ số đó xuất hiện ít nhất một lần ở vị trí tương ứng trong bất kỳ kết
quả nào của ngày. Vì vậy, ví dụ dự đoán chữ số `3` ở hàng đơn vị được tính
đúng nếu `3` xuất hiện ở hàng đơn vị của ít nhất một trong 27 kết quả.

## Mô hình

- `uniform_27_iid`: baseline Uniform ở cấp độ ngày, dùng
  `1 - (1 - 0,1)^27` cho xác suất xuất hiện ít nhất một lần trong 27 kết quả;
- `expanding_frequency`: tần suất xuất hiện tích lũy theo ngày;
- `rolling_frequency_w30`, `w90`, `w365`: tần suất trong cửa sổ gần nhất;
- `markov_presence`: xác suất xuất hiện hôm nay phụ thuộc trạng thái xuất hiện
  hoặc không xuất hiện của ngày trước;
- `random_forest`: học từ các vector target trễ 1, 2, 3, 7, 14 và 30 ngày.

Các mô hình thống kê được khởi tạo bằng dữ liệu trước giai đoạn test. Mỗi
fold tạo dự báo từ lịch sử trước ngày bắt đầu; final test 2025–2026 không được
dùng để lựa chọn mô hình.

## Chạy lại

```bash
python experiments/p2_models/00_prepare_daily_targets.py
python experiments/p2_models/01_daily_digit_models.py
```

Kết quả nằm tại:

```text
data/processed/daily_digit_targets.csv
artifacts/models_daily/model_daily_probabilities.csv.gz
artifacts/tables_daily/model_summary.csv
artifacts/tables_daily/model_daily_scores.csv.gz
artifacts/figures_daily/model_brier_final.png
```

