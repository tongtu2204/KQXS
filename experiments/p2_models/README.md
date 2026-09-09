# Phần 2 — Mô hình dự đoán pool và từng giải

Phần 2 được tách thành hai bài toán độc lập nhưng dùng chung dữ liệu đầy đủ
27 kết quả mỗi ngày trong `data/raw/kqxsmb_all_prizes_2007_2026.csv`.

## P2A — dự đoán pool chữ số của cả ngày

## Đơn vị đánh giá

Mỗi ngày có 27 kết quả từ toàn bộ các nhóm giải. Với mỗi vị trí trong số có
5 chữ số, target là vector nhị phân 10 chiều:

```text
target[position, digit] = 1
```

nếu chữ số đó xuất hiện ít nhất một lần ở vị trí tương ứng trong bất kỳ kết
quả nào của ngày. Các giải ngắn được căn phải: giải 4 chữ số chỉ đóng góp vào
4 vị trí cuối, không tạo chữ số `0` giả ở đầu. File chuẩn bị dữ liệu cũng lưu
`eligible_count`, số lần xuất hiện của từng chữ số và toàn bộ `pool_numbers`
để P3 sinh tổ hợp.

## Mô hình

- `uniform_27_iid`: baseline Uniform ở cấp độ ngày, dùng
  `1 - (1 - 0,1)^27` cho xác suất xuất hiện ít nhất một lần trong 27 kết quả;
- `expanding_frequency`: tần suất xuất hiện tích lũy theo ngày;
- `rolling_frequency_w30`, `w90`, `w365`: tần suất trong cửa sổ gần nhất;
- `markov_presence`: xác suất xuất hiện hôm nay phụ thuộc trạng thái xuất hiện
  hoặc không xuất hiện của ngày trước;
- `random_forest`: học từ các vector target trễ 1, 2, 3, 7, 14 và 30 ngày.

## P2B — dự đoán đích danh từng giải

Mỗi target là một slot cụ thể `prize + prize_index` (ví dụ `Giải ba_4`),
không chỉ là giải đặc biệt. Mô hình dự đoán phân phối chữ số theo vị trí của
đúng slot đó, sinh danh sách Top-k số và chấm hit theo đúng độ dài của giải.
Không dùng ký tự phụ đặc biệt; các pool/độ dài được lấy trực tiếp từ dữ liệu
full-prize.

Protocol của Phần 2:

- lịch sử huấn luyện: từ 2007 đến hết 2022;
- validation: 2023–2024;
- 2025–2026 không dùng trong Phần 2, được khóa riêng để đánh giá chiến lược
  thực tế ở Phần 3.

## Chạy lại

```bash
python experiments/p2_models/00_prepare_daily_targets.py
python experiments/p2_models/01_daily_digit_models.py
python experiments/p2_models/02_daily_model_evaluation.py
python experiments/p2_models/03_boosted_daily_models.py
python experiments/p2_models/04_prize_target_models.py
python experiments/p2_models/05_boosted_prize_target_models.py
```

Kết quả nằm tại:

```text
data/processed/daily_digit_targets.csv
artifacts/p2_models/model_summary.csv
artifacts/p2_models/daily_scores.csv.gz
artifacts/p2_models/actual_digit_ranks.csv.gz
artifacts/p2_models/rank_summary.csv
artifacts/p2_models/calibration.csv
artifacts/p2_models/boosted/boosted_summary.csv
artifacts/p2_models/boosted/boosted_calibration.csv
artifacts/p2_models/figures/
artifacts/p2_models/prize_target/prize_target_predictions.csv.gz
artifacts/p2_models/prize_target/prize_target_summary.csv
artifacts/p2_models/prize_target_boosted/boosted_prize_target_predictions.csv.gz
artifacts/p2_models/prize_target_boosted/boosted_prize_target_summary.csv
```


## Model boosting

- `03_boosted_daily_models.py`: XGBoost và CatBoost, mỗi model gồm 50 classifier nhị phân cho 5 vị trí × 10 chữ số.
- Cài dependency bằng `python -m pip install -r requirements.txt`.
- Kết quả được đánh giá cùng protocol và metric của `02_daily_model_evaluation.py`.
