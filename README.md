# KQXS — kiểm định ngẫu nhiên và backtest xác suất

Project nghiên cứu hai số cuối giải đặc biệt miền Bắc theo protocol thời gian cố định, walk-forward và không sử dụng dữ liệu tương lai.

## Protocol chính thức

| Giai đoạn | Thời gian | Số kỳ hiện có | Vai trò |
| --- | --- | ---: | --- |
| Train/development | 2002–2022 | 7.561 | Xây dựng feature và khởi tạo model |
| Validation | 2023–2024 | 723 | Chọn model và chiến lược |
| Final test | 2025–2026 | 595 | Đánh giá khóa, hiện đến 23/08/2026 |

Quy tắc chống leakage:

1. Prediction ngày `t` chỉ sử dụng dữ liệu đến trước ngày `t`.
2. Kết quả ngày `t` chỉ được cập nhật vào model sau khi prediction đã được tạo.
3. Model và chiến lược được chọn trên 2023–2024.
4. Cấu hình được lưu vào `frozen_strategy.json` và commit trước khi chạy 2025–2026.
5. Không chọn lại model, `m`, participation rate hoặc threshold theo final test.

Nhánh `backup/pre-refactor-split-20260828` lưu toàn bộ trạng thái cũ trước refactor.

## Model

- Uniform baseline.
- CatBoost static.
- CatBoost periodic retrain mỗi 30 ngày.
- CDM/Dirichlet–Multinomial.
- Rolling CDM với cửa sổ 30, 60, 90, 180 và 365 kỳ.
- Bayesian Markov.

Các model stateful cập nhật sau từng kết quả. CatBoost periodic chỉ retrain trên history kết thúc trước block dự đoán.

## Kết quả chính

Validation đã xét 1.250 cấu hình từ 10 model và khóa hai chiến lược:

| Chiến lược | Cấu hình khóa | Validation ROI | Final ROI | Final profit |
| --- | --- | ---: | ---: | ---: |
| Always Top-m | CDM, `m=1` | −0,41% | +34,45% | +2.050.000 VND |
| Selective Top-m | Bayesian Markov, `m=4`, target 20% | +79,31% | −14,29% | −800.000 VND |

Always Top‑1 CDM có lợi nhuận dương trong 595 kỳ final nhưng Wilson lower vẫn thấp hơn mức hòa vốn. Vì vậy chưa có bằng chứng đủ mạnh về edge bền vững. Chiến lược selective tốt nhất trên validation không tái lập trên final test.

Xem báo cáo đầy đủ tại:

```text
artifacts/refactor/reports/final_research_summary.md
```

## Artifact đã lưu trên Git

```text
artifacts/refactor/
├── validation_2023_2024/
│   ├── frozen_strategy.json
│   ├── probability_metrics.csv
│   ├── model_probabilities.csv.gz
│   ├── selected_strategies.csv
│   ├── selected_strategy_daily.csv.gz
│   └── strategy_candidates.csv.gz
├── final_test_2025_2026/
│   ├── frozen_config_used.json
│   ├── probability_metrics.csv
│   ├── model_probabilities.csv.gz
│   ├── strategy_results.csv
│   └── strategy_daily.csv.gz
├── figures/
│   ├── model_log_loss_validation_vs_final.png
│   └── cumulative_profit_validation_vs_final.png
└── reports/
    ├── model_output_validation.json
    ├── strategy_selection_validation.json
    ├── final_result_validation.json
    └── final_research_summary.md
```

Chỉ cần `git pull origin main` để xem toàn bộ kết quả trên. Không cần chạy lại model nếu chỉ đọc báo cáo, CSV và biểu đồ.

## Cài đặt

```bash
python -m pip install -r requirements.txt
```

Chuẩn hóa và kiểm tra dữ liệu:

```bash
python scripts/validate_protocol.py
python -m unittest discover -s tests
```

## Tái chạy toàn bộ

### 1. Chạy model

```bash
python experiments/p2_models/09_modern_ml_last2.py
python experiments/p2_models/09b_catboost_periodic_retrain.py
python experiments/p2_models/10_cdm_baseline.py
python experiments/p2_models/11_rolling_cdm.py
python experiments/p2_models/12_bayesian_markov.py
python experiments/p2_models/13_probability_ranking.py
python scripts/validate_model_outputs.py
```

CatBoost periodic retrain chạy CPU mất khoảng 17 phút trên môi trường kiểm thử hiện tại.

### 2. Chọn và khóa chiến lược trên validation

```bash
python experiments/p3_strategies/21_fixed_split_strategy.py --stage select
python scripts/validate_strategy_selection.py
```

Sau bước này cần review và commit `artifacts/refactor/validation_2023_2024/frozen_strategy.json` trước khi chạy final.

### 3. Chạy final test bằng cấu hình đã khóa

```bash
python experiments/p3_strategies/21_fixed_split_strategy.py --stage final
python scripts/export_model_probabilities.py
python experiments/p3_strategies/22_final_split_report.py
python scripts/validate_final_results.py
```

## Cập nhật dữ liệu mới

Raw data được lưu tại `data/raw/kqxsmb_*.csv`. Sau khi cập nhật file raw:

```bash
python scripts/validate_protocol.py
```

Script sẽ tạo lại `data/processed/kqxsmb_digits.csv`. File processed và output trung gian trong `artifacts/tables/` không được commit; các deliverable chuẩn trong `artifacts/refactor/` được commit để có thể pull và xem trực tiếp.
