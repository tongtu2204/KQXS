# Kết quả Phần 2 — mô hình chữ số theo ngày

## Protocol

- Target: 5 vị trí × 10 chữ số nhị phân; chữ số được tính là xuất hiện nếu có
  trong ít nhất một của 27 kết quả trong ngày.
- Dữ liệu phát triển/huấn luyện: 2007–2022.
- Validation để so sánh và chọn mô hình: 2023–2024 (723 ngày).
- 2025–2026 không được dùng trong Phần 2; dành riêng cho đánh giá chiến lược ở Phần 3.

## Brier score trên validation 2023–2024

| Model | Brier score | Log loss |
|---|---:|---:|
| Expanding Beta | 0.097820 | 0.329968 |
| Markov presence | 0.097853 | 0.330118 |
| Rolling Beta 365 | 0.097973 | 0.330935 |
| CatBoost | 0.098313 | 0.332414 |
| Random Forest | 0.098615 | 0.333515 |
| XGBoost | 0.099221 | 0.336660 |
| Rolling Beta 90 | 0.098779 | 0.335109 |
| Rolling Beta 30 | 0.100920 | 0.343546 |
| Dirichlet daily | 0.112701 | 0.397621 |
| Uniform 27-IID | 0.113875 | 0.408108 |

Expanding Beta là mô hình tốt nhất theo Brier/log loss; CatBoost là mô hình
machine-learning tốt nhất nhưng chưa vượt nhóm Bayesian/Markov. Top-k không
được dùng làm tiêu chí duy nhất vì target theo ngày có độ phủ rất cao.

Toàn bộ bảng chi tiết và hình được lưu dưới `artifacts/p2_models/`.
