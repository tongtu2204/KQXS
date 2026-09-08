# XGBoost/CatBoost — kết quả

| Model | Fold | Brier | Log loss |
| --- | --- | ---: | ---: |
| XGBoost | Validation 2023–2024 | 0,099221 | 0,336660 |
| CatBoost | Validation 2023–2024 | 0,098313 | 0,332414 |

CatBoost tốt hơn XGBoost trên validation. Tuy nhiên CatBoost vẫn kém
`expanding_beta` (Brier validation 0,097820; log loss validation 0,329968), nên
chưa phải model tốt nhất. XGBoost không vượt các model thống kê. Final
2025–2026 không được dùng để đánh giá model trong Phần 2.

