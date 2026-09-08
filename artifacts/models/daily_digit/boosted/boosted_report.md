# XGBoost/CatBoost — kết quả

| Model | Fold | Brier | Log loss |
| --- | --- | ---: | ---: |
| XGBoost | Validation 2023–2024 | 0,099221 | 0,336660 |
| CatBoost | Validation 2023–2024 | 0,098313 | 0,332414 |
| XGBoost | Final 2025–2026 | 0,097205 | 0,330035 |
| CatBoost | Final 2025–2026 | 0,096509 | 0,326312 |

CatBoost tốt hơn XGBoost ở cả hai fold. Tuy nhiên CatBoost vẫn kém
`expanding_beta` (Brier final 0,096307; log loss final 0,325100), nên chưa phải
model tốt nhất. XGBoost không vượt các model thống kê.

