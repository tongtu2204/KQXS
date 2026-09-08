# Báo cáo hoàn thiện Phần 2 — Model

## Protocol

- Đơn vị đánh giá: một ngày, không phải một giải riêng lẻ.
- Mỗi ngày có 27 kết quả; target gồm 5 vị trí × 10 chữ số nhị phân.
- Một chữ số được ghi nhận là actual nếu xuất hiện ít nhất một lần ở vị trí đó
  trong bất kỳ kết quả nào của ngày.
- Validation: 2023–2024, 723 ngày.
- Final test 2025–2026 không được dùng trong Phần 2; giai đoạn này dành riêng
  cho đánh giá chiến lược ở Phần 3.
- Với model thống kê, ngày `t` chỉ dùng dữ liệu trước `t`, sau đó mới cập nhật
  trạng thái cho ngày tiếp theo.

## Model

`uniform_27_iid`, `expanding_beta`, `rolling_beta_w30`, `rolling_beta_w90`,
`rolling_beta_w365`, `dirichlet_daily`, `markov_presence` và `random_forest`.

## Chỉ số

Daily multi-label Brier score, binary log loss, Brier theo từng vị trí,
Top-1/3/5/10, recall chữ số thực tế, calibration và rank của từng chữ số thực
tế. Các bảng chi tiết nằm trong cùng thư mục `artifacts/models/daily_digit`.

## Kết quả chính

| Fold | Model tốt nhất | Brier | Log loss |
| --- | --- | ---: | ---: |
| Validation 2023–2024 | expanding_beta | 0,097820 | 0,329968 |

`markov_presence` đứng rất sát `expanding_beta` trên validation. Random Forest
có tỷ lệ hit Top-1 tương đối cao nhưng không vượt ổn định về Brier/log loss.
Các vị trí hàng chục nghìn và hàng nghìn khó dự đoán hơn các vị trí thấp hơn;
điều này thể hiện qua Brier theo vị trí.

Do 27 kết quả trong ngày làm nhiều chữ số cùng xuất hiện, Top-3/Top-5 thường
đạt gần 100% ở tiêu chí “có ít nhất một chữ số đúng”. Vì vậy kết luận chính
phải dựa vào Brier, log loss, recall và rank; không diễn giải Top-k hit như
khả năng dự đoán đúng một giải thưởng cụ thể.

