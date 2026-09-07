# Báo cáo tiến trình Phần 2

## Trạng thái

Đã hoàn tất pipeline mô hình dự đoán chữ số theo ngày trên dữ liệu 7 nhóm giải.
Đơn vị đánh giá là ngày; một chữ số được ghi nhận là đúng nếu xuất hiện ở vị
trí tương ứng trong ít nhất một trong 27 kết quả của ngày đó.

## Dữ liệu và đánh giá

- Dữ liệu: 2007-01-01 đến 2026-09-06;
- 7.093 ngày;
- 27 kết quả/ngày;
- Validation: 2023-01-01 đến 2024-12-31, 723 ngày;
- Final test: 2025-01-01 đến 2026-09-06, 606 ngày;
- 5 vị trí × 10 chữ số = 50 target nhị phân/ngày.

## Chỉ số

- Daily multi-label Brier score;
- Tỷ lệ ít nhất một chữ số đúng trong Top-1, Top-3, Top-5 ở từng vị trí;
- Recall số chữ số thực tế nằm trong Top-k.

Bảng số liệu đầy đủ nằm trong `artifacts/tables_daily/model_summary.csv`.
Do mỗi ngày có 27 kết quả, nhiều chữ số thường cùng xuất hiện trong một vị trí;
vì vậy Top-k hit cần được đọc cùng với Brier score và recall, không được diễn
giải như xác suất dự đoán đúng một giải thưởng cụ thể.

## Kết quả sơ bộ

Theo daily multi-label Brier score, `markov_presence` tốt nhất trên validation
(0,097813), còn `markov_presence` và `expanding_frequency` gần như đồng hạng
trên final test (lần lượt 0,096306 và 0,096309). Random Forest không vượt ổn
định các mô hình thống kê. Đây là kết quả mô hình ban đầu; kết luận về tín hiệu
dự báo cần tiếp tục kiểm tra độ ổn định theo từng vị trí và so sánh với các
baseline tương ứng trước khi chuyển sang Phần 3.

