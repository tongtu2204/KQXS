# Thống kê và kiểm định chữ số mới

## Phạm vi

- Dữ liệu: `191,511` dòng giải, `7,093` ngày, từ `2007-01-01` đến `2026-09-06`.
- Phân tích gồm hai lớp: **gộp toàn bộ giải** và **tách theo từng giải**.
- Mức ý nghĩa: kiểm định hai phía với hiệu chỉnh nhiều kiểm định Benjamini–Hochberg (FDR), ngưỡng q < 0.05.
- Với đuôi 4/5 chữ số, các ô có kỳ vọng thấp được giữ ở dạng tần suất mô tả và không gán p-value Chi-square nếu không đủ điều kiện xấp xỉ.
- Thống kê trúng nhiều giải loại trừ rule tự động: vé trùng Giải đặc biệt không được tính đồng thời Giải phụ đặc biệt và Giải khuyến khích đặc biệt.

## Kết quả kiểm định

| Nhóm | Số kiểm định | Số bác bỏ H0 sau FDR |
|---|---:|---:|
| Phân phối chữ số | 45 | 1 |
| Phân phối đuôi số | 28 hợp lệ / 45 dòng | 5 |
| Độc lập giữa vị trí | 90 | 0 |
| Phụ thuộc theo thời gian | 45 | 0 |

## Trúng đồng thời nhiều giải

- Tỷ lệ ngày có ít nhất 2 giải có thể cùng trúng: **62.63%**.
- Tỷ lệ ngày có ít nhất 3 giải có thể cùng trúng: **0.25%**.
- Tỷ lệ ngày có ít nhất 4 giải có thể cùng trúng: **0.00%**.

Các bảng chi tiết và biểu đồ nằm trong cùng thư mục kết quả.
