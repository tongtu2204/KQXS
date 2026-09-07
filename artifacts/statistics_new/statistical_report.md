# Báo cáo thống kê tính ngẫu nhiên — bài toán mới

## Phạm vi dữ liệu

- Quy ước multi-win: **không tính** trường hợp tự động vé trùng Giải đặc biệt
  đồng thời trúng Giải phụ đặc biệt và Giải khuyến khích Đặc biệt.
- Khoảng ngày: `2007-01-01` đến `2026-09-06`
- Số ngày có kết quả: `7,093`
- Số dòng kết quả: `191,511`
- Số kết quả mỗi ngày: `[27]`
- Phân tích: gộp toàn bộ giải và tách theo từng giải.
- Vị trí chữ số: tính từ phải sang trái.

## Kiểm định chữ số

Kiểm định Chi-square so với phân phối đều 10 chữ số được lưu trong
`digit_distribution_tests.csv`. Kết quả pooled:

             position      n  chi2_statistic  p_value  entropy_bits
         unit 191511       14.935795 0.092715      3.321872
         tens 191511        4.877782 0.844829      3.321910
     hundreds 163139       18.298561 0.031864      3.321847
    thousands 141860       17.834767 0.037140      3.321838
ten_thousands  70930        8.064289 0.527681      3.321846

## Kiểm định hậu tố

Kết quả Chi-square cho hậu tố 2, 3, 4 và 5 chữ số được lưu trong
`suffix_distribution_tests.csv`.

          suffix      n  unique_values  chi2_statistic  p_value  entropy_bits
suffix_2 191511            100      107.902831 0.254074      6.643450
suffix_3 163139           1000     1080.242486 0.037106      9.961007
suffix_4 141860          10000    10194.983787 0.083541     13.235008
suffix_5  70930          50851    99818.625405 0.656189     15.497867

## Độc lập giữa các vị trí

Kiểm định Chi-square độc lập và Cramér's V được lưu trong
`position_independence_tests.csv`.

        left_position right_position  p_value  cramers_v
         unit           tens 0.281445   0.007141
         unit       hundreds 0.038348   0.008452
         unit      thousands 0.585018   0.007798
         unit  ten_thousands 0.864142   0.010260
         tens       hundreds 0.293658   0.007716
         tens      thousands 0.010375   0.009420
         tens  ten_thousands 0.573215   0.011055
     hundreds      thousands 0.747511   0.007519
     hundreds  ten_thousands 0.867400   0.010247
    thousands  ten_thousands 0.979696   0.009459

## Phụ thuộc theo thời gian

Tự tương quan theo lag 1, 2, 3, 7, 14 và 30 ngày được lưu trong
`temporal_dependence_tests.csv`.

## Khả năng trúng nhiều giải trong lịch sử thật

Kết quả theo từng ngày được lưu trong `multi_win_daily.csv`, tổng hợp theo năm
trong `multi_win_yearly.csv`. Đây là thống kê trên kết quả thực tế, không dùng
model và không phụ thuộc vào danh sách vé chiến thuật.

         year  total_draw_days  days_with_2plus_prizes  days_with_3plus_prizes  days_with_4plus_prizes  average_multi_win_patterns  max_simultaneous_prizes  average_max_total_payout  rate_days_with_2plus_prizes  rate_days_with_3plus_prizes  rate_days_with_4plus_prizes
 2007              363                     239                       0                       0                    0.980716                        2              2.500204e+07                     0.658402                     0.000000                          0.0
 2008              364                     235                       1                       0                    1.005495                        3              2.500220e+07                     0.645604                     0.002747                          0.0
 2009              363                     222                       0                       0                    1.030303                        2              2.500149e+07                     0.611570                     0.000000                          0.0
 2010              361                     221                       0                       0                    0.936288                        2              2.500150e+07                     0.612188                     0.000000                          0.0
 2011              361                     224                       1                       0                    0.947368                        3              2.500294e+07                     0.620499                     0.002770                          0.0
 2012              362                     232                       1                       0                    0.944751                        3              2.500133e+07                     0.640884                     0.002762                          0.0
 2013              361                     233                       3                       0                    1.027701                        3              2.500211e+07                     0.645429                     0.008310                          0.0
 2014              361                     215                       0                       0                    0.916898                        2              2.500139e+07                     0.595568                     0.000000                          0.0
 2015              361                     208                       2                       0                    0.900277                        3              2.500277e+07                     0.576177                     0.005540                          0.0
 2016              362                     231                       0                       0                    0.969613                        2              2.500293e+07                     0.638122                     0.000000                          0.0
 2017              361                     213                       0                       0                    0.900277                        2              2.500183e+07                     0.590028                     0.000000                          0.0
 2018              361                     196                       0                       0                    0.869806                        2              2.500316e+07                     0.542936                     0.000000                          0.0
 2019              361                     231                       0                       0                    1.013850                        2              2.500150e+07                     0.639889                     0.000000                          0.0
 2020              340                     219                       2                       0                    0.964706                        3              2.500241e+07                     0.644118                     0.005882                          0.0
 2021              361                     225                       1                       0                    0.983380                        3              2.500188e+07                     0.623269                     0.002770                          0.0
 2022              361                     226                       2                       0                    0.986150                        3              2.500283e+07                     0.626039                     0.005540                          0.0
 2023              361                     231                       3                       0                    1.013850                        3              2.500216e+07                     0.639889                     0.008310                          0.0
 2024              362                     248                       2                       0                    1.055249                        3              2.500199e+07                     0.685083                     0.005525                          0.0
 2025              361                     228                       0                       0                    0.983380                        2              2.500161e+07                     0.631579                     0.000000                          0.0
 2026              245                     165                       0                       0                    0.987755                        2              2.500506e+07                     0.673469                     0.000000                          0.0
