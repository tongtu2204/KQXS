# Tổng hợp kết quả nghiên cứu tính ngẫu nhiên và khả năng dự báo xổ số

## 1. Khung nghiên cứu

Chuỗi phân tích cuối cùng:

**Randomness → Predictability → Probability Ranking → Economic Strategy → Robustness → True Walk-Forward Deployment**

Mục tiêu không phải chứng minh xổ số hoàn toàn ngẫu nhiên, mà kiểm tra liệu dữ liệu lịch sử có chứa cấu trúc dự báo đủ mạnh và ổn định để khai thác hay không.

## 2. Probability ranking

| model | log_loss | log_loss_gain_vs_uniform | top_1_accuracy | top_5_accuracy | top_10_accuracy | top_20_accuracy | mean_true_rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| catboost | 4.604870 | +0.000300 | 1.1164% | 4.9937% | 10.6597% | 20.8571% | 49.58 |
| uniform | 4.605170 | +0.000000 | 1.0000% | 5.0000% | 10.0000% | 20.0000% | 50.50 |
| catboost_retrain | 4.605640 | -0.000470 | 1.1223% | 4.6450% | 9.9262% | 19.6957% | 50.92 |
| cdm | 4.611909 | -0.006739 | 1.3025% | 5.5329% | 10.5749% | 20.6118% | 50.76 |
| rolling_cdm_w30 | 4.667957 | -0.062787 | 1.1426% | 5.0036% | 10.1445% | 20.3060% | 50.28 |
| rolling_cdm_w365 | 4.705800 | -0.100630 | 0.8526% | 5.1666% | 10.1949% | 19.9682% | 51.15 |
| bayesian_markov | 4.706752 | -0.101582 | 1.2604% | 5.6518% | 10.4300% | 20.1602% | 50.28 |
| rolling_cdm_w60 | 4.710213 | -0.105042 | 0.9447% | 4.9189% | 9.9594% | 19.8733% | 50.82 |
| rolling_cdm_w90 | 4.733091 | -0.127920 | 0.9414% | 4.7746% | 9.5431% | 19.1600% | 51.24 |
| rolling_cdm_w180 | 4.738685 | -0.133515 | 0.7271% | 4.3416% | 9.5880% | 20.0399% | 51.27 |

Model có log loss tốt nhất là **catboost**, với gain so với Uniform `+0.000300`. Magnitude của gain cần được chú ý: lợi thế rất nhỏ không đồng nghĩa với khả năng dự báo kinh tế có ý nghĩa.

## 3. Strategy 14 — Always Top-m

| model | m | number_hits | hit_rate | break_even_hit_rate | total_profit | roi |
| --- | --- | --- | --- | --- | --- | --- |
| cdm | 1 | 30 | 1.2605% | 1.2500% | 200,000 | +0.84% |
| bayesian_markov | 2 | 58 | 2.4370% | 2.5000% | -1,200,000 | -2.52% |
| catboost | 1 | 28 | 1.1765% | 1.2500% | -1,400,000 | -5.88% |
| catboost_retrain | 2 | 56 | 2.3529% | 2.5000% | -2,800,000 | -5.88% |
| rolling_cdm_w30 | 2 | 53 | 2.2269% | 2.5000% | -5,200,000 | -10.92% |
| rolling_cdm_w60 | 6 | 151 | 6.3445% | 7.5000% | -22,000,000 | -15.41% |
| rolling_cdm_w365 | 5 | 125 | 5.2521% | 6.2500% | -19,000,000 | -15.97% |
| rolling_cdm_w90 | 2 | 50 | 2.1008% | 2.5000% | -7,600,000 | -15.97% |
| rolling_cdm_w180 | 25 | 605 | 25.4202% | 31.2500% | -111,000,000 | -18.66% |

Best observed Always Top-m là **cdm**, `m=1`, ROI **+0.84%**. Đây là kết quả quan sát sau khi quét m, không phải bằng chứng độc lập về edge.

## 4. Strategy 15 — Selective Top-m

| model | m | target_participation_rate | n_bets | n_hits | hit_rate | break_even_hit_rate | total_profit | roi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bayesian_markov | 1 | 30% | 327 | 8 | 2.4465% | 1.2500% | 3,130,000 | +95.72% |
| rolling_cdm_w30 | 1 | 20% | 249 | 6 | 2.4096% | 1.2500% | 2,310,000 | +92.77% |
| catboost | 1 | 10% | 132 | 3 | 2.2727% | 1.2500% | 1,080,000 | +81.82% |
| rolling_cdm_w60 | 2 | 10% | 279 | 9 | 3.2258% | 2.5000% | 1,620,000 | +29.03% |
| rolling_cdm_w90 | 2 | 30% | 502 | 14 | 2.7888% | 2.5000% | 1,160,000 | +11.55% |
| rolling_cdm_w365 | 3 | 50% | 819 | 34 | 4.1514% | 3.7500% | 2,630,000 | +10.70% |
| catboost_retrain | 2 | 30% | 302 | 8 | 2.6490% | 2.5000% | 360,000 | +5.96% |
| cdm | 3 | 10% | 126 | 5 | 3.9683% | 3.7500% | 220,000 | +5.82% |
| rolling_cdm_w180 | 10 | 10% | 52 | 6 | 11.5385% | 12.5000% | -400,000 | -7.69% |

Selective filtering tạo ra nhiều ROI dương hơn Always Top-m. Tuy nhiên đây cũng là nơi rủi ro data snooping cao nhất vì đã quét đồng thời model, m và participation rate.

## 5. Strategy 16 — Multiple-testing robustness

| total_configurations | positive_roi_configs | raw_significant_vs_random | raw_significant_vs_break_even | holm_significant_vs_random | holm_significant_vs_break_even | bonferroni_significant_vs_random | bonferroni_significant_vs_break_even | strict_robust_configs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1080 | 113 | 68 | 7 | 0 | 0 | 0 | 0 | 0 |

**Không có configuration nào còn significant so với break-even sau Holm correction, và không có strict robust config.**

Một số cấu hình nổi bật trước correction:

| model | m | target_participation_rate | n_bets | n_hits | hit_rate | break_even_hit_rate | wilson_lower | p_break_even_raw | p_break_even_holm | roi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bayesian_markov | 4 | 30% | 350 | 28 | 8.0000% | 5.0000% | 5.5927% | 0.010556 | 1 | +60.00% |
| bayesian_markov | 5 | 30% | 375 | 33 | 8.8000% | 6.2500% | 6.3346% | 0.0312963 | 1 | +40.80% |
| bayesian_markov | 1 | 30% | 327 | 8 | 2.4465% | 1.2500% | 1.2448% | 0.0553541 | 1 | +95.72% |
| bayesian_markov | 6 | 50% | 571 | 55 | 9.6322% | 7.5000% | 7.4751% | 0.0353934 | 1 | +28.43% |
| bayesian_markov | 3 | 50% | 616 | 32 | 5.1948% | 3.7500% | 3.7035% | 0.0425437 | 1 | +38.53% |
| bayesian_markov | 2 | 50% | 610 | 22 | 3.6066% | 2.5000% | 2.3936% | 0.058569 | 1 | +44.26% |
| bayesian_markov | 1 | 50% | 605 | 12 | 1.9835% | 1.2500% | 1.1382% | 0.0816506 | 1 | +58.68% |
| bayesian_markov | 4 | 50% | 589 | 39 | 6.6214% | 5.0000% | 4.8812% | 0.0480357 | 1 | +32.43% |
| rolling_cdm_w30 | 1 | 20% | 249 | 6 | 2.4096% | 1.2500% | 1.1089% | 0.0943943 | 1 | +92.77% |
| bayesian_markov | 6 | 30% | 358 | 36 | 10.0559% | 7.5000% | 7.3523% | 0.045801 | 1 | +34.08% |

## 6. Strategy 17 — Profit horizon

| criterion | window_type | model | m | target_participation_rate | horizon | n_windows | mean_roi | median_roi | profitable_window_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| max_median_roi | calendar | rolling_cdm_w30 | 1 | 20% | 365 | 1121 | +58.18% | +130.77% | 52.63% |
| max_profitable_window_rate | calendar | rolling_cdm_w30 | 5 | 30% | 365 | 1126 | +32.13% | +42.86% | 88.99% |
| max_median_roi | bet | bayesian_markov | 1 | 20% | 50 | 86 | +311.16% | +540.00% | 65.12% |
| max_profitable_window_rate | bet | bayesian_markov | 1 | 30% | 100 | 38 | +354.74% | +380.00% | 100.00% |

Các rolling windows chồng lấn mạnh, vì vậy `n_windows` không được diễn giải như số thí nghiệm độc lập. Profit horizon chỉ có vai trò mô tả temporal behavior.

## 7. Strategy 18 — Nested walk-forward, fixed threshold

| future_fold | selected_model | selected_m | selected_target_rate | history_n_bets | history_roi | future_n_bets | future_hits | future_hit_rate | future_break_even_hit_rate | future_total_profit | future_roi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-2023 | rolling_cdm_w60 | 1 | 20% | 120 | +100.00% | 172 | 1 | 0.5814% | 1.2500% | -920,000 | -53.49% |
| 2024-2026 | catboost | 1 | 10% | 143 | +235.66% | 0 | 0 | NA | 1.2500% | 0 | NA |

Fixed-threshold nested test cho thấy history performance không tái lập ổn định sang future. Một threshold tuyệt đối cũng có thể trở nên inactive khi probability scale thay đổi.

## 8. Strategy 19 — Nested walk-forward, adaptive threshold

| future_fold | selected_model | selected_m | selected_target_rate | history_n_bets | history_roi | future_n_bets | future_hits | future_hit_rate | future_break_even_hit_rate | future_total_profit | future_roi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-2023 | rolling_cdm_w30 | 12 | 30% | 219 | +33.94% | 237 | 26 | 10.9705% | 15.0000% | -7,640,000 | -26.86% |
| 2024-2026 | bayesian_markov | 3 | 10% | 138 | +93.24% | 93 | 4 | 4.3011% | 3.7500% | 410,000 | +14.70% |

## 9. So sánh true walk-forward

| strategy | n_future_folds | active_folds | inactive_folds | positive_active_folds | negative_active_folds | total_bets | total_hits | total_profit | aggregate_roi | random_expected_roi | roi_vs_random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_threshold | 2 | 1 | 1 | 0 | 1 | 172 | 1 | -920,000 | -53.49% | -20.00% | -33.49% |
| adaptive_threshold | 2 | 2 | 0 | 1 | 1 | 330 | 30 | -7,230,000 | -23.15% | -20.00% | -3.15% |

## 10. Key findings

| section | finding | value | interpretation |
| --- | --- | --- | --- |
| Probability | Best log-loss model | catboost | Lợi thế probability quality chỉ nên được coi là nhỏ nếu log-loss gain so với Uniform rất nhỏ. |
| Probability | Best model log-loss gain vs Uniform | +0.00030014 | Giá trị dương nghĩa là tốt hơn Uniform; cần xem magnitude chứ không chỉ dấu. |
| Always Top-m | Best observed configuration | cdm, m=1, ROI=+0.84% | Observed/post-hoc result; không phải validated edge. |
| Selective Top-m | Best observed configuration | bayesian_markov, m=1, r=30%, ROI=+95.72% | Được chọn sau khi quét nhiều configurations; chịu data-snooping / multiple-comparison risk. |
| Robustness | Holm significant vs break-even | 0 | Không có config sống sót Holm correction. |
| Robustness | Strict robust configs | 0 | Không có cấu hình đáp ứng strict robustness criteria. |
| Nested fixed | Aggregate ROI | -53.49% | Phải đọc cùng từng future fold; aggregate có thể che fold inactive. |
| Nested adaptive | Aggregate ROI | -23.15% | Adaptive threshold xử lý probability-scale drift nhưng chỉ có ý nghĩa nếu lợi nhuận tái lập future. |
| Final conclusion | Research conclusion | Chưa có bằng chứng đủ mạnh về predictive/economic edge có thể khai thác bền vững. | Không đồng nghĩa với việc chứng minh xổ số hoàn toàn ngẫu nhiên; chỉ nói về bằng chứng thực nghiệm trong dữ liệu và phương pháp đã thử. |

## 11. Kết luận cuối

**Chưa có bằng chứng đủ mạnh về predictive/economic edge có thể khai thác bền vững.**

Các mô hình và strategy tạo ra một số giai đoạn có ranking hoặc ROI quan sát được tốt hơn random. Tuy nhiên các lợi thế này không ổn định qua thời gian, không sống sót sau multiple-testing correction và không tái lập nhất quán trong nested walk-forward.

Do đó, trên tập dữ liệu và các phương pháp đã thử, **chưa có bằng chứng đủ mạnh về một predictive/economic edge có thể khai thác bền vững**.

Kết luận này không đồng nghĩa với việc chứng minh về mặt toán học rằng quá trình xổ số hoàn toàn ngẫu nhiên; nó chỉ phản ánh rằng các cấu trúc dự báo được thử nghiệm chưa tạo ra lợi thế ổn định và xác nhận được out-of-sample.