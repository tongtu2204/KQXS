"""
20 - FINAL RESEARCH SUMMARY

Tổng hợp toàn bộ nghiên cứu:

    Probability models
        -> Ranking quality
        -> Always Top-m
        -> Selective Top-m
        -> Multiple-testing robustness
        -> Profit horizon
        -> Nested walk-forward fixed threshold
        -> Nested walk-forward adaptive threshold

Mục tiêu
--------
Tạo các bảng cuối dùng cho:

    - báo cáo
    - slide
    - phần kết luận
    - appendix

Không train lại model.

Narrative cuối:
    Randomness
        ->
    Predictability
        ->
    Probability Ranking
        ->
    Economic Strategy
        ->
    Robustness
        ->
    True Walk-Forward Deployment

Các kết luận phải thận trọng:

    "Không tìm thấy bằng chứng đủ mạnh..."
không viết:
    "Xổ số hoàn toàn ngẫu nhiên."

Outputs
-------
artifacts/strategies/final_summary/

    01_probability_models.csv
    02_always_topm.csv
    03_selective_topm.csv
    04_robustness_summary.csv
    05_top_robustness_configs.csv
    06_profit_horizons.csv
    07_walk_forward_fixed.csv
    08_walk_forward_adaptive.csv
    09_walk_forward_comparison.csv
    10_key_findings.csv

    final_research_summary.md
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]


TABLE_DIR = (
    PROJECT_DIR
    / "artifacts"
    / "tables"
)


STRATEGY_DIR = (
    PROJECT_DIR
    / "artifacts"
    / "strategies"
)


OUTPUT_DIR = (
    STRATEGY_DIR
    / "final_summary"
)


# ============================================================
# INPUT FILES
# ============================================================

FILES = {
    # --------------------------------------------------------
    # Probability ranking
    # --------------------------------------------------------
    "probability": (
        TABLE_DIR
        / "probability_ranking_summary.csv"
    ),

    # --------------------------------------------------------
    # Strategy 14
    # --------------------------------------------------------
    "always": (
        STRATEGY_DIR
        / "always_topm_best_by_model.csv"
    ),

    # --------------------------------------------------------
    # Strategy 15
    # --------------------------------------------------------
    "selective": (
        STRATEGY_DIR
        / "selective_topm_best_by_model.csv"
    ),

    # --------------------------------------------------------
    # Strategy 16
    # --------------------------------------------------------
    "robustness": (
        STRATEGY_DIR
        / "selective_robustness_all.csv"
    ),

    "robustness_top": (
        STRATEGY_DIR
        / "selective_robustness_top_configs.csv"
    ),

    # --------------------------------------------------------
    # Strategy 17
    # --------------------------------------------------------
    "horizon": (
        STRATEGY_DIR
        / "profit_horizon_best_overall.csv"
    ),

    # --------------------------------------------------------
    # Strategy 18
    # --------------------------------------------------------
    "nested_fixed": (
        STRATEGY_DIR
        / "nested_strategy_test_results.csv"
    ),

    # --------------------------------------------------------
    # Strategy 19
    # --------------------------------------------------------
    "nested_adaptive": (
        STRATEGY_DIR
        / "adaptive_strategy_test_results.csv"
    ),
}


# ============================================================
# OUTPUT FILES
# ============================================================

OUTPUT_PROBABILITY = (
    OUTPUT_DIR
    / "01_probability_models.csv"
)


OUTPUT_ALWAYS = (
    OUTPUT_DIR
    / "02_always_topm.csv"
)


OUTPUT_SELECTIVE = (
    OUTPUT_DIR
    / "03_selective_topm.csv"
)


OUTPUT_ROBUSTNESS = (
    OUTPUT_DIR
    / "04_robustness_summary.csv"
)


OUTPUT_ROBUSTNESS_TOP = (
    OUTPUT_DIR
    / "05_top_robustness_configs.csv"
)


OUTPUT_HORIZON = (
    OUTPUT_DIR
    / "06_profit_horizons.csv"
)


OUTPUT_FIXED = (
    OUTPUT_DIR
    / "07_walk_forward_fixed.csv"
)


OUTPUT_ADAPTIVE = (
    OUTPUT_DIR
    / "08_walk_forward_adaptive.csv"
)


OUTPUT_WALK_FORWARD_COMPARISON = (
    OUTPUT_DIR
    / "09_walk_forward_comparison.csv"
)


OUTPUT_KEY_FINDINGS = (
    OUTPUT_DIR
    / "10_key_findings.csv"
)


OUTPUT_MARKDOWN = (
    OUTPUT_DIR
    / "final_research_summary.md"
)


# ============================================================
# CONSTANTS
# ============================================================

RANDOM_EXPECTED_ROI = (
    -0.20
)


NUMBER_OF_CLASSES = (
    100
)


# ============================================================
# HELPERS
# ============================================================

def require_file(
    name,
):

    file_path = (
        FILES[
            name
        ]
    )


    if not file_path.exists():

        raise FileNotFoundError(
            f"Không tìm thấy:\n"
            f"{file_path}\n\n"
            f"Hãy chạy experiment tương ứng trước."
        )


    return file_path


def safe_float(
    value,
):

    if pd.isna(
        value
    ):

        return np.nan


    return float(
        value
    )


def safe_roi(
    profit,
    cost,
):

    if (
        pd.isna(
            cost
        )
        or cost <= 0
    ):

        return np.nan


    return (
        profit
        / cost
    )


def format_pct(
    value,
    digits=2,
):

    if pd.isna(
        value
    ):

        return "NA"


    return (
        f"{value:.{digits}%}"
    )


def format_pct_signed(
    value,
    digits=2,
):

    if pd.isna(
        value
    ):

        return "NA"


    return (
        f"{value:+.{digits}%}"
    )


def format_number(
    value,
    digits=0,
):

    if pd.isna(
        value
    ):

        return "NA"


    return (
        f"{value:,.{digits}f}"
    )


def markdown_table(
    df,
    columns=None,
    formatters=None,
):
    """
    Tự tạo Markdown table.

    Không phụ thuộc package tabulate.
    """

    if columns is None:

        columns = list(
            df.columns
        )


    if formatters is None:

        formatters = {}


    if df.empty:

        return (
            "_Không có dữ liệu._"
        )


    lines = []


    # ========================================================
    # HEADER
    # ========================================================

    lines.append(
        "| "
        + " | ".join(
            columns
        )
        + " |"
    )


    lines.append(
        "| "
        + " | ".join(
            [
                "---"
                for _
                in columns
            ]
        )
        + " |"
    )


    # ========================================================
    # ROWS
    # ========================================================

    for _, row in df.iterrows():

        values = []


        for column in columns:

            value = (
                row[
                    column
                ]
            )


            if (
                column
                in formatters
            ):

                text = (
                    formatters[
                        column
                    ](
                        value
                    )
                )

            elif pd.isna(
                value
            ):

                text = (
                    "NA"
                )

            else:

                text = str(
                    value
                )


            text = (
                text
                .replace(
                    "|",
                    "\\|",
                )
            )


            values.append(
                text
            )


        lines.append(
            "| "
            + " | ".join(
                values
            )
            + " |"
        )


    return "\n".join(
        lines
    )


# ============================================================
# LOAD
# ============================================================

def load_inputs():

    for name in FILES:

        require_file(
            name
        )


    probability = pd.read_csv(
        FILES[
            "probability"
        ]
    )


    always = pd.read_csv(
        FILES[
            "always"
        ]
    )


    selective = pd.read_csv(
        FILES[
            "selective"
        ]
    )


    robustness = pd.read_csv(
        FILES[
            "robustness"
        ]
    )


    robustness_top = pd.read_csv(
        FILES[
            "robustness_top"
        ]
    )


    horizon = pd.read_csv(
        FILES[
            "horizon"
        ]
    )


    nested_fixed = pd.read_csv(
        FILES[
            "nested_fixed"
        ]
    )


    nested_adaptive = pd.read_csv(
        FILES[
            "nested_adaptive"
        ]
    )


    return {
        "probability": (
            probability
        ),

        "always": (
            always
        ),

        "selective": (
            selective
        ),

        "robustness": (
            robustness
        ),

        "robustness_top": (
            robustness_top
        ),

        "horizon": (
            horizon
        ),

        "nested_fixed": (
            nested_fixed
        ),

        "nested_adaptive": (
            nested_adaptive
        ),
    }


# ============================================================
# 01 - PROBABILITY TABLE
# ============================================================

def build_probability_table(
    df,
):

    wanted = [
        "model",
        "n_test",
        "log_loss",
        "log_loss_gain_vs_uniform",
        "relative_log_loss_gain_pct",
        "top_1_accuracy",
        "top_3_accuracy",
        "top_5_accuracy",
        "top_10_accuracy",
        "top_20_accuracy",
        "mean_true_rank",
        "mean_tie_size",
    ]


    columns = [
        column
        for column
        in wanted
        if column in df.columns
    ]


    output = (
        df[
            columns
        ]
        .copy()
    )


    output = (
        output
        .sort_values(
            [
                "log_loss",
                "mean_true_rank",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


    return output


# ============================================================
# 02 - ALWAYS TABLE
# ============================================================

def build_always_table(
    df,
):

    wanted = [
        "model",
        "m",
        "n_days",
        "number_hits",
        "hit_rate",
        "random_hit_rate",
        "break_even_hit_rate",
        "gap_to_break_even",
        "total_profit",
        "roi",
        "random_expected_roi",
        "roi_gain_vs_random",
        "max_drawdown",
    ]


    columns = [
        column
        for column
        in wanted
        if column in df.columns
    ]


    return (
        df[
            columns
        ]
        .sort_values(
            "roi",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 03 - SELECTIVE TABLE
# ============================================================

def build_selective_table(
    df,
):

    wanted = [
        "model",
        "m",
        "target_participation_rate",
        "n_test_days",
        "n_bets",
        "participation_rate",
        "n_hits",
        "hit_rate",
        "random_hit_rate",
        "break_even_hit_rate",
        "gap_to_break_even",
        "total_profit",
        "roi",
        "max_drawdown",
    ]


    columns = [
        column
        for column
        in wanted
        if column in df.columns
    ]


    return (
        df[
            columns
        ]
        .sort_values(
            "roi",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 04 - ROBUSTNESS SUMMARY
# ============================================================

def build_robustness_summary(
    df,
):

    n_configs = len(
        df
    )


    metrics = {
        "total_configurations": (
            n_configs
        ),

        "positive_roi_configs": int(
            df[
                "roi"
            ]
            .gt(
                0
            )
            .sum()
        ),

        "raw_significant_vs_random": int(
            df[
                "sig_random_raw"
            ]
            .astype(bool)
            .sum()
        ),

        "raw_significant_vs_break_even": int(
            df[
                "sig_break_even_raw"
            ]
            .astype(bool)
            .sum()
        ),

        "holm_significant_vs_random": int(
            df[
                "sig_random_holm"
            ]
            .astype(bool)
            .sum()
        ),

        "holm_significant_vs_break_even": int(
            df[
                "sig_break_even_holm"
            ]
            .astype(bool)
            .sum()
        ),

        "bonferroni_significant_vs_random": int(
            df[
                "sig_random_bonferroni"
            ]
            .astype(bool)
            .sum()
        ),

        "bonferroni_significant_vs_break_even": int(
            df[
                "sig_break_even_bonferroni"
            ]
            .astype(bool)
            .sum()
        ),

        "strict_robust_configs": int(
            df[
                "strict_robust"
            ]
            .astype(bool)
            .sum()
        ),
    }


    return pd.DataFrame(
        [
            metrics
        ]
    )


# ============================================================
# 05 - TOP ROBUSTNESS
# ============================================================

def build_top_robustness(
    df,
):

    wanted = [
        "model",
        "m",
        "target_participation_rate",
        "n_bets",
        "n_hits",
        "hit_rate",
        "break_even_hit_rate",
        "wilson_lower",
        "wilson_upper",
        "p_random_raw",
        "p_random_holm",
        "p_break_even_raw",
        "p_break_even_holm",
        "total_profit",
        "roi",
        "positive_folds",
        "negative_folds",
        "strict_robust",
    ]


    columns = [
        column
        for column
        in wanted
        if column in df.columns
    ]


    return (
        df[
            columns
        ]
        .head(
            30
        )
        .copy()
    )


# ============================================================
# 06 - HORIZON
# ============================================================

def build_horizon_table(
    df,
):

    wanted = [
        "criterion",
        "window_type",
        "model",
        "m",
        "target_participation_rate",
        "horizon",
        "n_windows",
        "mean_roi",
        "median_roi",
        "std_roi",
        "profitable_window_rate",
        "mean_n_bets",
        "mean_calendar_span_days",
    ]


    columns = [
        column
        for column
        in wanted
        if column in df.columns
    ]


    return (
        df[
            columns
        ]
        .copy()
    )


# ============================================================
# WALK-FORWARD STANDARDIZATION
# ============================================================

def standardize_fixed_walk_forward(
    df,
):

    output = pd.DataFrame(
        {
            "strategy": (
                "fixed_threshold"
            ),

            "future_fold": (
                df[
                    "future_fold"
                ]
            ),

            "selected_model": (
                df[
                    "selected_model"
                ]
            ),

            "selected_m": (
                df[
                    "selected_m"
                ]
            ),

            "selected_target_rate": (
                df[
                    "selected_target_rate"
                ]
            ),

            "history_n_bets": (
                df[
                    "history_n_bets"
                ]
            ),

            "history_roi": (
                df[
                    "history_roi"
                ]
            ),

            "future_n_days": (
                df[
                    "future_n_days"
                ]
            ),

            "future_n_bets": (
                df[
                    "future_n_bet_days"
                ]
            ),

            "future_participation_rate": (
                df[
                    "future_participation_rate"
                ]
            ),

            "future_hits": (
                df[
                    "future_number_hits"
                ]
            ),

            "future_hit_rate": (
                df[
                    "future_hit_rate"
                ]
            ),

            "future_break_even_hit_rate": (
                df[
                    "future_break_even_hit_rate"
                ]
            ),

            "future_total_cost": (
                df[
                    "future_total_cost"
                ]
            ),

            "future_total_revenue": (
                df[
                    "future_total_revenue"
                ]
            ),

            "future_total_profit": (
                df[
                    "future_total_profit"
                ]
            ),

            "future_roi": (
                df[
                    "future_roi"
                ]
            ),
        }
    )


    return output


def standardize_adaptive_walk_forward(
    df,
):

    output = pd.DataFrame(
        {
            "strategy": (
                "adaptive_threshold"
            ),

            "future_fold": (
                df[
                    "future_fold"
                ]
            ),

            "selected_model": (
                df[
                    "selected_model"
                ]
            ),

            "selected_m": (
                df[
                    "selected_m"
                ]
            ),

            "selected_target_rate": (
                df[
                    "selected_target_rate"
                ]
            ),

            "history_n_bets": (
                df[
                    "history_n_bets"
                ]
            ),

            "history_roi": (
                df[
                    "history_roi"
                ]
            ),

            "future_n_days": (
                df[
                    "future_n_days"
                ]
            ),

            "future_n_bets": (
                df[
                    "future_n_bet_days"
                ]
            ),

            "future_participation_rate": (
                df[
                    "future_participation_rate"
                ]
            ),

            "future_hits": (
                df[
                    "future_number_hits"
                ]
            ),

            "future_hit_rate": (
                df[
                    "future_hit_rate"
                ]
            ),

            "future_break_even_hit_rate": (
                df[
                    "future_break_even_hit_rate"
                ]
            ),

            "future_total_cost": (
                df[
                    "future_total_cost"
                ]
            ),

            "future_total_revenue": (
                df[
                    "future_total_revenue"
                ]
            ),

            "future_total_profit": (
                df[
                    "future_total_profit"
                ]
            ),

            "future_roi": (
                df[
                    "future_roi"
                ]
            ),
        }
    )


    if (
        "future_constant_confidence_days"
        in df.columns
    ):

        output[
            "constant_confidence_days"
        ] = (
            df[
                "future_constant_confidence_days"
            ]
        )


    return output


# ============================================================
# WALK-FORWARD AGGREGATE
# ============================================================

def aggregate_walk_forward(
    df,
    strategy_name,
):

    total_bets = int(
        df[
            "future_n_bets"
        ]
        .sum()
    )


    total_hits = int(
        df[
            "future_hits"
        ]
        .sum()
    )


    total_cost = float(
        df[
            "future_total_cost"
        ]
        .sum()
    )


    total_revenue = float(
        df[
            "future_total_revenue"
        ]
        .sum()
    )


    total_profit = (
        total_revenue
        - total_cost
    )


    aggregate_roi = (
        safe_roi(
            total_profit,
            total_cost,
        )
    )


    active = (
        df[
            "future_n_bets"
        ]
        .gt(
            0
        )
    )


    positive_active = (
        active
        & df[
            "future_roi"
        ]
        .gt(
            0
        )
    )


    negative_active = (
        active
        & df[
            "future_roi"
        ]
        .lt(
            0
        )
    )


    return {
        "strategy": (
            strategy_name
        ),

        "n_future_folds": (
            len(
                df
            )
        ),

        "active_folds": int(
            active.sum()
        ),

        "inactive_folds": int(
            (
                ~active
            )
            .sum()
        ),

        "positive_active_folds": int(
            positive_active.sum()
        ),

        "negative_active_folds": int(
            negative_active.sum()
        ),

        "total_bets": (
            total_bets
        ),

        "total_hits": (
            total_hits
        ),

        "total_cost": (
            total_cost
        ),

        "total_revenue": (
            total_revenue
        ),

        "total_profit": (
            total_profit
        ),

        "aggregate_roi": (
            aggregate_roi
        ),

        "random_expected_roi": (
            RANDOM_EXPECTED_ROI
        ),

        "roi_vs_random": (
            (
                aggregate_roi
                - RANDOM_EXPECTED_ROI
            )
            if pd.notna(
                aggregate_roi
            )
            else np.nan
        ),
    }


# ============================================================
# KEY FINDINGS
# ============================================================

def build_key_findings(
    probability,
    always,
    selective,
    robustness,
    fixed,
    adaptive,
    walk_comparison,
):

    findings = []


    # ========================================================
    # BEST PROBABILITY MODEL
    # ========================================================

    non_uniform = (
        probability.loc[
            probability[
                "model"
            ]
            .ne(
                "uniform"
            )
        ]
        .copy()
    )


    best_probability = (
        non_uniform
        .sort_values(
            "log_loss"
        )
        .iloc[0]
    )


    findings.append(
        {
            "section": (
                "Probability"
            ),

            "finding": (
                "Best log-loss model"
            ),

            "value": (
                best_probability[
                    "model"
                ]
            ),

            "interpretation": (
                "Lợi thế probability quality "
                "chỉ nên được coi là nhỏ nếu "
                "log-loss gain so với Uniform rất nhỏ."
            ),
        }
    )


    findings.append(
        {
            "section": (
                "Probability"
            ),

            "finding": (
                "Best model log-loss gain vs Uniform"
            ),

            "value": (
                f"{best_probability['log_loss_gain_vs_uniform']:+.8f}"
            ),

            "interpretation": (
                "Giá trị dương nghĩa là tốt hơn Uniform; "
                "cần xem magnitude chứ không chỉ dấu."
            ),
        }
    )


    # ========================================================
    # ALWAYS
    # ========================================================

    best_always = (
        always
        .sort_values(
            "roi",
            ascending=False,
        )
        .iloc[0]
    )


    findings.append(
        {
            "section": (
                "Always Top-m"
            ),

            "finding": (
                "Best observed configuration"
            ),

            "value": (
                f"{best_always['model']}, "
                f"m={int(best_always['m'])}, "
                f"ROI={best_always['roi']:+.2%}"
            ),

            "interpretation": (
                "Observed/post-hoc result; "
                "không phải validated edge."
            ),
        }
    )


    # ========================================================
    # SELECTIVE
    # ========================================================

    best_selective = (
        selective
        .sort_values(
            "roi",
            ascending=False,
        )
        .iloc[0]
    )


    findings.append(
        {
            "section": (
                "Selective Top-m"
            ),

            "finding": (
                "Best observed configuration"
            ),

            "value": (
                f"{best_selective['model']}, "
                f"m={int(best_selective['m'])}, "
                f"r={best_selective['target_participation_rate']:.0%}, "
                f"ROI={best_selective['roi']:+.2%}"
            ),

            "interpretation": (
                "Được chọn sau khi quét nhiều configurations; "
                "chịu data-snooping / multiple-comparison risk."
            ),
        }
    )


    # ========================================================
    # ROBUSTNESS
    # ========================================================

    holm_break_even = int(
        robustness[
            "holm_significant_vs_break_even"
        ]
        .iloc[0]
    )


    strict_robust = int(
        robustness[
            "strict_robust_configs"
        ]
        .iloc[0]
    )


    findings.append(
        {
            "section": (
                "Robustness"
            ),

            "finding": (
                "Holm significant vs break-even"
            ),

            "value": str(
                holm_break_even
            ),

            "interpretation": (
                (
                    "Không có config sống sót Holm correction."
                )
                if holm_break_even == 0
                else (
                    "Có config sống sót Holm correction; "
                    "cần kiểm tra nested walk-forward."
                )
            ),
        }
    )


    findings.append(
        {
            "section": (
                "Robustness"
            ),

            "finding": (
                "Strict robust configs"
            ),

            "value": str(
                strict_robust
            ),

            "interpretation": (
                (
                    "Không có cấu hình đáp ứng strict robustness criteria."
                )
                if strict_robust == 0
                else (
                    "Một số cấu hình vượt strict robustness criteria."
                )
            ),
        }
    )


    # ========================================================
    # FIXED
    # ========================================================

    fixed_aggregate = (
        walk_comparison.loc[
            walk_comparison[
                "strategy"
            ]
            .eq(
                "fixed_threshold"
            )
        ]
        .iloc[0]
    )


    findings.append(
        {
            "section": (
                "Nested fixed"
            ),

            "finding": (
                "Aggregate ROI"
            ),

            "value": (
                format_pct_signed(
                    fixed_aggregate[
                        "aggregate_roi"
                    ]
                )
            ),

            "interpretation": (
                "Phải đọc cùng từng future fold; "
                "aggregate có thể che fold inactive."
            ),
        }
    )


    # ========================================================
    # ADAPTIVE
    # ========================================================

    adaptive_aggregate = (
        walk_comparison.loc[
            walk_comparison[
                "strategy"
            ]
            .eq(
                "adaptive_threshold"
            )
        ]
        .iloc[0]
    )


    findings.append(
        {
            "section": (
                "Nested adaptive"
            ),

            "finding": (
                "Aggregate ROI"
            ),

            "value": (
                format_pct_signed(
                    adaptive_aggregate[
                        "aggregate_roi"
                    ]
                )
            ),

            "interpretation": (
                "Adaptive threshold xử lý probability-scale drift "
                "nhưng chỉ có ý nghĩa nếu lợi nhuận tái lập future."
            ),
        }
    )


    # ========================================================
    # FINAL
    # ========================================================

    adaptive_positive = int(
        adaptive_aggregate[
            "positive_active_folds"
        ]
    )


    adaptive_negative = int(
        adaptive_aggregate[
            "negative_active_folds"
        ]
    )


    adaptive_roi = (
        adaptive_aggregate[
            "aggregate_roi"
        ]
    )


    if (
        holm_break_even == 0
        and strict_robust == 0
        and (
            pd.isna(
                adaptive_roi
            )
            or adaptive_roi <= 0
        )
    ):

        final_conclusion = (
            "Chưa có bằng chứng đủ mạnh về predictive/economic edge "
            "có thể khai thác bền vững."
        )

    elif (
        adaptive_positive > 0
        and adaptive_negative > 0
    ):

        final_conclusion = (
            "Có một số future periods sinh lợi nhưng kết quả "
            "không ổn định qua thời gian; chưa đủ để xác nhận edge."
        )

    else:

        final_conclusion = (
            "Kết quả cần được diễn giải thận trọng; "
            "nested future sample còn nhỏ."
        )


    findings.append(
        {
            "section": (
                "Final conclusion"
            ),

            "finding": (
                "Research conclusion"
            ),

            "value": (
                final_conclusion
            ),

            "interpretation": (
                "Không đồng nghĩa với việc chứng minh "
                "xổ số hoàn toàn ngẫu nhiên; chỉ nói về "
                "bằng chứng thực nghiệm trong dữ liệu và phương pháp đã thử."
            ),
        }
    )


    return pd.DataFrame(
        findings
    )


# ============================================================
# MARKDOWN REPORT
# ============================================================

def build_markdown_report(
    probability,
    always,
    selective,
    robustness_summary,
    robustness_top,
    horizon,
    fixed,
    adaptive,
    walk_comparison,
    key_findings,
):

    lines = []


    # ========================================================
    # TITLE
    # ========================================================

    lines.append(
        "# Tổng hợp kết quả nghiên cứu tính ngẫu nhiên và khả năng dự báo xổ số"
    )


    lines.append(
        ""
    )


    lines.append(
        "## 1. Khung nghiên cứu"
    )


    lines.append(
        ""
    )


    lines.append(
        "Chuỗi phân tích cuối cùng:"
    )


    lines.append(
        ""
    )


    lines.append(
        "**Randomness → Predictability → Probability Ranking → "
        "Economic Strategy → Robustness → True Walk-Forward Deployment**"
    )


    lines.append(
        ""
    )


    lines.append(
        "Mục tiêu không phải chứng minh xổ số hoàn toàn ngẫu nhiên, "
        "mà kiểm tra liệu dữ liệu lịch sử có chứa cấu trúc dự báo "
        "đủ mạnh và ổn định để khai thác hay không."
    )


    # ========================================================
    # PROBABILITY
    # ========================================================

    lines.append(
        ""
    )


    lines.append(
        "## 2. Probability ranking"
    )


    lines.append(
        ""
    )


    probability_columns = [
        "model",
        "log_loss",
        "log_loss_gain_vs_uniform",
        "top_1_accuracy",
        "top_5_accuracy",
        "top_10_accuracy",
        "top_20_accuracy",
        "mean_true_rank",
    ]


    lines.append(
        markdown_table(
            probability.head(
                10
            ),

            columns=(
                probability_columns
            ),

            formatters={
                "log_loss": (
                    lambda x:
                    f"{x:.6f}"
                ),

                "log_loss_gain_vs_uniform": (
                    lambda x:
                    f"{x:+.6f}"
                ),

                "top_1_accuracy": (
                    lambda x:
                    format_pct(
                        x,
                        4,
                    )
                ),

                "top_5_accuracy": (
                    lambda x:
                    format_pct(
                        x,
                        4,
                    )
                ),

                "top_10_accuracy": (
                    lambda x:
                    format_pct(
                        x,
                        4,
                    )
                ),

                "top_20_accuracy": (
                    lambda x:
                    format_pct(
                        x,
                        4,
                    )
                ),

                "mean_true_rank": (
                    lambda x:
                    f"{x:.2f}"
                ),
            },
        )
    )


    best_probability = (
        probability.loc[
            probability[
                "model"
            ]
            .ne(
                "uniform"
            )
        ]
        .sort_values(
            "log_loss"
        )
        .iloc[0]
    )


    lines.append(
        ""
    )


    lines.append(
        (
            f"Model có log loss tốt nhất là **{best_probability['model']}**, "
            f"với gain so với Uniform "
            f"`{best_probability['log_loss_gain_vs_uniform']:+.6f}`. "
            "Magnitude của gain cần được chú ý: lợi thế rất nhỏ "
            "không đồng nghĩa với khả năng dự báo kinh tế có ý nghĩa."
        )
    )


    # ========================================================
    # ALWAYS
    # ========================================================

    lines.append(
        ""
    )


    lines.append(
        "## 3. Strategy 14 — Always Top-m"
    )


    lines.append(
        ""
    )


    always_columns = [
        "model",
        "m",
        "number_hits",
        "hit_rate",
        "break_even_hit_rate",
        "total_profit",
        "roi",
    ]


    lines.append(
        markdown_table(
            always,

            columns=(
                always_columns
            ),

            formatters={
                "hit_rate": (
                    lambda x:
                    format_pct(
                        x,
                        4,
                    )
                ),

                "break_even_hit_rate": (
                    lambda x:
                    format_pct(
                        x,
                        4,
                    )
                ),

                "total_profit": (
                    lambda x:
                    format_number(
                        x
                    )
                ),

                "roi": (
                    lambda x:
                    format_pct_signed(
                        x
                    )
                ),
            },
        )
    )


    best_always = (
        always
        .sort_values(
            "roi",
            ascending=False,
        )
        .iloc[0]
    )


    lines.append(
        ""
    )


    lines.append(
        (
            f"Best observed Always Top-m là **{best_always['model']}**, "
            f"`m={int(best_always['m'])}`, ROI "
            f"**{best_always['roi']:+.2%}**. "
            "Đây là kết quả quan sát sau khi quét m, không phải "
            "bằng chứng độc lập về edge."
        )
    )


    # ========================================================
    # SELECTIVE
    # ========================================================

    lines.append(
        ""
    )


    lines.append(
        "## 4. Strategy 15 — Selective Top-m"
    )


    lines.append(
        ""
    )


    selective_columns = [
        "model",
        "m",
        "target_participation_rate",
        "n_bets",
        "n_hits",
        "hit_rate",
        "break_even_hit_rate",
        "total_profit",
        "roi",
    ]


    lines.append(
        markdown_table(
            selective,

            columns=(
                selective_columns
            ),

            formatters={
                "target_participation_rate": (
                    lambda x:
                    format_pct(
                        x,
                        0,
                    )
                ),

                "hit_rate": (
                    lambda x:
                    format_pct(
                        x,
                        4,
                    )
                ),

                "break_even_hit_rate": (
                    lambda x:
                    format_pct(
                        x,
                        4,
                    )
                ),

                "total_profit": (
                    lambda x:
                    format_number(
                        x
                    )
                ),

                "roi": (
                    lambda x:
                    format_pct_signed(
                        x
                    )
                ),
            },
        )
    )


    lines.append(
        ""
    )


    lines.append(
        "Selective filtering tạo ra nhiều ROI dương hơn Always Top-m. "
        "Tuy nhiên đây cũng là nơi rủi ro data snooping cao nhất "
        "vì đã quét đồng thời model, m và participation rate."
    )


    # ========================================================
    # ROBUSTNESS
    # ========================================================

    lines.append(
        ""
    )


    lines.append(
        "## 5. Strategy 16 — Multiple-testing robustness"
    )


    lines.append(
        ""
    )


    lines.append(
        markdown_table(
            robustness_summary
        )
    )


    holm_be = int(
        robustness_summary[
            "holm_significant_vs_break_even"
        ]
        .iloc[0]
    )


    strict = int(
        robustness_summary[
            "strict_robust_configs"
        ]
        .iloc[0]
    )


    lines.append(
        ""
    )


    if (
        holm_be == 0
        and strict == 0
    ):

        lines.append(
            "**Không có configuration nào còn significant so với "
            "break-even sau Holm correction, và không có strict robust config.**"
        )

    else:

        lines.append(
            "Một số configuration vượt robustness criteria; "
            "cần ưu tiên nested walk-forward để xác nhận."
        )


    lines.append(
        ""
    )


    lines.append(
        "Một số cấu hình nổi bật trước correction:"
    )


    lines.append(
        ""
    )


    robustness_columns = [
        "model",
        "m",
        "target_participation_rate",
        "n_bets",
        "n_hits",
        "hit_rate",
        "break_even_hit_rate",
        "wilson_lower",
        "p_break_even_raw",
        "p_break_even_holm",
        "roi",
    ]


    lines.append(
        markdown_table(
            robustness_top.head(
                10
            ),

            columns=(
                robustness_columns
            ),

            formatters={
                "target_participation_rate": (
                    lambda x:
                    format_pct(
                        x,
                        0,
                    )
                ),

                "hit_rate": (
                    lambda x:
                    format_pct(
                        x,
                        4,
                    )
                ),

                "break_even_hit_rate": (
                    lambda x:
                    format_pct(
                        x,
                        4,
                    )
                ),

                "wilson_lower": (
                    lambda x:
                    format_pct(
                        x,
                        4,
                    )
                ),

                "p_break_even_raw": (
                    lambda x:
                    f"{x:.6g}"
                ),

                "p_break_even_holm": (
                    lambda x:
                    f"{x:.6g}"
                ),

                "roi": (
                    lambda x:
                    format_pct_signed(
                        x
                    )
                ),
            },
        )
    )


    # ========================================================
    # HORIZON
    # ========================================================

    lines.append(
        ""
    )


    lines.append(
        "## 6. Strategy 17 — Profit horizon"
    )


    lines.append(
        ""
    )


    horizon_columns = [
        "criterion",
        "window_type",
        "model",
        "m",
        "target_participation_rate",
        "horizon",
        "n_windows",
        "mean_roi",
        "median_roi",
        "profitable_window_rate",
    ]


    lines.append(
        markdown_table(
            horizon,

            columns=(
                horizon_columns
            ),

            formatters={
                "target_participation_rate": (
                    lambda x:
                    format_pct(
                        x,
                        0,
                    )
                ),

                "mean_roi": (
                    lambda x:
                    format_pct_signed(
                        x
                    )
                ),

                "median_roi": (
                    lambda x:
                    format_pct_signed(
                        x
                    )
                ),

                "profitable_window_rate": (
                    lambda x:
                    format_pct(
                        x
                    )
                ),
            },
        )
    )


    lines.append(
        ""
    )


    lines.append(
        "Các rolling windows chồng lấn mạnh, vì vậy `n_windows` "
        "không được diễn giải như số thí nghiệm độc lập. "
        "Profit horizon chỉ có vai trò mô tả temporal behavior."
    )


    # ========================================================
    # FIXED WALK FORWARD
    # ========================================================

    lines.append(
        ""
    )


    lines.append(
        "## 7. Strategy 18 — Nested walk-forward, fixed threshold"
    )


    lines.append(
        ""
    )


    fixed_columns = [
        "future_fold",
        "selected_model",
        "selected_m",
        "selected_target_rate",
        "history_n_bets",
        "history_roi",
        "future_n_bets",
        "future_hits",
        "future_hit_rate",
        "future_break_even_hit_rate",
        "future_total_profit",
        "future_roi",
    ]


    lines.append(
        markdown_table(
            fixed,

            columns=(
                fixed_columns
            ),

            formatters={
                "selected_target_rate": (
                    lambda x:
                    format_pct(
                        x,
                        0,
                    )
                ),

                "history_roi": (
                    lambda x:
                    format_pct_signed(
                        x
                    )
                ),

                "future_hit_rate": (
                    lambda x:
                    format_pct(
                        x,
                        4,
                    )
                ),

                "future_break_even_hit_rate": (
                    lambda x:
                    format_pct(
                        x,
                        4,
                    )
                ),

                "future_total_profit": (
                    lambda x:
                    format_number(
                        x
                    )
                ),

                "future_roi": (
                    lambda x:
                    format_pct_signed(
                        x
                    )
                ),
            },
        )
    )


    lines.append(
        ""
    )


    lines.append(
        "Fixed-threshold nested test cho thấy history performance "
        "không tái lập ổn định sang future. Một threshold tuyệt đối "
        "cũng có thể trở nên inactive khi probability scale thay đổi."
    )


    # ========================================================
    # ADAPTIVE WALK FORWARD
    # ========================================================

    lines.append(
        ""
    )


    lines.append(
        "## 8. Strategy 19 — Nested walk-forward, adaptive threshold"
    )


    lines.append(
        ""
    )


    adaptive_columns = [
        "future_fold",
        "selected_model",
        "selected_m",
        "selected_target_rate",
        "history_n_bets",
        "history_roi",
        "future_n_bets",
        "future_hits",
        "future_hit_rate",
        "future_break_even_hit_rate",
        "future_total_profit",
        "future_roi",
    ]


    lines.append(
        markdown_table(
            adaptive,

            columns=(
                adaptive_columns
            ),

            formatters={
                "selected_target_rate": (
                    lambda x:
                    format_pct(
                        x,
                        0,
                    )
                ),

                "history_roi": (
                    lambda x:
                    format_pct_signed(
                        x
                    )
                ),

                "future_hit_rate": (
                    lambda x:
                    format_pct(
                        x,
                        4,
                    )
                ),

                "future_break_even_hit_rate": (
                    lambda x:
                    format_pct(
                        x,
                        4,
                    )
                ),

                "future_total_profit": (
                    lambda x:
                    format_number(
                        x
                    )
                ),

                "future_roi": (
                    lambda x:
                    format_pct_signed(
                        x
                    )
                ),
            },
        )
    )


    # ========================================================
    # WALK COMPARISON
    # ========================================================

    lines.append(
        ""
    )


    lines.append(
        "## 9. So sánh true walk-forward"
    )


    lines.append(
        ""
    )


    walk_columns = [
        "strategy",
        "n_future_folds",
        "active_folds",
        "inactive_folds",
        "positive_active_folds",
        "negative_active_folds",
        "total_bets",
        "total_hits",
        "total_profit",
        "aggregate_roi",
        "random_expected_roi",
        "roi_vs_random",
    ]


    lines.append(
        markdown_table(
            walk_comparison,

            columns=(
                walk_columns
            ),

            formatters={
                "total_profit": (
                    lambda x:
                    format_number(
                        x
                    )
                ),

                "aggregate_roi": (
                    lambda x:
                    format_pct_signed(
                        x
                    )
                ),

                "random_expected_roi": (
                    lambda x:
                    format_pct_signed(
                        x
                    )
                ),

                "roi_vs_random": (
                    lambda x:
                    format_pct_signed(
                        x
                    )
                ),
            },
        )
    )


    # ========================================================
    # KEY FINDINGS
    # ========================================================

    lines.append(
        ""
    )


    lines.append(
        "## 10. Key findings"
    )


    lines.append(
        ""
    )


    lines.append(
        markdown_table(
            key_findings
        )
    )


    # ========================================================
    # FINAL CONCLUSION
    # ========================================================

    final_row = (
        key_findings.loc[
            key_findings[
                "section"
            ]
            .eq(
                "Final conclusion"
            )
        ]
        .iloc[0]
    )


    lines.append(
        ""
    )


    lines.append(
        "## 11. Kết luận cuối"
    )


    lines.append(
        ""
    )


    lines.append(
        f"**{final_row['value']}**"
    )


    lines.append(
        ""
    )


    lines.append(
        (
            "Các mô hình và strategy tạo ra một số giai đoạn "
            "có ranking hoặc ROI quan sát được tốt hơn random. "
            "Tuy nhiên các lợi thế này không ổn định qua thời gian, "
            "không sống sót sau multiple-testing correction và "
            "không tái lập nhất quán trong nested walk-forward."
        )
    )


    lines.append(
        ""
    )


    lines.append(
        (
            "Do đó, trên tập dữ liệu và các phương pháp đã thử, "
            "**chưa có bằng chứng đủ mạnh về một predictive/economic edge "
            "có thể khai thác bền vững**."
        )
    )


    lines.append(
        ""
    )


    lines.append(
        (
            "Kết luận này không đồng nghĩa với việc chứng minh về mặt toán học "
            "rằng quá trình xổ số hoàn toàn ngẫu nhiên; nó chỉ phản ánh rằng "
            "các cấu trúc dự báo được thử nghiệm chưa tạo ra lợi thế "
            "ổn định và xác nhận được out-of-sample."
        )
    )


    return "\n".join(
        lines
    )


# ============================================================
# PRINT FINAL
# ============================================================

def print_final_summary(
    probability,
    robustness_summary,
    walk_comparison,
):

    print(
        "\n"
        + "=" * 120
    )

    print(
        "FINAL RESEARCH SUMMARY"
    )

    print(
        "=" * 120
    )


    non_uniform = (
        probability.loc[
            probability[
                "model"
            ]
            .ne(
                "uniform"
            )
        ]
    )


    best_model = (
        non_uniform
        .sort_values(
            "log_loss"
        )
        .iloc[0]
    )


    print(
        "\n1. Probability quality"
    )


    print(
        f"   Best model       : "
        f"{best_model['model']}"
    )


    print(
        f"   Log loss         : "
        f"{best_model['log_loss']:.6f}"
    )


    print(
        f"   Gain vs Uniform  : "
        f"{best_model['log_loss_gain_vs_uniform']:+.6f}"
    )


    print(
        "\n2. Multiple testing"
    )


    print(
        f"   Configurations   : "
        f"{int(robustness_summary['total_configurations'].iloc[0]):,}"
    )


    print(
        f"   Holm vs BE       : "
        f"{int(robustness_summary['holm_significant_vs_break_even'].iloc[0])}"
    )


    print(
        f"   Strict robust    : "
        f"{int(robustness_summary['strict_robust_configs'].iloc[0])}"
    )


    print(
        "\n3. True walk-forward"
    )


    for _, row in (
        walk_comparison.iterrows()
    ):

        print(
            f"   {row['strategy']:<20} "
            f"bets={int(row['total_bets']):>4} | "
            f"profit={row['total_profit']:>12,.0f} | "
            f"ROI={format_pct_signed(row['aggregate_roi'])}"
        )


    print(
        "\n4. Research conclusion"
    )


    print(
        "   Chưa có bằng chứng đủ mạnh về predictive/economic "
        "edge có thể khai thác bền vững."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # LOAD
    # ========================================================

    data = (
        load_inputs()
    )


    # ========================================================
    # BUILD TABLES
    # ========================================================

    probability = (
        build_probability_table(
            data[
                "probability"
            ]
        )
    )


    always = (
        build_always_table(
            data[
                "always"
            ]
        )
    )


    selective = (
        build_selective_table(
            data[
                "selective"
            ]
        )
    )


    robustness_summary = (
        build_robustness_summary(
            data[
                "robustness"
            ]
        )
    )


    robustness_top = (
        build_top_robustness(
            data[
                "robustness_top"
            ]
        )
    )


    horizon = (
        build_horizon_table(
            data[
                "horizon"
            ]
        )
    )


    fixed = (
        standardize_fixed_walk_forward(
            data[
                "nested_fixed"
            ]
        )
    )


    adaptive = (
        standardize_adaptive_walk_forward(
            data[
                "nested_adaptive"
            ]
        )
    )


    # ========================================================
    # WALK-FORWARD COMPARISON
    # ========================================================

    walk_comparison = pd.DataFrame(
        [
            aggregate_walk_forward(
                fixed,
                "fixed_threshold",
            ),

            aggregate_walk_forward(
                adaptive,
                "adaptive_threshold",
            ),
        ]
    )


    # ========================================================
    # KEY FINDINGS
    # ========================================================

    key_findings = (
        build_key_findings(
            probability=(
                probability
            ),

            always=(
                always
            ),

            selective=(
                selective
            ),

            robustness=(
                robustness_summary
            ),

            fixed=(
                fixed
            ),

            adaptive=(
                adaptive
            ),

            walk_comparison=(
                walk_comparison
            ),
        )
    )


    # ========================================================
    # MARKDOWN
    # ========================================================

    markdown = (
        build_markdown_report(
            probability=(
                probability
            ),

            always=(
                always
            ),

            selective=(
                selective
            ),

            robustness_summary=(
                robustness_summary
            ),

            robustness_top=(
                robustness_top
            ),

            horizon=(
                horizon
            ),

            fixed=(
                fixed
            ),

            adaptive=(
                adaptive
            ),

            walk_comparison=(
                walk_comparison
            ),

            key_findings=(
                key_findings
            ),
        )
    )


    # ========================================================
    # SAVE CSV
    # ========================================================

    probability.to_csv(
        OUTPUT_PROBABILITY,
        index=False,
        encoding="utf-8-sig",
    )


    always.to_csv(
        OUTPUT_ALWAYS,
        index=False,
        encoding="utf-8-sig",
    )


    selective.to_csv(
        OUTPUT_SELECTIVE,
        index=False,
        encoding="utf-8-sig",
    )


    robustness_summary.to_csv(
        OUTPUT_ROBUSTNESS,
        index=False,
        encoding="utf-8-sig",
    )


    robustness_top.to_csv(
        OUTPUT_ROBUSTNESS_TOP,
        index=False,
        encoding="utf-8-sig",
    )


    horizon.to_csv(
        OUTPUT_HORIZON,
        index=False,
        encoding="utf-8-sig",
    )


    fixed.to_csv(
        OUTPUT_FIXED,
        index=False,
        encoding="utf-8-sig",
    )


    adaptive.to_csv(
        OUTPUT_ADAPTIVE,
        index=False,
        encoding="utf-8-sig",
    )


    walk_comparison.to_csv(
        OUTPUT_WALK_FORWARD_COMPARISON,
        index=False,
        encoding="utf-8-sig",
    )


    key_findings.to_csv(
        OUTPUT_KEY_FINDINGS,
        index=False,
        encoding="utf-8-sig",
    )


    # ========================================================
    # SAVE MARKDOWN
    # ========================================================

    OUTPUT_MARKDOWN.write_text(
        markdown,
        encoding="utf-8",
    )


    # ========================================================
    # PRINT
    # ========================================================

    print_final_summary(
        probability=(
            probability
        ),

        robustness_summary=(
            robustness_summary
        ),

        walk_comparison=(
            walk_comparison
        ),
    )


    print(
        "\n"
        + "=" * 100
    )

    print(
        "OUTPUT FILES"
    )

    print(
        "=" * 100
    )


    for file_path in [
        OUTPUT_PROBABILITY,
        OUTPUT_ALWAYS,
        OUTPUT_SELECTIVE,
        OUTPUT_ROBUSTNESS,
        OUTPUT_ROBUSTNESS_TOP,
        OUTPUT_HORIZON,
        OUTPUT_FIXED,
        OUTPUT_ADAPTIVE,
        OUTPUT_WALK_FORWARD_COMPARISON,
        OUTPUT_KEY_FINDINGS,
        OUTPUT_MARKDOWN,
    ]:

        print(
            file_path
        )


if __name__ == "__main__":
    main()