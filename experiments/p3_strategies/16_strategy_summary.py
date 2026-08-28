"""
Strategy 16 - Robustness Analysis

Mục tiêu
--------
Strategy 15 đã quét:

    9 models
    x 30 values of m
    x 4 participation rates
    = 1,080 configurations

Nếu chỉ lấy config có ROI cao nhất sau khi xem test outcome,
ta gặp multiple-comparison / data-snooping problem.

File này kiểm tra độ robust của toàn bộ 1,080 configs.

Các kiểm định
-------------
Với mỗi config:

1. Random null:

       H0:
           p_hit = m / 100

       H1:
           p_hit > m / 100

2. Economic break-even null:

       H0:
           p_hit = m / 80

       H1:
           p_hit > m / 80

Break-even null quan trọng hơn về kinh tế vì:

    ROI > 0
        iff
    hit_rate > m / 80

Multiple testing
----------------
Áp dụng trên TOÀN BỘ configs:

    - Holm correction
    - Bonferroni correction

Holm ít bảo thủ hơn Bonferroni,
nhưng vẫn kiểm soát family-wise error rate.

Confidence interval
-------------------
Wilson 95% confidence interval cho hit rate.

Robust candidate nên đồng thời có:

    - n_bets đủ lớn;
    - ROI > 0;
    - Wilson lower > break-even;
    - Holm-adjusted p-value vs break-even < 0.05;
    - kết quả không phụ thuộc hoàn toàn vào một fold.

Lưu ý
-----
Đây vẫn chưa phải validation cuối cùng.

Nested walk-forward ở Strategy 18
mới là kiểm tra deployment-style mạnh hơn.

Input
-----
artifacts/strategies/
    selective_topm_daily.csv

Output
------
artifacts/strategies/

    selective_robustness_all.csv

    selective_robustness_by_fold.csv

    selective_robustness_top_configs.csv

    selective_robustness_best_by_model.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]


STRATEGY_DIR = (
    PROJECT_DIR
    / "artifacts"
    / "strategies"
)


INPUT_DAILY = (
    STRATEGY_DIR
    / "selective_topm_daily.csv"
)


OUTPUT_ALL = (
    STRATEGY_DIR
    / "selective_robustness_all.csv"
)


OUTPUT_BY_FOLD = (
    STRATEGY_DIR
    / "selective_robustness_by_fold.csv"
)


OUTPUT_TOP = (
    STRATEGY_DIR
    / "selective_robustness_top_configs.csv"
)


OUTPUT_BEST_BY_MODEL = (
    STRATEGY_DIR
    / "selective_robustness_best_by_model.csv"
)


COST_PER_NUMBER = 10_000

PAYOUT_IF_HIT = 800_000

NUMBER_OF_CLASSES = 100


ALPHA = 0.05

CONFIDENCE_LEVEL = 0.95


# ------------------------------------------------------------
# Không dùng để loại khỏi Holm.
#
# Chỉ dùng để đánh dấu config có sample đủ lớn
# khi diễn giải robustness.
# ------------------------------------------------------------

MIN_BETS_FOR_ROBUSTNESS = 50


TOP_N_CONFIGS = 50


# ============================================================
# VALIDATE
# ============================================================

def validate_input(
    df,
):

    required = [
        "date",
        "fold",
        "model",
        "m",
        "target_participation_rate",
        "bet",
        "hit_if_bet",
        "cost",
        "revenue",
        "profit",
    ]


    missing = [
        column
        for column
        in required
        if column not in df.columns
    ]


    if missing:

        raise ValueError(
            "selective_topm_daily.csv "
            "thiếu các cột:\n"
            f"{missing}"
        )


# ============================================================
# WILSON INTERVAL
# ============================================================

def wilson_interval(
    successes,
    trials,
    z=1.959963984540054,
):
    """
    Wilson score interval 95%.

    Không dùng normal approximation đơn giản:

        p_hat +/- 1.96 * SE

    vì nhiều config có sample nhỏ / tỷ lệ thấp.
    """

    if trials <= 0:

        return (
            np.nan,
            np.nan,
        )


    p_hat = (
        successes
        / trials
    )


    z2 = (
        z ** 2
    )


    denominator = (
        1
        + z2
        / trials
    )


    center = (
        p_hat
        + z2
        / (
            2
            * trials
        )
    ) / denominator


    margin = (
        z
        / denominator
        * np.sqrt(
            (
                p_hat
                * (
                    1
                    - p_hat
                )
                / trials
            )
            +
            (
                z2
                / (
                    4
                    * trials ** 2
                )
            )
        )
    )


    lower = max(
        0.0,
        center - margin,
    )


    upper = min(
        1.0,
        center + margin,
    )


    return (
        float(
            lower
        ),
        float(
            upper
        ),
    )


# ============================================================
# HOLM CORRECTION
# ============================================================

def holm_adjust(
    p_values,
):
    """
    Holm-Bonferroni adjustment.

    Input:
        array p-values length M

    Procedure:
        sort ascending:

            p_(1) <= ... <= p_(M)

        adjusted raw:

            (M-i+1) * p_(i)

        sau đó enforce monotonicity.

    Return:
        adjusted p-values theo original order.
    """

    p_values = np.asarray(
        p_values,
        dtype=float,
    )


    m = len(
        p_values
    )


    if m == 0:

        return np.array(
            [],
            dtype=float,
        )


    order = np.argsort(
        p_values
    )


    sorted_p = (
        p_values[
            order
        ]
    )


    adjusted_sorted = np.empty(
        m,
        dtype=float,
    )


    running_max = 0.0


    for rank in range(
        m
    ):

        multiplier = (
            m
            - rank
        )


        adjusted = (
            multiplier
            * sorted_p[
                rank
            ]
        )


        adjusted = min(
            adjusted,
            1.0,
        )


        running_max = max(
            running_max,
            adjusted,
        )


        adjusted_sorted[
            rank
        ] = (
            running_max
        )


    adjusted_original = np.empty(
        m,
        dtype=float,
    )


    adjusted_original[
        order
    ] = (
        adjusted_sorted
    )


    return adjusted_original


# ============================================================
# MAX DRAWDOWN
# ============================================================

def calculate_max_drawdown(
    profit,
):

    profit = np.asarray(
        profit,
        dtype=float,
    )


    if len(
        profit
    ) == 0:

        return 0.0


    cumulative_profit = (
        np.cumsum(
            profit
        )
    )


    equity = np.concatenate(
        [
            np.array(
                [
                    0.0
                ]
            ),

            cumulative_profit,
        ]
    )


    running_max = (
        np.maximum.accumulate(
            equity
        )
    )


    drawdown = (
        running_max
        - equity
    )


    return float(
        drawdown.max()
    )


# ============================================================
# EXACT BINOMIAL P
# ============================================================

def one_sided_binomial_p(
    hits,
    bets,
    null_probability,
):
    """
    H1:
        true hit probability > null_probability
    """

    if bets <= 0:

        return 1.0


    return float(
        binomtest(
            k=int(
                hits
            ),

            n=int(
                bets
            ),

            p=float(
                null_probability
            ),

            alternative="greater",
        ).pvalue
    )


# ============================================================
# SUMMARIZE ONE CONFIG
# ============================================================

def summarize_config(
    subset,
):
    """
    Aggregate qua tất cả test folds.
    """

    model = (
        subset[
            "model"
        ]
        .iloc[0]
    )


    m = int(
        subset[
            "m"
        ]
        .iloc[0]
    )


    target_rate = float(
        subset[
            "target_participation_rate"
        ]
        .iloc[0]
    )


    n_test_days = len(
        subset
    )


    n_bets = int(
        subset[
            "bet"
        ]
        .sum()
    )


    n_hits = int(
        (
            subset[
                "bet"
            ]
            * subset[
                "hit_if_bet"
            ]
        )
        .sum()
    )


    participation_rate = (
        n_bets
        / n_test_days
        if n_test_days > 0
        else np.nan
    )


    hit_rate = (
        n_hits
        / n_bets
        if n_bets > 0
        else np.nan
    )


    random_hit_rate = (
        m
        / NUMBER_OF_CLASSES
    )


    break_even_hit_rate = (
        m
        * COST_PER_NUMBER
        / PAYOUT_IF_HIT
    )


    total_cost = float(
        subset[
            "cost"
        ]
        .sum()
    )


    total_revenue = float(
        subset[
            "revenue"
        ]
        .sum()
    )


    total_profit = (
        total_revenue
        - total_cost
    )


    roi = (
        total_profit
        / total_cost
        if total_cost > 0
        else np.nan
    )


    (
        wilson_lower,
        wilson_upper,
    ) = (
        wilson_interval(
            successes=(
                n_hits
            ),

            trials=(
                n_bets
            ),
        )
    )


    p_random_raw = (
        one_sided_binomial_p(
            hits=(
                n_hits
            ),

            bets=(
                n_bets
            ),

            null_probability=(
                random_hit_rate
            ),
        )
    )


    p_break_even_raw = (
        one_sided_binomial_p(
            hits=(
                n_hits
            ),

            bets=(
                n_bets
            ),

            null_probability=(
                break_even_hit_rate
            ),
        )
    )


    bet_profit = (
        subset.loc[
            subset[
                "bet"
            ]
            .eq(
                1
            ),
            "profit",
        ]
        .to_numpy(
            dtype=float
        )
    )


    return {
        "model": (
            model
        ),

        "m": (
            m
        ),

        "target_participation_rate": (
            target_rate
        ),

        "n_test_days": (
            n_test_days
        ),

        "n_bets": (
            n_bets
        ),

        "participation_rate": (
            participation_rate
        ),

        "n_hits": (
            n_hits
        ),

        "hit_rate": (
            hit_rate
        ),

        "random_hit_rate": (
            random_hit_rate
        ),

        "break_even_hit_rate": (
            break_even_hit_rate
        ),

        "hit_lift_vs_random_pp": (
            (
                hit_rate
                - random_hit_rate
            )
            if not np.isnan(
                hit_rate
            )
            else np.nan
        ),

        "hit_lift_vs_break_even_pp": (
            (
                hit_rate
                - break_even_hit_rate
            )
            if not np.isnan(
                hit_rate
            )
            else np.nan
        ),

        "wilson_lower": (
            wilson_lower
        ),

        "wilson_upper": (
            wilson_upper
        ),

        "wilson_lower_above_random": (
            (
                wilson_lower
                > random_hit_rate
            )
            if not np.isnan(
                wilson_lower
            )
            else False
        ),

        "wilson_lower_above_break_even": (
            (
                wilson_lower
                > break_even_hit_rate
            )
            if not np.isnan(
                wilson_lower
            )
            else False
        ),

        "p_random_raw": (
            p_random_raw
        ),

        "p_break_even_raw": (
            p_break_even_raw
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

        "roi": (
            roi
        ),

        "max_drawdown": (
            calculate_max_drawdown(
                bet_profit
            )
        ),

        "enough_bets": (
            n_bets
            >= MIN_BETS_FOR_ROBUSTNESS
        ),
    }


# ============================================================
# APPLY MULTIPLE TESTING
# ============================================================

def apply_multiple_testing(
    summary,
):
    """
    Correction trên TOÀN BỘ configuration set.

    Không filter theo ROI, n_bets, model...
    trước correction.
    """

    summary = (
        summary
        .copy()
        .reset_index(
            drop=True
        )
    )


    number_of_tests = len(
        summary
    )


    print(
        f"\nMultiple-testing family size: "
        f"{number_of_tests:,}"
    )


    # ========================================================
    # HOLM
    # ========================================================

    summary[
        "p_random_holm"
    ] = holm_adjust(
        summary[
            "p_random_raw"
        ]
        .to_numpy(
            dtype=float
        )
    )


    summary[
        "p_break_even_holm"
    ] = holm_adjust(
        summary[
            "p_break_even_raw"
        ]
        .to_numpy(
            dtype=float
        )
    )


    # ========================================================
    # BONFERRONI
    # ========================================================

    summary[
        "p_random_bonferroni"
    ] = np.minimum(
        summary[
            "p_random_raw"
        ]
        * number_of_tests,
        1.0,
    )


    summary[
        "p_break_even_bonferroni"
    ] = np.minimum(
        summary[
            "p_break_even_raw"
        ]
        * number_of_tests,
        1.0,
    )


    # ========================================================
    # SIGNIFICANCE FLAGS
    # ========================================================

    summary[
        "sig_random_raw"
    ] = (
        summary[
            "p_random_raw"
        ]
        < ALPHA
    )


    summary[
        "sig_break_even_raw"
    ] = (
        summary[
            "p_break_even_raw"
        ]
        < ALPHA
    )


    summary[
        "sig_random_holm"
    ] = (
        summary[
            "p_random_holm"
        ]
        < ALPHA
    )


    summary[
        "sig_break_even_holm"
    ] = (
        summary[
            "p_break_even_holm"
        ]
        < ALPHA
    )


    summary[
        "sig_random_bonferroni"
    ] = (
        summary[
            "p_random_bonferroni"
        ]
        < ALPHA
    )


    summary[
        "sig_break_even_bonferroni"
    ] = (
        summary[
            "p_break_even_bonferroni"
        ]
        < ALPHA
    )


    # ========================================================
    # ECONOMIC POSITIVE
    # ========================================================

    summary[
        "positive_profit"
    ] = (
        summary[
            "total_profit"
        ]
        > 0
    )


    summary[
        "positive_roi"
    ] = (
        summary[
            "roi"
        ]
        > 0
    )


    # ========================================================
    # STRICT ROBUST FLAG
    # ========================================================

    summary[
        "strict_robust"
    ] = (
        summary[
            "enough_bets"
        ]

        & summary[
            "positive_roi"
        ]

        & summary[
            "wilson_lower_above_break_even"
        ]

        & summary[
            "sig_break_even_holm"
        ]
    )


    return summary


# ============================================================
# BY FOLD
# ============================================================

def build_fold_summary(
    daily,
):

    records = []


    grouped = (
        daily
        .groupby(
            [
                "model",
                "m",
                "target_participation_rate",
                "fold",
            ],
            sort=False,
        )
    )


    for (
        model,
        m,
        target_rate,
        fold,
    ), subset in grouped:

        n_days = len(
            subset
        )


        n_bets = int(
            subset[
                "bet"
            ]
            .sum()
        )


        n_hits = int(
            (
                subset[
                    "bet"
                ]
                * subset[
                    "hit_if_bet"
                ]
            )
            .sum()
        )


        hit_rate = (
            n_hits
            / n_bets
            if n_bets > 0
            else np.nan
        )


        random_hit_rate = (
            int(
                m
            )
            / NUMBER_OF_CLASSES
        )


        break_even_hit_rate = (
            int(
                m
            )
            * COST_PER_NUMBER
            / PAYOUT_IF_HIT
        )


        total_cost = float(
            subset[
                "cost"
            ]
            .sum()
        )


        total_revenue = float(
            subset[
                "revenue"
            ]
            .sum()
        )


        total_profit = (
            total_revenue
            - total_cost
        )


        roi = (
            total_profit
            / total_cost
            if total_cost > 0
            else np.nan
        )


        (
            wilson_lower,
            wilson_upper,
        ) = (
            wilson_interval(
                successes=(
                    n_hits
                ),

                trials=(
                    n_bets
                ),
            )
        )


        p_random = (
            one_sided_binomial_p(
                hits=(
                    n_hits
                ),

                bets=(
                    n_bets
                ),

                null_probability=(
                    random_hit_rate
                ),
            )
        )


        p_break_even = (
            one_sided_binomial_p(
                hits=(
                    n_hits
                ),

                bets=(
                    n_bets
                ),

                null_probability=(
                    break_even_hit_rate
                ),
            )
        )


        records.append(
            {
                "model": (
                    model
                ),

                "m": (
                    int(
                        m
                    )
                ),

                "target_participation_rate": (
                    float(
                        target_rate
                    )
                ),

                "fold": (
                    fold
                ),

                "n_test_days": (
                    n_days
                ),

                "n_bets": (
                    n_bets
                ),

                "n_hits": (
                    n_hits
                ),

                "hit_rate": (
                    hit_rate
                ),

                "random_hit_rate": (
                    random_hit_rate
                ),

                "break_even_hit_rate": (
                    break_even_hit_rate
                ),

                "wilson_lower": (
                    wilson_lower
                ),

                "wilson_upper": (
                    wilson_upper
                ),

                "p_random_raw": (
                    p_random
                ),

                "p_break_even_raw": (
                    p_break_even
                ),

                "total_profit": (
                    total_profit
                ),

                "roi": (
                    roi
                ),

                "positive_roi": (
                    (
                        roi > 0
                    )
                    if not np.isnan(
                        roi
                    )
                    else False
                ),
            }
        )


    return pd.DataFrame(
        records
    )


# ============================================================
# TEMPORAL ROBUSTNESS
# ============================================================

def add_temporal_robustness(
    summary,
    fold_summary,
):
    """
    Đếm số fold:
        - có bet
        - ROI dương
        - ROI âm

    Không yêu cầu mọi fold positive,
    nhưng cung cấp thông tin stability.
    """

    records = []


    grouped = (
        fold_summary
        .groupby(
            [
                "model",
                "m",
                "target_participation_rate",
            ],
            sort=False,
        )
    )


    for (
        model,
        m,
        target_rate,
    ), subset in grouped:

        folds_with_bets = int(
            subset[
                "n_bets"
            ]
            .gt(
                0
            )
            .sum()
        )


        positive_folds = int(
            (
                subset[
                    "n_bets"
                ]
                .gt(
                    0
                )

                & subset[
                    "roi"
                ]
                .gt(
                    0
                )
            )
            .sum()
        )


        negative_folds = int(
            (
                subset[
                    "n_bets"
                ]
                .gt(
                    0
                )

                & subset[
                    "roi"
                ]
                .lt(
                    0
                )
            )
            .sum()
        )


        zero_bet_folds = int(
            subset[
                "n_bets"
            ]
            .eq(
                0
            )
            .sum()
        )


        roi_values = (
            subset.loc[
                subset[
                    "n_bets"
                ]
                .gt(
                    0
                ),
                "roi",
            ]
            .dropna()
        )


        if len(
            roi_values
        ) > 0:

            min_fold_roi = float(
                roi_values.min()
            )

            max_fold_roi = float(
                roi_values.max()
            )

            median_fold_roi = float(
                roi_values.median()
            )

        else:

            min_fold_roi = (
                np.nan
            )

            max_fold_roi = (
                np.nan
            )

            median_fold_roi = (
                np.nan
            )


        records.append(
            {
                "model": (
                    model
                ),

                "m": (
                    int(
                        m
                    )
                ),

                "target_participation_rate": (
                    float(
                        target_rate
                    )
                ),

                "n_folds": (
                    len(
                        subset
                    )
                ),

                "folds_with_bets": (
                    folds_with_bets
                ),

                "positive_folds": (
                    positive_folds
                ),

                "negative_folds": (
                    negative_folds
                ),

                "zero_bet_folds": (
                    zero_bet_folds
                ),

                "min_fold_roi": (
                    min_fold_roi
                ),

                "median_fold_roi": (
                    median_fold_roi
                ),

                "max_fold_roi": (
                    max_fold_roi
                ),
            }
        )


    temporal = pd.DataFrame(
        records
    )


    output = (
        summary
        .merge(
            temporal,
            on=[
                "model",
                "m",
                "target_participation_rate",
            ],
            how="left",
        )
    )


    output[
        "all_active_folds_positive"
    ] = (
        (
            output[
                "folds_with_bets"
            ]
            > 0
        )

        & (
            output[
                "negative_folds"
            ]
            == 0
        )
    )


    output[
        "positive_fold_share"
    ] = np.where(
        output[
            "folds_with_bets"
        ]
        > 0,

        (
            output[
                "positive_folds"
            ]
            / output[
                "folds_with_bets"
            ]
        ),

        np.nan,
    )


    return output


# ============================================================
# BUILD TOP TABLE
# ============================================================

def build_top_configs(
    summary,
):
    """
    Ranking descriptive.

    Ưu tiên:
        1. p_break_even_holm thấp
        2. Wilson lower cao hơn BE
        3. ROI
        4. n_bets

    Đây không phải model selection cho future deployment.
    """

    output = (
        summary
        .copy()
    )


    output[
        "wilson_margin_vs_break_even"
    ] = (
        output[
            "wilson_lower"
        ]
        - output[
            "break_even_hit_rate"
        ]
    )


    output = (
        output
        .sort_values(
            [
                "p_break_even_holm",
                "wilson_margin_vs_break_even",
                "roi",
                "n_bets",
            ],
            ascending=[
                True,
                False,
                False,
                False,
            ],
        )
        .head(
            TOP_N_CONFIGS
        )
        .reset_index(
            drop=True
        )
    )


    return output


# ============================================================
# BEST BY MODEL
# ============================================================

def build_best_by_model(
    summary,
):
    """
    Mỗi model lấy config tốt nhất theo robustness ranking.

    Không có nghĩa đây là config được phép chọn
    trong nested deployment.
    """

    temp = (
        summary
        .copy()
    )


    temp[
        "wilson_margin_vs_break_even"
    ] = (
        temp[
            "wilson_lower"
        ]
        - temp[
            "break_even_hit_rate"
        ]
    )


    best = (
        temp
        .sort_values(
            [
                "model",
                "p_break_even_holm",
                "wilson_margin_vs_break_even",
                "roi",
                "n_bets",
            ],
            ascending=[
                True,
                True,
                False,
                False,
                False,
            ],
        )
        .groupby(
            "model",
            as_index=False,
            sort=False,
        )
        .first()
    )


    return (
        best
        .sort_values(
            [
                "p_break_even_holm",
                "roi",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_global_summary(
    summary,
):

    number_configs = len(
        summary
    )


    raw_random = int(
        summary[
            "sig_random_raw"
        ]
        .sum()
    )


    raw_break_even = int(
        summary[
            "sig_break_even_raw"
        ]
        .sum()
    )


    holm_random = int(
        summary[
            "sig_random_holm"
        ]
        .sum()
    )


    holm_break_even = int(
        summary[
            "sig_break_even_holm"
        ]
        .sum()
    )


    bonf_random = int(
        summary[
            "sig_random_bonferroni"
        ]
        .sum()
    )


    bonf_break_even = int(
        summary[
            "sig_break_even_bonferroni"
        ]
        .sum()
    )


    strict = int(
        summary[
            "strict_robust"
        ]
        .sum()
    )


    print(
        "\n"
        + "=" * 100
    )

    print(
        "MULTIPLE TESTING SUMMARY"
    )

    print(
        "=" * 100
    )


    print(
        f"Configurations: "
        f"{number_configs:,}"
    )


    print(
        f"Raw p < {ALPHA:.2f} vs random: "
        f"{raw_random:,}"
    )


    print(
        f"Raw p < {ALPHA:.2f} vs break-even: "
        f"{raw_break_even:,}"
    )


    print(
        f"Holm significant vs random: "
        f"{holm_random:,}"
    )


    print(
        f"Holm significant vs break-even: "
        f"{holm_break_even:,}"
    )


    print(
        f"Bonferroni significant vs random: "
        f"{bonf_random:,}"
    )


    print(
        f"Bonferroni significant vs break-even: "
        f"{bonf_break_even:,}"
    )


    print(
        f"Strict robust configs: "
        f"{strict:,}"
    )


# ============================================================
# PRINT TOP
# ============================================================

def print_top_configs(
    top,
):

    columns = [
        "model",
        "m",
        "target_participation_rate",
        "n_bets",
        "n_hits",
        "hit_rate",
        "random_hit_rate",
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


    print(
        "\n"
        + "=" * 240
    )


    print(
        "TOP ROBUSTNESS CONFIGURATIONS"
    )


    print(
        "=" * 240
    )


    print(
        top[
            columns
        ]
        .head(
            30
        )
        .to_string(
            index=False,

            formatters={
                "target_participation_rate": (
                    "{:.0%}".format
                ),

                "hit_rate": (
                    "{:.4%}".format
                ),

                "random_hit_rate": (
                    "{:.4%}".format
                ),

                "break_even_hit_rate": (
                    "{:.4%}".format
                ),

                "wilson_lower": (
                    "{:.4%}".format
                ),

                "wilson_upper": (
                    "{:.4%}".format
                ),

                "p_random_raw": (
                    "{:.6g}".format
                ),

                "p_random_holm": (
                    "{:.6g}".format
                ),

                "p_break_even_raw": (
                    "{:.6g}".format
                ),

                "p_break_even_holm": (
                    "{:.6g}".format
                ),

                "total_profit": (
                    "{:,.0f}".format
                ),

                "roi": (
                    "{:+.2%}".format
                ),
            },
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT_DAILY.exists():

        raise FileNotFoundError(
            "Không tìm thấy:\n"
            f"{INPUT_DAILY}\n\n"
            "Hãy chạy Strategy 15 trước."
        )


    # ========================================================
    # LOAD
    # ========================================================

    daily = pd.read_csv(
        INPUT_DAILY,
        parse_dates=[
            "date"
        ],
    )


    validate_input(
        daily
    )


    print(
        f"Daily rows: "
        f"{len(daily):,}"
    )


    print(
        "Models:"
    )


    for model in (
        daily[
            "model"
        ]
        .drop_duplicates()
    ):

        print(
            f"  - {model}"
        )


    # ========================================================
    # CONFIG SUMMARY
    # ========================================================

    records = []


    grouped = (
        daily
        .groupby(
            [
                "model",
                "m",
                "target_participation_rate",
            ],
            sort=False,
        )
    )


    for (
        model,
        m,
        target_rate,
    ), subset in grouped:

        records.append(
            summarize_config(
                subset
            )
        )


    summary = pd.DataFrame(
        records
    )


    print(
        f"\nConfigurations found: "
        f"{len(summary):,}"
    )


    # Expected:
    # 9 * 30 * 4 = 1080
    expected_configs = (
        daily[
            "model"
        ]
        .nunique()

        * daily[
            "m"
        ]
        .nunique()

        * daily[
            "target_participation_rate"
        ]
        .nunique()
    )


    print(
        f"Expected configurations: "
        f"{expected_configs:,}"
    )


    if (
        len(
            summary
        )
        != expected_configs
    ):

        print(
            "WARNING: số config thực tế "
            "khác Cartesian product dự kiến."
        )


    # ========================================================
    # MULTIPLE TESTING
    # ========================================================

    summary = (
        apply_multiple_testing(
            summary
        )
    )


    # ========================================================
    # FOLD SUMMARY
    # ========================================================

    fold_summary = (
        build_fold_summary(
            daily
        )
    )


    # ========================================================
    # TEMPORAL ROBUSTNESS
    # ========================================================

    summary = (
        add_temporal_robustness(
            summary,
            fold_summary,
        )
    )


    # ========================================================
    # TABLES
    # ========================================================

    top_configs = (
        build_top_configs(
            summary
        )
    )


    best_by_model = (
        build_best_by_model(
            summary
        )
    )


    # ========================================================
    # PRINT
    # ========================================================

    print_global_summary(
        summary
    )


    print_top_configs(
        top_configs
    )


    # ========================================================
    # SAVE
    # ========================================================

    summary.to_csv(
        OUTPUT_ALL,
        index=False,
        encoding="utf-8-sig",
    )


    fold_summary.to_csv(
        OUTPUT_BY_FOLD,
        index=False,
        encoding="utf-8-sig",
    )


    top_configs.to_csv(
        OUTPUT_TOP,
        index=False,
        encoding="utf-8-sig",
    )


    best_by_model.to_csv(
        OUTPUT_BEST_BY_MODEL,
        index=False,
        encoding="utf-8-sig",
    )


    print(
        "\nĐã lưu:"
    )


    print(
        OUTPUT_ALL
    )


    print(
        OUTPUT_BY_FOLD
    )


    print(
        OUTPUT_TOP
    )


    print(
        OUTPUT_BEST_BY_MODEL
    )


if __name__ == "__main__":
    main()