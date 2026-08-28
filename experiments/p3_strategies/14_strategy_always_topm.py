"""
Strategy 14 - Always Top-m

Mỗi ngày:
    1. Xếp 100 số 00-99 theo probability giảm dần.
    2. Chọn Top-m, m = 1,...,30.
    3. Mua tất cả m số.

Payoff:
    cost mỗi số = 10,000 VND
    payout khi hit = 800,000 VND

Daily:
    cost = m * 10,000

    revenue =
        800,000 nếu actual thuộc Top-m
        0 nếu không

    profit = revenue - cost

Break-even hit rate:

    q_BE(m)
        = m * 10,000 / 800,000
        = m / 80

Random Top-m:

    P(hit) = m / 100

Expected ROI random:

    [(m/100)*800,000 - m*10,000]
    --------------------------------
               m*10,000

    = -20%

với mọi m.

Models:
    - CatBoost static
    - CatBoost periodic retrain
    - CDM
    - Rolling CDM:
          30 / 60 / 90 / 180 / 365
    - Bayesian Markov

TIE BREAK
---------
Strategy cần chọn số cụ thể.

Do đó không dùng fractional tie-aware như experiment 13.

Nếu probability bằng nhau:
    sử dụng một seeded priority cố định.

Tie-break:
    - không dùng actual;
    - không thay đổi theo ngày;
    - reproducible;
    - không tạo leakage.

Output:
    artifacts/strategies/
        always_topm_daily.csv
        always_topm_summary.csv
        always_topm_by_fold.csv
        always_topm_random_baseline.csv
        always_topm_best_by_model.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
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


COST_PER_NUMBER = 10_000

PAYOUT_IF_HIT = 800_000

NUMBER_OF_CLASSES = 100

RANDOM_STATE = 42


# ============================================================
# TOP-M GRID
# ============================================================

M_VALUES = list(
    range(
        1,
        31,
    )
)


PROBABILITY_COLUMNS = [
    f"p_{number:02d}"
    for number in range(
        NUMBER_OF_CLASSES
    )
]


# ============================================================
# INPUT FILES
# ============================================================

FILES = {
    "catboost": (
        TABLE_DIR
        / "modern_ml_last2_predictions.csv"
    ),

    "catboost_retrain": (
        TABLE_DIR
        / "catboost_retrain_predictions.csv"
    ),

    "cdm": (
        TABLE_DIR
        / "cdm_baseline_predictions.csv"
    ),

    "rolling_cdm": (
        TABLE_DIR
        / "rolling_cdm_predictions.csv"
    ),

    "bayesian_markov": (
        TABLE_DIR
        / "bayesian_markov_predictions.csv"
    ),
}


# ============================================================
# OUTPUT FILES
# ============================================================

OUTPUT_DAILY = (
    STRATEGY_DIR
    / "always_topm_daily.csv"
)

OUTPUT_SUMMARY = (
    STRATEGY_DIR
    / "always_topm_summary.csv"
)

OUTPUT_BY_FOLD = (
    STRATEGY_DIR
    / "always_topm_by_fold.csv"
)

OUTPUT_RANDOM_BASELINE = (
    STRATEGY_DIR
    / "always_topm_random_baseline.csv"
)

OUTPUT_BEST_BY_MODEL = (
    STRATEGY_DIR
    / "always_topm_best_by_model.csv"
)


# ============================================================
# TIE BREAK
# ============================================================

def create_tie_break_priority():
    """
    Tạo thứ tự ưu tiên cố định cho 100 class.

    Khi probability bằng nhau:
        class có priority nhỏ hơn
        được xếp trước.

    Priority hoàn toàn độc lập actual.
    """

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    permutation = rng.permutation(
        NUMBER_OF_CLASSES
    )

    priority = np.empty(
        NUMBER_OF_CLASSES,
        dtype=int,
    )

    priority[
        permutation
    ] = np.arange(
        NUMBER_OF_CLASSES
    )

    return priority


TIE_BREAK_PRIORITY = (
    create_tie_break_priority()
)


# ============================================================
# PROBABILITY NORMALIZATION
# ============================================================

def normalize_probability_matrix(
    probabilities,
):

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    probabilities = np.clip(
        probabilities,
        0.0,
        None,
    )

    row_sum = (
        probabilities
        .sum(
            axis=1,
            keepdims=True,
        )
    )

    if np.any(
        row_sum <= 0
    ):

        raise ValueError(
            "Có probability row có tổng <= 0."
        )

    return (
        probabilities
        / row_sum
    )


# ============================================================
# RANKING
# ============================================================

def rank_probability_matrix(
    probabilities,
):
    """
    Ranking cụ thể cho strategy:

        probability giảm dần
        -> tie-break seeded cố định.

    Return:
        order[row] =
            class ở rank 1,
            class ở rank 2,
            ...
    """

    probabilities = (
        normalize_probability_matrix(
            probabilities
        )
    )

    n_rows = (
        probabilities.shape[0]
    )

    classes = np.arange(
        NUMBER_OF_CLASSES
    )

    orders = np.empty(
        probabilities.shape,
        dtype=int,
    )


    for row_index in range(
        n_rows
    ):

        orders[
            row_index
        ] = np.lexsort(
            (
                TIE_BREAK_PRIORITY[
                    classes
                ],

                -probabilities[
                    row_index
                ],
            )
        )


    return orders


# ============================================================
# VALIDATION
# ============================================================

def validate_probability_columns(
    df,
    model_name,
):

    required_columns = [
        "date",
        "fold",
        "actual",
        *PROBABILITY_COLUMNS,
    ]

    missing = [
        column
        for column
        in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{model_name} thiếu cột:\n"
            f"{missing}"
        )


# ============================================================
# LOAD ONE MODEL
# ============================================================

def load_single_model(
    model_name,
    file_path,
):

    if not file_path.exists():

        raise FileNotFoundError(
            f"Không tìm thấy:\n"
            f"{file_path}"
        )


    print(
        f"Loading: "
        f"{model_name}"
    )


    df = pd.read_csv(
        file_path,
        parse_dates=[
            "date"
        ],
    ).copy()


    validate_probability_columns(
        df,
        model_name,
    )


    # ========================================================
    # ROLLING CDM
    # ========================================================

    if (
        model_name
        == "rolling_cdm"
    ):

        if (
            "window"
            not in df.columns
        ):

            raise ValueError(
                "rolling_cdm_predictions.csv "
                "không có cột window."
            )


        df[
            "model_key"
        ] = (
            "rolling_cdm_w"
            + df[
                "window"
            ]
            .astype(int)
            .astype(str)
        )


    else:

        df[
            "model_key"
        ] = (
            model_name
        )


    return df


# ============================================================
# LOAD ALL
# ============================================================

def load_predictions():

    frames = []


    for (
        model_name,
        file_path,
    ) in FILES.items():

        frames.append(
            load_single_model(
                model_name,
                file_path,
            )
        )


    predictions = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


    predictions = (
        predictions
        .sort_values(
            [
                "model_key",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    return predictions


# ============================================================
# PREPARE MODEL ONCE
# ============================================================

def prepare_model_data(
    df,
):
    """
    Ranking chỉ tính một lần cho mỗi model.

    Sau đó m=1..30 dùng chung ranking đó.

    Ngoài order còn tạo:
        actual_rank
    để xác định hit vectorized:

        hit = actual_rank <= m
    """

    df = (
        df
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
        .copy()
    )


    probabilities = (
        df[
            PROBABILITY_COLUMNS
        ]
        .to_numpy(
            dtype=float
        )
    )


    probabilities = (
        normalize_probability_matrix(
            probabilities
        )
    )


    actual = (
        df[
            "actual"
        ]
        .to_numpy(
            dtype=int
        )
    )


    # ========================================================
    # ORDER
    # ========================================================

    order = (
        rank_probability_matrix(
            probabilities
        )
    )


    # ========================================================
    # INVERSE RANK
    #
    # rank_matrix[row, class] = rank 1..100
    # ========================================================

    n_rows = len(
        df
    )


    inverse_rank = np.empty_like(
        order
    )


    row_index = np.arange(
        n_rows
    )[
        :,
        None,
    ]


    inverse_rank[
        row_index,
        order,
    ] = np.arange(
        1,
        NUMBER_OF_CLASSES + 1,
    )


    actual_rank = (
        inverse_rank[
            np.arange(
                n_rows
            ),
            actual,
        ]
    )


    # ========================================================
    # CUMULATIVE TOP-M PROBABILITY
    # ========================================================

    sorted_probabilities = (
        np.take_along_axis(
            probabilities,
            order,
            axis=1,
        )
    )


    cumulative_probability = (
        np.cumsum(
            sorted_probabilities,
            axis=1,
        )
    )


    return {
        "df": (
            df
        ),

        "probabilities": (
            probabilities
        ),

        "actual": (
            actual
        ),

        "order": (
            order
        ),

        "actual_rank": (
            actual_rank
        ),

        "cumulative_probability": (
            cumulative_probability
        ),
    }


# ============================================================
# DRAWDOWN
# ============================================================

def calculate_max_drawdown(
    cumulative_profit,
):

    cumulative_profit = np.asarray(
        cumulative_profit,
        dtype=float,
    )


    if len(
        cumulative_profit
    ) == 0:

        return 0.0


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
# RUN ONE MODEL / M
# ============================================================

def run_strategy(
    prepared,
    model_key,
    m,
):

    df = (
        prepared[
            "df"
        ]
    )

    actual = (
        prepared[
            "actual"
        ]
    )

    order = (
        prepared[
            "order"
        ]
    )

    actual_rank = (
        prepared[
            "actual_rank"
        ]
    )

    cumulative_probability = (
        prepared[
            "cumulative_probability"
        ]
    )


    # ========================================================
    # TOP-M
    # ========================================================

    top_m = (
        order[
            :,
            :m,
        ]
    )


    # ========================================================
    # MODEL CONFIDENCE
    #
    # q_m = sum probability của Top-m
    # ========================================================

    q_m = (
        cumulative_probability[
            :,
            m - 1,
        ]
    )


    # ========================================================
    # HIT
    # ========================================================

    hit = (
        actual_rank
        <= m
    ).astype(
        np.int8
    )


    # ========================================================
    # ECONOMICS
    # ========================================================

    daily_cost = (
        m
        * COST_PER_NUMBER
    )


    cost = np.full(
        len(
            df
        ),
        daily_cost,
        dtype=np.int64,
    )


    revenue = (
        hit.astype(
            np.int64
        )
        * PAYOUT_IF_HIT
    )


    profit = (
        revenue
        - cost
    )


    cumulative_profit = (
        np.cumsum(
            profit
        )
    )


    # ========================================================
    # SELECTED NUMBERS
    #
    # Giữ để audit concrete strategy.
    # ========================================================

    selected_numbers = [
        ",".join(
            f"{number:02d}"
            for number
            in selected
        )

        for selected
        in top_m
    ]


    output = pd.DataFrame(
        {
            "date": (
                df[
                    "date"
                ]
                .to_numpy()
            ),

            "fold": (
                df[
                    "fold"
                ]
                .to_numpy()
            ),

            "model": (
                model_key
            ),

            "m": (
                m
            ),

            "actual": (
                actual
            ),

            "actual_rank": (
                actual_rank
            ),

            "q_m": (
                q_m
            ),

            "hit": (
                hit
            ),

            "cost": (
                cost
            ),

            "revenue": (
                revenue
            ),

            "profit": (
                profit
            ),

            "cumulative_profit": (
                cumulative_profit
            ),

            "selected_numbers": (
                selected_numbers
            ),
        }
    )


    return output


# ============================================================
# SUMMARY ONE CONFIG
# ============================================================

def summarize_strategy(
    daily,
):

    model = (
        daily[
            "model"
        ]
        .iloc[0]
    )


    m = int(
        daily[
            "m"
        ]
        .iloc[0]
    )


    n_days = len(
        daily
    )


    number_hits = int(
        daily[
            "hit"
        ]
        .sum()
    )


    hit_rate = (
        number_hits
        / n_days
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
        daily[
            "cost"
        ]
        .sum()
    )


    total_revenue = float(
        daily[
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
    )


    random_expected_profit_per_day = (
        random_hit_rate
        * PAYOUT_IF_HIT

        - m
        * COST_PER_NUMBER
    )


    random_expected_roi = (
        random_expected_profit_per_day
        / (
            m
            * COST_PER_NUMBER
        )
    )


    mean_q_m = float(
        daily[
            "q_m"
        ]
        .mean()
    )


    return {
        "model": (
            model
        ),

        "m": (
            m
        ),

        "n_days": (
            n_days
        ),

        "number_hits": (
            number_hits
        ),

        "hit_rate": (
            hit_rate
        ),

        "random_hit_rate": (
            random_hit_rate
        ),

        "hit_rate_lift_pp": (
            hit_rate
            - random_hit_rate
        ),

        "hit_rate_lift_ratio": (
            hit_rate
            / random_hit_rate
        ),

        "break_even_hit_rate": (
            break_even_hit_rate
        ),

        "gap_to_break_even": (
            hit_rate
            - break_even_hit_rate
        ),

        "mean_q_m": (
            mean_q_m
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

        "random_expected_roi": (
            random_expected_roi
        ),

        "roi_gain_vs_random": (
            roi
            - random_expected_roi
        ),

        "max_drawdown": (
            calculate_max_drawdown(
                daily[
                    "cumulative_profit"
                ]
                .to_numpy()
            )
        ),
    }


# ============================================================
# SUMMARY BY FOLD
# ============================================================

def build_fold_summary(
    daily_results,
):

    records = []


    grouped = (
        daily_results
        .groupby(
            [
                "model",
                "m",
                "fold",
            ],
            sort=False,
        )
    )


    for (
        model,
        m,
        fold,
    ), subset in grouped:

        n_days = len(
            subset
        )


        number_hits = int(
            subset[
                "hit"
            ]
            .sum()
        )


        hit_rate = (
            number_hits
            / n_days
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
        )


        cumulative_profit = (
            subset[
                "profit"
            ]
            .cumsum()
            .to_numpy()
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

                "fold": (
                    fold
                ),

                "n_days": (
                    n_days
                ),

                "number_hits": (
                    number_hits
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

                "gap_to_break_even": (
                    hit_rate
                    - break_even_hit_rate
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
                        cumulative_profit
                    )
                ),
            }
        )


    return pd.DataFrame(
        records
    )


# ============================================================
# RANDOM BASELINE
# ============================================================

def build_random_baseline():

    records = []


    for m in M_VALUES:

        random_hit_rate = (
            m
            / NUMBER_OF_CLASSES
        )


        break_even_hit_rate = (
            m
            * COST_PER_NUMBER
            / PAYOUT_IF_HIT
        )


        expected_cost = (
            m
            * COST_PER_NUMBER
        )


        expected_revenue = (
            random_hit_rate
            * PAYOUT_IF_HIT
        )


        expected_profit = (
            expected_revenue
            - expected_cost
        )


        expected_roi = (
            expected_profit
            / expected_cost
        )


        records.append(
            {
                "m": (
                    m
                ),

                "random_hit_rate": (
                    random_hit_rate
                ),

                "break_even_hit_rate": (
                    break_even_hit_rate
                ),

                "expected_cost_per_day": (
                    expected_cost
                ),

                "expected_revenue_per_day": (
                    expected_revenue
                ),

                "expected_profit_per_day": (
                    expected_profit
                ),

                "expected_roi": (
                    expected_roi
                ),
            }
        )


    return pd.DataFrame(
        records
    )


# ============================================================
# BEST PER MODEL
# ============================================================

def build_best_by_model(
    summary,
):
    """
    Best observed m theo ROI.

    Đây chỉ là descriptive / post-hoc.
    Không phải OOS validated strategy.
    """

    best = (
        summary
        .sort_values(
            [
                "model",
                "roi",
                "total_profit",
            ],
            ascending=[
                True,
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
            "roi",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# PRINT
# ============================================================

def print_best(
    best,
):

    columns = [
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


    print(
        "\n"
        + "=" * 190
    )

    print(
        "ALWAYS TOP-M - BEST OBSERVED CONFIGURATION BY MODEL"
    )

    print(
        "=" * 190
    )


    print(
        best[
            columns
        ]
        .to_string(
            index=False,

            formatters={
                "hit_rate": (
                    "{:.4%}".format
                ),

                "random_hit_rate": (
                    "{:.4%}".format
                ),

                "break_even_hit_rate": (
                    "{:.4%}".format
                ),

                "gap_to_break_even": (
                    "{:+.4%}".format
                ),

                "total_profit": (
                    "{:,.0f}".format
                ),

                "roi": (
                    "{:+.2%}".format
                ),

                "random_expected_roi": (
                    "{:+.2%}".format
                ),

                "roi_gain_vs_random": (
                    "{:+.2%}".format
                ),

                "max_drawdown": (
                    "{:,.0f}".format
                ),
            },
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    STRATEGY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # LOAD
    # ========================================================

    predictions = (
        load_predictions()
    )


    model_keys = (
        predictions[
            "model_key"
        ]
        .unique()
    )


    print(
        "\nModels:"
    )

    for model in model_keys:

        print(
            f"  - {model}"
        )


    print(
        f"\nM grid: "
        f"{min(M_VALUES)} -> "
        f"{max(M_VALUES)}"
    )


    print(
        f"Cost per number: "
        f"{COST_PER_NUMBER:,}"
    )


    print(
        f"Payout if hit: "
        f"{PAYOUT_IF_HIT:,}"
    )


    # ========================================================
    # RUN
    # ========================================================

    daily_frames = []

    summary_records = []


    for (
        model_key,
        subset,
    ) in predictions.groupby(
        "model_key",
        sort=False,
    ):

        print(
            "\n"
            + "=" * 90
        )

        print(
            f"MODEL: "
            f"{model_key}"
        )

        print(
            "=" * 90
        )


        prepared = (
            prepare_model_data(
                subset
            )
        )


        for m in M_VALUES:

            daily = (
                run_strategy(
                    prepared=prepared,
                    model_key=model_key,
                    m=m,
                )
            )


            summary = (
                summarize_strategy(
                    daily
                )
            )


            daily_frames.append(
                daily
            )


            summary_records.append(
                summary
            )


        print(
            f"Completed m="
            f"{min(M_VALUES)}..."
            f"{max(M_VALUES)}"
        )


    # ========================================================
    # COMBINE
    # ========================================================

    daily_results = (
        pd.concat(
            daily_frames,
            ignore_index=True,
        )
    )


    summary = (
        pd.DataFrame(
            summary_records
        )
    )


    fold_summary = (
        build_fold_summary(
            daily_results
        )
    )


    random_baseline = (
        build_random_baseline()
    )


    best_by_model = (
        build_best_by_model(
            summary
        )
    )


    # ========================================================
    # PRINT
    # ========================================================

    print_best(
        best_by_model
    )


    # ========================================================
    # SAVE
    # ========================================================

    daily_results.to_csv(
        OUTPUT_DAILY,
        index=False,
        encoding="utf-8-sig",
    )


    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )


    fold_summary.to_csv(
        OUTPUT_BY_FOLD,
        index=False,
        encoding="utf-8-sig",
    )


    random_baseline.to_csv(
        OUTPUT_RANDOM_BASELINE,
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
        OUTPUT_DAILY
    )

    print(
        OUTPUT_SUMMARY
    )

    print(
        OUTPUT_BY_FOLD
    )

    print(
        OUTPUT_RANDOM_BASELINE
    )

    print(
        OUTPUT_BEST_BY_MODEL
    )


if __name__ == "__main__":
    main()