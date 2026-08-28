"""
Strategy 14: Always Top-m

Mỗi ngày:
    - Xếp 100 số 00-99 theo probability giảm dần.
    - Với mỗi m từ 10 đến 30:
        mua toàn bộ Top-m số.
    - Giá mỗi số: 10,000 VND.
    - Nếu actual nằm trong Top-m:
        payout = 800,000 VND.
    - Nếu không:
        payout = 0.

Profit mỗi ngày:
    profit = revenue - cost

Trong đó:
    cost = m * 10,000

Break-even hit rate:
    q_BE(m) = m * 10,000 / 800,000
            = m / 80

Random Top-m baseline:
    hit rate = m / 100

Expected random ROI:
    ((m/100)*800,000 - m*10,000)
    / (m*10,000)

Với payoff hiện tại:
    Random expected ROI = -20%
    với mọi m.
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

# Quét toàn bộ Top-1 -> Top-30
M_VALUES = list(range(1, 31))

RANDOM_STATE = 42


PROBABILITY_COLUMNS = [
    f"p_{number:02d}"
    for number in range(NUMBER_OF_CLASSES)
]


FILES = {
    "catboost": (
        TABLE_DIR
        / "modern_ml_last2_predictions.csv"
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
# TIE BREAK
# ============================================================

def create_tie_break_priority() -> np.ndarray:
    """
    Tạo thứ tự ưu tiên cố định cho 100 class.

    Mục đích:
        Khi nhiều số có probability bằng nhau,
        vẫn phải chọn ra Top-m cụ thể.

    Tie-break này:
        - cố định;
        - không phụ thuộc actual;
        - reproducible.
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


def rank_probability_matrix(
    probabilities: np.ndarray,
) -> np.ndarray:
    """
    Xếp hạng 100 số cho từng ngày.

    Thứ tự:
        1. probability giảm dần;
        2. nếu bằng nhau -> tie-break cố định.

    Return:
        shape = (n_days, 100)

    Mỗi row chứa class index theo thứ tự:
        rank 1, rank 2, ..., rank 100.
    """

    classes = np.arange(
        NUMBER_OF_CLASSES
    )

    orders = np.empty(
        probabilities.shape,
        dtype=int,
    )

    for row_index in range(
        len(probabilities)
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
# LOAD DATA
# ============================================================

def validate_probability_columns(
    df: pd.DataFrame,
    model_name: str,
) -> None:

    required_columns = [
        "date",
        "fold",
        "actual",
    ] + PROBABILITY_COLUMNS

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"\n{model_name} thiếu các cột:\n"
            f"{missing_columns}"
        )


def load_single_model(
    model_name: str,
    file_path: Path,
) -> pd.DataFrame:

    if not file_path.exists():

        raise FileNotFoundError(
            f"\nKhông tìm thấy file:\n"
            f"{file_path}"
        )

    df = pd.read_csv(
        file_path,
        parse_dates=["date"],
    ).copy()

    validate_probability_columns(
        df,
        model_name,
    )

    # Rolling CDM chứa nhiều window
    if model_name == "rolling_cdm":

        if "window" not in df.columns:

            raise ValueError(
                "rolling_cdm_predictions.csv "
                "không có cột window."
            )

        model_key = (
            "rolling_cdm_w"
            + df["window"]
            .astype(int)
            .astype(str)
        )

    else:

        model_key = pd.Series(
            model_name,
            index=df.index,
        )

    df = pd.concat(
        [
            df,
            model_key.rename(
                "model_key"
            ),
        ],
        axis=1,
    )

    return df


def load_predictions() -> pd.DataFrame:

    frames = []

    for (
        model_name,
        file_path,
    ) in FILES.items():

        print(
            f"Loading: {model_name}"
        )

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
# PREPARE MODEL RANKING
# ============================================================

def prepare_model_data(
    df: pd.DataFrame,
) -> dict:
    """
    Ranking chỉ tính một lần cho mỗi model.

    Sau đó dùng lại cho m=10..30,
    tránh sort lại 21 lần.
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

    actual = (
        df[
            "actual"
        ]
        .to_numpy(
            dtype=int
        )
    )

    order = (
        rank_probability_matrix(
            probabilities
        )
    )

    sorted_probabilities = (
        np.take_along_axis(
            probabilities,
            order,
            axis=1,
        )
    )

    # cumulative probability:
    # column 0 = q_1
    # column 9 = q_10
    # column 19 = q_20
    cumulative_probability = (
        np.cumsum(
            sorted_probabilities,
            axis=1,
        )
    )

    return {
        "df": df,
        "actual": actual,
        "order": order,
        "cumulative_probability": (
            cumulative_probability
        ),
    }


# ============================================================
# DRAWDOWN
# ============================================================

def calculate_max_drawdown(
    cumulative_profit: np.ndarray,
) -> float:

    if len(
        cumulative_profit
    ) == 0:

        return 0.0

    equity = np.concatenate(
        [
            np.array(
                [0.0]
            ),

            cumulative_profit.astype(
                float
            ),
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
# RUN ONE M
# ============================================================

def run_strategy(
    prepared: dict,
    model_key: str,
    m: int,
) -> pd.DataFrame:

    df = prepared["df"]

    actual = prepared[
        "actual"
    ]

    order = prepared[
        "order"
    ]

    cumulative_probability = (
        prepared[
            "cumulative_probability"
        ]
    )

    # Top-m số
    top_m = (
        order[
            :, :m
        ]
    )

    # Tổng probability của Top-m
    q_m = (
        cumulative_probability[
            :, m - 1
        ]
    )

    # Actual có nằm trong Top-m không?
    hit = np.array(
        [
            int(
                actual_value
                in selected_numbers
            )

            for (
                actual_value,
                selected_numbers,
            ) in zip(
                actual,
                top_m,
            )
        ],
        dtype=int,
    )

    daily_cost = (
        m
        * COST_PER_NUMBER
    )

    cost = np.full(
        len(df),
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

    selected_numbers = [
        ",".join(
            f"{number:02d}"
            for number in selected
        )

        for selected in top_m
    ]

    output = pd.DataFrame(
        {
            "date": (
                df["date"]
                .to_numpy()
            ),

            "fold": (
                df["fold"]
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
# SUMMARY
# ============================================================

def summarize_strategy(
    daily: pd.DataFrame,
) -> dict:

    model = (
        daily[
            "model"
        ].iloc[0]
    )

    m = int(
        daily[
            "m"
        ].iloc[0]
    )

    n_days = len(
        daily
    )

    number_hits = int(
        daily[
            "hit"
        ].sum()
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
        ].sum()
    )

    total_revenue = float(
        daily[
            "revenue"
        ].sum()
    )

    total_profit = float(
        daily[
            "profit"
        ].sum()
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
        ].mean()
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
    daily_results: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    grouped = (
        daily_results
        .groupby(
            [
                "model",
                "m",
                "fold",
            ]
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
            ].sum()
        )

        hit_rate = (
            number_hits
            / n_days
        )

        total_cost = float(
            subset[
                "cost"
            ].sum()
        )

        total_revenue = float(
            subset[
                "revenue"
            ].sum()
        )

        total_profit = (
            total_revenue
            - total_cost
        )

        roi = (
            total_profit
            / total_cost
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

        records.append(
            {
                "model": (
                    model
                ),

                "m": (
                    int(m)
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
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# RANDOM BASELINE
# ============================================================

def build_random_baseline() -> pd.DataFrame:

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

        daily_cost = (
            m
            * COST_PER_NUMBER
        )

        expected_revenue = (
            random_hit_rate
            * PAYOUT_IF_HIT
        )

        expected_profit = (
            expected_revenue
            - daily_cost
        )

        expected_roi = (
            expected_profit
            / daily_cost
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
                    daily_cost
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
# BEST CONFIG PER MODEL
# ============================================================

def get_best_configuration_per_model(
    summary: pd.DataFrame,
) -> pd.DataFrame:

    idx = (
        summary
        .groupby(
            "model"
        )[
            "roi"
        ]
        .idxmax()
    )

    best = (
        summary
        .loc[
            idx
        ]
        .sort_values(
            "roi",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return best


# ============================================================
# PRINT
# ============================================================

def print_full_summary(
    summary: pd.DataFrame,
) -> None:

    columns = [
        "model",
        "m",
        "n_days",
        "number_hits",
        "hit_rate",
        "random_hit_rate",
        "break_even_hit_rate",
        "mean_q_m",
        "total_profit",
        "roi",
        "random_expected_roi",
        "roi_gain_vs_random",
    ]

    printable = (
        summary[
            columns
        ]
        .sort_values(
            [
                "model",
                "m",
            ]
        )
    )

    print(
        "\n"
        + "=" * 185
    )

    print(
        "STRATEGY 14 - ALWAYS TOP-m"
    )

    print(
        "m = 10 ... 30"
    )

    print(
        "=" * 185
    )

    print(
        printable.to_string(
            index=False,
            formatters={
                "hit_rate": (
                    "{:.4%}".format
                ),

                "random_hit_rate": (
                    "{:.2%}".format
                ),

                "break_even_hit_rate": (
                    "{:.2%}".format
                ),

                "mean_q_m": (
                    "{:.4%}".format
                ),

                "total_profit": (
                    "{:,.0f}".format
                ),

                "roi": (
                    "{:.2%}".format
                ),

                "random_expected_roi": (
                    "{:.2%}".format
                ),

                "roi_gain_vs_random": (
                    "{:+.2%}".format
                ),
            },
        )
    )


def print_best_summary(
    best: pd.DataFrame,
) -> None:

    columns = [
        "model",
        "m",
        "number_hits",
        "hit_rate",
        "random_hit_rate",
        "break_even_hit_rate",
        "gap_to_break_even",
        "total_profit",
        "roi",
        "roi_gain_vs_random",
        "max_drawdown",
    ]

    print(
        "\n"
        + "=" * 175
    )

    print(
        "BEST m THEO TỪNG MODEL"
    )

    print(
        "=" * 175
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
                    "{:.2%}".format
                ),

                "break_even_hit_rate": (
                    "{:.2%}".format
                ),

                "gap_to_break_even": (
                    "{:+.2%}".format
                ),

                "total_profit": (
                    "{:,.0f}".format
                ),

                "roi": (
                    "{:.2%}".format
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

    predictions = (
        load_predictions()
    )

    print(
        f"\nPrediction rows: "
        f"{len(predictions):,}"
    )

    print(
        f"M range: "
        f"{M_VALUES[0]} -> {M_VALUES[-1]}"
    )

    print(
        f"Cost per number: "
        f"{COST_PER_NUMBER:,}"
    )

    print(
        f"Payout if hit: "
        f"{PAYOUT_IF_HIT:,}"
    )

    daily_frames = []

    summary_records = []

    model_groups = (
        predictions
        .groupby(
            "model_key"
        )
    )

    for (
        model_key,
        model_df,
    ) in model_groups:

        print(
            f"\nRunning model: "
            f"{model_key}"
        )

        prepared = (
            prepare_model_data(
                model_df
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

            daily_frames.append(
                daily
            )

            summary_records.append(
                summarize_strategy(
                    daily
                )
            )

    daily_results = pd.concat(
        daily_frames,
        ignore_index=True,
    )

    summary = pd.DataFrame(
        summary_records
    )

    by_fold = (
        build_fold_summary(
            daily_results
        )
    )

    random_baseline = (
        build_random_baseline()
    )

    best_by_model = (
        get_best_configuration_per_model(
            summary
        )
    )

    # ========================================================
    # PRINT
    # ========================================================

    print_full_summary(
        summary
    )

    print_best_summary(
        best_by_model
    )

    # ========================================================
    # SAVE
    # ========================================================

    daily_path = (
        STRATEGY_DIR
        / "always_topm_daily.csv"
    )

    summary_path = (
        STRATEGY_DIR
        / "always_topm_summary.csv"
    )

    fold_path = (
        STRATEGY_DIR
        / "always_topm_by_fold.csv"
    )

    random_path = (
        STRATEGY_DIR
        / "always_topm_random_baseline.csv"
    )

    best_path = (
        STRATEGY_DIR
        / "always_topm_best_by_model.csv"
    )

    daily_results.to_csv(
        daily_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    by_fold.to_csv(
        fold_path,
        index=False,
        encoding="utf-8-sig",
    )

    random_baseline.to_csv(
        random_path,
        index=False,
        encoding="utf-8-sig",
    )

    best_by_model.to_csv(
        best_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nĐã lưu:"
    )

    print(
        daily_path
    )

    print(
        summary_path
    )

    print(
        fold_path
    )

    print(
        random_path
    )

    print(
        best_path
    )


if __name__ == "__main__":
    main()