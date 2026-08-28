"""
18 - NESTED WALK-FORWARD STRATEGY

Mục tiêu
--------
Không nhìn kết quả tương lai để chọn strategy.

Với mỗi outer test fold:

    Test 2022-2023:
        dùng kết quả OOS 2020-2021 làm HISTORY.

    Test 2024-2026:
        dùng kết quả OOS 2020-2021 + 2022-2023 làm HISTORY.

Trên HISTORY:
    - thử tất cả model
    - m = 1..30
    - r = 10%, 20%, 30%, 50%
    - tính confidence:
          q_m = sum probability Top-m
    - xác định threshold từ HISTORY
    - backtest từng config trên HISTORY
    - chọn config tốt nhất

Sau đó FREEZE:
    model
    m
    r
    confidence threshold
    tie acceptance rate

và áp dụng nguyên xi sang FUTURE TEST FOLD.

Không dùng actual của future test để:
    - chọn model
    - chọn m
    - chọn r
    - chọn threshold

Payoff:
    10,000 VND / số
    payout hit = 800,000 VND

Random expected ROI = -20%.

IMPORTANT
---------
Kết quả file này mới gần với deployment thực tế hơn Strategy 15,
vì config được chọn trước khi nhìn kết quả fold tương lai.
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

NUMBER_OF_CLASSES = 100

COST_PER_NUMBER = 10_000
PAYOUT_IF_HIT = 800_000

M_VALUES = list(
    range(
        1,
        31,
    )
)

TARGET_PARTICIPATION_RATES = [
    0.10,
    0.20,
    0.30,
    0.50,
]

RANDOM_STATE = 42


# ------------------------------------------------------------
# Tránh chọn config chỉ có vài bet rồi may mắn
# ------------------------------------------------------------

MIN_HISTORY_BETS = 50


# ------------------------------------------------------------
# Selection method
#
# "max_roi":
#       config ROI history cao nhất
#
# "robust_roi":
#       ưu tiên ROI nhưng phạt config có ít bet
#
# robust_score =
#       ROI * sqrt(n_bet / max_n_bet)
# ------------------------------------------------------------

SELECTION_METHOD = (
    "robust_roi"
)


PROBABILITY_COLUMNS = [
    f"p_{number:02d}"
    for number in range(
        NUMBER_OF_CLASSES
    )
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


# ------------------------------------------------------------
# Outer walk-forward folds
# ------------------------------------------------------------

OUTER_FOLDS = [
    {
        "test_fold": (
            "2022-2023"
        ),

        "history_folds": [
            "2020-2021",
        ],
    },

    {
        "test_fold": (
            "2024-2026"
        ),

        "history_folds": [
            "2020-2021",
            "2022-2023",
        ],
    },
]


# ------------------------------------------------------------
# Outputs
# ------------------------------------------------------------

OUTPUT_SELECTION = (
    STRATEGY_DIR
    / "nested_strategy_selection.csv"
)

OUTPUT_TEST = (
    STRATEGY_DIR
    / "nested_strategy_test_results.csv"
)

OUTPUT_DAILY = (
    STRATEGY_DIR
    / "nested_strategy_test_daily.csv"
)

OUTPUT_ALL_HISTORY_CONFIGS = (
    STRATEGY_DIR
    / "nested_strategy_history_configs.csv"
)

OUTPUT_SUMMARY = (
    STRATEGY_DIR
    / "nested_strategy_summary.csv"
)


# ============================================================
# TIE BREAK FOR NUMBER RANKING
# ============================================================

def create_tie_break_priority():

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
    probabilities,
):

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
# LOAD
# ============================================================

def validate_probability_columns(
    df,
    model_name,
):

    required = [
        "date",
        "fold",
        "actual",
    ] + PROBABILITY_COLUMNS

    missing = [
        column
        for column
        in required
        if column
        not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{model_name} thiếu:\n"
            f"{missing}"
        )


def load_single_model(
    model_name,
    file_path,
):

    if not file_path.exists():

        raise FileNotFoundError(
            file_path
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

    if model_name == "rolling_cdm":

        if "window" not in df.columns:

            raise ValueError(
                "rolling_cdm thiếu window"
            )

        model_key = (
            "rolling_cdm_w"
            + df[
                "window"
            ]
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


def load_predictions():

    frames = []

    for (
        model_name,
        file_path,
    ) in FILES.items():

        print(
            f"Loading: "
            f"{model_name}"
        )

        frames.append(
            load_single_model(
                model_name,
                file_path,
            )
        )

    result = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    return (
        result
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


# ============================================================
# PREPARE MODEL
# ============================================================

def prepare_model_data(
    df,
):

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

        "actual": (
            actual
        ),

        "order": (
            order
        ),

        "cum_prob": (
            cumulative_probability
        ),
    }


# ============================================================
# BUILD TOP-m DATA
# ============================================================

def build_topm_data(
    prepared,
    m,
):

    df = prepared[
        "df"
    ]

    actual = prepared[
        "actual"
    ]

    order = prepared[
        "order"
    ]

    cumulative_probability = (
        prepared[
            "cum_prob"
        ]
    )

    top_m = (
        order[
            :, :m
        ]
    )

    q_m = (
        cumulative_probability[
            :, m - 1
        ]
    )

    hit = np.array(
        [
            int(
                actual_value
                in selected
            )

            for (
                actual_value,
                selected,
            ) in zip(
                actual,
                top_m,
            )
        ],
        dtype=int,
    )

    return pd.DataFrame(
        {
            "date": (
                df[
                    "date"
                ].to_numpy()
            ),

            "fold": (
                df[
                    "fold"
                ].to_numpy()
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

            "selected_numbers": [
                ",".join(
                    f"{number:02d}"
                    for number
                    in selected
                )

                for selected
                in top_m
            ],
        }
    )


# ============================================================
# CONFIDENCE RULE
# ============================================================

def calculate_confidence_rule(
    q_values,
    target_rate,
):

    q = np.asarray(
        q_values,
        dtype=float,
    )

    if len(q) == 0:

        return (
            np.nan,
            0.0,
        )

    if np.allclose(
        q,
        q[0],
        rtol=1e-12,
        atol=1e-15,
    ):

        return (
            np.nan,
            0.0,
        )

    threshold = float(
        np.quantile(
            q,
            1
            - target_rate,
        )
    )

    target_count = (
        target_rate
        * len(q)
    )

    greater_count = int(
        np.sum(
            q
            > threshold
        )
    )

    equal_mask = (
        np.isclose(
            q,
            threshold,
            rtol=1e-12,
            atol=1e-15,
        )
    )

    equal_count = int(
        equal_mask.sum()
    )

    needed_from_ties = (
        target_count
        - greater_count
    )

    if equal_count > 0:

        tie_acceptance_rate = (
            needed_from_ties
            / equal_count
        )

    else:

        tie_acceptance_rate = (
            0.0
        )

    tie_acceptance_rate = float(
        np.clip(
            tie_acceptance_rate,
            0,
            1,
        )
    )

    return (
        threshold,
        tie_acceptance_rate,
    )


def apply_confidence_rule(
    q_values,
    threshold,
    tie_acceptance_rate,
    seed,
):

    q = np.asarray(
        q_values,
        dtype=float,
    )

    bet = np.zeros(
        len(q),
        dtype=int,
    )

    if np.isnan(
        threshold
    ):

        return bet


    greater = (
        q
        > threshold
    )

    equal = (
        np.isclose(
            q,
            threshold,
            rtol=1e-12,
            atol=1e-15,
        )
    )


    bet[
        greater
    ] = 1


    tie_indices = (
        np.flatnonzero(
            equal
        )
    )


    if (
        len(
            tie_indices
        ) > 0
        and tie_acceptance_rate > 0
    ):

        rng = np.random.default_rng(
            seed
        )

        accept = (
            rng.random(
                len(
                    tie_indices
                )
            )
            < tie_acceptance_rate
        )

        bet[
            tie_indices[
                accept
            ]
        ] = 1


    return bet


# ============================================================
# EVALUATE BETS
# ============================================================

def evaluate_bets(
    df,
    bet,
    m,
):

    n_days = len(
        df
    )

    n_bets = int(
        np.sum(
            bet
        )
    )

    hit_array = (
        df[
            "hit"
        ]
        .to_numpy(
            dtype=int
        )
    )

    number_hits = int(
        np.sum(
            hit_array
            * bet
        )
    )

    cost_per_bet_day = (
        m
        * COST_PER_NUMBER
    )

    total_cost = (
        n_bets
        * cost_per_bet_day
    )

    total_revenue = (
        number_hits
        * PAYOUT_IF_HIT
    )

    total_profit = (
        total_revenue
        - total_cost
    )


    if total_cost > 0:

        roi = (
            total_profit
            / total_cost
        )

    else:

        roi = np.nan


    if n_bets > 0:

        hit_rate = (
            number_hits
            / n_bets
        )

    else:

        hit_rate = np.nan


    return {
        "n_days": (
            n_days
        ),

        "n_bet_days": (
            n_bets
        ),

        "participation_rate": (
            n_bets / n_days
            if n_days > 0
            else np.nan
        ),

        "number_hits": (
            number_hits
        ),

        "hit_rate": (
            hit_rate
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


# ============================================================
# HISTORY SEARCH
# ============================================================

def evaluate_history_configs(
    predictions,
    history_folds,
    outer_test_fold,
):

    """
    Search toàn bộ:
        model × m × r

    chỉ trên history folds.
    """

    records = []


    for (
        model_key,
        model_df,
    ) in predictions.groupby(
        "model_key"
    ):

        history_df = (
            model_df.loc[
                model_df[
                    "fold"
                ]
                .isin(
                    history_folds
                )
            ]
            .copy()
        )


        if history_df.empty:

            continue


        prepared = (
            prepare_model_data(
                history_df
            )
        )


        for m in M_VALUES:

            topm = (
                build_topm_data(
                    prepared,
                    m,
                )
            )


            q_values = (
                topm[
                    "q_m"
                ]
                .to_numpy()
            )


            for target_rate in (
                TARGET_PARTICIPATION_RATES
            ):

                (
                    threshold,
                    tie_rate,
                ) = (
                    calculate_confidence_rule(
                        q_values,
                        target_rate,
                    )
                )


                seed = (
                    RANDOM_STATE
                    + m * 10_000
                    + int(
                        target_rate
                        * 1000
                    )
                    + sum(
                        ord(x)
                        for x
                        in str(
                            outer_test_fold
                        )
                    )
                )


                bet = (
                    apply_confidence_rule(
                        q_values=q_values,
                        threshold=threshold,
                        tie_acceptance_rate=(
                            tie_rate
                        ),
                        seed=seed,
                    )
                )


                perf = (
                    evaluate_bets(
                        df=topm,
                        bet=bet,
                        m=m,
                    )
                )


                break_even = (
                    m
                    * COST_PER_NUMBER
                    / PAYOUT_IF_HIT
                )


                record = {
                    "outer_test_fold": (
                        outer_test_fold
                    ),

                    "history_folds": (
                        ",".join(
                            history_folds
                        )
                    ),

                    "model": (
                        model_key
                    ),

                    "m": (
                        m
                    ),

                    "target_participation_rate": (
                        target_rate
                    ),

                    "threshold": (
                        threshold
                    ),

                    "tie_acceptance_rate": (
                        tie_rate
                    ),

                    "break_even_hit_rate": (
                        break_even
                    ),
                }


                record.update(
                    perf
                )


                records.append(
                    record
                )


    result = pd.DataFrame(
        records
    )


    # ========================================================
    # ROBUST SCORE
    # ========================================================

    valid = (
        result.loc[
            result[
                "n_bet_days"
            ]
            .ge(
                MIN_HISTORY_BETS
            )
            & result[
                "roi"
            ]
            .notna()
        ]
        .copy()
    )


    if not valid.empty:

        max_bets = float(
            valid[
                "n_bet_days"
            ].max()
        )


        valid[
            "robust_score"
        ] = (
            valid[
                "roi"
            ]
            * np.sqrt(
                valid[
                    "n_bet_days"
                ]
                / max_bets
            )
        )


        result = (
            result.merge(
                valid[
                    [
                        "model",
                        "m",
                        "target_participation_rate",
                        "robust_score",
                    ]
                ],
                on=[
                    "model",
                    "m",
                    "target_participation_rate",
                ],
                how="left",
            )
        )

    else:

        result[
            "robust_score"
        ] = np.nan


    return result


# ============================================================
# SELECT BEST HISTORY CONFIG
# ============================================================

def select_best_config(
    history_results,
):

    valid = (
        history_results.loc[
            history_results[
                "n_bet_days"
            ]
            .ge(
                MIN_HISTORY_BETS
            )
            & history_results[
                "roi"
            ]
            .notna()
        ]
        .copy()
    )


    if valid.empty:

        raise ValueError(
            "Không có history config "
            "đủ MIN_HISTORY_BETS."
        )


    if (
        SELECTION_METHOD
        == "max_roi"
    ):

        best = (
            valid
            .sort_values(
                [
                    "roi",
                    "n_bet_days",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .iloc[0]
        )


    elif (
        SELECTION_METHOD
        == "robust_roi"
    ):

        best = (
            valid
            .sort_values(
                [
                    "robust_score",
                    "roi",
                    "n_bet_days",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )
            .iloc[0]
        )


    else:

        raise ValueError(
            "SELECTION_METHOD phải là "
            "'max_roi' hoặc 'robust_roi'"
        )


    return best


# ============================================================
# APPLY FROZEN CONFIG TO FUTURE
# ============================================================

def test_future_fold(
    predictions,
    best,
    test_fold,
):

    model = (
        best[
            "model"
        ]
    )

    m = int(
        best[
            "m"
        ]
    )

    r = float(
        best[
            "target_participation_rate"
        ]
    )

    threshold = float(
        best[
            "threshold"
        ]
    )

    tie_rate = float(
        best[
            "tie_acceptance_rate"
        ]
    )


    model_test = (
        predictions.loc[
            predictions[
                "model_key"
            ]
            .eq(
                model
            )
            & predictions[
                "fold"
            ]
            .eq(
                test_fold
            )
        ]
        .copy()
    )


    if model_test.empty:

        raise ValueError(
            f"Không có dữ liệu "
            f"{model} / {test_fold}"
        )


    prepared = (
        prepare_model_data(
            model_test
        )
    )


    topm = (
        build_topm_data(
            prepared,
            m,
        )
    )


    seed = (
        RANDOM_STATE
        + m * 10_000
        + int(
            r
            * 1000
        )
        + sum(
            ord(x)
            for x
            in str(
                test_fold
            )
        )
        + 1_000_000
    )


    bet = (
        apply_confidence_rule(
            q_values=(
                topm[
                    "q_m"
                ].to_numpy()
            ),
            threshold=(
                threshold
            ),
            tie_acceptance_rate=(
                tie_rate
            ),
            seed=seed,
        )
    )


    perf = (
        evaluate_bets(
            df=topm,
            bet=bet,
            m=m,
        )
    )


    daily_cost = (
        m
        * COST_PER_NUMBER
    )


    daily = topm.copy()

    daily[
        "model"
    ] = (
        model
    )

    daily[
        "m"
    ] = (
        m
    )

    daily[
        "target_participation_rate"
    ] = (
        r
    )

    daily[
        "threshold"
    ] = (
        threshold
    )

    daily[
        "bet"
    ] = (
        bet
    )

    daily[
        "cost"
    ] = (
        bet
        * daily_cost
    )

    daily[
        "revenue"
    ] = (
        bet
        * daily[
            "hit"
        ]
        * PAYOUT_IF_HIT
    )

    daily[
        "profit"
    ] = (
        daily[
            "revenue"
        ]
        - daily[
            "cost"
        ]
    )

    daily[
        "outer_test_fold"
    ] = (
        test_fold
    )


    result = {
        "test_fold": (
            test_fold
        ),

        "selected_model": (
            model
        ),

        "selected_m": (
            m
        ),

        "selected_r": (
            r
        ),

        "frozen_threshold": (
            threshold
        ),

        "history_roi": float(
            best[
                "roi"
            ]
        ),

        "history_bets": int(
            best[
                "n_bet_days"
            ]
        ),

        "history_hits": int(
            best[
                "number_hits"
            ]
        ),

        "history_hit_rate": float(
            best[
                "hit_rate"
            ]
        ),

        "break_even_hit_rate": (
            m
            * COST_PER_NUMBER
            / PAYOUT_IF_HIT
        ),

        "random_hit_rate": (
            m
            / NUMBER_OF_CLASSES
        ),
    }


    for key, value in (
        perf.items()
    ):

        result[
            f"test_{key}"
        ] = value


    return (
        result,
        daily,
    )


# ============================================================
# FINAL AGGREGATE
# ============================================================

def build_final_summary(
    test_results,
):

    total_bets = int(
        test_results[
            "test_n_bet_days"
        ].sum()
    )

    total_hits = int(
        test_results[
            "test_number_hits"
        ].sum()
    )

    total_cost = float(
        test_results[
            "test_total_cost"
        ].sum()
    )

    total_revenue = float(
        test_results[
            "test_total_revenue"
        ].sum()
    )

    total_profit = (
        total_revenue
        - total_cost
    )


    if total_cost > 0:

        roi = (
            total_profit
            / total_cost
        )

    else:

        roi = np.nan


    if total_bets > 0:

        hit_rate = (
            total_hits
            / total_bets
        )

    else:

        hit_rate = np.nan


    profitable_folds = int(
        (
            test_results[
                "test_total_profit"
            ]
            > 0
        )
        .sum()
    )


    return pd.DataFrame(
        [
            {
                "selection_method": (
                    SELECTION_METHOD
                ),

                "min_history_bets": (
                    MIN_HISTORY_BETS
                ),

                "n_future_folds": (
                    len(
                        test_results
                    )
                ),

                "profitable_future_folds": (
                    profitable_folds
                ),

                "total_bet_days": (
                    total_bets
                ),

                "total_hits": (
                    total_hits
                ),

                "aggregate_hit_rate": (
                    hit_rate
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
                    roi
                ),

                "random_expected_roi": (
                    -0.20
                ),

                "roi_gain_vs_random": (
                    roi + 0.20
                    if not np.isnan(
                        roi
                    )
                    else np.nan
                ),
            }
        ]
    )


# ============================================================
# PRINT
# ============================================================

def print_selection(
    selection_df,
):

    print(
        "\n"
        + "=" * 190
    )

    print(
        "NESTED WALK-FORWARD "
        "- CONFIG CHOSEN FROM PAST ONLY"
    )

    print(
        "=" * 190
    )


    columns = [
        "outer_test_fold",
        "history_folds",
        "model",
        "m",
        "target_participation_rate",
        "n_bet_days",
        "number_hits",
        "hit_rate",
        "roi",
        "robust_score",
        "threshold",
    ]


    print(
        selection_df[
            columns
        ]
        .to_string(
            index=False,
            formatters={
                "target_participation_rate": (
                    "{:.0%}".format
                ),

                "hit_rate": (
                    "{:.4%}".format
                ),

                "roi": (
                    "{:.2%}".format
                ),

                "robust_score": (
                    "{:.4f}".format
                ),

                "threshold": (
                    "{:.6f}".format
                ),
            },
        )
    )


def print_test(
    test_df,
):

    print(
        "\n"
        + "=" * 200
    )

    print(
        "TRUE FUTURE TEST RESULTS"
    )

    print(
        "=" * 200
    )


    columns = [
        "test_fold",
        "selected_model",
        "selected_m",
        "selected_r",
        "test_n_days",
        "test_n_bet_days",
        "test_participation_rate",
        "test_number_hits",
        "test_hit_rate",
        "break_even_hit_rate",
        "test_total_profit",
        "test_roi",
    ]


    print(
        test_df[
            columns
        ]
        .to_string(
            index=False,
            formatters={
                "selected_r": (
                    "{:.0%}".format
                ),

                "test_participation_rate": (
                    "{:.2%}".format
                ),

                "test_hit_rate": (
                    "{:.4%}".format
                ),

                "break_even_hit_rate": (
                    "{:.2%}".format
                ),

                "test_total_profit": (
                    "{:,.0f}".format
                ),

                "test_roi": (
                    "{:.2%}".format
                ),
            },
        )
    )


def print_final(
    summary,
):

    print(
        "\n"
        + "=" * 160
    )

    print(
        "FINAL NESTED WALK-FORWARD SUMMARY"
    )

    print(
        "=" * 160
    )


    print(
        summary.to_string(
            index=False,
            formatters={
                "aggregate_hit_rate": (
                    "{:.4%}".format
                ),

                "total_cost": (
                    "{:,.0f}".format
                ),

                "total_revenue": (
                    "{:,.0f}".format
                ),

                "total_profit": (
                    "{:,.0f}".format
                ),

                "aggregate_roi": (
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
        f"Selection method: "
        f"{SELECTION_METHOD}"
    )

    print(
        f"MIN_HISTORY_BETS: "
        f"{MIN_HISTORY_BETS}"
    )


    all_history_frames = []

    selection_records = []

    future_results = []

    future_daily_frames = []


    # ========================================================
    # OUTER WALK-FORWARD
    # ========================================================

    for outer in (
        OUTER_FOLDS
    ):

        test_fold = (
            outer[
                "test_fold"
            ]
        )

        history_folds = (
            outer[
                "history_folds"
            ]
        )


        print(
            "\n"
            + "-" * 100
        )

        print(
            f"Future test fold: "
            f"{test_fold}"
        )

        print(
            f"History folds: "
            f"{history_folds}"
        )


        # ----------------------------------------------------
        # Search history only
        # ----------------------------------------------------

        history_results = (
            evaluate_history_configs(
                predictions=(
                    predictions
                ),
                history_folds=(
                    history_folds
                ),
                outer_test_fold=(
                    test_fold
                ),
            )
        )


        all_history_frames.append(
            history_results
        )


        # ----------------------------------------------------
        # Choose config
        # ----------------------------------------------------

        best = (
            select_best_config(
                history_results
            )
        )


        selection_records.append(
            best.to_dict()
        )


        print(
            "\nSelected from past:"
        )

        print(
            f"  model = "
            f"{best['model']}"
        )

        print(
            f"  m     = "
            f"{int(best['m'])}"
        )

        print(
            f"  r     = "
            f"{best['target_participation_rate']:.0%}"
        )

        print(
            f"  history ROI = "
            f"{best['roi']:.2%}"
        )

        print(
            f"  history bets = "
            f"{int(best['n_bet_days'])}"
        )


        # ----------------------------------------------------
        # Freeze -> future
        # ----------------------------------------------------

        (
            test_result,
            test_daily,
        ) = test_future_fold(
            predictions=(
                predictions
            ),
            best=best,
            test_fold=(
                test_fold
            ),
        )


        future_results.append(
            test_result
        )

        future_daily_frames.append(
            test_daily
        )


        print(
            "\nFuture result:"
        )

        print(
            f"  bets = "
            f"{test_result['test_n_bet_days']}"
        )

        print(
            f"  hits = "
            f"{test_result['test_number_hits']}"
        )

        print(
            f"  ROI  = "
            f"{test_result['test_roi']:.2%}"
        )


    # ========================================================
    # BUILD OUTPUTS
    # ========================================================

    all_history = pd.concat(
        all_history_frames,
        ignore_index=True,
    )


    selection_df = pd.DataFrame(
        selection_records
    )


    test_df = pd.DataFrame(
        future_results
    )


    daily_df = pd.concat(
        future_daily_frames,
        ignore_index=True,
    )


    final_summary = (
        build_final_summary(
            test_df
        )
    )


    # ========================================================
    # PRINT
    # ========================================================

    print_selection(
        selection_df
    )


    print_test(
        test_df
    )


    print_final(
        final_summary
    )


    # ========================================================
    # SAVE
    # ========================================================

    all_history.to_csv(
        OUTPUT_ALL_HISTORY_CONFIGS,
        index=False,
        encoding="utf-8-sig",
    )


    selection_df.to_csv(
        OUTPUT_SELECTION,
        index=False,
        encoding="utf-8-sig",
    )


    test_df.to_csv(
        OUTPUT_TEST,
        index=False,
        encoding="utf-8-sig",
    )


    daily_df.to_csv(
        OUTPUT_DAILY,
        index=False,
        encoding="utf-8-sig",
    )


    final_summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )


    print(
        "\nĐã lưu:"
    )

    print(
        OUTPUT_ALL_HISTORY_CONFIGS
    )

    print(
        OUTPUT_SELECTION
    )

    print(
        OUTPUT_TEST
    )

    print(
        OUTPUT_DAILY
    )

    print(
        OUTPUT_SUMMARY
    )


if __name__ == "__main__":
    main()