"""
19 - ADAPTIVE PERCENTILE THRESHOLD STRATEGY

Mục tiêu
--------
Strategy 18 freeze confidence threshold tuyệt đối:

    q_m(t) >= tau_history

Điều này có thể thất bại khi scale của probability/confidence
thay đổi giữa history và future.

Strategy 19 thay threshold tuyệt đối bằng threshold percentile
được cập nhật chỉ từ confidence QUÁ KHỨ:

    tau_t =
        quantile(
            q_{t-W}, ..., q_{t-1},
            1 - target_rate
        )

Không dùng actual tương lai để cập nhật threshold.

Ví dụ:
    target_rate = 20%

thì ngày t ta so sánh q_t với percentile 80%
của q trong W evaluation days trước đó.

QUAN TRỌNG
----------
Không ép model phải bet đúng 20% nếu confidence đã collapse.

Nếu q history gần như constant:

    max(q) - min(q) <= DISPERSION_TOLERANCE

=> strategy inactive.

Điều này đặc biệt quan trọng cho CatBoost khi probability
bị blend thành Uniform.

Nested walk-forward
-------------------
Future 2022-2023:
    chọn model / m / r
    chỉ từ 2020-2021.

Future 2024-2026:
    chọn model / m / r
    chỉ từ 2020-2021 + 2022-2023.

KHÁC Strategy 18:
    Strategy 18 chọn fixed threshold.

    Strategy 19:
        chọn model, m, r từ history
        nhưng threshold được update động
        chỉ bằng past q.

Selection trên history cũng phải dùng adaptive rule,
để strategy selection và future deployment nhất quán.

Models
------
- CatBoost static
- CatBoost periodic retrain
- CDM
- Rolling CDM:
      30 / 60 / 90 / 180 / 365
- Bayesian Markov

Outputs
-------
artifacts/strategies/

    adaptive_strategy_selection.csv
    adaptive_strategy_history_configs.csv
    adaptive_strategy_test_results.csv
    adaptive_strategy_test_daily.csv
    adaptive_strategy_summary.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


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

RANDOM_STATE = 42


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


# ============================================================
# ADAPTIVE THRESHOLD CONFIG
# ============================================================

# q history gần nhất dùng để tính percentile
ADAPTIVE_WINDOW = 180


# Ít nhất bao nhiêu q quá khứ mới bắt đầu bet
MIN_CONFIDENCE_HISTORY = 60


# Nếu q gần như constant => model không có confidence signal
DISPERSION_TOLERANCE = 1e-10


# Ít nhất bao nhiêu history bets mới được chọn config
MIN_HISTORY_BETS = 50


PROBABILITY_COLUMNS = [
    f"p_{number:02d}"
    for number
    in range(
        NUMBER_OF_CLASSES
    )
]


# ============================================================
# INPUT
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
# OUTER WALK FORWARD
# ============================================================

OUTER_FOLDS = [
    {
        "test_fold": "2022-2023",

        "history_folds": [
            "2020-2021",
        ],
    },

    {
        "test_fold": "2024-2026",

        "history_folds": [
            "2020-2021",
            "2022-2023",
        ],
    },
]


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_SELECTION = (
    STRATEGY_DIR
    / "adaptive_strategy_selection.csv"
)

OUTPUT_HISTORY_CONFIGS = (
    STRATEGY_DIR
    / "adaptive_strategy_history_configs.csv"
)

OUTPUT_TEST_RESULTS = (
    STRATEGY_DIR
    / "adaptive_strategy_test_results.csv"
)

OUTPUT_TEST_DAILY = (
    STRATEGY_DIR
    / "adaptive_strategy_test_daily.csv"
)

OUTPUT_SUMMARY = (
    STRATEGY_DIR
    / "adaptive_strategy_summary.csv"
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


# ============================================================
# NORMALIZE
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
        probabilities.sum(
            axis=1,
            keepdims=True,
        )
    )

    if np.any(
        row_sum <= 0
    ):

        raise ValueError(
            "Có probability row tổng <= 0."
        )

    return (
        probabilities
        / row_sum
    )


# ============================================================
# RANK
# ============================================================

def rank_probability_matrix(
    probabilities,
):

    probabilities = (
        normalize_probability_matrix(
            probabilities
        )
    )

    classes = np.arange(
        NUMBER_OF_CLASSES
    )

    orders = np.empty(
        probabilities.shape,
        dtype=int,
    )

    for row_index in range(
        len(
            probabilities
        )
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

def validate_columns(
    df,
    model_name,
):

    required = [
        "date",
        "fold",
        "actual",
        *PROBABILITY_COLUMNS,
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
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

    validate_columns(
        df,
        model_name,
    )

    if (
        model_name
        == "rolling_cdm"
    ):

        if (
            "window"
            not in df.columns
        ):

            raise ValueError(
                "rolling_cdm thiếu window."
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

    order = (
        rank_probability_matrix(
            probabilities
        )
    )


    # ========================================================
    # ACTUAL RANK
    # ========================================================

    inverse_rank = np.empty_like(
        order
    )

    inverse_rank[
        np.arange(
            len(
                df
            )
        )[
            :,
            None,
        ],
        order,
    ] = np.arange(
        1,
        NUMBER_OF_CLASSES + 1,
    )

    actual_rank = (
        inverse_rank[
            np.arange(
                len(
                    df
                )
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
        "df": df,
        "actual": actual,
        "order": order,
        "actual_rank": actual_rank,
        "cum_prob": cumulative_probability,
    }


# ============================================================
# BUILD TOP-M
# ============================================================

def build_topm_data(
    prepared,
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
            "cum_prob"
        ]
    )

    top_m = (
        order[
            :,
            :m,
        ]
    )

    q_m = (
        cumulative_probability[
            :,
            m - 1,
        ]
    )

    hit = (
        actual_rank
        <= m
    ).astype(
        np.int8
    )

    selected_numbers = [
        ",".join(
            f"{number:02d}"
            for number
            in selected
        )

        for selected
        in top_m
    ]

    return pd.DataFrame(
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

            "selected_numbers": (
                selected_numbers
            ),
        }
    )


# ============================================================
# STABLE SEED
# ============================================================

def stable_seed(
    text,
):

    value = sum(
        (
            index + 1
        )
        * ord(
            character
        )

        for (
            index,
            character,
        )
        in enumerate(
            text
        )
    )

    return (
        RANDOM_STATE
        + value
    )


# ============================================================
# ADAPTIVE RULE
# ============================================================

def run_adaptive_rule(
    df,
    target_rate,
    initial_q_history=None,
    seed_label="",
):
    """
    Sequential adaptive threshold.

    Khi quyết định ngày t:

        history_q =
            q của các ngày trước t

        threshold_t =
            percentile 1-r của recent history_q

    Sau khi quyết định:
        q_t được append vào q history.

    actual_t KHÔNG dùng để update threshold.

    Return:
        bet
        threshold
        q_history_size
        q_history_range
        status
    """

    df = (
        df
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )

    q_current = (
        df[
            "q_m"
        ]
        .to_numpy(
            dtype=float
        )
    )


    # ========================================================
    # INITIAL HISTORY
    # ========================================================

    if initial_q_history is None:

        q_history = []

    else:

        q_history = list(
            np.asarray(
                initial_q_history,
                dtype=float,
            )
        )


    n_rows = len(
        df
    )


    bet = np.zeros(
        n_rows,
        dtype=np.int8,
    )


    thresholds = np.full(
        n_rows,
        np.nan,
        dtype=float,
    )


    history_sizes = np.zeros(
        n_rows,
        dtype=int,
    )


    history_ranges = np.full(
        n_rows,
        np.nan,
        dtype=float,
    )


    status = np.empty(
        n_rows,
        dtype=object,
    )


    rng = np.random.default_rng(
        stable_seed(
            seed_label
        )
    )


    # ========================================================
    # SEQUENTIAL
    # ========================================================

    for i in range(
        n_rows
    ):

        recent_history = np.asarray(
            q_history[
                -ADAPTIVE_WINDOW:
            ],
            dtype=float,
        )


        history_sizes[
            i
        ] = len(
            recent_history
        )


        # ====================================================
        # NOT ENOUGH HISTORY
        # ====================================================

        if (
            len(
                recent_history
            )
            < MIN_CONFIDENCE_HISTORY
        ):

            status[
                i
            ] = (
                "warmup"
            )


            q_history.append(
                q_current[
                    i
                ]
            )

            continue


        # ====================================================
        # DISPERSION GATE
        # ====================================================

        q_range = float(
            np.max(
                recent_history
            )
            - np.min(
                recent_history
            )
        )


        history_ranges[
            i
        ] = (
            q_range
        )


        if (
            q_range
            <= DISPERSION_TOLERANCE
        ):

            status[
                i
            ] = (
                "constant_confidence"
            )


            q_history.append(
                q_current[
                    i
                ]
            )

            continue


        # ====================================================
        # ADAPTIVE THRESHOLD
        # ====================================================

        threshold = float(
            np.quantile(
                recent_history,
                1.0
                - target_rate,
            )
        )


        thresholds[
            i
        ] = (
            threshold
        )


        current_q = (
            q_current[
                i
            ]
        )


        # ====================================================
        # ABOVE THRESHOLD
        # ====================================================

        if (
            current_q
            > threshold
        ):

            bet[
                i
            ] = 1

            status[
                i
            ] = (
                "bet_above"
            )


        # ====================================================
        # EXACT TIE
        # ====================================================

        elif np.isclose(
            current_q,
            threshold,
            rtol=1e-12,
            atol=1e-15,
        ):

            greater_count = int(
                np.sum(
                    recent_history
                    > threshold
                )
            )


            equal_count = int(
                np.sum(
                    np.isclose(
                        recent_history,
                        threshold,
                        rtol=1e-12,
                        atol=1e-15,
                    )
                )
            )


            target_count = (
                target_rate
                * len(
                    recent_history
                )
            )


            if (
                equal_count
                > 0
            ):

                tie_acceptance = (
                    (
                        target_count
                        - greater_count
                    )
                    / equal_count
                )

            else:

                tie_acceptance = (
                    0.0
                )


            tie_acceptance = float(
                np.clip(
                    tie_acceptance,
                    0.0,
                    1.0,
                )
            )


            if (
                rng.random()
                < tie_acceptance
            ):

                bet[
                    i
                ] = 1

                status[
                    i
                ] = (
                    "bet_tie"
                )

            else:

                status[
                    i
                ] = (
                    "skip_tie"
                )


        else:

            status[
                i
            ] = (
                "below_threshold"
            )


        # ====================================================
        # q_t becomes history only AFTER decision
        # ====================================================

        q_history.append(
            current_q
        )


    return {
        "bet": (
            bet
        ),

        "threshold": (
            thresholds
        ),

        "history_size": (
            history_sizes
        ),

        "history_range": (
            history_ranges
        ),

        "status": (
            status
        ),
    }


# ============================================================
# WILSON
# ============================================================

def wilson_interval(
    hits,
    bets,
    z=1.959963984540054,
):

    if bets <= 0:

        return (
            np.nan,
            np.nan,
        )

    p_hat = (
        hits
        / bets
    )

    z2 = (
        z ** 2
    )

    denominator = (
        1.0
        + z2
        / bets
    )

    center = (
        p_hat
        + z2
        / (
            2.0
            * bets
        )
    ) / denominator

    margin = (
        z
        / denominator
        * np.sqrt(
            (
                p_hat
                * (
                    1.0
                    - p_hat
                )
                / bets
            )
            +
            (
                z2
                / (
                    4.0
                    * bets ** 2
                )
            )
        )
    )

    return (
        max(
            0.0,
            float(
                center
                - margin
            ),
        ),

        min(
            1.0,
            float(
                center
                + margin
            ),
        ),
    )


# ============================================================
# ECONOMIC EVALUATION
# ============================================================

def evaluate_bets(
    df,
    bet,
    m,
):

    bet = np.asarray(
        bet,
        dtype=int,
    )

    hit = (
        df[
            "hit"
        ]
        .to_numpy(
            dtype=int
        )
    )

    n_days = len(
        df
    )

    n_bets = int(
        bet.sum()
    )

    n_hits = int(
        np.sum(
            bet
            * hit
        )
    )

    participation_rate = (
        n_bets
        / n_days
        if n_days > 0
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

    total_cost = (
        n_bets
        * m
        * COST_PER_NUMBER
    )

    total_revenue = (
        n_hits
        * PAYOUT_IF_HIT
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
            n_hits,
            n_bets,
        )
    )

    if n_bets > 0:

        p_break_even = float(
            binomtest(
                k=n_hits,
                n=n_bets,
                p=break_even_hit_rate,
                alternative="greater",
            ).pvalue
        )

    else:

        p_break_even = (
            1.0
        )

    return {
        "n_days": (
            n_days
        ),

        "n_bet_days": (
            n_bets
        ),

        "participation_rate": (
            participation_rate
        ),

        "number_hits": (
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

        "wilson_margin_vs_break_even": (
            (
                wilson_lower
                - break_even_hit_rate
            )
            if pd.notna(
                wilson_lower
            )
            else np.nan
        ),

        "p_break_even_raw": (
            p_break_even
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
# HISTORY CONFIG SEARCH
# ============================================================

def evaluate_history_configs(
    prepared_models,
    history_folds,
    future_fold,
):

    records = []

    history_label = (
        "+".join(
            history_folds
        )
    )


    for (
        model_key,
        prepared,
    ) in prepared_models.items():

        print(
            f"  searching "
            f"{model_key}"
        )


        for m in M_VALUES:

            topm = (
                build_topm_data(
                    prepared,
                    m,
                )
            )


            history = (
                topm.loc[
                    topm[
                        "fold"
                    ]
                    .isin(
                        history_folds
                    )
                ]
                .copy()
                .sort_values(
                    "date"
                )
                .reset_index(
                    drop=True
                )
            )


            if history.empty:
                continue


            for target_rate in (
                TARGET_PARTICIPATION_RATES
            ):

                adaptive = (
                    run_adaptive_rule(
                        df=(
                            history
                        ),

                        target_rate=(
                            target_rate
                        ),

                        initial_q_history=None,

                        seed_label=(
                            f"history|"
                            f"{history_label}|"
                            f"{model_key}|"
                            f"{m}|"
                            f"{target_rate}"
                        ),
                    )
                )


                metrics = (
                    evaluate_bets(
                        df=(
                            history
                        ),

                        bet=(
                            adaptive[
                                "bet"
                            ]
                        ),

                        m=(
                            m
                        ),
                    )
                )


                records.append(
                    {
                        "future_fold": (
                            future_fold
                        ),

                        "history_folds": (
                            history_label
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

                        "adaptive_window": (
                            ADAPTIVE_WINDOW
                        ),

                        "min_confidence_history": (
                            MIN_CONFIDENCE_HISTORY
                        ),

                        "enough_history_bets": (
                            metrics[
                                "n_bet_days"
                            ]
                            >= MIN_HISTORY_BETS
                        ),

                        **metrics,
                    }
                )


    return pd.DataFrame(
        records
    )


# ============================================================
# SELECT CONFIG
# ============================================================

def select_history_config(
    history_configs,
):

    eligible = (
        history_configs.loc[
            history_configs[
                "enough_history_bets"
            ]
        ]
        .copy()
    )

    fallback = False


    if eligible.empty:

        eligible = (
            history_configs
            .copy()
        )

        fallback = True


    eligible[
        "selection_score"
    ] = (
        eligible[
            "wilson_margin_vs_break_even"
        ]
        .fillna(
            -np.inf
        )
    )


    selected = (
        eligible
        .sort_values(
            [
                "selection_score",
                "p_break_even_raw",
                "roi",
                "n_bet_days",
            ],
            ascending=[
                False,
                True,
                False,
                False,
            ],
        )
        .iloc[0]
        .copy()
    )


    selected[
        "selection_fallback"
    ] = (
        fallback
    )


    return selected


# ============================================================
# GET Q HISTORY BEFORE FUTURE
# ============================================================

def get_initial_future_q_history(
    prepared,
    m,
    history_folds,
):

    topm = (
        build_topm_data(
            prepared,
            m,
        )
    )


    history = (
        topm.loc[
            topm[
                "fold"
            ]
            .isin(
                history_folds
            )
        ]
        .sort_values(
            "date"
        )
    )


    return (
        history[
            "q_m"
        ]
        .to_numpy(
            dtype=float
        )
    )


# ============================================================
# FUTURE DEPLOYMENT
# ============================================================

def evaluate_future(
    selected,
    prepared_models,
):

    model_key = (
        selected[
            "model"
        ]
    )

    m = int(
        selected[
            "m"
        ]
    )

    target_rate = float(
        selected[
            "target_participation_rate"
        ]
    )

    future_fold = (
        selected[
            "future_fold"
        ]
    )

    history_folds = (
        str(
            selected[
                "history_folds"
            ]
        )
        .split(
            "+"
        )
    )


    prepared = (
        prepared_models[
            model_key
        ]
    )


    topm = (
        build_topm_data(
            prepared,
            m,
        )
    )


    future = (
        topm.loc[
            topm[
                "fold"
            ]
            .eq(
                future_fold
            )
        ]
        .copy()
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # INITIAL q HISTORY
    #
    # Chỉ q từ các past folds.
    # ========================================================

    initial_q = (
        get_initial_future_q_history(
            prepared=(
                prepared
            ),

            m=(
                m
            ),

            history_folds=(
                history_folds
            ),
        )
    )


    adaptive = (
        run_adaptive_rule(
            df=(
                future
            ),

            target_rate=(
                target_rate
            ),

            initial_q_history=(
                initial_q
            ),

            seed_label=(
                f"future|"
                f"{future_fold}|"
                f"{model_key}|"
                f"{m}|"
                f"{target_rate}"
            ),
        )
    )


    bet = (
        adaptive[
            "bet"
        ]
    )


    metrics = (
        evaluate_bets(
            df=(
                future
            ),

            bet=(
                bet
            ),

            m=(
                m
            ),
        )
    )


    # ========================================================
    # DAILY OUTPUT
    # ========================================================

    future[
        "bet"
    ] = (
        bet
    )


    future[
        "adaptive_threshold"
    ] = (
        adaptive[
            "threshold"
        ]
    )


    future[
        "confidence_history_size"
    ] = (
        adaptive[
            "history_size"
        ]
    )


    future[
        "confidence_history_range"
    ] = (
        adaptive[
            "history_range"
        ]
    )


    future[
        "adaptive_status"
    ] = (
        adaptive[
            "status"
        ]
    )


    future[
        "cost"
    ] = (
        future[
            "bet"
        ]
        * m
        * COST_PER_NUMBER
    )


    future[
        "revenue"
    ] = (
        future[
            "bet"
        ]
        * future[
            "hit"
        ]
        * PAYOUT_IF_HIT
    )


    future[
        "profit"
    ] = (
        future[
            "revenue"
        ]
        - future[
            "cost"
        ]
    )


    future[
        "cumulative_profit"
    ] = (
        future[
            "profit"
        ]
        .cumsum()
    )


    future[
        "selected_model"
    ] = (
        model_key
    )


    future[
        "selected_m"
    ] = (
        m
    )


    future[
        "selected_target_rate"
    ] = (
        target_rate
    )


    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    status_counts = (
        future[
            "adaptive_status"
        ]
        .value_counts()
        .to_dict()
    )


    constant_days = int(
        status_counts.get(
            "constant_confidence",
            0,
        )
    )


    warmup_days = int(
        status_counts.get(
            "warmup",
            0,
        )
    )


    result = {
        "future_fold": (
            future_fold
        ),

        "history_folds": (
            selected[
                "history_folds"
            ]
        ),

        "selected_model": (
            model_key
        ),

        "selected_m": (
            m
        ),

        "selected_target_rate": (
            target_rate
        ),

        "history_selection_score": (
            selected[
                "selection_score"
            ]
        ),

        "history_n_bets": (
            selected[
                "n_bet_days"
            ]
        ),

        "history_hits": (
            selected[
                "number_hits"
            ]
        ),

        "history_hit_rate": (
            selected[
                "hit_rate"
            ]
        ),

        "history_break_even": (
            selected[
                "break_even_hit_rate"
            ]
        ),

        "history_wilson_lower": (
            selected[
                "wilson_lower"
            ]
        ),

        "history_roi": (
            selected[
                "roi"
            ]
        ),

        "future_constant_confidence_days": (
            constant_days
        ),

        "future_warmup_days": (
            warmup_days
        ),

        **{
            f"future_{key}": value
            for (
                key,
                value,
            )
            in metrics.items()
        },
    }


    return (
        result,
        future
    )


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    results,
):

    total_bets = int(
        results[
            "future_n_bet_days"
        ]
        .sum()
    )


    total_hits = int(
        results[
            "future_number_hits"
        ]
        .sum()
    )


    total_cost = float(
        results[
            "future_total_cost"
        ]
        .sum()
    )


    total_revenue = float(
        results[
            "future_total_revenue"
        ]
        .sum()
    )


    total_profit = (
        total_revenue
        - total_cost
    )


    aggregate_roi = (
        total_profit
        / total_cost
        if total_cost > 0
        else np.nan
    )


    active_folds = int(
        results[
            "future_n_bet_days"
        ]
        .gt(
            0
        )
        .sum()
    )


    positive_folds = int(
        (
            results[
                "future_n_bet_days"
            ]
            .gt(
                0
            )

            & results[
                "future_roi"
            ]
            .gt(
                0
            )
        )
        .sum()
    )


    negative_folds = int(
        (
            results[
                "future_n_bet_days"
            ]
            .gt(
                0
            )

            & results[
                "future_roi"
            ]
            .lt(
                0
            )
        )
        .sum()
    )


    return pd.DataFrame(
        [
            {
                "n_future_folds": (
                    len(
                        results
                    )
                ),

                "active_future_folds": (
                    active_folds
                ),

                "inactive_future_folds": (
                    len(
                        results
                    )
                    - active_folds
                ),

                "positive_active_folds": (
                    positive_folds
                ),

                "negative_active_folds": (
                    negative_folds
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

                "note": (
                    "Aggregate ROI is secondary; "
                    "inspect each future fold."
                ),
            }
        ]
    )


# ============================================================
# PRINT SELECTION
# ============================================================

def print_selection(
    selected,
):

    print(
        "\nSELECTED ADAPTIVE STRATEGY"
    )


    print(
        f"  Future fold       : "
        f"{selected['future_fold']}"
    )

    print(
        f"  History folds     : "
        f"{selected['history_folds']}"
    )

    print(
        f"  Model             : "
        f"{selected['model']}"
    )

    print(
        f"  m                 : "
        f"{int(selected['m'])}"
    )

    print(
        f"  target rate       : "
        f"{selected['target_participation_rate']:.0%}"
    )

    print(
        f"  history bets      : "
        f"{int(selected['n_bet_days'])}"
    )

    print(
        f"  history hits      : "
        f"{int(selected['number_hits'])}"
    )

    print(
        f"  history hit rate  : "
        f"{selected['hit_rate']:.4%}"
    )

    print(
        f"  break-even        : "
        f"{selected['break_even_hit_rate']:.4%}"
    )

    print(
        f"  Wilson lower      : "
        f"{selected['wilson_lower']:.4%}"
    )

    print(
        f"  selection score   : "
        f"{selected['selection_score']:+.6f}"
    )

    print(
        f"  history ROI       : "
        f"{selected['roi']:+.2%}"
    )


# ============================================================
# PRINT FUTURE
# ============================================================

def print_future(
    result,
):

    print(
        "\nADAPTIVE FUTURE RESULT"
    )

    print(
        f"  Fold              : "
        f"{result['future_fold']}"
    )

    print(
        f"  Days              : "
        f"{result['future_n_days']:,}"
    )

    print(
        f"  Bets              : "
        f"{result['future_n_bet_days']:,}"
    )

    print(
        f"  Participation     : "
        f"{result['future_participation_rate']:.2%}"
    )

    print(
        f"  Constant conf days: "
        f"{result['future_constant_confidence_days']:,}"
    )

    print(
        f"  Hits              : "
        f"{result['future_number_hits']:,}"
    )


    if (
        result[
            "future_n_bet_days"
        ]
        > 0
    ):

        print(
            f"  Hit rate          : "
            f"{result['future_hit_rate']:.4%}"
        )

        print(
            f"  Break-even        : "
            f"{result['future_break_even_hit_rate']:.4%}"
        )

        print(
            f"  Wilson lower      : "
            f"{result['future_wilson_lower']:.4%}"
        )

        print(
            f"  Profit            : "
            f"{result['future_total_profit']:,.0f}"
        )

        print(
            f"  ROI               : "
            f"{result['future_roi']:+.2%}"
        )

    else:

        print(
            "  STATUS            : "
            "NO BETS"
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
        "\nAdaptive config:"
    )

    print(
        f"  Window            : "
        f"{ADAPTIVE_WINDOW}"
    )

    print(
        f"  Min q history     : "
        f"{MIN_CONFIDENCE_HISTORY}"
    )

    print(
        f"  Dispersion tol    : "
        f"{DISPERSION_TOLERANCE}"
    )


    # ========================================================
    # PREPARE
    # ========================================================

    prepared_models = {}


    for (
        model_key,
        subset,
    ) in predictions.groupby(
        "model_key",
        sort=False,
    ):

        print(
            f"Preparing: "
            f"{model_key}"
        )

        prepared_models[
            model_key
        ] = (
            prepare_model_data(
                subset
            )
        )


    # ========================================================
    # NESTED
    # ========================================================

    selection_records = []

    history_frames = []

    future_records = []

    daily_frames = []


    for outer in OUTER_FOLDS:

        future_fold = (
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
            + "=" * 110
        )

        print(
            f"FUTURE: "
            f"{future_fold}"
        )

        print(
            f"HISTORY: "
            f"{history_folds}"
        )

        print(
            "=" * 110
        )


        history_configs = (
            evaluate_history_configs(
                prepared_models=(
                    prepared_models
                ),

                history_folds=(
                    history_folds
                ),

                future_fold=(
                    future_fold
                ),
            )
        )


        history_frames.append(
            history_configs
        )


        selected = (
            select_history_config(
                history_configs
            )
        )


        print_selection(
            selected
        )


        selection_records.append(
            selected.to_dict()
        )


        (
            result,
            future_daily,
        ) = (
            evaluate_future(
                selected=(
                    selected
                ),

                prepared_models=(
                    prepared_models
                ),
            )
        )


        print_future(
            result
        )


        future_records.append(
            result
        )


        daily_frames.append(
            future_daily
        )


    # ========================================================
    # OUTPUT DATAFRAMES
    # ========================================================

    selection_df = pd.DataFrame(
        selection_records
    )


    history_df = pd.concat(
        history_frames,
        ignore_index=True,
    )


    future_df = pd.DataFrame(
        future_records
    )


    daily_df = pd.concat(
        daily_frames,
        ignore_index=True,
    )


    summary_df = (
        build_summary(
            future_df
        )
    )


    # ========================================================
    # PRINT
    # ========================================================

    print(
        "\n"
        + "=" * 180
    )

    print(
        "ADAPTIVE NESTED WALK-FORWARD RESULTS"
    )

    print(
        "=" * 180
    )


    columns = [
        "future_fold",
        "selected_model",
        "selected_m",
        "selected_target_rate",
        "history_n_bets",
        "history_roi",
        "future_n_bet_days",
        "future_participation_rate",
        "future_constant_confidence_days",
        "future_number_hits",
        "future_hit_rate",
        "future_break_even_hit_rate",
        "future_total_profit",
        "future_roi",
    ]


    print(
        future_df[
            columns
        ]
        .to_string(
            index=False,

            formatters={
                "selected_target_rate": (
                    "{:.0%}".format
                ),

                "history_roi": (
                    "{:+.2%}".format
                ),

                "future_participation_rate": (
                    "{:.2%}".format
                ),

                "future_hit_rate": (
                    lambda x:
                    (
                        f"{x:.4%}"
                        if pd.notna(
                            x
                        )
                        else "NA"
                    )
                ),

                "future_break_even_hit_rate": (
                    "{:.4%}".format
                ),

                "future_total_profit": (
                    "{:,.0f}".format
                ),

                "future_roi": (
                    lambda x:
                    (
                        f"{x:+.2%}"
                        if pd.notna(
                            x
                        )
                        else "NA"
                    )
                ),
            },
        )
    )


    print(
        "\n"
        + "=" * 100
    )

    print(
        "ADAPTIVE AGGREGATE - READ WITH CAUTION"
    )

    print(
        "=" * 100
    )

    print(
        summary_df.to_string(
            index=False
        )
    )


    # ========================================================
    # SAVE
    # ========================================================

    selection_df.to_csv(
        OUTPUT_SELECTION,
        index=False,
        encoding="utf-8-sig",
    )


    history_df.to_csv(
        OUTPUT_HISTORY_CONFIGS,
        index=False,
        encoding="utf-8-sig",
    )


    future_df.to_csv(
        OUTPUT_TEST_RESULTS,
        index=False,
        encoding="utf-8-sig",
    )


    daily_df.to_csv(
        OUTPUT_TEST_DAILY,
        index=False,
        encoding="utf-8-sig",
    )


    summary_df.to_csv(
        OUTPUT_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )


    print(
        "\nĐã lưu:"
    )

    print(
        OUTPUT_SELECTION
    )

    print(
        OUTPUT_HISTORY_CONFIGS
    )

    print(
        OUTPUT_TEST_RESULTS
    )

    print(
        OUTPUT_TEST_DAILY
    )

    print(
        OUTPUT_SUMMARY
    )


if __name__ == "__main__":
    main()