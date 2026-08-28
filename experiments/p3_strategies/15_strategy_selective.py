"""
Strategy 15 - Selective Top-m

Ý tưởng
-------
Always Top-m chơi mỗi ngày.

Selective Top-m:
    chỉ chơi những ngày model có confidence cao.

Với mỗi:
    model
    m = 1..30
    target participation rate:
        10%, 20%, 30%, 50%

Ta tính:

    q_m(t)
        = tổng probability của Top-m số
          tại ngày t.

Trong từng outer fold:

    50% ngày đầu:
        TUNE

    50% ngày sau:
        TEST

Threshold chỉ được chọn từ TUNE.

Sau đó freeze threshold và áp dụng cho TEST.

Không dùng kết quả actual của TEST
để chọn threshold.

Payoff
------
cost mỗi số:
    10,000 VND

payout nếu hit:
    800,000 VND

Nếu chọn m số:

    cost/bet day = m * 10,000

Break-even:

    hit_rate_BE
        = m * 10,000 / 800,000
        = m / 80

Random Top-m:

    P(hit) = m / 100

Expected random ROI:

    -20%

TIE BREAK
---------
Strategy phải chọn số cụ thể.

Nếu probability bằng nhau:
    dùng seeded deterministic priority.

Tie-break:
    - không dùng actual;
    - reproducible;
    - không leakage.

CONFIDENCE TIES
---------------
Nếu nhiều ngày tune có cùng q_m tại threshold:

    threshold được xác định từ quantile.

Tie ở đúng threshold được xử lý bằng
tie_acceptance_rate.

Khi áp dụng sang TEST:
    tie-break bằng seeded RNG độc lập actual.

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

    selective_topm_daily.csv

    selective_topm_bets.csv

    selective_topm_summary.csv

    selective_topm_thresholds.csv

    selective_topm_best_by_model.csv
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


TUNE_RATIO = 0.50


PROBABILITY_COLUMNS = [
    f"p_{number:02d}"
    for number
    in range(
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
# OUTPUT
# ============================================================

OUTPUT_DAILY = (
    STRATEGY_DIR
    / "selective_topm_daily.csv"
)


OUTPUT_BETS = (
    STRATEGY_DIR
    / "selective_topm_bets.csv"
)


OUTPUT_SUMMARY = (
    STRATEGY_DIR
    / "selective_topm_summary.csv"
)


OUTPUT_THRESHOLDS = (
    STRATEGY_DIR
    / "selective_topm_thresholds.csv"
)


OUTPUT_BEST_BY_MODEL = (
    STRATEGY_DIR
    / "selective_topm_best_by_model.csv"
)


# ============================================================
# TIE BREAK PRIORITY
# ============================================================

def create_tie_break_priority():
    """
    Priority cố định cho 100 class.

    Không dùng actual.
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
            "Có probability row tổng <= 0."
        )


    return (
        probabilities
        / row_sum
    )


# ============================================================
# RANK PROBABILITY
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
# VALIDATE
# ============================================================

def validate_probability_columns(
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
        for column
        in required
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


    return (
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


# ============================================================
# PREPARE MODEL
# ============================================================

def prepare_model_data(
    df,
):
    """
    Tính ranking chỉ một lần cho mỗi model.

    Sau đó m=1..30 dùng lại.
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


    order = (
        rank_probability_matrix(
            probabilities
        )
    )


    # ========================================================
    # INVERSE RANK
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
    # SORTED PROBABILITY
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
# PREPARE TOP-M
# ============================================================

def prepare_topm(
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

            "m": (
                m
            ),

            "q_m": (
                q_m
            ),

            "hit_if_bet": (
                hit
            ),

            "selected_numbers": (
                selected_numbers
            ),
        }
    )


# ============================================================
# CHRONOLOGICAL TUNE / TEST
# ============================================================

def chronological_split(
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
    )


    split_index = int(
        len(
            df
        )
        * TUNE_RATIO
    )


    tune = (
        df
        .iloc[
            :split_index
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    test = (
        df
        .iloc[
            split_index:
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    return (
        tune,
        test,
    )


# ============================================================
# CONFIDENCE RULE
# ============================================================

def calculate_confidence_rule(
    tune,
    target_rate,
):
    """
    Tìm threshold sao cho khoảng target_rate
    của tune observations có q_m cao nhất.

    Return:
        threshold
        tie_acceptance_rate
        actual_tune_participation
    """

    q = (
        tune[
            "q_m"
        ]
        .to_numpy(
            dtype=float
        )
    )


    if len(
        q
    ) == 0:

        return (
            np.nan,
            0.0,
            0.0,
        )


    # ========================================================
    # CONSTANT CONFIDENCE
    #
    # Không có khả năng phân biệt confidence.
    # Không bet.
    # ========================================================

    if np.allclose(
        q,
        q[0],
        rtol=1e-12,
        atol=1e-15,
    ):

        return (
            np.nan,
            0.0,
            0.0,
        )


    threshold = float(
        np.quantile(
            q,
            1.0
            - target_rate,
        )
    )


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


    greater_count = int(
        greater.sum()
    )


    equal_count = int(
        equal.sum()
    )


    target_count = (
        target_rate
        * len(
            q
        )
    )


    needed_from_ties = (
        target_count
        - greater_count
    )


    if (
        equal_count
        > 0
    ):

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
            0.0,
            1.0,
        )
    )


    expected_count = (
        greater_count

        + tie_acceptance_rate
        * equal_count
    )


    actual_tune_participation = (
        expected_count
        / len(
            q
        )
    )


    return (
        threshold,
        tie_acceptance_rate,
        actual_tune_participation,
    )


# ============================================================
# APPLY CONFIDENCE RULE
# ============================================================

def apply_confidence_rule(
    test,
    threshold,
    tie_acceptance_rate,
    seed,
):

    q = (
        test[
            "q_m"
        ]
        .to_numpy(
            dtype=float
        )
    )


    bet = np.zeros(
        len(
            test
        ),
        dtype=np.int8,
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


    tie_indices = np.flatnonzero(
        equal
    )


    if (
        len(
            tie_indices
        )
        > 0

        and tie_acceptance_rate
        > 0
    ):

        rng = np.random.default_rng(
            seed
        )


        random_values = rng.random(
            len(
                tie_indices
            )
        )


        accepted = (
            tie_indices[
                random_values
                < tie_acceptance_rate
            ]
        )


        bet[
            accepted
        ] = 1


    return bet


# ============================================================
# STABLE SEED
# ============================================================

def build_seed(
    model_key,
    fold,
    m,
    target_rate,
):
    """
    Seed deterministic.

    Không dùng Python hash()
    vì hash có thể thay đổi giữa process.
    """

    text = (
        f"{model_key}|"
        f"{fold}|"
        f"{m}|"
        f"{target_rate:.6f}"
    )


    text_value = sum(
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
        + text_value
    )


# ============================================================
# RUN ONE FOLD
# ============================================================

def run_selective_fold(
    prepared_topm,
    model_key,
    fold,
    m,
    target_rate,
):

    fold_data = (
        prepared_topm
        .loc[
            prepared_topm[
                "fold"
            ]
            .eq(
                fold
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


    (
        tune,
        test,
    ) = (
        chronological_split(
            fold_data
        )
    )


    (
        threshold,
        tie_acceptance_rate,
        tune_participation_rate,
    ) = (
        calculate_confidence_rule(
            tune,
            target_rate,
        )
    )


    # ========================================================
    # APPLY TO TEST
    # ========================================================

    seed = (
        build_seed(
            model_key=(
                model_key
            ),

            fold=(
                fold
            ),

            m=(
                m
            ),

            target_rate=(
                target_rate
            ),
        )
    )


    bet = (
        apply_confidence_rule(
            test=(
                test
            ),

            threshold=(
                threshold
            ),

            tie_acceptance_rate=(
                tie_acceptance_rate
            ),

            seed=(
                seed
            ),
        )
    )


    test[
        "bet"
    ] = (
        bet
    )


    # ========================================================
    # ECONOMICS
    # ========================================================

    daily_cost = (
        m
        * COST_PER_NUMBER
    )


    test[
        "cost"
    ] = (
        test[
            "bet"
        ]
        * daily_cost
    )


    test[
        "revenue"
    ] = (
        test[
            "bet"
        ]

        * test[
            "hit_if_bet"
        ]

        * PAYOUT_IF_HIT
    )


    test[
        "profit"
    ] = (
        test[
            "revenue"
        ]
        - test[
            "cost"
        ]
    )


    test[
        "model"
    ] = (
        model_key
    )


    test[
        "m"
    ] = (
        m
    )


    test[
        "target_participation_rate"
    ] = (
        target_rate
    )


    test[
        "threshold"
    ] = (
        threshold
    )


    test[
        "tie_acceptance_rate"
    ] = (
        tie_acceptance_rate
    )


    test[
        "tune_participation_rate"
    ] = (
        tune_participation_rate
    )


    test[
        "threshold_status"
    ] = np.where(
        np.isnan(
            threshold
        ),
        "constant_confidence",
        "valid",
    )


    # ========================================================
    # THRESHOLD RECORD
    # ========================================================

    threshold_record = {
        "model": (
            model_key
        ),

        "fold": (
            fold
        ),

        "m": (
            m
        ),

        "target_participation_rate": (
            target_rate
        ),

        "n_fold": (
            len(
                fold_data
            )
        ),

        "n_tune": (
            len(
                tune
            )
        ),

        "n_test": (
            len(
                test
            )
        ),

        "tune_start": (
            tune[
                "date"
            ]
            .min()
        ),

        "tune_end": (
            tune[
                "date"
            ]
            .max()
        ),

        "test_start": (
            test[
                "date"
            ]
            .min()
        ),

        "test_end": (
            test[
                "date"
            ]
            .max()
        ),

        "threshold": (
            threshold
        ),

        "tie_acceptance_rate": (
            tie_acceptance_rate
        ),

        "target_tune_participation_rate": (
            target_rate
        ),

        "expected_tune_participation_rate": (
            tune_participation_rate
        ),

        "test_bet_count": int(
            test[
                "bet"
            ]
            .sum()
        ),

        "test_participation_rate": float(
            test[
                "bet"
            ]
            .mean()
        ),

        "threshold_status": (
            (
                "constant_confidence"
                if np.isnan(
                    threshold
                )
                else "valid"
            )
        ),
    }


    return (
        test,
        threshold_record,
    )


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


    cumulative = np.cumsum(
        profit
    )


    equity = np.concatenate(
        [
            np.array(
                [
                    0.0
                ]
            ),

            cumulative,
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
# SUMMARIZE CONFIG
# ============================================================

def summarize_config(
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


    target_rate = float(
        daily[
            "target_participation_rate"
        ]
        .iloc[0]
    )


    n_test_days = len(
        daily
    )


    n_bets = int(
        daily[
            "bet"
        ]
        .sum()
    )


    n_hits = int(
        (
            daily[
                "bet"
            ]

            * daily[
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
        if total_cost > 0
        else np.nan
    )


    random_expected_roi = (
        -0.20
    )


    mean_q_all = float(
        daily[
            "q_m"
        ]
        .mean()
    )


    bet_subset = (
        daily.loc[
            daily[
                "bet"
            ]
            .eq(
                1
            )
        ]
    )


    if len(
        bet_subset
    ) > 0:

        mean_q_bet = float(
            bet_subset[
                "q_m"
            ]
            .mean()
        )

    else:

        mean_q_bet = (
            np.nan
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

        "hit_rate_lift_pp": (
            (
                hit_rate
                - random_hit_rate
            )
            if not np.isnan(
                hit_rate
            )
            else np.nan
        ),

        "break_even_hit_rate": (
            break_even_hit_rate
        ),

        "gap_to_break_even": (
            (
                hit_rate
                - break_even_hit_rate
            )
            if not np.isnan(
                hit_rate
            )
            else np.nan
        ),

        "mean_q_all_test": (
            mean_q_all
        ),

        "mean_q_bet_days": (
            mean_q_bet
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
            (
                roi
                - random_expected_roi
            )
            if not np.isnan(
                roi
            )
            else np.nan
        ),

        "max_drawdown": (
            calculate_max_drawdown(
                daily[
                    "profit"
                ]
                .to_numpy()
            )
        ),
    }


# ============================================================
# BEST BY MODEL
# ============================================================

def build_best_by_model(
    summary,
):
    """
    Best observed selective config.

    Post-hoc descriptive only.

    Robustness sẽ kiểm tra ở bước 16.
    """

    valid = (
        summary
        .loc[
            summary[
                "n_bets"
            ]
            .gt(
                0
            )
        ]
        .copy()
    )


    if valid.empty:

        return valid


    best = (
        valid
        .sort_values(
            [
                "model",
                "roi",
                "total_profit",
                "n_bets",
            ],
            ascending=[
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

    if best.empty:

        print(
            "Không có config nào phát sinh bet."
        )

        return


    columns = [
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
        "random_expected_roi",
        "roi_gain_vs_random",
        "max_drawdown",
    ]


    print(
        "\n"
        + "=" * 210
    )


    print(
        "SELECTIVE TOP-M - "
        "BEST OBSERVED CONFIGURATION BY MODEL"
    )


    print(
        "=" * 210
    )


    print(
        best[
            columns
        ]
        .to_string(
            index=False,

            formatters={
                "target_participation_rate": (
                    "{:.0%}".format
                ),

                "participation_rate": (
                    "{:.2%}".format
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


    print(
        "\nModels:"
    )


    for model_key in (
        predictions[
            "model_key"
        ]
        .unique()
    ):

        print(
            f"  - {model_key}"
        )


    print(
        f"\nM grid: "
        f"{min(M_VALUES)} -> "
        f"{max(M_VALUES)}"
    )


    print(
        "Participation rates: "
        + ", ".join(
            f"{rate:.0%}"
            for rate
            in TARGET_PARTICIPATION_RATES
        )
    )


    print(
        f"Tune ratio: "
        f"{TUNE_RATIO:.0%}"
    )


    # ========================================================
    # RUN
    # ========================================================

    daily_frames = []

    threshold_records = []


    for (
        model_key,
        model_subset,
    ) in predictions.groupby(
        "model_key",
        sort=False,
    ):

        print(
            "\n"
            + "=" * 100
        )

        print(
            f"MODEL: "
            f"{model_key}"
        )

        print(
            "=" * 100
        )


        prepared = (
            prepare_model_data(
                model_subset
            )
        )


        folds = (
            prepared[
                "df"
            ][
                "fold"
            ]
            .drop_duplicates()
            .tolist()
        )


        for m in M_VALUES:

            topm_data = (
                prepare_topm(
                    prepared,
                    m,
                )
            )


            for target_rate in (
                TARGET_PARTICIPATION_RATES
            ):

                for fold in folds:

                    (
                        daily,
                        threshold_record,
                    ) = (
                        run_selective_fold(
                            prepared_topm=(
                                topm_data
                            ),

                            model_key=(
                                model_key
                            ),

                            fold=(
                                fold
                            ),

                            m=(
                                m
                            ),

                            target_rate=(
                                target_rate
                            ),
                        )
                    )


                    daily_frames.append(
                        daily
                    )


                    threshold_records.append(
                        threshold_record
                    )


        print(
            f"Completed "
            f"{len(M_VALUES)} m values x "
            f"{len(TARGET_PARTICIPATION_RATES)} rates"
        )


    # ========================================================
    # COMBINE DAILY
    # ========================================================

    daily_results = pd.concat(
        daily_frames,
        ignore_index=True,
    )


    thresholds = pd.DataFrame(
        threshold_records
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary_records = []


    for (
        model,
        m,
        target_rate,
    ), subset in daily_results.groupby(
        [
            "model",
            "m",
            "target_participation_rate",
        ],
        sort=False,
    ):

        summary_records.append(
            summarize_config(
                subset
            )
        )


    summary = pd.DataFrame(
        summary_records
    )


    # ========================================================
    # BET-ONLY TABLE
    # ========================================================

    bets = (
        daily_results
        .loc[
            daily_results[
                "bet"
            ]
            .eq(
                1
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # BEST
    # ========================================================

    best_by_model = (
        build_best_by_model(
            summary
        )
    )


    # ========================================================
    # PRINT
    # ========================================================

    print(
        "\n"
        f"Total configurations: "
        f"{len(summary):,}"
    )


    print(
        f"Total daily rows: "
        f"{len(daily_results):,}"
    )


    print(
        f"Total bet rows: "
        f"{len(bets):,}"
    )


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


    bets.to_csv(
        OUTPUT_BETS,
        index=False,
        encoding="utf-8-sig",
    )


    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )


    thresholds.to_csv(
        OUTPUT_THRESHOLDS,
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
        OUTPUT_BETS
    )


    print(
        OUTPUT_SUMMARY
    )


    print(
        OUTPUT_THRESHOLDS
    )


    print(
        OUTPUT_BEST_BY_MODEL
    )


if __name__ == "__main__":
    main()