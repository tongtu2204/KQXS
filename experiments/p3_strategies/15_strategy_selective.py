"""
Strategy 15: Selective Top-m

Quét:
    m = 1 ... 30

Với mỗi m:
    - Xếp 100 số theo probability.
    - Tính:
          q_m(t) = tổng probability của Top-m.
    - Chỉ chơi vào những ngày có q_m cao.

Selective rates:
    10%, 20%, 30%, 50%

Trong từng fold:
    - 50% đầu: tune threshold
    - 50% sau: test

Payoff:
    cost per number = 10,000 VND
    payout if hit   = 800,000 VND

Break-even:
    hit_rate_BE = m * 10,000 / 800,000
                = m / 80

Random Top-m:
    hit rate = m / 100
    expected ROI = -20%
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

M_VALUES = list(range(1, 31))

TARGET_PARTICIPATION_RATES = [
    0.10,
    0.20,
    0.30,
    0.50,
]

TUNE_RATIO = 0.50

RANDOM_STATE = 42


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


# ============================================================
# TIE BREAK
# ============================================================

def create_tie_break_priority() -> np.ndarray:

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
    df: pd.DataFrame,
    model_name: str,
) -> None:

    required = [
        "date",
        "fold",
        "actual",
    ] + PROBABILITY_COLUMNS

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{model_name} thiếu cột:\n"
            f"{missing}"
        )


def load_single_model(
    model_name: str,
    file_path: Path,
) -> pd.DataFrame:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Không tìm thấy:\n"
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

    if model_name == "rolling_cdm":

        if "window" not in df.columns:

            raise ValueError(
                "rolling_cdm thiếu cột window."
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
    df: pd.DataFrame,
) -> dict:

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
        "df": df,
        "actual": actual,
        "order": order,
        "cumulative_probability": (
            cumulative_probability
        ),
    }


# ============================================================
# PREPARE ONE m
# ============================================================

def prepare_topm(
    prepared: dict,
    m: int,
) -> pd.DataFrame:

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
            "cumulative_probability"
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

    selected_numbers = [
        ",".join(
            f"{number:02d}"
            for number in selected
        )

        for selected
        in top_m
    ]

    return pd.DataFrame(
        {
            "date": (
                df["date"]
                .to_numpy()
            ),

            "fold": (
                df["fold"]
                .to_numpy()
            ),

            "actual": (
                actual
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
# SPLIT TUNE / TEST
# ============================================================

def chronological_split(
    df: pd.DataFrame,
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
        len(df)
        * TUNE_RATIO
    )

    tune = (
        df.iloc[
            :split_index
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    test = (
        df.iloc[
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
# CONFIDENCE THRESHOLD
# ============================================================

def calculate_confidence_rule(
    tune: pd.DataFrame,
    target_rate: float,
):

    q = (
        tune[
            "q_m"
        ]
        .to_numpy(
            dtype=float
        )
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
            1 - target_rate,
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
            0.0,
            1.0,
        )
    )

    return (
        threshold,
        tie_acceptance_rate,
    )


def apply_confidence_rule(
    test: pd.DataFrame,
    threshold: float,
    tie_acceptance_rate: float,
    seed: int,
) -> np.ndarray:

    q = (
        test[
            "q_m"
        ]
        .to_numpy(
            dtype=float
        )
    )

    bet = np.zeros(
        len(test),
        dtype=int,
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

    bet[
        greater
    ] = 1

    tie_indices = (
        np.flatnonzero(
            equal
        )
    )

    if (
        len(tie_indices) > 0
        and tie_acceptance_rate > 0
    ):

        rng = np.random.default_rng(
            seed
        )

        random_values = (
            rng.random(
                len(
                    tie_indices
                )
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
# RUN ONE FOLD
# ============================================================

def run_selective_fold(
    prepared_topm: pd.DataFrame,
    model_key: str,
    fold,
    m: int,
    target_rate: float,
):

    fold_data = (
        prepared_topm.loc[
            prepared_topm[
                "fold"
            ].eq(
                fold
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    (
        tune,
        test,
    ) = chronological_split(
        fold_data
    )

    (
        threshold,
        tie_acceptance_rate,
    ) = calculate_confidence_rule(
        tune,
        target_rate,
    )

    if np.isnan(
        threshold
    ):

        test[
            "bet"
        ] = 0

        threshold_status = (
            "constant_confidence"
        )

    else:

        seed = (
            RANDOM_STATE
            + m * 10_000
            + int(
                target_rate
                * 1_000
            )
            + sum(
                ord(character)
                for character
                in str(fold)
            )
        )

        test[
            "bet"
        ] = (
            apply_confidence_rule(
                test=test,
                threshold=threshold,
                tie_acceptance_rate=(
                    tie_acceptance_rate
                ),
                seed=seed,
            )
        )

        threshold_status = (
            "valid"
        )

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
        "threshold_status"
    ] = (
        threshold_status
    )

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

        "threshold": (
            threshold
        ),

        "tie_acceptance_rate": (
            tie_acceptance_rate
        ),

        "threshold_status": (
            threshold_status
        ),

        "tune_size": (
            len(tune)
        ),

        "test_size": (
            len(test)
        ),

        "actual_participation_rate": float(
            test[
                "bet"
            ].mean()
        ),
    }

    return (
        test,
        threshold_record,
    )


# ============================================================
# SUMMARY
# ============================================================

def summarize_configuration(
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

    target_rate = float(
        daily[
            "target_participation_rate"
        ].iloc[0]
    )

    n_eval_days = len(
        daily
    )

    bets = (
        daily.loc[
            daily[
                "bet"
            ].eq(1)
        ]
        .copy()
    )

    n_bet_days = len(
        bets
    )

    number_hits = int(
        bets[
            "hit_if_bet"
        ].sum()
    )

    if n_bet_days > 0:

        hit_rate = (
            number_hits
            / n_bet_days
        )

        mean_q_m_when_bet = float(
            bets[
                "q_m"
            ].mean()
        )

    else:

        hit_rate = np.nan

        mean_q_m_when_bet = (
            np.nan
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

    random_hit_rate = (
        m
        / NUMBER_OF_CLASSES
    )

    break_even_hit_rate = (
        m
        * COST_PER_NUMBER
        / PAYOUT_IF_HIT
    )

    random_expected_roi = (
        (
            random_hit_rate
            * PAYOUT_IF_HIT
            - m
            * COST_PER_NUMBER
        )
        / (
            m
            * COST_PER_NUMBER
        )
    )

    actual_participation_rate = (
        n_bet_days
        / n_eval_days
        if n_eval_days > 0
        else np.nan
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

        "n_eval_days": (
            n_eval_days
        ),

        "n_bet_days": (
            n_bet_days
        ),

        "actual_participation_rate": (
            actual_participation_rate
        ),

        "number_hits": (
            number_hits
        ),

        "hit_rate_when_bet": (
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
            if not np.isnan(
                hit_rate
            )
            else np.nan
        ),

        "mean_q_m_when_bet": (
            mean_q_m_when_bet
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
            if not np.isnan(
                roi
            )
            else np.nan
        ),
    }


# ============================================================
# BEST CONFIG PER MODEL
# ============================================================

def get_best_configuration_per_model(
    summary: pd.DataFrame,
) -> pd.DataFrame:

    valid = (
        summary.loc[
            summary[
                "roi"
            ].notna()
        ]
        .copy()
    )

    idx = (
        valid
        .groupby(
            "model"
        )[
            "roi"
        ]
        .idxmax()
    )

    return (
        valid
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


# ============================================================
# PRINT
# ============================================================

def print_best_summary(
    best: pd.DataFrame,
) -> None:

    columns = [
        "model",
        "m",
        "target_participation_rate",
        "n_bet_days",
        "actual_participation_rate",
        "number_hits",
        "hit_rate_when_bet",
        "random_hit_rate",
        "break_even_hit_rate",
        "gap_to_break_even",
        "mean_q_m_when_bet",
        "total_profit",
        "roi",
        "roi_gain_vs_random",
    ]

    print(
        "\n"
        + "=" * 200
    )

    print(
        "BEST SELECTIVE CONFIGURATION "
        "THEO TỪNG MODEL"
    )

    print(
        "=" * 200
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

                "actual_participation_rate": (
                    "{:.2%}".format
                ),

                "hit_rate_when_bet": (
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

                "mean_q_m_when_bet": (
                    "{:.4%}".format
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
            },
        )
    )


def print_top_configurations(
    summary: pd.DataFrame,
    n: int = 30,
) -> None:

    valid = (
        summary.loc[
            summary[
                "roi"
            ].notna()
        ]
        .sort_values(
            "roi",
            ascending=False,
        )
        .head(
            n
        )
    )

    columns = [
        "model",
        "m",
        "target_participation_rate",
        "n_bet_days",
        "actual_participation_rate",
        "number_hits",
        "hit_rate_when_bet",
        "break_even_hit_rate",
        "total_profit",
        "roi",
        "roi_gain_vs_random",
    ]

    print(
        "\n"
        + "=" * 180
    )

    print(
        f"TOP {n} SELECTIVE CONFIGURATIONS"
    )

    print(
        "=" * 180
    )

    print(
        valid[
            columns
        ]
        .to_string(
            index=False,
            formatters={
                "target_participation_rate": (
                    "{:.0%}".format
                ),

                "actual_participation_rate": (
                    "{:.2%}".format
                ),

                "hit_rate_when_bet": (
                    "{:.4%}".format
                ),

                "break_even_hit_rate": (
                    "{:.2%}".format
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
        f"{M_VALUES[0]} -> "
        f"{M_VALUES[-1]}"
    )

    print(
        "Selective rates: "
        + ", ".join(
            f"{rate:.0%}"
            for rate
            in TARGET_PARTICIPATION_RATES
        )
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

    threshold_records = []

    for (
        model_key,
        model_df,
    ) in predictions.groupby(
        "model_key"
    ):

        print(
            f"\nRunning model: "
            f"{model_key}"
        )

        prepared_model = (
            prepare_model_data(
                model_df
            )
        )

        for m in M_VALUES:

            prepared_topm = (
                prepare_topm(
                    prepared_model,
                    m,
                )
            )

            folds = (
                prepared_topm[
                    "fold"
                ]
                .drop_duplicates()
                .tolist()
            )

            for fold in folds:

                for target_rate in (
                    TARGET_PARTICIPATION_RATES
                ):

                    (
                        daily,
                        threshold_record,
                    ) = run_selective_fold(
                        prepared_topm=(
                            prepared_topm
                        ),
                        model_key=(
                            model_key
                        ),
                        fold=fold,
                        m=m,
                        target_rate=(
                            target_rate
                        ),
                    )

                    daily_frames.append(
                        daily
                    )

                    threshold_records.append(
                        threshold_record
                    )

    daily_results = pd.concat(
        daily_frames,
        ignore_index=True,
    )

    threshold_df = pd.DataFrame(
        threshold_records
    )

    summary_records = []

    grouped = (
        daily_results
        .groupby(
            [
                "model",
                "m",
                "target_participation_rate",
            ]
        )
    )

    for _, subset in grouped:

        summary_records.append(
            summarize_configuration(
                subset
            )
        )

    summary = pd.DataFrame(
        summary_records
    )

    best_by_model = (
        get_best_configuration_per_model(
            summary
        )
    )

    print_best_summary(
        best_by_model
    )

    print_top_configurations(
        summary,
        n=30,
    )

    # ========================================================
    # SAVE
    # ========================================================

    daily_path = (
        STRATEGY_DIR
        / "selective_topm_daily.csv"
    )

    bets_path = (
        STRATEGY_DIR
        / "selective_topm_bets.csv"
    )

    summary_path = (
        STRATEGY_DIR
        / "selective_topm_summary.csv"
    )

    threshold_path = (
        STRATEGY_DIR
        / "selective_topm_thresholds.csv"
    )

    best_path = (
        STRATEGY_DIR
        / "selective_topm_best_by_model.csv"
    )

    daily_results.to_csv(
        daily_path,
        index=False,
        encoding="utf-8-sig",
    )

    daily_results.loc[
        daily_results[
            "bet"
        ].eq(1)
    ].to_csv(
        bets_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    threshold_df.to_csv(
        threshold_path,
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
        bets_path
    )

    print(
        summary_path
    )

    print(
        threshold_path
    )

    print(
        best_path
    )


if __name__ == "__main__":
    main()