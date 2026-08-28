"""
13 - PROBABILITY RANKING

So sánh probability ranking của các mô hình 00-99.

Models
------
- Uniform
- CatBoost static
- CatBoost periodic retrain
- CDM
- Rolling CDM:
      30 / 60 / 90 / 180 / 365
- Bayesian Markov

Mục tiêu
--------
1. So sánh Log Loss.
2. So sánh mean probability của actual.
3. Đánh giá ranking của actual.
4. Top-k theo tie-aware fractional scoring.
5. So sánh với Uniform baseline.
6. Chuẩn bị model probability cho strategy.

TIE-AWARE
---------
Nếu probability của actual bị tie với nhiều class:

    n_greater
        = số class có probability > actual probability

    n_equal
        = số class có probability = actual probability
          bao gồm actual.

Average rank:

    rank =
        1
        + n_greater
        + (n_equal - 1) / 2

Ví dụ Uniform:

    n_greater = 0
    n_equal = 100

=> average rank = 50.5

Fractional Top-k:

Nếu một tie group cắt qua ngưỡng k:

    score =
        phần xác suất actual được nằm trong Top-k
        nếu tie được phá ngẫu nhiên.

Uniform vì vậy cho chính xác:

    Top-1  = 1%
    Top-5  = 5%
    Top-10 = 10%

Không còn phụ thuộc vào thứ tự argsort.
"""

from pathlib import Path

import matplotlib.pyplot as plt
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

FIGURE_DIR = (
    PROJECT_DIR
    / "artifacts"
    / "figures"
)


NUMBER_OF_CLASSES = 100

EPSILON = 1e-15


UNIFORM_LOG_LOSS = (
    np.log(
        NUMBER_OF_CLASSES
    )
)


TOP_K_VALUES = [
    1,
    3,
    5,
    10,
    20,
]


PROBABILITY_COLUMNS = [
    f"p_{number:02d}"
    for number
    in range(
        NUMBER_OF_CLASSES
    )
]


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
    TABLE_DIR
    / "probability_ranking_daily.csv"
)

OUTPUT_SUMMARY = (
    TABLE_DIR
    / "probability_ranking_summary.csv"
)

OUTPUT_FOLD_SUMMARY = (
    TABLE_DIR
    / "probability_ranking_by_fold.csv"
)

OUTPUT_RANK_CDF = (
    TABLE_DIR
    / "probability_rank_cdf.csv"
)

OUTPUT_SELECTED_RANKS = (
    TABLE_DIR
    / "probability_selected_ranks.csv"
)

OUTPUT_STRATEGY_CANDIDATES = (
    TABLE_DIR
    / "probability_strategy_candidates.csv"
)


# ============================================================
# VALIDATE
# ============================================================

def validate_file(
    file_path: Path,
):

    if not file_path.exists():

        raise FileNotFoundError(
            f"Không tìm thấy file:\n"
            f"{file_path}"
        )


def validate_columns(
    df: pd.DataFrame,
    model_name: str,
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
# LOAD MODEL
# ============================================================

def load_single_model(
    model_name: str,
    file_path: Path,
):

    validate_file(
        file_path
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


    # ========================================================
    # MODEL KEY
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
                "thiếu cột window."
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
# NORMALIZE PROBABILITY
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
# TIE-AWARE RANKING
# ============================================================

def tie_aware_metrics(
    probabilities,
    actual,
):
    """
    Return:
        actual_probability
        average_rank
        n_greater
        n_equal
        top-k fractional scores
    """

    probabilities = (
        normalize_probability_matrix(
            probabilities
        )
    )

    actual = np.asarray(
        actual,
        dtype=int,
    )


    n_rows = len(
        actual
    )


    actual_probability = (
        probabilities[
            np.arange(
                n_rows
            ),
            actual,
        ]
    )


    # ========================================================
    # COMPARE WITH ACTUAL PROBABILITY
    # ========================================================

    actual_probability_column = (
        actual_probability[
            :, None
        ]
    )


    greater_mask = (
        probabilities
        > (
            actual_probability_column
            + EPSILON
        )
    )


    equal_mask = (
        np.isclose(
            probabilities,
            actual_probability_column,
            rtol=1e-12,
            atol=1e-15,
        )
    )


    n_greater = (
        greater_mask
        .sum(
            axis=1
        )
        .astype(int)
    )


    n_equal = (
        equal_mask
        .sum(
            axis=1
        )
        .astype(int)
    )


    # safety
    n_equal = np.maximum(
        n_equal,
        1,
    )


    # ========================================================
    # AVERAGE RANK
    # ========================================================

    average_rank = (
        1.0
        + n_greater
        + (
            n_equal
            - 1
        )
        / 2.0
    )


    # ========================================================
    # FRACTIONAL TOP-K
    # ========================================================

    top_k_scores = {}


    for k in TOP_K_VALUES:

        scores = np.zeros(
            n_rows,
            dtype=float,
        )


        # -----------------------------------------------
        # actual tie group entirely inside Top-k
        # -----------------------------------------------

        completely_inside = (
            (
                n_greater
                + n_equal
            )
            <= k
        )


        scores[
            completely_inside
        ] = 1.0


        # -----------------------------------------------
        # tie group intersects boundary
        # -----------------------------------------------

        crossing = (
            (n_greater < k)
            & (
                (
                    n_greater
                    + n_equal
                )
                > k
            )
        )


        scores[
            crossing
        ] = (
            (
                k
                - n_greater[
                    crossing
                ]
            )
            / n_equal[
                crossing
            ]
        )


        # -----------------------------------------------
        # if n_greater >= k:
        # score remains 0
        # -----------------------------------------------

        top_k_scores[
            k
        ] = scores


    return (
        actual_probability,
        average_rank,
        n_greater,
        n_equal,
        top_k_scores,
    )


# ============================================================
# LOG LOSS
# ============================================================

def row_log_loss(
    actual_probability,
):

    actual_probability = (
        np.clip(
            actual_probability,
            EPSILON,
            1.0,
        )
    )


    return (
        -np.log(
            actual_probability
        )
    )


# ============================================================
# PROCESS MODEL
# ============================================================

def process_model(
    model_key,
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


    (
        actual_probability,
        average_rank,
        n_greater,
        n_equal,
        top_k_scores,
    ) = (
        tie_aware_metrics(
            probabilities,
            actual,
        )
    )


    model_loss = (
        row_log_loss(
            actual_probability
        )
    )


    uniform_loss = np.full(
        len(
            df
        ),
        UNIFORM_LOG_LOSS,
        dtype=float,
    )


    improvement = (
        uniform_loss
        - model_loss
    )


    # ========================================================
    # DAILY OUTPUT
    # ========================================================

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

            "actual": (
                actual
            ),

            "actual_probability": (
                actual_probability
            ),

            "actual_rank": (
                average_rank
            ),

            "n_greater": (
                n_greater
            ),

            "n_equal": (
                n_equal
            ),

            "model_loss": (
                model_loss
            ),

            "uniform_loss": (
                uniform_loss
            ),

            "improvement": (
                improvement
            ),
        }
    )


    for k in TOP_K_VALUES:

        output[
            f"top_{k}_score"
        ] = (
            top_k_scores[
                k
            ]
        )


    return output


# ============================================================
# BUILD DAILY
# ============================================================

def build_daily_ranking(
    predictions,
):

    frames = []


    for (
        model_key,
        subset,
    ) in predictions.groupby(
        "model_key",
        sort=False,
    ):

        print(
            f"Ranking: "
            f"{model_key}"
        )


        frames.append(
            process_model(
                model_key,
                subset,
            )
        )


    daily = pd.concat(
        frames,
        ignore_index=True,
    )


    return (
        daily
        .sort_values(
            [
                "model",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# SUMMARY
# ============================================================

def calculate_summary(
    df,
    model_key,
):

    ranks = (
        df[
            "actual_rank"
        ]
        .to_numpy(
            dtype=float
        )
    )


    record = {
        "model": (
            model_key
        ),

        "n_test": (
            len(
                df
            )
        ),

        "log_loss": float(
            df[
                "model_loss"
            ]
            .mean()
        ),

        "uniform_log_loss": (
            UNIFORM_LOG_LOSS
        ),

        "log_loss_gain_vs_uniform": float(
            UNIFORM_LOG_LOSS
            - df[
                "model_loss"
            ]
            .mean()
        ),

        "relative_log_loss_gain_pct": float(
            (
                UNIFORM_LOG_LOSS
                - df[
                    "model_loss"
                ]
                .mean()
            )
            / UNIFORM_LOG_LOSS
            * 100
        ),

        "mean_true_rank": float(
            np.mean(
                ranks
            )
        ),

        "median_true_rank": float(
            np.median(
                ranks
            )
        ),

        "true_rank_q25": float(
            np.quantile(
                ranks,
                0.25,
            )
        ),

        "true_rank_q75": float(
            np.quantile(
                ranks,
                0.75,
            )
        ),

        "true_rank_q90": float(
            np.quantile(
                ranks,
                0.90,
            )
        ),

        "mean_actual_probability": float(
            df[
                "actual_probability"
            ]
            .mean()
        ),

        "mean_tie_size": float(
            df[
                "n_equal"
            ]
            .mean()
        ),

        "max_tie_size": int(
            df[
                "n_equal"
            ]
            .max()
        ),
    }


    for k in TOP_K_VALUES:

        accuracy = float(
            df[
                f"top_{k}_score"
            ]
            .mean()
        )


        random_accuracy = (
            k
            / NUMBER_OF_CLASSES
        )


        record[
            f"top_{k}_accuracy"
        ] = (
            accuracy
        )


        record[
            f"top_{k}_random"
        ] = (
            random_accuracy
        )


        record[
            f"top_{k}_lift_pp"
        ] = (
            accuracy
            - random_accuracy
        )


        record[
            f"top_{k}_lift_ratio"
        ] = (
            accuracy
            / random_accuracy
        )


    return record


def build_model_summary(
    daily,
):

    records = []


    for (
        model_key,
        subset,
    ) in daily.groupby(
        "model",
        sort=False,
    ):

        records.append(
            calculate_summary(
                subset,
                model_key,
            )
        )


    # ========================================================
    # UNIFORM THEORETICAL
    # ========================================================

    uniform = {
        "model": (
            "uniform"
        ),

        "n_test": (
            daily[
                "date"
            ]
            .nunique()
        ),

        "log_loss": (
            UNIFORM_LOG_LOSS
        ),

        "uniform_log_loss": (
            UNIFORM_LOG_LOSS
        ),

        "log_loss_gain_vs_uniform": (
            0.0
        ),

        "relative_log_loss_gain_pct": (
            0.0
        ),

        "mean_true_rank": (
            50.5
        ),

        "median_true_rank": (
            50.5
        ),

        "true_rank_q25": (
            50.5
        ),

        "true_rank_q75": (
            50.5
        ),

        "true_rank_q90": (
            50.5
        ),

        "mean_actual_probability": (
            0.01
        ),

        "mean_tie_size": (
            100.0
        ),

        "max_tie_size": (
            100
        ),
    }


    for k in TOP_K_VALUES:

        random_accuracy = (
            k
            / NUMBER_OF_CLASSES
        )


        uniform[
            f"top_{k}_accuracy"
        ] = (
            random_accuracy
        )

        uniform[
            f"top_{k}_random"
        ] = (
            random_accuracy
        )

        uniform[
            f"top_{k}_lift_pp"
        ] = (
            0.0
        )

        uniform[
            f"top_{k}_lift_ratio"
        ] = (
            1.0
        )


    records.append(
        uniform
    )


    summary = pd.DataFrame(
        records
    )


    return (
        summary
        .sort_values(
            [
                "log_loss",
                "mean_true_rank",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# FOLD SUMMARY
# ============================================================

def build_fold_summary(
    daily,
):

    records = []


    for (
        model_key,
        fold,
    ), subset in daily.groupby(
        [
            "model",
            "fold",
        ],
        sort=False,
    ):

        record = {
            "model": (
                model_key
            ),

            "fold": (
                fold
            ),

            "n_test": (
                len(
                    subset
                )
            ),

            "log_loss": float(
                subset[
                    "model_loss"
                ]
                .mean()
            ),

            "log_loss_gain_vs_uniform": float(
                UNIFORM_LOG_LOSS
                - subset[
                    "model_loss"
                ]
                .mean()
            ),

            "mean_true_rank": float(
                subset[
                    "actual_rank"
                ]
                .mean()
            ),

            "median_true_rank": float(
                subset[
                    "actual_rank"
                ]
                .median()
            ),

            "mean_actual_probability": float(
                subset[
                    "actual_probability"
                ]
                .mean()
            ),
        }


        for k in TOP_K_VALUES:

            record[
                f"top_{k}_accuracy"
            ] = float(
                subset[
                    f"top_{k}_score"
                ]
                .mean()
            )


        records.append(
            record
        )


    return pd.DataFrame(
        records
    )


# ============================================================
# RANK CDF
# ============================================================

def build_rank_cdf(
    daily,
):
    """
    Tie-aware expected CDF.

    Với fractional average rank không nên dùng:
        rank <= k

    trực tiếp.

    Ta dùng luôn top-k fractional score đã tính.

    Với k ngoài TOP_K_VALUES:
        tính fractional score trực tiếp từ n_greater/n_equal.
    """

    records = []


    for (
        model_key,
        subset,
    ) in daily.groupby(
        "model",
        sort=False,
    ):

        n_greater = (
            subset[
                "n_greater"
            ]
            .to_numpy(
                dtype=int
            )
        )

        n_equal = (
            subset[
                "n_equal"
            ]
            .to_numpy(
                dtype=int
            )
        )


        for k in range(
            1,
            NUMBER_OF_CLASSES + 1,
        ):

            score = np.zeros(
                len(
                    subset
                ),
                dtype=float,
            )


            completely_inside = (
                (
                    n_greater
                    + n_equal
                )
                <= k
            )


            score[
                completely_inside
            ] = 1.0


            crossing = (
                (n_greater < k)
                & (
                    (
                        n_greater
                        + n_equal
                    )
                    > k
                )
            )


            score[
                crossing
            ] = (
                (
                    k
                    - n_greater[
                        crossing
                    ]
                )
                / n_equal[
                    crossing
                ]
            )


            empirical_cdf = float(
                score.mean()
            )


            uniform_cdf = (
                k
                / NUMBER_OF_CLASSES
            )


            records.append(
                {
                    "model": (
                        model_key
                    ),

                    "rank": (
                        k
                    ),

                    "empirical_cdf": (
                        empirical_cdf
                    ),

                    "uniform_cdf": (
                        uniform_cdf
                    ),

                    "lift_pp": (
                        empirical_cdf
                        - uniform_cdf
                    ),

                    "lift_ratio": (
                        empirical_cdf
                        / uniform_cdf
                    ),
                }
            )


    # ========================================================
    # UNIFORM
    # ========================================================

    for k in range(
        1,
        NUMBER_OF_CLASSES + 1,
    ):

        value = (
            k
            / NUMBER_OF_CLASSES
        )


        records.append(
            {
                "model": (
                    "uniform"
                ),

                "rank": (
                    k
                ),

                "empirical_cdf": (
                    value
                ),

                "uniform_cdf": (
                    value
                ),

                "lift_pp": (
                    0.0
                ),

                "lift_ratio": (
                    1.0
                ),
            }
        )


    return pd.DataFrame(
        records
    )


# ============================================================
# SELECTED RANK TABLE
# ============================================================

def build_selected_rank_table(
    rank_cdf,
):

    selected = (
        rank_cdf.loc[
            rank_cdf[
                "rank"
            ]
            .isin(
                TOP_K_VALUES
            )
        ]
        .copy()
    )


    selected[
        "empirical_pct"
    ] = (
        selected[
            "empirical_cdf"
        ]
        * 100
    )


    selected[
        "uniform_pct"
    ] = (
        selected[
            "uniform_cdf"
        ]
        * 100
    )


    selected[
        "lift_pp_pct"
    ] = (
        selected[
            "lift_pp"
        ]
        * 100
    )


    return selected[
        [
            "model",
            "rank",
            "empirical_pct",
            "uniform_pct",
            "lift_pp_pct",
            "lift_ratio",
        ]
    ]


# ============================================================
# STRATEGY CANDIDATE TABLE
# ============================================================

def build_strategy_candidate_table(
    summary,
):

    columns = [
        "model",
        "n_test",
        "log_loss",
        "log_loss_gain_vs_uniform",
        "top_1_accuracy",
        "top_3_accuracy",
        "top_5_accuracy",
        "top_10_accuracy",
        "top_20_accuracy",
        "mean_true_rank",
        "median_true_rank",
        "mean_actual_probability",
        "mean_tie_size",
    ]


    return (
        summary[
            columns
        ]
        .copy()
    )


# ============================================================
# PLOT
# ============================================================

def plot_rank_cdf(
    rank_cdf,
):

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    fig, ax = plt.subplots(
        figsize=(
            11,
            7,
        ),
        constrained_layout=True,
    )


    for (
        model_key,
        subset,
    ) in rank_cdf.groupby(
        "model",
        sort=False,
    ):

        ax.plot(
            subset[
                "rank"
            ],

            subset[
                "empirical_cdf"
            ],

            label=(
                model_key
            ),
        )


    ax.set_xlabel(
        "Top-k"
    )

    ax.set_ylabel(
        "Expected P(actual nằm trong Top-k)"
    )

    ax.set_title(
        "Tie-aware probability ranking"
    )

    ax.set_xlim(
        1,
        100,
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        fontsize=8
    )


    output_file = (
        FIGURE_DIR
        / "probability_rank_cdf.png"
    )


    fig.savefig(
        output_file,
        dpi=160,
    )

    plt.close(
        fig
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    summary,
):

    columns = [
        "model",
        "n_test",
        "log_loss",
        "log_loss_gain_vs_uniform",
        "top_1_accuracy",
        "top_3_accuracy",
        "top_5_accuracy",
        "top_10_accuracy",
        "top_20_accuracy",
        "mean_true_rank",
        "mean_tie_size",
    ]


    print(
        "\n"
        + "=" * 190
    )

    print(
        "PROBABILITY RANKING SUMMARY - TIE AWARE"
    )

    print(
        "=" * 190
    )


    print(
        summary[
            columns
        ]
        .to_string(
            index=False,

            formatters={
                "log_loss": (
                    "{:.6f}".format
                ),

                "log_loss_gain_vs_uniform": (
                    "{:+.6f}".format
                ),

                "top_1_accuracy": (
                    "{:.4%}".format
                ),

                "top_3_accuracy": (
                    "{:.4%}".format
                ),

                "top_5_accuracy": (
                    "{:.4%}".format
                ),

                "top_10_accuracy": (
                    "{:.4%}".format
                ),

                "top_20_accuracy": (
                    "{:.4%}".format
                ),

                "mean_true_rank": (
                    "{:.2f}".format
                ),

                "mean_tie_size": (
                    "{:.2f}".format
                ),
            },
        )
    )


# ============================================================
# PRINT CATBOOST COMPARISON
# ============================================================

def print_catboost_comparison(
    fold_summary,
):

    subset = (
        fold_summary.loc[
            fold_summary[
                "model"
            ]
            .isin(
                [
                    "catboost",
                    "catboost_retrain",
                ]
            )
        ]
        .copy()
    )


    if subset.empty:

        return


    columns = [
        "model",
        "fold",
        "n_test",
        "log_loss",
        "log_loss_gain_vs_uniform",
        "top_1_accuracy",
        "top_5_accuracy",
        "top_10_accuracy",
        "top_20_accuracy",
        "mean_true_rank",
    ]


    print(
        "\n"
        + "=" * 180
    )

    print(
        "CATBOOST STATIC VS RETRAIN - TIE AWARE"
    )

    print(
        "=" * 180
    )


    print(
        subset[
            columns
        ]
        .sort_values(
            [
                "fold",
                "model",
            ]
        )
        .to_string(
            index=False,

            formatters={
                "log_loss": (
                    "{:.6f}".format
                ),

                "log_loss_gain_vs_uniform": (
                    "{:+.6f}".format
                ),

                "top_1_accuracy": (
                    "{:.4%}".format
                ),

                "top_5_accuracy": (
                    "{:.4%}".format
                ),

                "top_10_accuracy": (
                    "{:.4%}".format
                ),

                "top_20_accuracy": (
                    "{:.4%}".format
                ),

                "mean_true_rank": (
                    "{:.2f}".format
                ),
            },
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
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
        f"\nRaw prediction rows: "
        f"{len(predictions):,}"
    )


    print(
        "Models:"
    )

    for model_name in (
        predictions[
            "model_key"
        ]
        .unique()
    ):

        print(
            f"  - {model_name}"
        )


    # ========================================================
    # DAILY TIE-AWARE
    # ========================================================

    daily = (
        build_daily_ranking(
            predictions
        )
    )


    print(
        f"\nRanking daily rows: "
        f"{len(daily):,}"
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = (
        build_model_summary(
            daily
        )
    )


    fold_summary = (
        build_fold_summary(
            daily
        )
    )


    rank_cdf = (
        build_rank_cdf(
            daily
        )
    )


    selected_ranks = (
        build_selected_rank_table(
            rank_cdf
        )
    )


    strategy_candidates = (
        build_strategy_candidate_table(
            summary
        )
    )


    # ========================================================
    # PRINT
    # ========================================================

    print_summary(
        summary
    )


    print_catboost_comparison(
        fold_summary
    )


    # ========================================================
    # SAVE
    # ========================================================

    daily.to_csv(
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
        OUTPUT_FOLD_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )


    rank_cdf.to_csv(
        OUTPUT_RANK_CDF,
        index=False,
        encoding="utf-8-sig",
    )


    selected_ranks.to_csv(
        OUTPUT_SELECTED_RANKS,
        index=False,
        encoding="utf-8-sig",
    )


    strategy_candidates.to_csv(
        OUTPUT_STRATEGY_CANDIDATES,
        index=False,
        encoding="utf-8-sig",
    )


    plot_rank_cdf(
        rank_cdf
    )


    # ========================================================
    # FINAL
    # ========================================================

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
        OUTPUT_FOLD_SUMMARY
    )

    print(
        OUTPUT_RANK_CDF
    )

    print(
        OUTPUT_SELECTED_RANKS
    )

    print(
        OUTPUT_STRATEGY_CANDIDATES
    )


if __name__ == "__main__":
    main()