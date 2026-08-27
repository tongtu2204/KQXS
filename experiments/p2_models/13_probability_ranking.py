"""Tổng hợp và so sánh probability ranking của các mô hình 00-99.

Models:
    - CatBoost 100-class
    - CDM full-history
    - Rolling CDM
    - Bayesian Markov
    - Uniform baseline

Mục tiêu:
    1. So sánh Top-k accuracy.
    2. Phân tích rank của kết quả thật.
    3. So sánh empirical rank CDF với Uniform:
           P(R <= k) = k / 100
    4. Tính lift so với random.
    5. Chuẩn bị output cho các strategy 14-16.
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

UNIFORM_LOG_LOSS = np.log(
    NUMBER_OF_CLASSES
)

TOP_K_VALUES = [
    1,
    3,
    5,
    10,
    20,
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
# LOAD
# ============================================================


def validate_file(
    file_path: Path,
) -> None:
    """Kiểm tra file input có tồn tại."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file:\n"
            f"{file_path}\n\n"
            "Hãy chạy các experiment 09-12 trước."
        )


def validate_columns(
    df: pd.DataFrame,
    model_name: str,
) -> None:
    """Kiểm tra các cột bắt buộc."""

    required_columns = [
        "date",
        "fold",
        "actual",
        "actual_probability",
        "actual_rank",
        "model_loss",
        "uniform_loss",
        "improvement",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{model_name} thiếu các cột: "
            f"{missing}"
        )


def load_single_model(
    model_name: str,
    file_path: Path,
) -> pd.DataFrame:
    """Đọc một probability prediction file."""

    validate_file(
        file_path
    )

    df = pd.read_csv(
        file_path,
        parse_dates=["date"],
    )

    validate_columns(
        df,
        model_name,
    )

    df = df.copy()

    df["source_model"] = (
        model_name
    )

    return df


def load_predictions() -> pd.DataFrame:
    """Đọc toàn bộ prediction output từ experiment 09-12."""

    frames = []

    for model_name, file_path in FILES.items():

        df = load_single_model(
            model_name,
            file_path,
        )

        # --------------------------------
        # Standard model key
        # --------------------------------

        if model_name == "rolling_cdm":

            if "window" not in df.columns:
                raise ValueError(
                    "rolling_cdm_predictions.csv "
                    "không có cột window."
                )

            df["model_key"] = (
                "rolling_cdm_w"
                + df["window"]
                .astype(int)
                .astype(str)
            )

        else:

            df["model_key"] = (
                model_name
            )

        frames.append(
            df
        )

    predictions = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    predictions = (
        predictions.sort_values(
            [
                "model_key",
                "date",
            ]
        )
        .reset_index(drop=True)
    )

    return predictions


# ============================================================
# BASIC METRICS
# ============================================================


def top_k_accuracy_from_rank(
    ranks: np.ndarray,
    k: int,
) -> float:
    """Top-k accuracy từ actual rank."""

    return float(
        np.mean(
            ranks <= k
        )
    )


def calculate_summary(
    df: pd.DataFrame,
    model_key: str,
) -> dict:
    """Tính summary cho một model."""

    ranks = (
        df["actual_rank"]
        .to_numpy()
    )

    record = {
        "model": model_key,

        "n_test": len(
            df
        ),

        "log_loss": float(
            df[
                "model_loss"
            ].mean()
        ),

        "uniform_log_loss": (
            UNIFORM_LOG_LOSS
        ),

        "log_loss_gain_vs_uniform": float(
            UNIFORM_LOG_LOSS
            - df[
                "model_loss"
            ].mean()
        ),

        "relative_log_loss_gain_pct": float(
            (
                UNIFORM_LOG_LOSS
                - df[
                    "model_loss"
                ].mean()
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
            ].mean()
        ),
    }

    for k in TOP_K_VALUES:

        accuracy = (
            top_k_accuracy_from_rank(
                ranks,
                k,
            )
        )

        random_accuracy = (
            k
            / NUMBER_OF_CLASSES
        )

        record[
            f"top_{k}_accuracy"
        ] = accuracy

        record[
            f"top_{k}_random"
        ] = random_accuracy

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
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Tổng hợp tất cả model."""

    records = []

    for model_key, subset in (
        predictions.groupby(
            "model_key"
        )
    ):

        records.append(
            calculate_summary(
                subset,
                model_key,
            )
        )

    # --------------------------------
    # Uniform theoretical baseline
    # --------------------------------

    uniform_record = {
        "model": "uniform",

        "n_test": (
            predictions[
                "date"
            ].nunique()
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
            25.25
        ),

        "true_rank_q75": (
            75.75
        ),

        "true_rank_q90": (
            90.1
        ),

        "mean_actual_probability": (
            1
            / NUMBER_OF_CLASSES
        ),
    }

    for k in TOP_K_VALUES:

        random_accuracy = (
            k
            / NUMBER_OF_CLASSES
        )

        uniform_record[
            f"top_{k}_accuracy"
        ] = random_accuracy

        uniform_record[
            f"top_{k}_random"
        ] = random_accuracy

        uniform_record[
            f"top_{k}_lift_pp"
        ] = 0.0

        uniform_record[
            f"top_{k}_lift_ratio"
        ] = 1.0

    records.append(
        uniform_record
    )

    summary = pd.DataFrame(
        records
    )

    summary = (
        summary.sort_values(
            "log_loss"
        )
        .reset_index(drop=True)
    )

    return summary


# ============================================================
# FOLD SUMMARY
# ============================================================


def build_fold_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """So sánh ranking theo từng fold."""

    records = []

    for (
        model_key,
        fold,
    ), subset in predictions.groupby(
        [
            "model_key",
            "fold",
        ]
    ):

        ranks = (
            subset[
                "actual_rank"
            ]
            .to_numpy()
        )

        record = {
            "model": model_key,
            "fold": fold,

            "n_test": len(
                subset
            ),

            "log_loss": float(
                subset[
                    "model_loss"
                ].mean()
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
        }

        for k in TOP_K_VALUES:

            record[
                f"top_{k}_accuracy"
            ] = (
                top_k_accuracy_from_rank(
                    ranks,
                    k,
                )
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
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tính empirical CDF:

        P(actual_rank <= k)

    với k = 1,...,100.
    """

    records = []

    for model_key, subset in (
        predictions.groupby(
            "model_key"
        )
    ):

        ranks = (
            subset[
                "actual_rank"
            ]
            .to_numpy()
        )

        for k in range(
            1,
            NUMBER_OF_CLASSES + 1,
        ):

            empirical_cdf = float(
                np.mean(
                    ranks <= k
                )
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

                    "rank": k,

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

    return pd.DataFrame(
        records
    )


# ============================================================
# SELECTED RANK TABLE
# ============================================================


def build_selected_rank_table(
    rank_cdf: pd.DataFrame,
) -> pd.DataFrame:
    """Bảng CDF tại các rank quan trọng."""

    selected = (
        rank_cdf.loc[
            rank_cdf[
                "rank"
            ].isin(
                TOP_K_VALUES
            )
        ]
        .copy()
    )

    selected["empirical_pct"] = (
        selected[
            "empirical_cdf"
        ]
        * 100
    )

    selected["uniform_pct"] = (
        selected[
            "uniform_cdf"
        ]
        * 100
    )

    selected["lift_pp_pct"] = (
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
# MODEL COMPARISON FOR STRATEGY
# ============================================================


def build_strategy_candidate_table(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tạo bảng ngắn để quyết định model nào
    đáng chuyển sang strategy 14-16.
    """

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
    ]

    output = (
        summary[
            columns
        ]
        .copy()
    )

    return output


# ============================================================
# PLOTS
# ============================================================


def plot_rank_cdf(
    rank_cdf: pd.DataFrame,
) -> None:
    """Vẽ empirical rank CDF của tất cả model."""

    fig, ax = plt.subplots(
        figsize=(11, 7),
        constrained_layout=True,
    )

    for model_key, subset in (
        rank_cdf.groupby(
            "model"
        )
    ):

        ax.plot(
            subset["rank"],
            subset[
                "empirical_cdf"
            ],
            label=model_key,
        )

    ranks = np.arange(
        1,
        NUMBER_OF_CLASSES + 1,
    )

    ax.plot(
        ranks,
        ranks / NUMBER_OF_CLASSES,
        linestyle="--",
        linewidth=2,
        label="Uniform",
    )

    ax.set_xlabel(
        "Rank k"
    )

    ax.set_ylabel(
        "P(actual rank <= k)"
    )

    ax.set_title(
        "CDF của rank kết quả thực tế",
        fontsize=15,
        fontweight="bold",
    )

    ax.set_xlim(
        1,
        NUMBER_OF_CLASSES,
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
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.show()

    plt.close(
        fig
    )

    print(
        f"Đã lưu: {output_file}"
    )


def plot_top_k_accuracy(
    summary: pd.DataFrame,
) -> None:
    """Vẽ Top-k accuracy theo model."""

    plot_data = (
        summary.loc[
            summary[
                "model"
            ].ne(
                "uniform"
            )
        ]
        .copy()
    )

    x = np.arange(
        len(
            plot_data
        )
    )

    fig, ax = plt.subplots(
        figsize=(13, 7),
        constrained_layout=True,
    )

    width = 0.15

    for index, k in enumerate(
        TOP_K_VALUES
    ):

        offset = (
            index
            - (
                len(
                    TOP_K_VALUES
                )
                - 1
            )
            / 2
        ) * width

        ax.bar(
            x + offset,
            plot_data[
                f"top_{k}_accuracy"
            ],
            width,
            label=f"Top-{k}",
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        plot_data[
            "model"
        ],
        rotation=30,
        ha="right",
    )

    ax.set_ylabel(
        "Accuracy"
    )

    ax.set_title(
        "Top-k accuracy theo mô hình",
        fontsize=15,
        fontweight="bold",
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    output_file = (
        FIGURE_DIR
        / "probability_topk_comparison.png"
    )

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.show()

    plt.close(
        fig
    )

    print(
        f"Đã lưu: {output_file}"
    )


def plot_mean_rank(
    summary: pd.DataFrame,
) -> None:
    """So sánh mean rank với baseline Uniform 50.5."""

    plot_data = (
        summary.loc[
            summary[
                "model"
            ].ne(
                "uniform"
            )
        ]
        .sort_values(
            "mean_true_rank"
        )
        .copy()
    )

    fig, ax = plt.subplots(
        figsize=(11, 6),
        constrained_layout=True,
    )

    ax.bar(
        plot_data[
            "model"
        ],
        plot_data[
            "mean_true_rank"
        ],
    )

    ax.axhline(
        50.5,
        linestyle="--",
        linewidth=2,
        label="Uniform expected rank = 50.5",
    )

    ax.set_ylabel(
        "Mean actual rank"
    )

    ax.set_title(
        "Mean rank của kết quả thực tế",
        fontsize=15,
        fontweight="bold",
    )

    ax.tick_params(
        axis="x",
        rotation=30,
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    output_file = (
        FIGURE_DIR
        / "probability_mean_rank.png"
    )

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.show()

    plt.close(
        fig
    )

    print(
        f"Đã lưu: {output_file}"
    )


# ============================================================
# PRINT
# ============================================================


def print_summary(
    summary: pd.DataFrame,
) -> None:
    """In bảng tổng hợp chính."""

    display_columns = [
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
    ]

    print(
        "\n"
        + "=" * 175
    )

    print(
        "TỔNG HỢP PROBABILITY RANKING"
    )

    print(
        "=" * 175
    )

    print(
        summary[
            display_columns
        ].to_string(
            index=False,
            formatters={
                "log_loss": (
                    "{:.6f}".format
                ),

                "log_loss_gain_vs_uniform": (
                    "{:.6f}".format
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

                "median_true_rank": (
                    "{:.2f}".format
                ),
            },
        )
    )


def print_top_k_lift(
    selected_rank_table: pd.DataFrame,
) -> None:
    """In lift so với random ở các rank quan trọng."""

    print(
        "\n"
        + "=" * 120
    )

    print(
        "LIFT CDF SO VỚI UNIFORM"
    )

    print(
        "=" * 120
    )

    print(
        selected_rank_table.to_string(
            index=False,
            formatters={
                "empirical_pct": (
                    "{:.3f}".format
                ),

                "uniform_pct": (
                    "{:.3f}".format
                ),

                "lift_pp_pct": (
                    "{:+.3f}".format
                ),

                "lift_ratio": (
                    "{:.3f}".format
                ),
            },
        )
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    """Chạy probability ranking analysis."""

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------
    # Load
    # --------------------------------

    predictions = (
        load_predictions()
    )

    print(
        f"Loaded "
        f"{len(predictions):,} "
        f"prediction rows."
    )

    print(
        "\nModels:"
    )

    for model_key, count in (
        predictions[
            "model_key"
        ]
        .value_counts()
        .sort_index()
        .items()
    ):

        print(
            f"  {model_key}: "
            f"{count:,}"
        )

    # --------------------------------
    # Summary
    # --------------------------------

    model_summary = (
        build_model_summary(
            predictions
        )
    )

    fold_summary = (
        build_fold_summary(
            predictions
        )
    )

    rank_cdf = (
        build_rank_cdf(
            predictions
        )
    )

    selected_rank_table = (
        build_selected_rank_table(
            rank_cdf
        )
    )

    strategy_candidates = (
        build_strategy_candidate_table(
            model_summary
        )
    )

    # --------------------------------
    # Print
    # --------------------------------

    print_summary(
        model_summary
    )

    print_top_k_lift(
        selected_rank_table
    )

    # --------------------------------
    # Save
    # --------------------------------

    model_summary.to_csv(
        TABLE_DIR
        / "probability_ranking_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fold_summary.to_csv(
        TABLE_DIR
        / "probability_ranking_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rank_cdf.to_csv(
        TABLE_DIR
        / "probability_rank_cdf.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selected_rank_table.to_csv(
        TABLE_DIR
        / "probability_rank_selected.csv",
        index=False,
        encoding="utf-8-sig",
    )

    strategy_candidates.to_csv(
        TABLE_DIR
        / "strategy_model_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------
    # Figures
    # --------------------------------

    plot_rank_cdf(
        rank_cdf
    )

    plot_top_k_accuracy(
        model_summary
    )

    plot_mean_rank(
        model_summary
    )


if __name__ == "__main__":
    main()