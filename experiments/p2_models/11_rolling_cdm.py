"""Rolling CDM cho bài toán dự đoán hai chữ số cuối 00-99.

Mô hình:
    P_t(j) = (n_j^(W) + alpha) / (W + 100 * alpha)

Trong đó:
    n_j^(W) : số lần j xuất hiện trong W ngày gần nhất
    alpha   : symmetric Dirichlet prior

Đánh giá:
    walk-forward, không dùng future information.
"""

from collections import deque
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import log_loss


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import STATEFUL_EVALUATION_FOLDS

DATA_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "kqxsmb_digits.csv"
)

TABLE_DIR = PROJECT_DIR / "artifacts" / "tables"
FIGURE_DIR = PROJECT_DIR / "artifacts" / "figures"

NUMBER_OF_CLASSES = 100

ALPHA = 1.0

WINDOWS = [
    30,
    60,
    90,
    180,
    365,
]

RANDOM_STATE = 42
NUMBER_OF_BOOTSTRAPS = 20_000

FOLDS = STATEFUL_EVALUATION_FOLDS


# ============================================================
# DATA
# ============================================================


def read_data() -> pd.DataFrame:
    """Đọc dữ liệu và chuẩn hóa target 00-99."""

    df = pd.read_csv(
        DATA_FILE,
        dtype={
            "full_result": str,
            "last_2_digits": str,
        },
        parse_dates=["date"],
    )

    df = (
        df.sort_values("date")
        .reset_index(drop=True)
    )

    df["last_2_target"] = (
        df["last_2_digits"]
        .str.zfill(2)
        .astype(int)
    )

    return df


# ============================================================
# PROBABILITY
# ============================================================


def rolling_cdm_probability(
    counts: np.ndarray,
    alpha: float = ALPHA,
) -> np.ndarray:
    """Posterior predictive probability từ rolling counts."""

    counts = np.asarray(
        counts,
        dtype=float,
    )

    denominator = (
        counts.sum()
        + NUMBER_OF_CLASSES * alpha
    )

    return (
        counts + alpha
    ) / denominator


# ============================================================
# RANKING
# ============================================================


def create_tie_break_priority() -> np.ndarray:
    """Tạo tie-break reproducible cho các probability bằng nhau."""

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

    priority[permutation] = np.arange(
        NUMBER_OF_CLASSES
    )

    return priority


TIE_BREAK_PRIORITY = (
    create_tie_break_priority()
)


def rank_classes(
    probabilities: np.ndarray,
) -> np.ndarray:
    """Sort class theo probability giảm dần."""

    classes = np.arange(
        NUMBER_OF_CLASSES
    )

    order = np.lexsort(
        (
            TIE_BREAK_PRIORITY[
                classes
            ],
            -probabilities,
        )
    )

    return order


def actual_rank(
    actual: int,
    order: np.ndarray,
) -> int:
    """Rank 1-100 của kết quả thật."""

    return int(
        np.where(
            order == actual
        )[0][0]
        + 1
    )


# ============================================================
# WALK-FORWARD
# ============================================================


def run_fold_window(
    df: pd.DataFrame,
    fold: dict,
    window: int,
) -> pd.DataFrame:
    """
    Chạy rolling CDM cho một fold và một window.

    Mỗi ngày:
        dùng đúng W kết quả gần nhất trước ngày đó
        -> predict
        -> observe actual
        -> update rolling history.
    """

    test_start = fold[
        "test_start"
    ]

    test_end = fold[
        "test_end"
    ]

    history_mask = (
        df["date"].dt.year
        < test_start
    )

    test_mask = (
        df["date"].dt.year
        .between(
            test_start,
            test_end,
        )
    )

    history_values = (
        df.loc[
            history_mask,
            "last_2_target",
        ]
        .to_numpy()
    )

    test = (
        df.loc[
            test_mask,
            [
                "date",
                "last_2_target",
            ],
        ]
        .copy()
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(history_values) < window:
        raise ValueError(
            f"Không đủ history cho window={window}: "
            f"{len(history_values)} observations."
        )

    initial_window = (
        history_values[-window:]
    )

    rolling_history = deque(
        initial_window.tolist(),
        maxlen=window,
    )

    counts = np.bincount(
        initial_window,
        minlength=NUMBER_OF_CLASSES,
    ).astype(float)

    print(
        f"\nFold {fold['name']} | "
        f"window={window}: "
        f"history={len(history_values):,}, "
        f"test={len(test):,}"
    )

    records = []

    for _, row in test.iterrows():

        date = row["date"]

        actual = int(
            row["last_2_target"]
        )

        probabilities = (
            rolling_cdm_probability(
                counts,
                alpha=ALPHA,
            )
        )

        order = rank_classes(
            probabilities
        )

        sorted_probabilities = (
            probabilities[order]
        )

        predicted = int(
            order[0]
        )

        true_probability = float(
            probabilities[actual]
        )

        true_rank = actual_rank(
            actual,
            order,
        )

        record = {
            "date": date,
            "fold": fold["name"],
            "model": "rolling_cdm",
            "window": window,
            "alpha": ALPHA,

            "actual": actual,
            "pred_top1": predicted,

            "actual_probability": (
                true_probability
            ),

            "actual_rank": (
                true_rank
            ),

            "top1_number": int(
                order[0]
            ),

            "top1_probability": float(
                sorted_probabilities[0]
            ),

            "top2_number": int(
                order[1]
            ),

            "top2_probability": float(
                sorted_probabilities[1]
            ),

            "top3_number": int(
                order[2]
            ),

            "top3_probability": float(
                sorted_probabilities[2]
            ),

            "top5_probability_sum": float(
                sorted_probabilities[
                    :5
                ].sum()
            ),

            "top10_probability_sum": float(
                sorted_probabilities[
                    :10
                ].sum()
            ),

            "top1_hit": int(
                true_rank <= 1
            ),

            "top3_hit": int(
                true_rank <= 3
            ),

            "top5_hit": int(
                true_rank <= 5
            ),

            "top10_hit": int(
                true_rank <= 10
            ),

            "top20_hit": int(
                true_rank <= 20
            ),
        }

        for number in range(
            NUMBER_OF_CLASSES
        ):
            record[
                f"p_{number:02d}"
            ] = float(
                probabilities[number]
            )

        record["model_loss"] = float(
            -np.log(
                np.clip(
                    true_probability,
                    1e-15,
                    1,
                )
            )
        )

        record["uniform_loss"] = float(
            np.log(
                NUMBER_OF_CLASSES
            )
        )

        record["improvement"] = (
            record["uniform_loss"]
            - record["model_loss"]
        )

        records.append(
            record
        )

        # ---------------------------------
        # Update rolling window AFTER actual
        # ---------------------------------

        oldest = rolling_history[0]

        counts[oldest] -= 1

        rolling_history.append(
            actual
        )

        counts[actual] += 1

    return pd.DataFrame(
        records
    )


# ============================================================
# EVALUATION
# ============================================================


def evaluate_predictions(
    predictions: pd.DataFrame,
) -> dict:
    """Tính metric cho một fold-window."""

    probability_columns = [
        f"p_{number:02d}"
        for number
        in range(
            NUMBER_OF_CLASSES
        )
    ]

    probabilities = (
        predictions[
            probability_columns
        ]
        .to_numpy()
    )

    y_true = (
        predictions["actual"]
        .to_numpy()
    )

    ranks = (
        predictions["actual_rank"]
        .to_numpy()
    )

    return {
        "fold": (
            predictions[
                "fold"
            ].iloc[0]
        ),

        "window": int(
            predictions[
                "window"
            ].iloc[0]
        ),

        "alpha": ALPHA,

        "test_size": len(
            predictions
        ),

        "top_1_accuracy": float(
            np.mean(
                ranks <= 1
            )
        ),

        "top_3_accuracy": float(
            np.mean(
                ranks <= 3
            )
        ),

        "top_5_accuracy": float(
            np.mean(
                ranks <= 5
            )
        ),

        "top_10_accuracy": float(
            np.mean(
                ranks <= 10
            )
        ),

        "top_20_accuracy": float(
            np.mean(
                ranks <= 20
            )
        ),

        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=list(
                    range(
                        NUMBER_OF_CLASSES
                    )
                ),
            )
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
    }


def overall_window_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Tổng hợp metric theo window trên toàn bộ các fold."""

    records = []

    for window in WINDOWS:

        subset = (
            predictions.loc[
                predictions[
                    "window"
                ].eq(window)
            ]
            .copy()
        )

        probabilities = (
            subset[
                [
                    f"p_{number:02d}"
                    for number
                    in range(
                        NUMBER_OF_CLASSES
                    )
                ]
            ]
            .to_numpy()
        )

        y_true = (
            subset[
                "actual"
            ]
            .to_numpy()
        )

        ranks = (
            subset[
                "actual_rank"
            ]
            .to_numpy()
        )

        uniform_log_loss = np.log(
            NUMBER_OF_CLASSES
        )

        model_log_loss = float(
            log_loss(
                y_true,
                probabilities,
                labels=list(
                    range(
                        NUMBER_OF_CLASSES
                    )
                ),
            )
        )

        records.append(
            {
                "window": window,

                "n_test": len(
                    subset
                ),

                "top_1_accuracy": float(
                    np.mean(
                        ranks <= 1
                    )
                ),

                "top_3_accuracy": float(
                    np.mean(
                        ranks <= 3
                    )
                ),

                "top_5_accuracy": float(
                    np.mean(
                        ranks <= 5
                    )
                ),

                "top_10_accuracy": float(
                    np.mean(
                        ranks <= 10
                    )
                ),

                "top_20_accuracy": float(
                    np.mean(
                        ranks <= 20
                    )
                ),

                "log_loss": (
                    model_log_loss
                ),

                "uniform_log_loss": (
                    uniform_log_loss
                ),

                "log_loss_gain_vs_uniform": (
                    uniform_log_loss
                    - model_log_loss
                ),

                "relative_log_loss_gain_pct": (
                    (
                        uniform_log_loss
                        - model_log_loss
                    )
                    / uniform_log_loss
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
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# STATISTICAL TEST
# ============================================================


def statistical_test_window(
    predictions: pd.DataFrame,
    window: int,
) -> dict:
    """Kiểm định một rolling window so với Uniform."""

    subset = (
        predictions.loc[
            predictions[
                "window"
            ].eq(window)
        ]
        .copy()
    )

    number_tested = len(
        subset
    )

    number_correct = int(
        subset[
            "actual_rank"
        ]
        .le(1)
        .sum()
    )

    accuracy_p_value = (
        binomtest(
            k=number_correct,
            n=number_tested,
            p=1 / NUMBER_OF_CLASSES,
            alternative="greater",
        ).pvalue
    )

    yearly_improvement = (
        subset.assign(
            year=subset[
                "date"
            ].dt.year
        )
        .groupby(
            "year"
        )[
            "improvement"
        ]
        .mean()
    )

    values = (
        yearly_improvement
        .to_numpy()
    )

    rng = np.random.default_rng(
        RANDOM_STATE
        + window
    )

    bootstrap_means = np.empty(
        NUMBER_OF_BOOTSTRAPS
    )

    for iteration in range(
        NUMBER_OF_BOOTSTRAPS
    ):

        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )

        bootstrap_means[
            iteration
        ] = sample.mean()

    lower, upper = np.quantile(
        bootstrap_means,
        [
            0.025,
            0.975,
        ],
    )

    return {
        "window": window,
        "number_tested": (
            number_tested
        ),
        "number_correct": (
            number_correct
        ),
        "top_1_accuracy": float(
            number_correct
            / number_tested
        ),
        "binomial_p_value": float(
            accuracy_p_value
        ),
        "mean_log_loss_improvement": float(
            subset[
                "improvement"
            ].mean()
        ),
        "bootstrap_ci_low": float(
            lower
        ),
        "bootstrap_ci_high": float(
            upper
        ),
    }


# ============================================================
# PLOT
# ============================================================


def plot_window_log_loss(
    window_summary: pd.DataFrame,
) -> None:
    """Vẽ log loss theo rolling window."""

    fig, ax = plt.subplots(
        figsize=(10, 6),
        constrained_layout=True,
    )

    ax.plot(
        window_summary[
            "window"
        ],
        window_summary[
            "log_loss"
        ],
        marker="o",
        label="Rolling CDM",
    )

    ax.axhline(
        np.log(
            NUMBER_OF_CLASSES
        ),
        linestyle="--",
        label="Uniform",
    )

    ax.set_xlabel(
        "Rolling window (days)"
    )

    ax.set_ylabel(
        "Log loss"
    )

    ax.set_title(
        "Rolling CDM: Log loss theo cửa sổ lịch sử",
        fontsize=15,
        fontweight="bold",
    )

    ax.legend()

    ax.grid(
        alpha=0.25,
    )

    output_file = (
        FIGURE_DIR
        / "rolling_cdm_log_loss.png"
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
# MAIN
# ============================================================


def main() -> None:
    """Chạy toàn bộ rolling CDM experiment."""

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = read_data()

    prediction_frames = []
    fold_records = []

    for fold in FOLDS:

        for window in WINDOWS:

            fold_predictions = (
                run_fold_window(
                    df,
                    fold,
                    window,
                )
            )

            fold_metrics = (
                evaluate_predictions(
                    fold_predictions
                )
            )

            prediction_frames.append(
                fold_predictions
            )

            fold_records.append(
                fold_metrics
            )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    fold_results = pd.DataFrame(
        fold_records
    )

    uniform_log_loss = np.log(
        NUMBER_OF_CLASSES
    )

    fold_results[
        "uniform_log_loss"
    ] = uniform_log_loss

    fold_results[
        "log_loss_gain_vs_uniform"
    ] = (
        uniform_log_loss
        - fold_results[
            "log_loss"
        ]
    )

    fold_results[
        "relative_log_loss_gain_pct"
    ] = (
        fold_results[
            "log_loss_gain_vs_uniform"
        ]
        / uniform_log_loss
        * 100
    )

    window_summary = (
        overall_window_summary(
            predictions
        )
    )

    significance_records = [
        statistical_test_window(
            predictions,
            window,
        )
        for window
        in WINDOWS
    ]

    significance = pd.DataFrame(
        significance_records
    )

    # ---------------------------------
    # Print
    # ---------------------------------

    print(
        "\n"
        + "=" * 170
    )

    print(
        "KẾT QUẢ ROLLING CDM - THEO WINDOW"
    )

    print(
        "=" * 170
    )

    print(
        window_summary.to_string(
            index=False,
            formatters={
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
                "log_loss": (
                    "{:.6f}".format
                ),
                "uniform_log_loss": (
                    "{:.6f}".format
                ),
                "log_loss_gain_vs_uniform": (
                    "{:.6f}".format
                ),
                "relative_log_loss_gain_pct": (
                    "{:.4f}".format
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

    print(
        "\nKiểm định theo window:"
    )

    print(
        significance.to_string(
            index=False,
        )
    )

    # ---------------------------------
    # Save
    # ---------------------------------

    fold_results.to_csv(
        TABLE_DIR
        / "rolling_cdm_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    window_summary.to_csv(
        TABLE_DIR
        / "rolling_cdm_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    significance.to_csv(
        TABLE_DIR
        / "rolling_cdm_significance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions.to_csv(
        TABLE_DIR
        / "rolling_cdm_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_window_log_loss(
        window_summary
    )


if __name__ == "__main__":
    main()
