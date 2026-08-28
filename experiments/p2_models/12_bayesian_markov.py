"""Bayesian Markov cho bài toán dự đoán hai chữ số cuối 00-99.

Mô hình:
    P(X_t = j | X_{t-1} = i)
        = (n_ij + alpha)
          / (sum_k n_ik + 100 * alpha)

Đánh giá theo walk-forward:
    dùng transition counts từ quá khứ
    -> dự đoán ngày t từ kết quả ngày t-1
    -> quan sát actual
    -> cập nhật transition.
"""

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
# MARKOV
# ============================================================


def build_transition_counts(
    values: np.ndarray,
) -> np.ndarray:
    """Tạo ma trận transition count 100x100."""

    counts = np.zeros(
        (
            NUMBER_OF_CLASSES,
            NUMBER_OF_CLASSES,
        ),
        dtype=float,
    )

    for previous, current in zip(
        values[:-1],
        values[1:],
    ):
        counts[
            int(previous),
            int(current),
        ] += 1

    return counts


def markov_probability(
    transition_counts: np.ndarray,
    previous_state: int,
    alpha: float = ALPHA,
) -> np.ndarray:
    """Posterior predictive của Bayesian Markov."""

    row = transition_counts[
        previous_state
    ]

    denominator = (
        row.sum()
        + NUMBER_OF_CLASSES * alpha
    )

    probabilities = (
        row + alpha
    ) / denominator

    return probabilities


# ============================================================
# RANKING
# ============================================================


def create_tie_break_priority() -> np.ndarray:
    """Tie-break cố định khi nhiều class cùng probability."""

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


TIE_BREAK_PRIORITY = create_tie_break_priority()


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
    """Rank 1-100 của actual."""

    return int(
        np.where(
            order == actual
        )[0][0]
        + 1
    )


# ============================================================
# WALK-FORWARD
# ============================================================


def run_fold(
    df: pd.DataFrame,
    fold: dict,
) -> pd.DataFrame:
    """Chạy Bayesian Markov walk-forward cho một fold."""

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

    history = (
        df.loc[
            history_mask,
            [
                "date",
                "last_2_target",
            ],
        ]
        .copy()
        .sort_values("date")
        .reset_index(drop=True)
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

    history_values = (
        history[
            "last_2_target"
        ]
        .to_numpy()
    )

    transition_counts = (
        build_transition_counts(
            history_values
        )
    )

    previous_state = int(
        history_values[-1]
    )

    print(
        f"\nFold {fold['name']}: "
        f"history={len(history):,}, "
        f"test={len(test):,}"
    )

    records = []

    for _, row in test.iterrows():

        date = row["date"]

        actual = int(
            row["last_2_target"]
        )

        # ---------------------------------
        # Predict from previous state
        # ---------------------------------
        probabilities = (
            markov_probability(
                transition_counts,
                previous_state,
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

        transition_row_count = int(
            transition_counts[
                previous_state
            ].sum()
        )

        record = {
            "date": date,
            "fold": fold["name"],
            "model": "bayesian_markov",
            "alpha": ALPHA,

            "previous_state": (
                previous_state
            ),

            "transition_row_count": (
                transition_row_count
            ),

            "actual": actual,

            "pred_top1": (
                predicted
            ),

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
        # Update AFTER observing actual
        # ---------------------------------
        transition_counts[
            previous_state,
            actual,
        ] += 1

        previous_state = actual

    return pd.DataFrame(
        records
    )


# ============================================================
# EVALUATION
# ============================================================


def evaluate_fold(
    predictions: pd.DataFrame,
) -> dict:
    """Tính metrics cho một fold."""

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
        predictions[
            "actual"
        ]
        .to_numpy()
    )

    ranks = (
        predictions[
            "actual_rank"
        ]
        .to_numpy()
    )

    return {
        "fold": (
            predictions[
                "fold"
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


def statistical_tests(
    predictions: pd.DataFrame,
) -> dict:
    """Kiểm định tổng Bayesian Markov vs Uniform."""

    number_tested = len(
        predictions
    )

    number_correct = int(
        predictions[
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
        predictions.assign(
            year=predictions[
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

    ranks = (
        predictions[
            "actual_rank"
        ]
        .to_numpy()
    )

    return {
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
            predictions[
                "improvement"
            ].mean()
        ),

        "bootstrap_ci_low": float(
            lower
        ),

        "bootstrap_ci_high": float(
            upper
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


# ============================================================
# PLOT
# ============================================================


def plot_fold_results(
    fold_results: pd.DataFrame,
) -> None:
    """So sánh Bayesian Markov vs Uniform."""

    fig, ax = plt.subplots(
        figsize=(10, 6),
        constrained_layout=True,
    )

    positions = np.arange(
        len(fold_results)
    )

    width = 0.35

    ax.bar(
        positions - width / 2,
        fold_results[
            "log_loss"
        ],
        width,
        label="Bayesian Markov",
    )

    ax.bar(
        positions + width / 2,
        np.full(
            len(
                fold_results
            ),
            np.log(
                NUMBER_OF_CLASSES
            ),
        ),
        width,
        label="Uniform",
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        fold_results[
            "fold"
        ]
    )

    ax.set_ylabel(
        "Log loss"
    )

    ax.set_title(
        "Bayesian Markov vs Uniform",
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
        / "bayesian_markov_log_loss.png"
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
    """Chạy Bayesian Markov experiment."""

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

        fold_predictions = (
            run_fold(
                df,
                fold,
            )
        )

        fold_metrics = (
            evaluate_fold(
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

    significance = (
        statistical_tests(
            predictions
        )
    )

    print(
        "\n"
        + "=" * 170
    )

    print(
        "KẾT QUẢ BAYESIAN MARKOV"
    )

    print(
        "=" * 170
    )

    print(
        fold_results.to_string(
            index=False,
            formatters={
                "alpha": (
                    "{:.2f}".format
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

                "log_loss": (
                    "{:.6f}".format
                ),

                "mean_true_rank": (
                    "{:.2f}".format
                ),

                "median_true_rank": (
                    "{:.2f}".format
                ),

                "true_rank_q25": (
                    "{:.2f}".format
                ),

                "true_rank_q75": (
                    "{:.2f}".format
                ),

                "true_rank_q90": (
                    "{:.2f}".format
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
            },
        )
    )

    print(
        "\nKiểm định tổng hợp:"
    )

    for key, value in (
        significance.items()
    ):
        print(
            f"{key}: {value}"
        )

    # ---------------------------------
    # SAVE
    # ---------------------------------

    fold_results.to_csv(
        TABLE_DIR
        / "bayesian_markov_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions.to_csv(
        TABLE_DIR
        / "bayesian_markov_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [significance]
    ).to_csv(
        TABLE_DIR
        / "bayesian_markov_significance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_fold_results(
        fold_results
    )


if __name__ == "__main__":
    main()
