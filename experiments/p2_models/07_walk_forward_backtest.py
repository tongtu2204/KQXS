"""Walk-forward backtest theo năm cho Uniform, Markov và Random Forest."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss


PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "kqxsmb_digits.csv"
)

TABLE_DIR = PROJECT_DIR / "artifacts" / "tables"
FIGURE_DIR = PROJECT_DIR / "artifacts" / "figures"

DIGIT_COLUMNS = [
    "digit_1",
    "digit_2",
    "digit_3",
    "digit_4",
    "digit_5",
]

LAGS = [1, 2, 3, 7, 14, 19, 30]
TEST_YEARS = range(2019, 2027)

RANDOM_STATE = 42


def read_data() -> pd.DataFrame:
    """Đọc dữ liệu theo thời gian."""

    return (
        pd.read_csv(
            DATA_FILE,
            dtype={
                "full_result": str,
                "last_2_digits": str,
            },
            parse_dates=["date"],
        )
        .sort_values("date")
        .reset_index(drop=True)
    )


def build_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Tạo feature chỉ từ dữ liệu quá khứ."""

    features = pd.DataFrame(
        index=df.index
    )

    for column in DIGIT_COLUMNS:
        for lag in LAGS:
            features[f"{column}_lag_{lag}"] = (
                df[column].shift(lag)
            )

    day_of_week = df["date"].dt.dayofweek
    month = df["date"].dt.month

    features["day_of_week_sin"] = np.sin(
        2 * np.pi * day_of_week / 7
    )
    features["day_of_week_cos"] = np.cos(
        2 * np.pi * day_of_week / 7
    )
    features["month_sin"] = np.sin(
        2 * np.pi * month / 12
    )
    features["month_cos"] = np.cos(
        2 * np.pi * month / 12
    )

    return features


def prepare_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Chuẩn bị feature, target và ngày."""

    features = build_features(df)
    valid = features.notna().all(axis=1)

    return (
        features.loc[valid]
        .astype(float)
        .reset_index(drop=True),
        df.loc[valid, DIGIT_COLUMNS]
        .astype(int)
        .reset_index(drop=True),
        df.loc[valid, "date"]
        .reset_index(drop=True),
    )


def create_model() -> RandomForestClassifier:
    """Khởi tạo Random Forest."""

    return RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def align_probabilities(
    model: RandomForestClassifier,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Căn probability theo chữ số 0-9."""

    aligned = np.zeros(
        (len(probabilities), 10)
    )

    for model_column, digit in enumerate(
        model.classes_
    ):
        aligned[:, int(digit)] = (
            probabilities[:, model_column]
        )

    return aligned


def build_markov_matrix(
    digits: np.ndarray,
    smoothing: float = 1.0,
) -> np.ndarray:
    """Ước lượng ma trận Markov có Laplace smoothing."""

    counts = np.full(
        (10, 10),
        smoothing,
        dtype=float,
    )

    for previous_digit, next_digit in zip(
        digits[:-1],
        digits[1:],
    ):
        counts[
            int(previous_digit),
            int(next_digit),
        ] += 1

    return counts / counts.sum(
        axis=1,
        keepdims=True,
    )


def multiclass_brier_score(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Tính Brier score đa lớp."""

    one_hot = np.eye(10)[y_true]

    return np.mean(
        np.sum(
            (probabilities - one_hot) ** 2,
            axis=1,
        )
    )


def top_3_accuracy(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Tính Top-3 accuracy."""

    top_digits = np.argsort(
        probabilities,
        axis=1,
    )[:, -3:]

    return np.mean(
        [
            true_digit in candidates
            for true_digit, candidates
            in zip(y_true, top_digits)
        ]
    )


def evaluate(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """Đánh giá một phân phối dự đoán."""

    predictions = probabilities.argmax(
        axis=1
    )

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "top_3_accuracy": top_3_accuracy(
            y_true,
            probabilities,
        ),
        "log_loss": log_loss(
            y_true,
            probabilities,
            labels=list(range(10)),
        ),
        "brier_score": multiclass_brier_score(
            y_true,
            probabilities,
        ),
    }

    if np.allclose(probabilities, 0.10):
        metrics["accuracy"] = 0.10
        metrics["top_3_accuracy"] = 0.30

    return metrics


def run_walk_forward(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    dates: pd.Series,
) -> pd.DataFrame:
    """Huấn luyện bằng quá khứ và kiểm tra từng năm."""

    records = []

    for test_year in TEST_YEARS:
        train_mask = dates.dt.year < test_year
        test_mask = dates.dt.year == test_year

        x_train = features.loc[train_mask]
        x_test = features.loc[test_mask]

        y_train = targets.loc[train_mask]
        y_test = targets.loc[test_mask]

        if len(x_test) == 0:
            continue

        print(
            f"\nNăm {test_year}: "
            f"train={len(x_train):,}, "
            f"test={len(x_test):,}"
        )

        for position, target_column in enumerate(
            DIGIT_COLUMNS,
            start=1,
        ):
            print(
                f"  Vị trí {position}...",
                end="",
                flush=True,
            )

            train_target = (
                y_train[target_column]
                .to_numpy()
            )

            test_target = (
                y_test[target_column]
                .to_numpy()
            )

            # ----------------------------------------
            # Uniform
            # ----------------------------------------

            uniform_probabilities = np.full(
                (len(x_test), 10),
                0.10,
            )

            # ----------------------------------------
            # Markov
            # ----------------------------------------

            transition_matrix = (
                build_markov_matrix(
                    train_target
                )
            )

            previous_test_digits = (
                x_test[
                    f"{target_column}_lag_1"
                ]
                .to_numpy()
                .astype(int)
            )

            markov_probabilities = (
                transition_matrix[
                    previous_test_digits
                ]
            )

            # ----------------------------------------
            # Random Forest
            # ----------------------------------------

            model = create_model()

            model.fit(
                x_train,
                train_target,
            )

            random_forest_probabilities = (
                align_probabilities(
                    model,
                    model.predict_proba(
                        x_test
                    ),
                )
            )

            # Co Random Forest về phân phối đều
            shrunk_rf_probabilities = (
                0.75 * uniform_probabilities
                + 0.25
                * random_forest_probabilities
            )

            # Thử kết hợp cả RF và Markov
            enhanced_probabilities = (
                0.50 * uniform_probabilities
                + 0.25
                * random_forest_probabilities
                + 0.25
                * markov_probabilities
            )

            probability_sets = {
                "Uniform": (
                    uniform_probabilities
                ),
                "Markov": (
                    markov_probabilities
                ),
                "Random Forest": (
                    random_forest_probabilities
                ),
                "Shrunk RF": (
                    shrunk_rf_probabilities
                ),
                "RF + Markov": (
                    enhanced_probabilities
                ),
            }

            for model_name, probabilities in (
                probability_sets.items()
            ):
                metrics = evaluate(
                    test_target,
                    probabilities,
                )

                records.append(
                    {
                        "test_year": test_year,
                        "position": position,
                        "model": model_name,
                        "number_of_test_draws": (
                            len(x_test)
                        ),
                        **metrics,
                    }
                )

            print(" xong")

    return pd.DataFrame(records)


def weighted_average(
    group: pd.DataFrame,
    column: str,
) -> float:
    """Trung bình có trọng số theo số kỳ test."""

    return np.average(
        group[column],
        weights=group[
            "number_of_test_draws"
        ],
    )


def aggregate_results(
    fold_results: pd.DataFrame,
) -> pd.DataFrame:
    """Tổng hợp toàn bộ các năm."""

    records = []

    for (
        model_name,
        position,
    ), group in fold_results.groupby(
        ["model", "position"]
    ):
        records.append(
            {
                "model": model_name,
                "position": position,
                "number_of_test_draws": (
                    group[
                        "number_of_test_draws"
                    ].sum()
                ),
                "accuracy": weighted_average(
                    group,
                    "accuracy",
                ),
                "top_3_accuracy": (
                    weighted_average(
                        group,
                        "top_3_accuracy",
                    )
                ),
                "log_loss": weighted_average(
                    group,
                    "log_loss",
                ),
                "brier_score": weighted_average(
                    group,
                    "brier_score",
                ),
                "years_better_than_uniform": (
                    group["log_loss"]
                    .lt(np.log(10) - 1e-12)
                    .sum()
                ),
                "number_of_years": len(group),
            }
        )

    return pd.DataFrame(records)


def print_results(
    aggregate: pd.DataFrame,
) -> None:
    """In kết quả tổng hợp."""

    print("\n" + "=" * 125)
    print("KẾT QUẢ WALK-FORWARD TỔNG HỢP")
    print("=" * 125)

    print(
        aggregate.to_string(
            index=False,
            formatters={
                "accuracy": "{:.4%}".format,
                "top_3_accuracy": "{:.4%}".format,
                "log_loss": "{:.6f}".format,
                "brier_score": "{:.6f}".format,
            },
        )
    )


def plot_results(
    aggregate: pd.DataFrame,
) -> None:
    """Vẽ log loss tổng hợp."""

    pivot = aggregate.pivot(
        index="position",
        columns="model",
        values="log_loss",
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(15, 7),
        width=0.80,
    )

    ax.axhline(
        np.log(10),
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Uniform = log(10)",
    )

    ax.set_title(
        "Walk-forward backtest theo năm",
        fontsize=16,
        fontweight="bold",
    )

    ax.set_xlabel("Vị trí")
    ax.set_ylabel("Log loss")
    ax.tick_params(
        axis="x",
        rotation=0,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend(
        title="Mô hình",
        fontsize=9,
    )

    fig = ax.get_figure()
    fig.tight_layout()

    output_file = (
        FIGURE_DIR
        / "walk_forward_log_loss.png"
    )

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.show()
    plt.close(fig)

    print(f"Đã lưu: {output_file}")


def main() -> None:
    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = read_data()

    features, targets, dates = prepare_data(
        df
    )

    fold_results = run_walk_forward(
        features,
        targets,
        dates,
    )

    aggregate = aggregate_results(
        fold_results
    )

    print_results(aggregate)

    fold_results.to_csv(
        TABLE_DIR
        / "walk_forward_yearly_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    aggregate.to_csv(
        TABLE_DIR
        / "walk_forward_aggregate_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_results(aggregate)


if __name__ == "__main__":
    main()