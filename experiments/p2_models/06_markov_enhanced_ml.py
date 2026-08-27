"""Kết hợp xác suất Uniform, Random Forest và Markov."""

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

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
RANDOM_STATE = 42

WEIGHT_VALUES = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]


def read_data() -> pd.DataFrame:
    """Đọc dữ liệu theo thứ tự thời gian."""

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
    """Tạo đặc trưng quá khứ và lịch."""

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
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
]:
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
    """Khởi tạo Random Forest thống nhất."""

    return RandomForestClassifier(
        n_estimators=400,
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
    """Căn xác suất theo các chữ số 0-9."""

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
    """Ước lượng ma trận chuyển Markov 10 × 10."""

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


def markov_probabilities(
    transition_matrix: np.ndarray,
    previous_digits: np.ndarray,
) -> np.ndarray:
    """Lấy phân phối kỳ sau theo chữ số kỳ trước."""

    return transition_matrix[
        previous_digits.astype(int)
    ]


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


def top_k_accuracy(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    k: int = 3,
) -> float:
    """Tính Top-k accuracy."""

    top_k = np.argsort(
        probabilities,
        axis=1,
    )[:, -k:]

    return np.mean(
        [
            true_digit in candidate_digits
            for true_digit, candidate_digits
            in zip(y_true, top_k)
        ]
    )


def evaluate(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """Tính các metrics."""

    predictions = probabilities.argmax(
        axis=1
    )

    return {
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "top_3_accuracy": top_k_accuracy(
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


def combine_probabilities(
    uniform: np.ndarray,
    random_forest: np.ndarray,
    markov: np.ndarray,
    random_forest_weight: float,
    markov_weight: float,
) -> np.ndarray:
    """Kết hợp ba phân phối xác suất."""

    uniform_weight = (
        1
        - random_forest_weight
        - markov_weight
    )

    return (
        uniform_weight * uniform
        + random_forest_weight * random_forest
        + markov_weight * markov
    )


def select_weights(
    y_validation: np.ndarray,
    uniform_validation: np.ndarray,
    random_forest_validation: np.ndarray,
    markov_validation: np.ndarray,
) -> dict:
    """Chọn trọng số bằng log loss trên validation."""

    candidates = []

    for rf_weight in WEIGHT_VALUES:
        for markov_weight in WEIGHT_VALUES:
            if rf_weight + markov_weight > 1:
                continue

            combined = combine_probabilities(
                uniform_validation,
                random_forest_validation,
                markov_validation,
                random_forest_weight=rf_weight,
                markov_weight=markov_weight,
            )

            candidates.append(
                {
                    "uniform_weight": (
                        1
                        - rf_weight
                        - markov_weight
                    ),
                    "random_forest_weight": rf_weight,
                    "markov_weight": markov_weight,
                    "validation_log_loss": log_loss(
                        y_validation,
                        combined,
                        labels=list(range(10)),
                    ),
                }
            )

    candidates = pd.DataFrame(candidates)

    return (
        candidates.sort_values(
            "validation_log_loss"
        )
        .iloc[0]
        .to_dict()
    )


def run_experiment(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    dates: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Chạy validation và đánh giá trên test."""

    test_start = int(
        len(features) * (1 - TEST_SIZE)
    )

    train_validation_end = test_start

    fit_end = int(
        train_validation_end
        * (1 - VALIDATION_SIZE)
    )

    x_fit = features.iloc[:fit_end]
    x_validation = features.iloc[
        fit_end:train_validation_end
    ]

    x_train = features.iloc[
        :train_validation_end
    ]

    x_test = features.iloc[test_start:]

    y_fit_all = targets.iloc[:fit_end]
    y_validation_all = targets.iloc[
        fit_end:train_validation_end
    ]

    y_train_all = targets.iloc[
        :train_validation_end
    ]

    y_test_all = targets.iloc[test_start:]

    print(f"Fit       : {len(x_fit):,}")
    print(f"Validation: {len(x_validation):,}")
    print(f"Train tổng: {len(x_train):,}")
    print(f"Test      : {len(x_test):,}")

    print(
        "Khoảng test: "
        f"{dates.iloc[test_start:].min().date()} đến "
        f"{dates.iloc[test_start:].max().date()}"
    )

    metric_records = []
    weight_records = []

    combined_predictions = pd.DataFrame(
        {
            "date": dates.iloc[
                test_start:
            ].to_numpy(),
        }
    )

    for position, target_column in enumerate(
        DIGIT_COLUMNS,
        start=1,
    ):
        print(
            f"\nĐang xử lý vị trí {position}..."
        )

        # ====================================================
        # Chọn trọng số trên validation
        # ====================================================

        validation_model = create_model()

        validation_model.fit(
            x_fit,
            y_fit_all[target_column],
        )

        rf_validation = align_probabilities(
            validation_model,
            validation_model.predict_proba(
                x_validation
            ),
        )

        fit_markov_matrix = build_markov_matrix(
            y_fit_all[target_column].to_numpy()
        )

        previous_validation = (
            x_validation[
                f"{target_column}_lag_1"
            ].to_numpy()
        )

        markov_validation = (
            markov_probabilities(
                fit_markov_matrix,
                previous_validation,
            )
        )

        uniform_validation = np.full(
            (len(x_validation), 10),
            0.10,
        )

        selected_weights = select_weights(
            y_validation=(
                y_validation_all[
                    target_column
                ].to_numpy()
            ),
            uniform_validation=(
                uniform_validation
            ),
            random_forest_validation=(
                rf_validation
            ),
            markov_validation=(
                markov_validation
            ),
        )

        weight_records.append(
            {
                "position": position,
                **selected_weights,
            }
        )

        # ====================================================
        # Huấn luyện lại trên toàn bộ train
        # ====================================================

        final_model = create_model()

        final_model.fit(
            x_train,
            y_train_all[target_column],
        )

        rf_test = align_probabilities(
            final_model,
            final_model.predict_proba(
                x_test
            ),
        )

        train_markov_matrix = build_markov_matrix(
            y_train_all[target_column]
            .to_numpy()
        )

        previous_test = (
            x_test[
                f"{target_column}_lag_1"
            ].to_numpy()
        )

        markov_test = markov_probabilities(
            train_markov_matrix,
            previous_test,
        )

        uniform_test = np.full(
            (len(x_test), 10),
            0.10,
        )

        enhanced_test = combine_probabilities(
            uniform_test,
            rf_test,
            markov_test,
            random_forest_weight=(
                selected_weights[
                    "random_forest_weight"
                ]
            ),
            markov_weight=(
                selected_weights[
                    "markov_weight"
                ]
            ),
        )

        test_target = (
            y_test_all[target_column]
            .to_numpy()
        )

        probability_sets = {
            "Uniform": uniform_test,
            "Markov": markov_test,
            "Random Forest": rf_test,
            "Enhanced": enhanced_test,
        }

        for model_name, probabilities in (
            probability_sets.items()
        ):
            metrics = evaluate(
                test_target,
                probabilities,
            )

            # Accuracy của Uniform dùng kỳ vọng lý thuyết
            if np.allclose(
                probabilities,
                uniform_test,
            ):
                metrics["accuracy"] = 0.10
                metrics["top_3_accuracy"] = 0.30

            metric_records.append(
                {
                    "model": model_name,
                    "position": position,
                    **metrics,
                }
            )

        combined_predictions[
            f"actual_digit_{position}"
        ] = test_target

        combined_predictions[
            f"predicted_digit_{position}"
        ] = enhanced_test.argmax(axis=1)

    return (
        pd.DataFrame(metric_records),
        pd.DataFrame(weight_records),
        combined_predictions,
    )


def print_results(
    metrics: pd.DataFrame,
    weights: pd.DataFrame,
) -> None:
    """In kết quả."""

    print("\n" + "=" * 100)
    print("TRỌNG SỐ ĐƯỢC CHỌN TRÊN VALIDATION")
    print("=" * 100)

    print(
        weights.to_string(
            index=False,
            formatters={
                "uniform_weight": "{:.2f}".format,
                "random_forest_weight": "{:.2f}".format,
                "markov_weight": "{:.2f}".format,
                "validation_log_loss": "{:.6f}".format,
            },
        )
    )

    print("\n" + "=" * 110)
    print("KẾT QUẢ TRÊN TẬP TEST")
    print("=" * 110)

    print(
        metrics.to_string(
            index=False,
            formatters={
                "accuracy": "{:.4%}".format,
                "top_3_accuracy": "{:.4%}".format,
                "log_loss": "{:.6f}".format,
                "brier_score": "{:.6f}".format,
            },
        )
    )


def plot_log_loss(
    metrics: pd.DataFrame,
) -> None:
    """So sánh log loss."""

    pivot = metrics.pivot(
        index="position",
        columns="model",
        values="log_loss",
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(14, 6),
        width=0.78,
    )

    ax.axhline(
        np.log(10),
        color="red",
        linestyle="--",
        label="Uniform log loss",
    )

    ax.set_title(
        "Random Forest và Markov trên tập test",
        fontsize=15,
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

    ax.legend()

    fig = ax.get_figure()
    fig.tight_layout()

    output_file = (
        FIGURE_DIR
        / "markov_enhanced_ml_log_loss.png"
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

    (
        metrics,
        weights,
        predictions,
    ) = run_experiment(
        features,
        targets,
        dates,
    )

    print_results(
        metrics,
        weights,
    )

    metrics.to_csv(
        TABLE_DIR
        / "markov_enhanced_ml_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    weights.to_csv(
        TABLE_DIR
        / "markov_enhanced_weights.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions.to_csv(
        TABLE_DIR
        / "markov_enhanced_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_log_loss(metrics)


if __name__ == "__main__":
    main()