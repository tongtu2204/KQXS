"""Baseline ML dự đoán riêng từng chữ số của giải đặc biệt."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss


# ============================================================
# Cấu hình
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "kqxsmb_digits.csv"
)

TABLE_DIR = PROJECT_DIR / "artifacts" / "tables"
FIGURE_DIR = PROJECT_DIR / "artifacts" / "figures"
MODEL_DIR = PROJECT_DIR / "artifacts" / "models"

DIGIT_COLUMNS = [
    "digit_1",
    "digit_2",
    "digit_3",
    "digit_4",
    "digit_5",
]

LAGS = [1, 2, 3, 7, 14, 19, 30]

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# Đọc và tạo đặc trưng
# ============================================================

def read_data() -> pd.DataFrame:
    """Đọc dữ liệu theo thứ tự thời gian."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {DATA_FILE}."
        )

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
    """Tạo đặc trưng chỉ từ thông tin có trước kỳ cần dự đoán."""

    features = pd.DataFrame(
        index=df.index
    )

    # Lịch sử chữ số tại cả 5 vị trí
    for column in DIGIT_COLUMNS:
        for lag in LAGS:
            features[f"{column}_lag_{lag}"] = (
                df[column].shift(lag)
            )

    # Biểu diễn chu kỳ của thứ trong tuần
    day_of_week = df["date"].dt.dayofweek

    features["day_of_week_sin"] = np.sin(
        2 * np.pi * day_of_week / 7
    )

    features["day_of_week_cos"] = np.cos(
        2 * np.pi * day_of_week / 7
    )

    # Biểu diễn chu kỳ của tháng
    month = df["date"].dt.month

    features["month_sin"] = np.sin(
        2 * np.pi * month / 12
    )

    features["month_cos"] = np.cos(
        2 * np.pi * month / 12
    )

    return features


def prepare_model_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Loại các dòng chưa đủ lịch sử lag."""

    features = build_features(df)

    valid = features.notna().all(axis=1)

    features = (
        features.loc[valid]
        .astype(float)
        .reset_index(drop=True)
    )

    targets = (
        df.loc[valid, DIGIT_COLUMNS]
        .astype(int)
        .reset_index(drop=True)
    )

    dates = (
        df.loc[valid, "date"]
        .reset_index(drop=True)
    )

    return features, targets, dates


# ============================================================
# Metrics
# ============================================================

def align_probabilities(
    model: RandomForestClassifier,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Đảm bảo probability có đủ 10 cột tương ứng 0-9."""

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


def multiclass_brier_score(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Tính multiclass Brier score."""

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
    k: int,
) -> float:
    """Tỷ lệ chữ số thật nằm trong k dự đoán cao nhất."""

    top_k = np.argsort(
        probabilities,
        axis=1,
    )[:, -k:]

    correct = np.array(
        [
            true_digit in candidates
            for true_digit, candidates
            in zip(y_true, top_k)
        ]
    )

    return correct.mean()


def evaluate_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """Tính các chỉ số đánh giá xác suất."""

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
            k=3,
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


# ============================================================
# Huấn luyện và đánh giá
# ============================================================

def train_and_evaluate(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    dates: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:
    """Huấn luyện 5 mô hình và đánh giá trên tập cuối."""

    split_index = int(
        len(features) * (1 - TEST_SIZE)
    )

    x_train = features.iloc[:split_index]
    x_test = features.iloc[split_index:]

    y_train = targets.iloc[:split_index]
    y_test = targets.iloc[split_index:]

    test_dates = dates.iloc[split_index:]

    print(f"Số quan sát train: {len(x_train):,}")
    print(f"Số quan sát test : {len(x_test):,}")
    print(
        "Khoảng test      : "
        f"{test_dates.min().date()} đến "
        f"{test_dates.max().date()}"
    )

    metric_records = []

    prediction_table = pd.DataFrame(
        {
            "date": test_dates.to_numpy(),
        }
    )

    trained_models = {}

    for position, target_column in enumerate(
        DIGIT_COLUMNS,
        start=1,
    ):
        train_target = (
            y_train[target_column]
            .to_numpy()
        )

        test_target = (
            y_test[target_column]
            .to_numpy()
        )

        # --------------------------------------------
        # Baseline ngẫu nhiên đều
        # --------------------------------------------

        uniform_probabilities = np.full(
            (len(y_test), 10),
            0.10,
        )

        uniform_metrics = {
            "accuracy": 0.10,
            "top_3_accuracy": 0.30,
            "log_loss": np.log(10),
            "brier_score": 0.90,
        }

        metric_records.append(
            {
                "model": "Uniform",
                "position": position,
                **uniform_metrics,
            }
        )

        # --------------------------------------------
        # Baseline tần suất trong train
        # --------------------------------------------

        prior_probabilities = (
            pd.Series(train_target)
            .value_counts(normalize=True)
            .reindex(range(10), fill_value=0)
            .to_numpy()
        )

        frequency_probabilities = np.tile(
            prior_probabilities,
            (len(y_test), 1),
        )

        frequency_metrics = evaluate_probabilities(
            test_target,
            frequency_probabilities,
        )

        metric_records.append(
            {
                "model": "Training frequency",
                "position": position,
                **frequency_metrics,
            }
        )

        # --------------------------------------------
        # Random Forest
        # --------------------------------------------

        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        print(
            f"Đang huấn luyện mô hình vị trí "
            f"{position}..."
        )

        model.fit(
            x_train,
            train_target,
        )

        model_probabilities = align_probabilities(
            model,
            model.predict_proba(x_test),
        )

        model_metrics = evaluate_probabilities(
            test_target,
            model_probabilities,
        )

        metric_records.append(
            {
                "model": "Random Forest",
                "position": position,
                **model_metrics,
            }
        )

        prediction_table[
            f"actual_digit_{position}"
        ] = test_target

        prediction_table[
            f"predicted_digit_{position}"
        ] = model_probabilities.argmax(
            axis=1
        )

        trained_models[target_column] = model

    metrics = pd.DataFrame(
        metric_records
    )

    return (
        metrics,
        prediction_table,
        x_test,
        trained_models,
    )


# ============================================================
# Đánh giá kết quả ghép 5 chữ số
# ============================================================

def evaluate_combined_predictions(
    prediction_table: pd.DataFrame,
) -> pd.DataFrame:
    """Đánh giá khi ghép dự đoán của 5 mô hình."""

    actual_columns = [
        f"actual_digit_{position}"
        for position in range(1, 6)
    ]

    predicted_columns = [
        f"predicted_digit_{position}"
        for position in range(1, 6)
    ]

    actual = prediction_table[
        actual_columns
    ].to_numpy()

    predicted = prediction_table[
        predicted_columns
    ].to_numpy()

    correct_matrix = (
        actual == predicted
    )

    prediction_table["correct_digits"] = (
        correct_matrix.sum(axis=1)
    )

    prediction_table["actual_full_result"] = [
        "".join(map(str, row))
        for row in actual
    ]

    prediction_table[
        "predicted_full_result"
    ] = [
        "".join(map(str, row))
        for row in predicted
    ]

    prediction_table["actual_last_2"] = (
        prediction_table[
            "actual_full_result"
        ].str[-2:]
    )

    prediction_table["predicted_last_2"] = (
        prediction_table[
            "predicted_full_result"
        ].str[-2:]
    )

    summary = pd.DataFrame(
        [
            {
                "metric": "Mean correct digits",
                "value": (
                    prediction_table[
                        "correct_digits"
                    ].mean()
                ),
                "random_expectation": 0.5,
            },
            {
                "metric": "Exact five digits",
                "value": (
                    prediction_table[
                        "actual_full_result"
                    ]
                    == prediction_table[
                        "predicted_full_result"
                    ]
                ).mean(),
                "random_expectation": 1 / 100_000,
            },
            {
                "metric": "Exact last two digits",
                "value": (
                    prediction_table[
                        "actual_last_2"
                    ]
                    == prediction_table[
                        "predicted_last_2"
                    ]
                ).mean(),
                "random_expectation": 1 / 100,
            },
        ]
    )

    return summary


# ============================================================
# Dự đoán kỳ tiếp theo
# ============================================================

def predict_next_draw(
    df: pd.DataFrame,
    features: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """Huấn luyện lại trên toàn bộ dữ liệu và dự đoán kỳ kế tiếp."""

    next_date = (
        df["date"].max()
        + pd.Timedelta(days=1)
    )

    next_row = {
        "date": next_date,
    }

    for column in DIGIT_COLUMNS:
        next_row[column] = np.nan

    extended_df = pd.concat(
        [
            df[["date", *DIGIT_COLUMNS]],
            pd.DataFrame([next_row]),
        ],
        ignore_index=True,
    )

    next_features = (
        build_features(extended_df)
        .iloc[[-1]]
        .astype(float)
    )

    records = []
    final_models = {}

    for position, target_column in enumerate(
        DIGIT_COLUMNS,
        start=1,
    ):
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        model.fit(
            features,
            targets[target_column],
        )

        probabilities = align_probabilities(
            model,
            model.predict_proba(next_features),
        )[0]

        top_digits = np.argsort(
            probabilities
        )[::-1][:3]

        records.append(
            {
                "date": next_date,
                "position": position,
                "predicted_digit": int(
                    top_digits[0]
                ),
                "top_1_probability": (
                    probabilities[top_digits[0]]
                ),
                "top_2_digit": int(
                    top_digits[1]
                ),
                "top_2_probability": (
                    probabilities[top_digits[1]]
                ),
                "top_3_digit": int(
                    top_digits[2]
                ),
                "top_3_probability": (
                    probabilities[top_digits[2]]
                ),
            }
        )

        final_models[target_column] = model

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "models": final_models,
            "feature_columns": features.columns.tolist(),
            "lags": LAGS,
        },
        MODEL_DIR / "random_forest_digit_models.joblib",
    )

    return pd.DataFrame(records)


# ============================================================
# Biểu đồ
# ============================================================

def plot_accuracy(
    metrics: pd.DataFrame,
) -> None:
    """So sánh accuracy của các mô hình."""

    pivot = metrics.pivot(
        index="position",
        columns="model",
        values="accuracy",
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(13, 6),
        width=0.75,
        color=[
            "steelblue",
            "darkorange",
            "seagreen",
        ],
    )

    ax.axhline(
        0.10,
        color="red",
        linestyle="--",
        label="Mức ngẫu nhiên 10%",
    )

    ax.set_title(
        "Độ chính xác dự đoán từng chữ số trên tập test",
        fontsize=15,
        fontweight="bold",
    )

    ax.set_xlabel("Vị trí")
    ax.set_ylabel("Accuracy")
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
    )

    fig = ax.get_figure()
    fig.tight_layout()

    output_file = (
        FIGURE_DIR
        / "ml_digit_baseline_accuracy.png"
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


# ============================================================
# Chạy chương trình
# ============================================================

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

    features, targets, dates = (
        prepare_model_data(df)
    )

    (
        metrics,
        prediction_table,
        _,
        _,
    ) = train_and_evaluate(
        features,
        targets,
        dates,
    )

    combined_summary = (
        evaluate_combined_predictions(
            prediction_table
        )
    )

    next_prediction = predict_next_draw(
        df,
        features,
        targets,
    )

    metrics.to_csv(
        TABLE_DIR / "ml_digit_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    prediction_table.to_csv(
        TABLE_DIR / "ml_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    combined_summary.to_csv(
        TABLE_DIR / "ml_combined_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    next_prediction.to_csv(
        TABLE_DIR / "next_draw_prediction.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 95)
    print("KẾT QUẢ DỰ ĐOÁN TỪNG VỊ TRÍ")
    print("=" * 95)

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

    print("\nKết quả ghép 5 chữ số:")

    print(
        combined_summary.to_string(
            index=False,
            formatters={
                "value": "{:.6f}".format,
                "random_expectation": "{:.6f}".format,
            },
        )
    )

    print("\nDự đoán cho kỳ tiếp theo:")

    print(
        next_prediction.to_string(
            index=False,
            formatters={
                "top_1_probability": "{:.4%}".format,
                "top_2_probability": "{:.4%}".format,
                "top_3_probability": "{:.4%}".format,
            },
        )
    )

    predicted_number = "".join(
        next_prediction[
            "predicted_digit"
        ]
        .astype(str)
        .tolist()
    )

    print(
        f"\nSố 5 chữ số dự đoán: "
        f"{predicted_number}"
    )

    print(
        f"Hai chữ số cuối dự đoán: "
        f"{predicted_number[-2:]}"
    )

    plot_accuracy(metrics)


if __name__ == "__main__":
    main()