"""Dự đoán trực tiếp hai chữ số cuối bằng CatBoost 100 lớp."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import binomtest
from sklearn.metrics import log_loss


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

LAGS = [1, 2, 3, 7, 14, 19, 30, 60]
ROLLING_WINDOWS = [30, 90, 365]

RANDOM_STATE = 42
NUMBER_OF_CLASSES = 100
NUMBER_OF_BOOTSTRAPS = 20_000

# Ba tập test không giao nhau
FOLDS = [
    {
        "name": "2020-2021",
        "train_end": 2017,
        "validation_start": 2018,
        "validation_end": 2019,
        "test_start": 2020,
        "test_end": 2021,
    },
    {
        "name": "2022-2023",
        "train_end": 2019,
        "validation_start": 2020,
        "validation_end": 2021,
        "test_start": 2022,
        "test_end": 2023,
    },
    {
        "name": "2024-2026",
        "train_end": 2021,
        "validation_start": 2022,
        "validation_end": 2023,
        "test_start": 2024,
        "test_end": 2026,
    },
]


def read_data() -> pd.DataFrame:
    """Đọc và chuẩn hóa target 00-99."""

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


def build_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Tạo feature không sử dụng thông tin tương lai."""

    feature_data = {}

    # Lag của từng vị trí
    for column in DIGIT_COLUMNS:
        for lag in LAGS:
            feature_data[
                f"{column}_lag_{lag}"
            ] = df[column].shift(lag)

    # Lag trực tiếp của số 00-99
    for lag in LAGS:
        feature_data[
            f"last_2_lag_{lag}"
        ] = df["last_2_target"].shift(lag)

    # Giá trị đã shift để không dùng target hiện tại
    historical_target = (
        df["last_2_target"].shift(1)
    )

    # Tần suất của từng số trong các cửa sổ quá khứ
    for window in ROLLING_WINDOWS:
        for number in range(100):
            feature_data[
                f"freq_{number:02d}_w{window}"
            ] = (
                historical_target
                .eq(number)
                .rolling(
                    window=window,
                    min_periods=window,
                )
                .mean()
            )

    day_of_week = df["date"].dt.dayofweek
    month = df["date"].dt.month

    feature_data["day_of_week_sin"] = np.sin(
        2 * np.pi * day_of_week / 7
    )
    feature_data["day_of_week_cos"] = np.cos(
        2 * np.pi * day_of_week / 7
    )
    feature_data["month_sin"] = np.sin(
        2 * np.pi * month / 12
    )
    feature_data["month_cos"] = np.cos(
        2 * np.pi * month / 12
    )

    return pd.DataFrame(
        feature_data,
        index=df.index,
    )


def prepare_data(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """Chuẩn bị feature, target và ngày."""

    features = build_features(df)

    valid = features.notna().all(axis=1)

    return (
        features.loc[valid]
        .astype(float)
        .reset_index(drop=True),
        df.loc[valid, "last_2_target"]
        .astype(int)
        .reset_index(drop=True),
        df.loc[valid, "date"]
        .reset_index(drop=True),
    )


def create_model() -> CatBoostClassifier:
    """Khởi tạo mô hình 100 lớp."""

    return CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        iterations=1_000,
        learning_rate=0.04,
        depth=7,
        l2_leaf_reg=15,
        random_strength=1,
        random_seed=RANDOM_STATE,
        thread_count=-1,
        early_stopping_rounds=100,
        allow_writing_files=False,
        verbose=100,
    )


def align_probabilities(
    model: CatBoostClassifier,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Căn xác suất theo đầy đủ 100 lớp."""

    aligned = np.zeros(
        (
            len(probabilities),
            NUMBER_OF_CLASSES,
        )
    )

    for model_column, label in enumerate(
        model.classes_
    ):
        aligned[:, int(label)] = (
            probabilities[:, model_column]
        )

    return aligned


def blend_with_uniform(
    model_probabilities: np.ndarray,
    model_weight: float,
) -> np.ndarray:
    """Co xác suất của mô hình về Uniform."""

    uniform = np.full_like(
        model_probabilities,
        1 / NUMBER_OF_CLASSES,
    )

    return (
        model_weight * model_probabilities
        + (1 - model_weight) * uniform
    )


def select_blend_weight(
    y_validation: np.ndarray,
    validation_probabilities: np.ndarray,
) -> tuple[float, float]:
    """Chọn trọng số hoàn toàn trên validation."""

    candidates = []

    for model_weight in np.arange(
        0,
        1.01,
        0.05,
    ):
        blended = blend_with_uniform(
            validation_probabilities,
            model_weight,
        )

        score = log_loss(
            y_validation,
            blended,
            labels=list(
                range(NUMBER_OF_CLASSES)
            ),
        )

        candidates.append(
            {
                "model_weight": model_weight,
                "validation_log_loss": score,
            }
        )

    candidates = pd.DataFrame(candidates)

    best = candidates.sort_values(
        "validation_log_loss"
    ).iloc[0]

    return (
        float(best["model_weight"]),
        float(best["validation_log_loss"]),
    )


def top_k_accuracy(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    k: int,
) -> float:
    """Tính Top-k accuracy."""

    top_k = np.argsort(
        probabilities,
        axis=1,
    )[:, -k:]

    return np.mean(
        [
            true_label in candidates
            for true_label, candidates
            in zip(y_true, top_k)
        ]
    )


def multiclass_brier_score(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Tính Brier score 100 lớp."""

    one_hot = np.eye(
        NUMBER_OF_CLASSES
    )[y_true]

    return np.mean(
        np.sum(
            (probabilities - one_hot) ** 2,
            axis=1,
        )
    )


def evaluate(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """Đánh giá mô hình."""

    predictions = probabilities.argmax(
        axis=1
    )

    return {
        "top_1_accuracy": np.mean(
            predictions == y_true
        ),
        "top_5_accuracy": top_k_accuracy(
            y_true,
            probabilities,
            k=5,
        ),
        "top_10_accuracy": top_k_accuracy(
            y_true,
            probabilities,
            k=10,
        ),
        "log_loss": log_loss(
            y_true,
            probabilities,
            labels=list(range(100)),
        ),
        "brier_score": multiclass_brier_score(
            y_true,
            probabilities,
        ),
    }


def run_fold(
    features: pd.DataFrame,
    target: pd.Series,
    dates: pd.Series,
    fold: dict,
) -> tuple[dict, pd.DataFrame]:
    """Huấn luyện và kiểm tra một fold thời gian."""

    years = dates.dt.year

    train_mask = years.le(
        fold["train_end"]
    )

    validation_mask = years.between(
        fold["validation_start"],
        fold["validation_end"],
    )

    test_mask = years.between(
        fold["test_start"],
        fold["test_end"],
    )

    x_train = features.loc[train_mask]
    y_train = target.loc[train_mask]

    x_validation = features.loc[
        validation_mask
    ]

    y_validation = target.loc[
        validation_mask
    ]

    x_test = features.loc[test_mask]
    y_test = target.loc[test_mask]

    print(
        f"\nFold {fold['name']}: "
        f"train={len(x_train):,}, "
        f"validation={len(x_validation):,}, "
        f"test={len(x_test):,}"
    )

    model = create_model()

    model.fit(
        x_train,
        y_train,
        eval_set=(
            x_validation,
            y_validation,
        ),
        use_best_model=True,
    )

    validation_probabilities = (
        align_probabilities(
            model,
            model.predict_proba(
                x_validation
            ),
        )
    )

    (
        selected_weight,
        validation_log_loss,
    ) = select_blend_weight(
        y_validation.to_numpy(),
        validation_probabilities,
    )

    test_probabilities = (
        align_probabilities(
            model,
            model.predict_proba(x_test),
        )
    )

    test_probabilities = (
        blend_with_uniform(
            test_probabilities,
            selected_weight,
        )
    )

    metrics = evaluate(
        y_test.to_numpy(),
        test_probabilities,
    )

    result = {
        "fold": fold["name"],
        "train_size": len(x_train),
        "validation_size": len(x_validation),
        "test_size": len(x_test),
        "best_iteration": (
            model.get_best_iteration()
        ),
        "model_weight": selected_weight,
        "validation_log_loss": (
            validation_log_loss
        ),
        **metrics,
    }

    predictions = pd.DataFrame(
        {
            "date": dates.loc[
                test_mask
            ].to_numpy(),
            "actual": y_test.to_numpy(),
            "predicted": (
                test_probabilities.argmax(
                    axis=1
                )
            ),
        }
    )

    true_probabilities = (
        test_probabilities[
            np.arange(len(y_test)),
            y_test.to_numpy(),
        ]
    )

    predictions["model_loss"] = (
        -np.log(
            np.clip(
                true_probabilities,
                1e-15,
                1,
            )
        )
    )

    predictions["uniform_loss"] = (
        np.log(NUMBER_OF_CLASSES)
    )

    predictions["improvement"] = (
        predictions["uniform_loss"]
        - predictions["model_loss"]
    )

    predictions["fold"] = fold["name"]

    return result, predictions


def statistical_tests(
    predictions: pd.DataFrame,
) -> dict:
    """Kiểm định accuracy và bootstrap log-loss."""

    number_correct = (
        predictions["actual"]
        .eq(predictions["predicted"])
        .sum()
    )

    number_tested = len(predictions)

    accuracy_p_value = binomtest(
        k=number_correct,
        n=number_tested,
        p=0.01,
        alternative="greater",
    ).pvalue

    # Block bootstrap theo năm
    yearly_improvement = (
        predictions.assign(
            year=predictions["date"].dt.year
        )
        .groupby("year")["improvement"]
        .mean()
    )

    values = yearly_improvement.to_numpy()

    random_generator = (
        np.random.default_rng(
            RANDOM_STATE
        )
    )

    bootstrap_means = np.empty(
        NUMBER_OF_BOOTSTRAPS
    )

    for iteration in range(
        NUMBER_OF_BOOTSTRAPS
    ):
        sample = random_generator.choice(
            values,
            size=len(values),
            replace=True,
        )

        bootstrap_means[iteration] = (
            sample.mean()
        )

    lower, upper = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
    )

    return {
        "number_tested": number_tested,
        "number_correct": int(
            number_correct
        ),
        "top_1_accuracy": (
            number_correct / number_tested
        ),
        "binomial_p_value": (
            accuracy_p_value
        ),
        "mean_log_loss_improvement": (
            predictions[
                "improvement"
            ].mean()
        ),
        "bootstrap_ci_low": lower,
        "bootstrap_ci_high": upper,
    }


def plot_fold_results(
    fold_results: pd.DataFrame,
) -> None:
    """Vẽ log loss theo fold."""

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
        fold_results["log_loss"],
        width,
        label="CatBoost",
        color="steelblue",
    )

    ax.bar(
        positions + width / 2,
        np.log(100),
        width,
        label="Uniform",
        color="gray",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        fold_results["fold"]
    )

    ax.set_ylabel("Log loss")

    ax.set_title(
        "CatBoost dự đoán trực tiếp 00–99",
        fontsize=15,
        fontweight="bold",
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    output_file = (
        FIGURE_DIR
        / "modern_ml_last2_log_loss.png"
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

    features, target, dates = (
        prepare_data(df)
    )

    result_records = []
    prediction_frames = []

    for fold in FOLDS:
        result, predictions = run_fold(
            features,
            target,
            dates,
            fold,
        )

        result_records.append(result)
        prediction_frames.append(
            predictions
        )

    fold_results = pd.DataFrame(
        result_records
    )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    significance = statistical_tests(
        predictions
    )

    print("\n" + "=" * 130)
    print("KẾT QUẢ CATBOOST 100 LỚP")
    print("=" * 130)

    print(
        fold_results.to_string(
            index=False,
            formatters={
                "model_weight": "{:.2f}".format,
                "validation_log_loss": (
                    "{:.6f}".format
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
                "log_loss": "{:.6f}".format,
                "brier_score": "{:.6f}".format,
            },
        )
    )

    print("\nKiểm định tổng hợp:")

    for key, value in significance.items():
        print(f"{key}: {value}")

    fold_results.to_csv(
        TABLE_DIR
        / "modern_ml_last2_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions.to_csv(
        TABLE_DIR
        / "modern_ml_last2_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [significance]
    ).to_csv(
        TABLE_DIR
        / "modern_ml_last2_significance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_fold_results(fold_results)


if __name__ == "__main__":
    main()