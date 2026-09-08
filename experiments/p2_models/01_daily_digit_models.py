"""Mô hình dự đoán chữ số theo ngày cho Phần 2.

Mỗi dự báo có 5 vị trí x 10 chữ số. Một chữ số được coi là actual nếu nó xuất
hiện ở vị trí tương ứng trong ít nhất một trong 27 kết quả của ngày.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

POSITION_NAMES = ("ten_thousands", "thousands", "hundreds", "tens", "units")

DATA_FILE = PROJECT_DIR / "data" / "processed" / "daily_digit_targets.csv"
ARTIFACT_DIR = PROJECT_DIR / "artifacts" / "p2_models" / "baseline"
TABLE_DIR = ARTIFACT_DIR
FIGURE_DIR = ARTIFACT_DIR / "figures"
FOLDS = {
    "validation_2023_2024": ("2023-01-01", "2024-12-31", "2022-12-31"),
}


def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_FILE, parse_dates=["date"])
    data = data.sort_values("date").reset_index(drop=True)
    return data


def target_columns() -> list[str]:
    return [f"{position}_d{digit}" for position in POSITION_NAMES for digit in range(10)]


def model_probability(name: str, history: pd.DataFrame, dates: pd.Series) -> np.ndarray:
    """Return n_days x 50 probabilities using history strictly before dates."""
    columns = target_columns()
    if name == "uniform_27_iid":
        # Baseline Uniform ở cấp độ ngày: P(digit xuất hiện >= 1 lần trong
        # 27 kết quả) = 1 - (1 - 0.1)^27.
        daily_probability = 1 - (1 - 0.1) ** 27
        return np.full((len(dates), len(columns)), daily_probability)
    if name == "expanding_frequency":
        p = history[columns].mean().to_numpy(dtype=float)
        return np.tile(p, (len(dates), 1))
    if name.startswith("rolling_frequency_w"):
        window = int(name.rsplit("w", 1)[1])
        p = history[columns].tail(window).mean().to_numpy(dtype=float)
        return np.tile(p, (len(dates), 1))
    if name == "markov_presence":
        # Binary Markov model for presence/absence, with Laplace smoothing.
        all_data = history[columns].to_numpy(dtype=int)
        result = []
        for index in range(0, len(columns), 10):
            block = all_data[:, index:index + 10]
            previous, current = block[:-1], block[1:]
            probs = []
            for digit in range(10):
                a = previous[:, digit]
                b = current[:, digit]
                n1 = ((a == 1) & (b == 1)).sum()
                d1 = (a == 1).sum()
                n0 = ((a == 0) & (b == 1)).sum()
                d0 = (a == 0).sum()
                last = block[-1, digit]
                probs.append((n1 + 1) / (d1 + 2) if last else (n0 + 1) / (d0 + 2))
            result.extend(probs)
        return np.tile(np.asarray(result), (len(dates), 1))
    raise ValueError(name)


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    columns = target_columns()
    feature_parts = []
    for lag in (1, 2, 3, 7, 14, 30):
        part = data[columns].shift(lag).copy()
        part.columns = [f"{column}_lag{lag}" for column in columns]
        feature_parts.append(part)
    calendar = pd.DataFrame({
        "dow_sin": np.sin(2 * np.pi * data.date.dt.dayofweek / 7),
        "dow_cos": np.cos(2 * np.pi * data.date.dt.dayofweek / 7),
        "month_sin": np.sin(2 * np.pi * data.date.dt.month / 12),
        "month_cos": np.cos(2 * np.pi * data.date.dt.month / 12),
    }, index=data.index)
    return pd.concat([*feature_parts, calendar], axis=1)


def random_forest_probability(data: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, history_end: pd.Timestamp) -> tuple[pd.Series, np.ndarray]:
    features = build_features(data)
    columns = target_columns()
    train = data.date <= history_end
    test = data.date.between(start, end)
    valid_train = train & features.notna().all(axis=1)
    valid_test = test & features.notna().all(axis=1)
    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=40, min_samples_leaf=5, n_jobs=-1, random_state=42))
    model.fit(features.loc[valid_train], data.loc[valid_train, columns])
    probabilities = []
    for estimator, classes in zip(model.estimators_, [model.classes_ for model in model.estimators_]):
        raw = estimator.predict_proba(features.loc[valid_test])
        aligned = np.zeros((len(raw), 2))
        for j, cls in enumerate(classes):
            aligned[:, int(cls)] = raw[:, j]
        probabilities.append(aligned[:, 1])
    return data.loc[valid_test, "date"].reset_index(drop=True), np.column_stack(probabilities)


def score(probabilities: np.ndarray, actual: np.ndarray, model: str, fold: str, dates: pd.Series) -> tuple[pd.DataFrame, dict]:
    rows = []
    for i, date in enumerate(dates):
        p, y = probabilities[i], actual[i]
        row = {"date": date, "model": model, "fold": fold, "brier_score": float(np.mean((p - y) ** 2))}
        for k in (1, 3, 5):
            top = np.argsort(-p, axis=1)[:, :k]
            hits = [bool(y[pos, top[pos]].any()) for pos in range(5)]
            row[f"any_hit_top{k}"] = float(np.mean(hits))
            row[f"digit_recall_top{k}"] = float(sum(y[pos, top[pos]].sum() for pos in range(5)) / max(y.sum(), 1))
        rows.append(row)
    daily = pd.DataFrame(rows)
    summary = {"model": model, "fold": fold, "n_days": len(daily), "brier_score": daily.brier_score.mean()}
    for k in (1, 3, 5):
        summary[f"any_hit_top{k}"] = daily[f"any_hit_top{k}"].mean()
        summary[f"digit_recall_top{k}"] = daily[f"digit_recall_top{k}"].mean()
    return daily, summary


def main() -> None:
    data = load_data()
    columns = target_columns()
    all_summary, all_daily, all_probability = [], [], []
    model_names = ["uniform_27_iid", "expanding_frequency", "rolling_frequency_w30", "rolling_frequency_w90", "rolling_frequency_w365", "markov_presence"]
    for fold, (start_text, end_text, history_text) in FOLDS.items():
        start, end, history_end = map(pd.Timestamp, (start_text, end_text, history_text))
        test = data.date.between(start, end)
        history = data.loc[data.date <= history_end]
        actual = data.loc[test, columns].to_numpy(dtype=int).reshape(-1, 5, 10)
        dates = data.loc[test, "date"].reset_index(drop=True)
        for model in model_names:
            p = model_probability(model, history, dates).reshape(-1, 5, 10)
            daily, summary = score(p, actual, model, fold, dates)
            all_daily.append(daily)
            all_summary.append(summary)
            all_probability.append(pd.DataFrame(p.reshape(len(p), 50), columns=[f"p_{c}" for c in columns]).assign(date=dates, model=model, fold=fold))
        rf_dates, rf_p = random_forest_probability(data, start, end, history_end)
        rf_actual = data.loc[data.date.isin(rf_dates), columns].to_numpy(dtype=int).reshape(-1, 5, 10)
        daily, summary = score(rf_p.reshape(-1, 5, 10), rf_actual, "random_forest", fold, rf_dates)
        all_daily.append(daily); all_summary.append(summary)
        all_probability.append(pd.DataFrame(rf_p, columns=[f"p_{c}" for c in columns]).assign(date=rf_dates, model="random_forest", fold=fold))

    summary = pd.DataFrame(all_summary)
    daily = pd.concat(all_daily, ignore_index=True)
    probabilities = pd.concat(all_probability, ignore_index=True)
    for directory in (ARTIFACT_DIR, TABLE_DIR, FIGURE_DIR): directory.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLE_DIR / "model_summary.csv", index=False)
    daily.to_csv(TABLE_DIR / "model_daily_scores.csv.gz", index=False, compression="gzip")
    probabilities.to_csv(ARTIFACT_DIR / "model_daily_probabilities.csv.gz", index=False, compression="gzip")
    plot = summary[summary.fold.eq("validation_2023_2024")].sort_values("brier_score")
    plt.figure(figsize=(10, 5)); plt.bar(plot.model, plot.brier_score); plt.xticks(rotation=30, ha="right"); plt.ylabel("Daily multi-label Brier score"); plt.title("Phần 2: so sánh baseline trên validation 2023–2024"); plt.tight_layout(); plt.savefig(FIGURE_DIR / "baseline_brier_validation.png", dpi=180); plt.close()
    print(summary.to_string(index=False))
    print(f"\nĐã ghi artifacts vào: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
