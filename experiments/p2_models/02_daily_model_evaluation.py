"""Đánh giá đầy đủ các model chữ số theo ngày bằng walk-forward.

Target của mỗi ngày là 5 vị trí x 10 chữ số nhị phân. Tất cả model đều phát
ra xác suất cho cùng 50 target để có thể so sánh công bằng và chuyển tiếp sang
phần ranking/strategy.
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

POSITIONS = ("ten_thousands", "thousands", "hundreds", "tens", "units")
DATA_FILE = PROJECT_DIR / "data" / "processed" / "daily_digit_targets.csv"
TABLE_DIR = PROJECT_DIR / "artifacts" / "models" / "daily_digit"
FIGURE_DIR = TABLE_DIR / "figures"
FOLDS = {
    "validation_2023_2024": ("2023-01-01", "2024-12-31", "2022-12-31"),
}
MODEL_NAMES = (
    "uniform_27_iid",
    "expanding_beta",
    "rolling_beta_w30",
    "rolling_beta_w90",
    "rolling_beta_w365",
    "dirichlet_daily",
    "markov_presence",
    "random_forest",
)


def columns() -> list[str]:
    return [f"{position}_d{digit}" for position in POSITIONS for digit in range(10)]


def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Chưa có {DATA_FILE}; hãy chạy 00_prepare_daily_targets.py")
    return pd.read_csv(DATA_FILE, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


def clip_probability(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def statistical_probability(model: str, history: pd.DataFrame) -> np.ndarray:
    result = np.zeros((5, 10), dtype=float)
    for position_index, position in enumerate(POSITIONS):
        x = history[[f"{position}_d{digit}" for digit in range(10)]].to_numpy(dtype=float)
        if model == "uniform_27_iid":
            result[position_index] = 1 - 0.9 ** 27
        elif model == "expanding_beta":
            result[position_index] = (x.sum(axis=0) + 1) / (len(x) + 2)
        elif model.startswith("rolling_beta_w"):
            window = int(model.rsplit("w", 1)[1])
            x = x[-min(window, len(x)):]
            result[position_index] = (x.sum(axis=0) + 1) / (len(x) + 2)
        elif model == "dirichlet_daily":
            # Dirichlet posterior trên số lần một chữ số xuất hiện trong ngày,
            # sau đó đổi sang xác suất xuất hiện >= 1 lần trong 27 kết quả.
            counts = x.sum(axis=0) + 1
            per_draw = counts / counts.sum()
            result[position_index] = 1 - (1 - per_draw) ** 27
        elif model == "markov_presence":
            previous, current = x[:-1], x[1:]
            last = x[-1]
            for digit in range(10):
                state = previous[:, digit] == int(last[digit])
                numerator = ((current[:, digit] == 1) & state).sum() + 1
                denominator = state.sum() + 2
                result[position_index, digit] = numerator / denominator
        else:
            raise ValueError(model)
    return clip_probability(result)


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    cols = columns()
    parts = []
    for lag in (1, 2, 3, 7, 14, 30):
        part = data[cols].shift(lag).copy()
        part.columns = [f"{c}_lag{lag}" for c in cols]
        parts.append(part)
    date = data["date"]
    parts.append(pd.DataFrame({
        "dow_sin": np.sin(2 * np.pi * date.dt.dayofweek / 7),
        "dow_cos": np.cos(2 * np.pi * date.dt.dayofweek / 7),
        "month_sin": np.sin(2 * np.pi * date.dt.month / 12),
        "month_cos": np.cos(2 * np.pi * date.dt.month / 12),
    }, index=data.index))
    return pd.concat(parts, axis=1)


def fit_random_forest(data: pd.DataFrame, history_end: pd.Timestamp, test_mask: pd.Series) -> tuple[pd.Series, np.ndarray]:
    cols = columns()
    features = build_features(data)
    train_mask = (data.date <= history_end) & features.notna().all(axis=1)
    valid_mask = test_mask & features.notna().all(axis=1)
    model = MultiOutputClassifier(RandomForestClassifier(
        n_estimators=80, min_samples_leaf=5, n_jobs=-1, random_state=42,
    ))
    model.fit(features.loc[train_mask], data.loc[train_mask, cols])
    output = []
    for estimator in model.estimators_:
        raw = estimator.predict_proba(features.loc[valid_mask])
        aligned = np.zeros((len(raw), 2))
        for index, cls in enumerate(estimator.classes_):
            aligned[:, int(cls)] = raw[:, index]
        output.append(aligned[:, 1])
    return data.loc[valid_mask, "date"].reset_index(drop=True), clip_probability(np.column_stack(output).reshape(-1, 5, 10))


def evaluate(model: str, fold: str, dates: pd.Series, probabilities: np.ndarray, actual: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_rows, rank_rows, calibration_rows = [], [], []
    for row_index, date in enumerate(dates):
        p, y = probabilities[row_index], actual[row_index]
        daily = {"date": date, "model": model, "fold": fold, "brier_score": float(np.mean((p - y) ** 2)), "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))}
        for k in (1, 3, 5, 10):
            top = np.argsort(-p, axis=1)[:, :k]
            daily[f"hit_any_top{k}"] = np.mean([bool(y[pos, top[pos]].any()) for pos in range(5)])
            daily[f"recall_top{k}"] = sum(y[pos, top[pos]].sum() for pos in range(5)) / max(y.sum(), 1)
        for pos, position in enumerate(POSITIONS):
            daily[f"brier_{position}"] = np.mean((p[pos] - y[pos]) ** 2)
            order = np.argsort(-p[pos])
            for digit in np.where(y[pos] == 1)[0]:
                rank_rows.append({"date": date, "model": model, "fold": fold, "position": position, "digit": int(digit), "rank": int(np.where(order == digit)[0][0] + 1)})
        daily_rows.append(daily)
        flat_p, flat_y = p.ravel(), y.ravel()
        for bin_id in range(10):
            low, high = bin_id / 10, (bin_id + 1) / 10
            mask = (flat_p >= low) & (flat_p < high if bin_id < 9 else flat_p <= high)
            if mask.any():
                calibration_rows.append({"model": model, "fold": fold, "bin": bin_id, "mean_predicted": flat_p[mask].mean(), "observed_rate": flat_y[mask].mean(), "n": int(mask.sum())})
    return pd.DataFrame(daily_rows), pd.DataFrame(rank_rows), pd.DataFrame(calibration_rows)


def main() -> None:
    data = load_data()
    cols = columns()
    daily_all, rank_all, calibration_all, summaries = [], [], [], []
    for fold, (start_text, end_text, history_text) in FOLDS.items():
        start, end, history_end = map(pd.Timestamp, (start_text, end_text, history_text))
        test_mask = data.date.between(start, end)
        history = data.loc[data.date < start].copy()
        dates = data.loc[test_mask, "date"].reset_index(drop=True)
        actual = data.loc[test_mask, cols].to_numpy(dtype=int).reshape(-1, 5, 10)
        for model in MODEL_NAMES:
            if model == "random_forest":
                model_dates, p = fit_random_forest(data, history_end, test_mask)
                y = data.loc[data.date.isin(model_dates), cols].to_numpy(dtype=int).reshape(-1, 5, 10)
                d, r, c = evaluate(model, fold, model_dates, p, y)
            else:
                predictions = []
                for date in dates:
                    current_history = data.loc[data.date < date]
                    predictions.append(statistical_probability(model, current_history))
                d, r, c = evaluate(model, fold, dates, np.asarray(predictions), actual)
            daily_all.append(d); rank_all.append(r); calibration_all.append(c)
            summaries.append({"model": model, "fold": fold, "n_days": len(d), **{column: d[column].mean() for column in d.columns if column not in {"date", "model", "fold"}}})

    summary = pd.DataFrame(summaries)
    daily = pd.concat(daily_all, ignore_index=True)
    ranks = pd.concat(rank_all, ignore_index=True)
    calibration = pd.concat(calibration_all, ignore_index=True)
    rank_summary = ranks.groupby(["model", "fold"]).agg(mean_rank=("rank", "mean"), median_rank=("rank", "median"), top1=("rank", lambda x: np.mean(x <= 1)), top3=("rank", lambda x: np.mean(x <= 3)), top5=("rank", lambda x: np.mean(x <= 5)), top10=("rank", lambda x: np.mean(x <= 10))).reset_index()
    TABLE_DIR.mkdir(parents=True, exist_ok=True); FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLE_DIR / "model_summary.csv", index=False)
    daily.to_csv(TABLE_DIR / "daily_scores.csv.gz", index=False, compression="gzip")
    ranks.to_csv(TABLE_DIR / "actual_digit_ranks.csv.gz", index=False, compression="gzip")
    rank_summary.to_csv(TABLE_DIR / "rank_summary.csv", index=False)
    calibration.to_csv(TABLE_DIR / "calibration.csv", index=False)
    final = summary[summary.fold.eq("validation_2023_2024")].sort_values("brier_score")
    plt.figure(figsize=(11, 5)); plt.bar(final.model, final.brier_score); plt.xticks(rotation=35, ha="right"); plt.ylabel("Daily multi-label Brier score"); plt.title("Phần 2 — so sánh model trên validation 2023–2024"); plt.tight_layout(); plt.savefig(FIGURE_DIR / "brier_validation.png", dpi=180); plt.close()
    print(summary.sort_values(["fold", "brier_score"]).to_string(index=False))


if __name__ == "__main__":
    main()

