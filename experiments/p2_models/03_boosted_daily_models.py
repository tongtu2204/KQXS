"""XGBoost và CatBoost cho target chữ số xuất hiện theo ngày.

Mỗi model được huấn luyện cho 50 target nhị phân (5 vị trí x 10 chữ số).
Fold test là frozen theo thời gian: train chỉ dùng dữ liệu trước ngày test.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

POSITIONS = ("ten_thousands", "thousands", "hundreds", "tens", "units")
DATA_FILE = PROJECT_DIR / "data" / "processed" / "daily_digit_targets.csv"
OUTPUT_DIR = PROJECT_DIR / "artifacts" / "models" / "daily_digit" / "boosted"
FOLDS = {
    "validation_2023_2024": ("2023-01-01", "2024-12-31", "2022-12-31"),
}


def target_columns():
    return [f"{p}_d{d}" for p in POSITIONS for d in range(10)]


def build_features(data):
    cols = target_columns()
    parts = []
    for lag in (1, 2, 3, 7, 14, 30):
        part = data[cols].shift(lag)
        part.columns = [f"{c}_lag{lag}" for c in cols]
        parts.append(part)
    date = data.date
    parts.append(pd.DataFrame({
        "dow_sin": np.sin(2 * np.pi * date.dt.dayofweek / 7),
        "dow_cos": np.cos(2 * np.pi * date.dt.dayofweek / 7),
        "month_sin": np.sin(2 * np.pi * date.dt.month / 12),
        "month_cos": np.cos(2 * np.pi * date.dt.month / 12),
    }, index=data.index))
    return pd.concat(parts, axis=1)


def make_model(model_name):
    if model_name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
            reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", n_jobs=-1, random_state=42,
        )
    if model_name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=250, depth=5, learning_rate=0.03, loss_function="Logloss",
            eval_metric="Logloss", l2_leaf_reg=5, random_seed=42,
            thread_count=-1, verbose=False, allow_writing_files=False,
        )
    raise ValueError(model_name)


def aligned_probability(model, x):
    p = model.predict_proba(x)
    classes = model.classes_.astype(int)
    output = np.zeros(len(x))
    for i, cls in enumerate(classes):
        if cls == 1:
            output = p[:, i]
    return np.clip(output, 1e-6, 1 - 1e-6)


def evaluate(model_name, fold, dates, probabilities, actual):
    rows = []
    for i, date in enumerate(dates):
        p, y = probabilities[i], actual[i]
        row = {
            "date": date, "model": model_name, "fold": fold,
            "brier_score": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        }
        for k in (1, 3, 5, 10):
            top = np.argsort(-p, axis=1)[:, :k]
            row[f"hit_any_top{k}"] = np.mean([bool(y[pos, top[pos]].any()) for pos in range(5)])
            row[f"recall_top{k}"] = sum(y[pos, top[pos]].sum() for pos in range(5)) / max(y.sum(), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    data = pd.read_csv(DATA_FILE, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    cols = target_columns()
    features = build_features(data)
    summary, probabilities, daily = [], [], []
    for fold, (start_text, end_text, history_text) in FOLDS.items():
        start, end, history_end = map(pd.Timestamp, (start_text, end_text, history_text))
        train = (data.date <= history_end) & features.notna().all(axis=1)
        test = data.date.between(start, end) & features.notna().all(axis=1)
        x_train, x_test = features.loc[train], features.loc[test]
        y_train, y_test = data.loc[train, cols], data.loc[test, cols]
        dates = data.loc[test, "date"].reset_index(drop=True)
        for model_name in ("xgboost", "catboost"):
            output = []
            for column in cols:
                # Một số chữ số xuất hiện trong gần như mọi ngày ở các vị trí
                # thấp. Khi train chỉ có một class, tree booster không thể fit;
                # dùng posterior Beta-smoothed làm dự báo hợp lệ.
                unique = y_train[column].unique()
                if len(unique) < 2:
                    constant = (y_train[column].sum() + 1) / (len(y_train) + 2)
                    output.append(np.full(len(x_test), constant))
                    continue
                model = make_model(model_name)
                model.fit(x_train, y_train[column])
                output.append(aligned_probability(model, x_test))
            p = np.column_stack(output).reshape(-1, 5, 10)
            y = y_test.to_numpy(dtype=int).reshape(-1, 5, 10)
            d = evaluate(model_name, fold, dates, p, y)
            daily.append(d)
            summary.append({"model": model_name, "fold": fold, "n_days": len(d), **{c: d[c].mean() for c in d.columns if c not in {"date", "model", "fold"}}})
            probabilities.append(pd.DataFrame(p.reshape(len(p), 50), columns=[f"p_{c}" for c in cols]).assign(date=dates, model=model_name, fold=fold))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "boosted_summary.csv", index=False)
    pd.concat(daily, ignore_index=True).to_csv(OUTPUT_DIR / "boosted_daily_scores.csv.gz", index=False, compression="gzip")
    pd.concat(probabilities, ignore_index=True).to_csv(OUTPUT_DIR / "boosted_probabilities.csv.gz", index=False, compression="gzip")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()

