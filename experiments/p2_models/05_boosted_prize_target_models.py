"""P2B boosted models: named prize-slot prediction on validation only.

Each of the 27 prize/index streams is modeled separately.  For every valid
right-aligned position and digit, XGBoost and CatBoost estimate a binary
presence probability from lagged pool-independent features of that stream.
The final 2025-2026 period is deliberately excluded; it is reserved for P3.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_DIR / "data" / "raw" / "kqxsmb_all_prizes_2007_2026.csv"
OUTPUT_DIR = PROJECT_DIR / "artifacts" / "p2_models" / "prize_target_boosted"
POSITIONS = ("ten_thousands", "thousands", "hundreds", "tens", "units")
VALIDATION_START = pd.Timestamp("2023-01-01")
VALIDATION_END = pd.Timestamp("2024-12-31")
HISTORY_END = pd.Timestamp("2022-12-31")
TOP_NUMBERS = (1, 5, 10, 20, 100)
TICKET_COST = 10_000


def load_data() -> pd.DataFrame:
    data = pd.read_csv(RAW_FILE, dtype={"number": str, "prize_index": str}, encoding="utf-8-sig")
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data["number"] = data["number"].str.strip()
    data["length"] = data["number"].str.len()
    data["target"] = data["prize"] + "_" + data["prize_index"]
    for index, position in enumerate(POSITIONS):
        data[position] = data["number"].map(
            lambda value, i=index: int(value[i - (5 - len(value))]) if len(value) >= 5 - i else np.nan
        )
    return data.sort_values(["target", "date"]).reset_index(drop=True)


def features(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for lag in (1, 2, 3, 7, 14, 30):
        part = frame[list(POSITIONS)].shift(lag)
        part.columns = [f"{column}_lag{lag}" for column in POSITIONS]
        parts.append(part)
    date = frame["date"]
    parts.append(pd.DataFrame({
        "dow_sin": np.sin(2 * np.pi * date.dt.dayofweek / 7),
        "dow_cos": np.cos(2 * np.pi * date.dt.dayofweek / 7),
        "month_sin": np.sin(2 * np.pi * date.dt.month / 12),
        "month_cos": np.cos(2 * np.pi * date.dt.month / 12),
    }, index=frame.index))
    # Missing leading positions are structural for 2/3/4-digit prizes, not
    # missing observations.  Keep them as a separate sentinel so short-prize
    # targets remain in the 27-slot evaluation.
    return pd.concat(parts, axis=1).fillna(-1)


def boosted_model(name: str):
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=160, max_depth=3, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
            reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", n_jobs=-1, random_state=42, verbosity=0,
        )
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=160, depth=5, learning_rate=0.03, loss_function="Logloss",
        l2_leaf_reg=5, random_seed=42, thread_count=-1,
        verbose=False, allow_writing_files=False,
    )


def candidate_numbers(probability: np.ndarray, length: int, top_n: int) -> list[str]:
    choices = [[0] if i < 5 - length else list(np.argsort(-probability[i], kind="stable")[:3]) for i in range(5)]
    digits = np.asarray(list(itertools.product(*choices)), dtype=int)
    valid = np.arange(5 - length, 5)
    scores = np.prod(probability[valid, digits[:, valid]], axis=1)
    order = np.argsort(-scores, kind="stable")[:top_n]
    return ["".join(str(int(x)) for x in digits[i, -length:]) for i in order]


def run_model(model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_data()
    prediction_rows = []
    for target, target_data in data.groupby("target", sort=True):
        target_data = target_data.sort_values("date").reset_index(drop=True)
        length = int(target_data["length"].mode().iloc[0])
        x = features(target_data)
        train = (target_data["date"] <= HISTORY_END) & x.notna().all(axis=1)
        test = target_data["date"].between(VALIDATION_START, VALIDATION_END) & x.notna().all(axis=1)
        if not test.any():
            continue
        probabilities = np.full((int(test.sum()), 5, 10), np.nan)
        for position_index, position in enumerate(POSITIONS):
            if position_index < 5 - length:
                continue
            values = target_data[position]
            for digit in range(10):
                y = (values == digit).astype(int)
                if y.loc[train].nunique() < 2:
                    probabilities[:, position_index, digit] = (y.loc[train].sum() + 1) / (train.sum() + 2)
                    continue
                model = boosted_model(model_name)
                model.fit(x.loc[train], y.loc[train])
                raw = model.predict_proba(x.loc[test])
                classes = model.classes_.astype(int)
                probabilities[:, position_index, digit] = next(raw[:, i] for i, cls in enumerate(classes) if cls == 1)
        for row_number, (_, actual) in enumerate(target_data.loc[test].iterrows()):
            top = candidate_numbers(probabilities[row_number], length, max(TOP_NUMBERS))
            actual_suffix = str(actual["number"])[-length:]
            record = {"date": actual["date"], "phase": "validation_2023_2024", "target": target,
                      "prize": actual["prize"], "prize_index": actual["prize_index"],
                      "model": model_name, "actual": str(actual["number"]).zfill(5), "length": length}
            for k in TOP_NUMBERS:
                selected = top[:k]
                record[f"top{k}_hit"] = int(actual_suffix in set(selected))
                record[f"top{k}_candidate_count"] = len(selected)
                record[f"top{k}_cost"] = len(selected) * TICKET_COST
            prediction_rows.append(record)
    predictions = pd.DataFrame(prediction_rows)
    summary_rows = []
    for keys, frame in predictions.groupby(["phase", "prize", "prize_index", "model"], sort=True):
        row = dict(zip(["phase", "prize", "prize_index", "model"], keys))
        row["n_predictions"] = len(frame)
        for k in TOP_NUMBERS:
            row[f"top{k}_hit_rate"] = frame[f"top{k}_hit"].mean()
            row[f"top{k}_cost"] = frame[f"top{k}_cost"].mean()
        summary_rows.append(row)
    return predictions, pd.DataFrame(summary_rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_predictions, all_summaries = [], []
    for model_name in ("xgboost", "catboost"):
        print(f"Running P2B {model_name} ...", flush=True)
        predictions, summary = run_model(model_name)
        all_predictions.append(predictions)
        all_summaries.append(summary)
    predictions = pd.concat(all_predictions, ignore_index=True)
    summary = pd.concat(all_summaries, ignore_index=True)
    predictions.to_csv(OUTPUT_DIR / "boosted_prize_target_predictions.csv.gz", index=False, compression="gzip", encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "boosted_prize_target_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.groupby("model")[["top1_hit_rate", "top5_hit_rate", "top10_hit_rate", "top20_hit_rate", "top100_hit_rate"]].mean().to_string())


if __name__ == "__main__":
    main()
