"""P2B: predict the named result of each prize/draw slot.

This is separate from P2A's daily pool target.  A target is identified by
``prize + prize_index``.  Probabilities are factorized by the five
right-aligned digit positions; Top-k full numbers are then generated from the
Top-3 digit choices.  The script reports exact/suffix hit rates by prize.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

RAW_FILE = PROJECT_DIR / "data" / "raw" / "kqxsmb_all_prizes_2007_2026.csv"
OUTPUT_DIR = PROJECT_DIR / "artifacts" / "p2_models" / "prize_target"
POSITIONS = ("ten_thousands", "thousands", "hundreds", "tens", "units")
WINDOWS = (30, 90, 365)
TOP_NUMBERS = (1, 5, 10, 20, 100)
TICKET_COST = 10_000
FOLDS = {
    "validation_2023_2024": (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")),
}


def load_data() -> pd.DataFrame:
    data = pd.read_csv(RAW_FILE, dtype={"number": str, "prize_index": str}, encoding="utf-8-sig")
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data["number"] = data["number"].str.strip()
    data["number_5"] = data["number"].str.zfill(5)
    data["length"] = data["number"].str.len()
    data["target"] = data["prize"] + "_" + data["prize_index"]
    for index, position in enumerate(POSITIONS):
        data[position] = data["number"].map(lambda value, i=index: int(value[-5 + i]) if len(value) >= 5 - i else np.nan)
    return data.sort_values(["date", "prize", "prize_index"]).reset_index(drop=True)


def phase_mask(date: pd.Timestamp, phase: str) -> bool:
    start, end = FOLDS[phase]
    return start <= date <= end


def probabilities(history: pd.DataFrame, model: str, length: int) -> np.ndarray:
    result = np.full((5, 10), np.nan)
    source = history.tail(int(model.rsplit("w", 1)[1])) if model.startswith("rolling_w") else history
    for position_index, position in enumerate(POSITIONS):
        if position_index < 5 - length:
            continue
        values = source[position].dropna().astype(int)
        counts = np.bincount(values, minlength=10) if len(values) else np.zeros(10)
        result[position_index] = (counts + 1) / (counts.sum() + 10)
    return result


def candidates(probability: np.ndarray, length: int, top_n: int) -> list[str]:
    choices = []
    for position_index in range(5):
        if position_index < 5 - length:
            choices.append([0])
        else:
            choices.append(list(np.argsort(-probability[position_index], kind="stable")[:3]))
    digits = np.asarray(list(itertools.product(*choices)), dtype=int)
    valid_positions = np.arange(5 - length, 5)
    scores = np.prod(probability[valid_positions, digits[:, valid_positions]], axis=1)
    order = np.argsort(-scores, kind="stable")[:top_n]
    return ["".join(str(int(x)) for x in digits[index, -length:]) for index in order]


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_data()
    models = ["expanding", *(f"rolling_w{window}" for window in WINDOWS)]
    prediction_rows, summary_rows = [], []
    for phase in FOLDS:
        for target, target_data in data.groupby("target", sort=True):
            target_data = target_data.sort_values("date").reset_index(drop=True)
            length = int(target_data["length"].mode().iloc[0])
            for model in models:
                for row_index, actual in target_data.iterrows():
                    date = actual["date"]
                    if not phase_mask(date, phase):
                        continue
                    history = target_data.iloc[:row_index]
                    p = probabilities(history, model, length)
                    top = candidates(p, length, max(TOP_NUMBERS))
                    actual_suffix = str(actual["number"])[-length:]
                    record = {"date": date, "phase": phase, "target": target, "prize": actual["prize"], "prize_index": actual["prize_index"], "model": model, "actual": str(actual["number_5"]), "length": length}
                    for k in TOP_NUMBERS:
                        selected = top[:k]
                        record[f"top{k}_hit"] = int(actual_suffix in {x[-length:] for x in selected})
                        record[f"top{k}_candidate_count"] = len(selected)
                        record[f"top{k}_cost"] = len(selected) * TICKET_COST
                    for position_index, position in enumerate(POSITIONS):
                        for digit in range(10):
                            record[f"p_{position}_{digit}"] = float(p[position_index, digit]) if np.isfinite(p[position_index, digit]) else np.nan
                    prediction_rows.append(record)
    predictions = pd.DataFrame(prediction_rows)
    for keys, frame in predictions.groupby(["phase", "prize", "prize_index", "model"], sort=True):
        row = dict(zip(["phase", "prize", "prize_index", "model"], keys))
        row["n_predictions"] = len(frame)
        for k in TOP_NUMBERS:
            row[f"top{k}_hit_rate"] = frame[f"top{k}_hit"].mean()
        summary_rows.append(row)
    return predictions, pd.DataFrame(summary_rows)


def main() -> None:
    predictions, summary = run()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_DIR / "prize_target_predictions.csv.gz", index=False, compression="gzip", encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "prize_target_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"\nĐã ghi kết quả P2B vào: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
