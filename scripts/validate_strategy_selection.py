"""Validate that frozen strategies were selected only on 2023-2024."""

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import REPORT_ARTIFACT_DIR, VALIDATION, VALIDATION_ARTIFACT_DIR
from src.strategy import M_VALUES, TARGET_PARTICIPATION_RATES


def main() -> None:
    frozen_path = VALIDATION_ARTIFACT_DIR / "frozen_strategy.json"
    candidate_path = VALIDATION_ARTIFACT_DIR / "strategy_candidates.csv.gz"
    selected_path = VALIDATION_ARTIFACT_DIR / "selected_strategies.csv"
    daily_path = VALIDATION_ARTIFACT_DIR / "selected_strategy_daily.csv.gz"

    config = json.loads(frozen_path.read_text(encoding="utf-8"))
    candidates = pd.read_csv(candidate_path)
    selected = pd.read_csv(selected_path)
    daily = pd.read_csv(daily_path, parse_dates=["date"])

    if config["selection_period"] != ["2023-01-01", "2024-12-31"]:
        raise ValueError("Frozen config has the wrong selection period")
    if not daily["date"].between(VALIDATION.start, VALIDATION.end).all():
        raise ValueError("Selected-strategy daily output contains non-validation dates")
    if set(selected["strategy"]) != {"always_top_m", "selective_top_m"}:
        raise ValueError("Expected exactly the two canonical strategy types")
    if len(config["strategies"]) != 2:
        raise ValueError("Frozen config must contain exactly two strategies")
    if config["primary_strategy"] not in set(selected["strategy"]):
        raise ValueError("Primary strategy is not one of the frozen strategies")

    model_count = candidates["model"].nunique()
    expected_candidates = model_count * len(M_VALUES) * (
        1 + len(TARGET_PARTICIPATION_RATES)
    )
    if len(candidates) != expected_candidates:
        raise ValueError(
            f"Expected {expected_candidates} candidates, found {len(candidates)}"
        )

    for _, row in selected.iterrows():
        match = candidates[
            candidates["strategy"].eq(row["strategy"])
            & candidates["model"].eq(row["model"])
            & candidates["m"].eq(row["m"])
        ]
        if pd.notna(row["target_participation_rate"]):
            match = match[
                match["target_participation_rate"].sub(
                    row["target_participation_rate"]
                ).abs().lt(1e-12)
            ]
        if len(match) != 1:
            raise ValueError("Frozen strategy does not map to one validation candidate")

    summary = {
        "selection_period": config["selection_period"],
        "model_count": model_count,
        "candidate_count": len(candidates),
        "selected_strategy_count": len(selected),
        "daily_min_date": daily["date"].min().date().isoformat(),
        "daily_max_date": daily["date"].max().date().isoformat(),
        "primary_strategy": config["primary_strategy"],
        "final_test_metrics_read": False,
    }
    output = REPORT_ARTIFACT_DIR / "strategy_selection_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

