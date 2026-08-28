"""Final integrity checks for frozen-strategy evaluation artifacts."""

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import (
    FIGURE_ARTIFACT_DIR,
    FINAL_ARTIFACT_DIR,
    FINAL_TEST,
    REPORT_ARTIFACT_DIR,
    VALIDATION_ARTIFACT_DIR,
)


def main() -> None:
    frozen = json.loads(
        (VALIDATION_ARTIFACT_DIR / "frozen_strategy.json").read_text(encoding="utf-8")
    )
    used = json.loads(
        (FINAL_ARTIFACT_DIR / "frozen_config_used.json").read_text(encoding="utf-8")
    )
    if frozen != used:
        raise ValueError("Final test did not use the committed frozen configuration")

    results = pd.read_csv(FINAL_ARTIFACT_DIR / "strategy_results.csv")
    daily = pd.read_csv(FINAL_ARTIFACT_DIR / "strategy_daily.csv.gz", parse_dates=["date"])
    probability = pd.read_csv(
        FINAL_ARTIFACT_DIR / "model_probabilities.csv.gz",
        parse_dates=["date"],
    )

    expected_strategies = {item["strategy"] for item in frozen["strategies"]}
    if set(results["strategy"]) != expected_strategies:
        raise ValueError("Final results do not contain every frozen strategy")
    if set(daily["strategy"]) != expected_strategies:
        raise ValueError("Final daily output does not contain every frozen strategy")
    if not daily["date"].between(FINAL_TEST.start, FINAL_TEST.end).all():
        raise ValueError("Final daily file contains a date outside 2025-2026")
    if not probability["date"].between(FINAL_TEST.start, FINAL_TEST.end).all():
        raise ValueError("Final probability file contains a date outside 2025-2026")
    if probability["model"].nunique() != 10:
        raise ValueError("Expected ten canonical probability models")
    if probability["date"].nunique() != 595:
        raise ValueError("Expected 595 final-test prediction dates")

    required_files = [
        REPORT_ARTIFACT_DIR / "final_research_summary.md",
        FIGURE_ARTIFACT_DIR / "model_log_loss_validation_vs_final.png",
        FIGURE_ARTIFACT_DIR / "cumulative_profit_validation_vs_final.png",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing final deliverables: {missing}")

    summary = {
        "frozen_config_identical": True,
        "final_start": daily["date"].min().date().isoformat(),
        "final_end": daily["date"].max().date().isoformat(),
        "final_unique_dates": int(daily["date"].nunique()),
        "canonical_model_count": int(probability["model"].nunique()),
        "probability_rows": len(probability),
        "strategy_count": len(results),
        "report_and_figures_present": True,
    }
    output = REPORT_ARTIFACT_DIR / "final_result_validation.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

