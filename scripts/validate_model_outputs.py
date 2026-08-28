"""Validate all model outputs before probability ranking or strategy work."""

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import REPORT_ARTIFACT_DIR
from src.probability import validate_prediction_frame


TABLE_DIR = PROJECT_DIR / "artifacts" / "tables"
OUTPUT_FILE = REPORT_ARTIFACT_DIR / "model_output_validation.json"

FILES = {
    "catboost_static": ("modern_ml_last2_predictions.csv", ()),
    "catboost_retrain": ("catboost_retrain_predictions.csv", ()),
    "cdm": ("cdm_baseline_predictions.csv", ()),
    "rolling_cdm": ("rolling_cdm_predictions.csv", ("window",)),
    "bayesian_markov": ("bayesian_markov_predictions.csv", ()),
}


def main() -> None:
    summaries = []

    for source, (file_name, key_columns) in FILES.items():
        path = TABLE_DIR / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing model output: {path}")
        frame = pd.read_csv(path, parse_dates=["date"])
        summaries.append(
            validate_prediction_frame(
                frame,
                source=source,
                key_columns=key_columns,
            )
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

