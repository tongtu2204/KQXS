"""Export canonical compressed daily probabilities for both evaluation phases."""

import sys
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import (
    FINAL_ARTIFACT_DIR,
    FINAL_TEST,
    PROBABILITY_COLUMNS,
    VALIDATION,
    VALIDATION_ARTIFACT_DIR,
)
from src.probability import validate_prediction_frame


TABLE_DIR = PROJECT_DIR / "artifacts" / "tables"
MODEL_FILES = {
    "catboost_static": "modern_ml_last2_predictions.csv",
    "catboost_retrain": "catboost_retrain_predictions.csv",
    "cdm": "cdm_baseline_predictions.csv",
    "rolling_cdm": "rolling_cdm_predictions.csv",
    "bayesian_markov": "bayesian_markov_predictions.csv",
}


def export_phase(phase: str, output_dir: Path) -> Path:
    frames = []
    reference = None

    for base_model, file_name in MODEL_FILES.items():
        source = pd.read_csv(TABLE_DIR / file_name, parse_dates=["date"])
        source = source.loc[source["fold"].eq(phase)].copy()
        groups = source.groupby("window", sort=True) if base_model == "rolling_cdm" else [(None, source)]

        for window, subset in groups:
            model = f"rolling_cdm_w{int(window)}" if window is not None else base_model
            canonical = subset[["date", "fold", "actual", *PROBABILITY_COLUMNS]].copy()
            canonical.insert(2, "model", model)
            canonical = canonical.sort_values("date").reset_index(drop=True)
            validate_prediction_frame(
                canonical.drop(columns="model"),
                source=f"export:{model}:{phase}",
            )
            frames.append(canonical)
            if reference is None:
                reference = canonical[["date", "fold", "actual"]].copy()

    if reference is None:
        raise RuntimeError(f"No rows loaded for {phase}")

    uniform_probability = pd.DataFrame(
        0.01,
        index=reference.index,
        columns=PROBABILITY_COLUMNS,
    )
    uniform = pd.concat(
        [reference.reset_index(drop=True), uniform_probability], axis=1
    )
    uniform.insert(2, "model", "uniform")
    frames.append(uniform)

    output = pd.concat(frames, ignore_index=True).sort_values(["date", "model"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "model_probabilities.csv.gz"
    output.to_csv(
        output_file,
        index=False,
        compression="gzip",
        encoding="utf-8-sig",
        float_format="%.12g",
    )
    print(
        f"{phase}: rows={len(output):,}, models={output['model'].nunique()}, "
        f"dates={output['date'].nunique()} -> {output_file}"
    )
    return output_file


def main() -> None:
    export_phase(VALIDATION.name, VALIDATION_ARTIFACT_DIR)
    export_phase(FINAL_TEST.name, FINAL_ARTIFACT_DIR)


if __name__ == "__main__":
    main()

