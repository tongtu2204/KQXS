"""Select strategies on 2023-2024 and evaluate frozen rules on 2025-2026.

Run selection first:
    python experiments/p3_strategies/21_fixed_split_strategy.py --stage select

Only after committing the frozen configuration, run final evaluation:
    python experiments/p3_strategies/21_fixed_split_strategy.py --stage final
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import (
    FINAL_ARTIFACT_DIR,
    FINAL_TEST,
    PROBABILITY_COLUMNS,
    VALIDATION,
    VALIDATION_ARTIFACT_DIR,
)
from src.strategy import (
    MIN_VALIDATION_BETS,
    M_VALUES,
    TARGET_PARTICIPATION_RATES,
    FrozenStrategy,
    apply_selective_threshold,
    evaluate_bets,
    fit_selective_threshold,
    frozen_strategy_from_row,
    frozen_strategy_to_dict,
    model_metrics,
    prepare_ranking,
)


TABLE_DIR = PROJECT_DIR / "artifacts" / "tables"
FROZEN_CONFIG_FILE = VALIDATION_ARTIFACT_DIR / "frozen_strategy.json"

MODEL_FILES = {
    "catboost_static": "modern_ml_last2_predictions.csv",
    "catboost_retrain": "catboost_retrain_predictions.csv",
    "cdm": "cdm_baseline_predictions.csv",
    "rolling_cdm": "rolling_cdm_predictions.csv",
    "bayesian_markov": "bayesian_markov_predictions.csv",
}


def load_phase_predictions(phase: str) -> dict[str, pd.DataFrame]:
    if phase not in {VALIDATION.name, FINAL_TEST.name}:
        raise ValueError(f"Unsupported phase: {phase}")

    models: dict[str, pd.DataFrame] = {}
    actual_reference: pd.DataFrame | None = None

    for base_model, file_name in MODEL_FILES.items():
        path = TABLE_DIR / file_name
        data = pd.read_csv(path, parse_dates=["date"])
        data = data.loc[data["fold"].eq(phase)].copy()

        groups = data.groupby("window", sort=True) if base_model == "rolling_cdm" else [(None, data)]
        for window, subset in groups:
            model_key = (
                f"rolling_cdm_w{int(window)}" if window is not None else base_model
            )
            subset = subset[["date", "actual", *PROBABILITY_COLUMNS]].copy()
            subset["model_key"] = model_key
            subset = subset.sort_values("date").reset_index(drop=True)
            models[model_key] = subset

            reference = subset[["date", "actual"]]
            if actual_reference is None:
                actual_reference = reference
            elif not reference.equals(actual_reference):
                raise ValueError(f"Date/actual mismatch for {model_key} in {phase}")

    if actual_reference is None:
        raise RuntimeError("No model predictions were loaded")

    uniform_probability = pd.DataFrame(
        0.01,
        index=actual_reference.index,
        columns=PROBABILITY_COLUMNS,
    )
    uniform = pd.concat(
        [actual_reference.reset_index(drop=True), uniform_probability],
        axis=1,
    )
    uniform["model_key"] = "uniform"
    models["uniform"] = uniform

    return models


def select_on_validation() -> None:
    models = load_phase_predictions(VALIDATION.name)
    VALIDATION_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    probability_rows = [model_metrics(frame) for frame in models.values()]
    probability_metrics = pd.DataFrame(probability_rows).sort_values("log_loss")
    probability_metrics.to_csv(
        VALIDATION_ARTIFACT_DIR / "probability_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    candidate_rows: list[dict[str, object]] = []
    prepared_models = {key: prepare_ranking(frame) for key, frame in models.items()}

    for model_key, prepared in prepared_models.items():
        frame = prepared["frame"]
        for m in M_VALUES:
            always_metrics, _ = evaluate_bets(
                prepared,
                m=m,
                bet=np.ones(len(frame), dtype=bool),
            )
            candidate_rows.append(
                {
                    "strategy": "always_top_m",
                    "model": model_key,
                    "target_participation_rate": np.nan,
                    "confidence_threshold": np.nan,
                    "tie_acceptance_rate": np.nan,
                    **always_metrics,
                }
            )

            confidence = np.asarray(prepared["cumulative_probability"])[:, m - 1]
            for target_rate in TARGET_PARTICIPATION_RATES:
                threshold, tie_rate = fit_selective_threshold(confidence, target_rate)
                label = f"validation|{model_key}|{m}|{target_rate}"
                bet = apply_selective_threshold(
                    frame["date"], confidence, threshold, tie_rate, label
                )
                selective_metrics, _ = evaluate_bets(prepared, m=m, bet=bet)
                candidate_rows.append(
                    {
                        "strategy": "selective_top_m",
                        "model": model_key,
                        "target_participation_rate": target_rate,
                        "confidence_threshold": threshold,
                        "tie_acceptance_rate": tie_rate,
                        **selective_metrics,
                    }
                )

    candidates = pd.DataFrame(candidate_rows)
    candidates.to_csv(
        VALIDATION_ARTIFACT_DIR / "strategy_candidates.csv.gz",
        index=False,
        compression="gzip",
        encoding="utf-8-sig",
    )

    eligible = candidates.loc[candidates["n_bets"].ge(MIN_VALIDATION_BETS)].copy()
    if eligible.empty:
        raise RuntimeError("No eligible validation strategy")

    selected_rows = []
    for strategy_name, subset in eligible.groupby("strategy", sort=True):
        selected = subset.sort_values(
            ["selection_score", "total_profit", "m", "model"],
            ascending=[False, False, True, True],
        ).iloc[0]
        selected_rows.append(selected)

    selected = pd.DataFrame(selected_rows).sort_values("strategy").reset_index(drop=True)
    selected.to_csv(
        VALIDATION_ARTIFACT_DIR / "selected_strategies.csv",
        index=False,
        encoding="utf-8-sig",
    )

    frozen = [frozen_strategy_from_row(row) for _, row in selected.iterrows()]
    primary = max(frozen, key=lambda item: item.validation_selection_score)
    config = {
        "selection_period": [
            VALIDATION.start.date().isoformat(),
            VALIDATION.end.date().isoformat(),
        ],
        "final_test_period": [
            FINAL_TEST.start.date().isoformat(),
            FINAL_TEST.end.date().isoformat(),
        ],
        "selection_rule": (
            "max Wilson 95% lower bound minus break-even hit rate; "
            "tie-break total profit, smaller m, model name"
        ),
        "minimum_validation_bets": MIN_VALIDATION_BETS,
        "primary_strategy": primary.strategy,
        "strategies": [frozen_strategy_to_dict(item) for item in frozen],
    }
    FROZEN_CONFIG_FILE.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    daily_frames = []
    for item in frozen:
        prepared = prepared_models[item.model]
        frame = prepared["frame"]
        confidence = np.asarray(prepared["cumulative_probability"])[:, item.m - 1]
        if item.strategy == "always_top_m":
            bet = np.ones(len(frame), dtype=bool)
        else:
            bet = apply_selective_threshold(
                frame["date"],
                confidence,
                float(item.confidence_threshold),
                float(item.tie_acceptance_rate),
                f"validation|{item.model}|{item.m}|{item.target_participation_rate}",
            )
        _, daily = evaluate_bets(prepared, m=item.m, bet=bet)
        daily.insert(0, "model", item.model)
        daily.insert(0, "strategy", item.strategy)
        daily_frames.append(daily)

    pd.concat(daily_frames, ignore_index=True).to_csv(
        VALIDATION_ARTIFACT_DIR / "selected_strategy_daily.csv.gz",
        index=False,
        compression="gzip",
        encoding="utf-8-sig",
    )

    print("VALIDATION PROBABILITY METRICS")
    print(probability_metrics.to_string(index=False))
    print("\nFROZEN STRATEGIES")
    print(selected.to_string(index=False))
    print(f"\nSaved frozen config: {FROZEN_CONFIG_FILE}")


def load_frozen_strategies() -> tuple[dict[str, object], list[FrozenStrategy]]:
    if not FROZEN_CONFIG_FILE.exists():
        raise FileNotFoundError(
            "Frozen validation strategy is missing; run --stage select and commit it first"
        )
    config = json.loads(FROZEN_CONFIG_FILE.read_text(encoding="utf-8"))
    strategies = [FrozenStrategy(**item) for item in config["strategies"]]
    return config, strategies


def evaluate_final() -> None:
    config, strategies = load_frozen_strategies()
    models = load_phase_predictions(FINAL_TEST.name)
    FINAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    probability_metrics = pd.DataFrame(
        [model_metrics(frame) for frame in models.values()]
    ).sort_values("log_loss")
    probability_metrics.to_csv(
        FINAL_ARTIFACT_DIR / "probability_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    result_rows = []
    daily_frames = []
    for item in strategies:
        prepared = prepare_ranking(models[item.model])
        frame = prepared["frame"]
        confidence = np.asarray(prepared["cumulative_probability"])[:, item.m - 1]
        if item.strategy == "always_top_m":
            bet = np.ones(len(frame), dtype=bool)
        else:
            bet = apply_selective_threshold(
                frame["date"],
                confidence,
                float(item.confidence_threshold),
                float(item.tie_acceptance_rate),
                f"final_test|{item.model}|{item.m}|{item.target_participation_rate}",
            )
        metrics, daily = evaluate_bets(prepared, m=item.m, bet=bet)
        result_rows.append(
            {
                "strategy": item.strategy,
                "model": item.model,
                "target_participation_rate": item.target_participation_rate,
                "frozen_confidence_threshold": item.confidence_threshold,
                "validation_selection_score": item.validation_selection_score,
                "validation_roi": item.validation_roi,
                **metrics,
            }
        )
        daily.insert(0, "model", item.model)
        daily.insert(0, "strategy", item.strategy)
        daily_frames.append(daily)

    results = pd.DataFrame(result_rows)
    results.to_csv(
        FINAL_ARTIFACT_DIR / "strategy_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(daily_frames, ignore_index=True).to_csv(
        FINAL_ARTIFACT_DIR / "strategy_daily.csv.gz",
        index=False,
        compression="gzip",
        encoding="utf-8-sig",
    )
    (FINAL_ARTIFACT_DIR / "frozen_config_used.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("FINAL PROBABILITY METRICS")
    print(probability_metrics.to_string(index=False))
    print("\nFINAL FROZEN STRATEGY RESULTS")
    print(results.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("select", "final"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "select":
        select_on_validation()
    else:
        evaluate_final()


if __name__ == "__main__":
    main()
