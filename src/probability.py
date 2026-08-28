"""Validation helpers for 100-class daily probability outputs."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.config import FINAL_TEST, NUMBER_OF_CLASSES, PROBABILITY_COLUMNS, VALIDATION


def validate_prediction_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    key_columns: Sequence[str] = (),
) -> dict[str, object]:
    """Validate probability shape, phase dates, uniqueness, and no-lookahead."""

    required = {"date", "fold", "actual", *PROBABILITY_COLUMNS, *key_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source}: missing columns {sorted(missing)}")

    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"])

    probability_columns = [
        column for column in data.columns if column.startswith("p_")
    ]
    if tuple(probability_columns) != PROBABILITY_COLUMNS:
        raise ValueError(
            f"{source}: expected {NUMBER_OF_CLASSES} ordered probability columns"
        )

    probabilities = data[probability_columns].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{source}: non-finite probabilities")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError(f"{source}: probabilities outside [0, 1]")

    row_sums = probabilities.sum(axis=1)
    maximum_sum_error = float(np.max(np.abs(row_sums - 1.0)))
    if maximum_sum_error > 1e-10:
        raise ValueError(
            f"{source}: probability rows do not sum to one; "
            f"max error={maximum_sum_error}"
        )

    if not data["actual"].between(0, NUMBER_OF_CLASSES - 1).all():
        raise ValueError(f"{source}: actual class outside 0..99")

    allowed_folds = {VALIDATION.name, FINAL_TEST.name}
    unexpected_folds = set(data["fold"].unique()).difference(allowed_folds)
    if unexpected_folds:
        raise ValueError(f"{source}: unexpected folds {sorted(unexpected_folds)}")

    validation_rows = data["fold"].eq(VALIDATION.name)
    final_rows = data["fold"].eq(FINAL_TEST.name)
    if not data.loc[validation_rows, "date"].between(
        VALIDATION.start, VALIDATION.end
    ).all():
        raise ValueError(f"{source}: validation date outside 2023-2024")
    if not data.loc[final_rows, "date"].between(
        FINAL_TEST.start, FINAL_TEST.end
    ).all():
        raise ValueError(f"{source}: final-test date outside 2025-2026")

    duplicate_key = ["fold", "date", *key_columns]
    if data.duplicated(duplicate_key).any():
        raise ValueError(f"{source}: duplicate prediction key {duplicate_key}")

    if "history_end" in data.columns:
        history_end = pd.to_datetime(data["history_end"])
        if not history_end.lt(data["date"]).all():
            raise ValueError(f"{source}: history_end must be before prediction date")

    counts = data.groupby("fold").size().astype(int).to_dict()
    unique_dates = data.groupby("fold")["date"].nunique().astype(int).to_dict()

    return {
        "source": source,
        "rows": len(data),
        "probability_columns": len(probability_columns),
        "maximum_probability_sum_error": maximum_sum_error,
        "rows_by_fold": counts,
        "unique_dates_by_fold": unique_dates,
        "history_end_checked": "history_end" in data.columns,
    }
