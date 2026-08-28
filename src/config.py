"""Canonical experiment protocol for the KQXS project.

The final-test interval is intentionally fixed and must never be used for
model, hyperparameter, or betting-strategy selection.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "kqxsmb_digits.csv"

ARTIFACT_DIR = PROJECT_DIR / "artifacts" / "refactor"
VALIDATION_ARTIFACT_DIR = ARTIFACT_DIR / "validation_2023_2024"
FINAL_ARTIFACT_DIR = ARTIFACT_DIR / "final_test_2025_2026"
REPORT_ARTIFACT_DIR = ARTIFACT_DIR / "reports"
FIGURE_ARTIFACT_DIR = ARTIFACT_DIR / "figures"

NUMBER_OF_CLASSES = 100
RANDOM_STATE = 42
PROBABILITY_COLUMNS = tuple(
    f"p_{number:02d}" for number in range(NUMBER_OF_CLASSES)
)


@dataclass(frozen=True)
class Period:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp

    def mask(self, dates: pd.Series) -> pd.Series:
        return dates.between(self.start, self.end, inclusive="both")


TRAIN_DEVELOPMENT = Period(
    name="train_development",
    start=pd.Timestamp("2002-01-01"),
    end=pd.Timestamp("2022-12-31"),
)

VALIDATION = Period(
    name="validation",
    start=pd.Timestamp("2023-01-01"),
    end=pd.Timestamp("2024-12-31"),
)

FINAL_TEST = Period(
    name="final_test",
    start=pd.Timestamp("2025-01-01"),
    end=pd.Timestamp("2026-12-31"),
)

EVALUATION_PERIODS = (VALIDATION, FINAL_TEST)

# Stateful models initialize from every draw before test_start and then update
# only after emitting the current day's prediction.
STATEFUL_EVALUATION_FOLDS = (
    {
        "name": VALIDATION.name,
        "test_start": VALIDATION.start.year,
        "test_end": VALIDATION.end.year,
    },
    {
        "name": FINAL_TEST.name,
        "test_start": FINAL_TEST.start.year,
        "test_end": FINAL_TEST.end.year,
    },
)

# The static CatBoost baseline needs an inner early-stopping interval. For the
# final test, 2023-2024 is the only model-selection interval; 2025-2026 is
# never used in fit or early stopping.
STATIC_CATBOOST_FOLDS = (
    {
        "name": VALIDATION.name,
        "train_end": 2020,
        "validation_start": 2021,
        "validation_end": 2022,
        "test_start": 2023,
        "test_end": 2024,
    },
    {
        "name": FINAL_TEST.name,
        "train_end": 2022,
        "validation_start": 2023,
        "validation_end": 2024,
        "test_start": 2025,
        "test_end": 2026,
    },
)


def validate_protocol() -> None:
    """Fail fast if experiment periods overlap or are not chronological."""

    periods = (TRAIN_DEVELOPMENT, VALIDATION, FINAL_TEST)

    for previous, current in zip(periods, periods[1:]):
        if previous.end >= current.start:
            raise ValueError(
                f"Overlapping periods: {previous.name} and {current.name}"
            )

        expected_start = previous.end + pd.Timedelta(days=1)
        if current.start != expected_start:
            raise ValueError(
                f"Gap between {previous.name} and {current.name}: "
                f"expected {expected_start.date()}, got {current.start.date()}"
            )


def phase_for_date(value: pd.Timestamp) -> str:
    date = pd.Timestamp(value)

    for period in (TRAIN_DEVELOPMENT, VALIDATION, FINAL_TEST):
        if period.start <= date <= period.end:
            return period.name

    return "outside_protocol"


validate_protocol()
