"""Data loading, preparation, and protocol validation."""

from pathlib import Path

import pandas as pd

from src.config import (
    FINAL_TEST,
    PROCESSED_DATA_FILE,
    RAW_DATA_DIR,
    TRAIN_DEVELOPMENT,
    VALIDATION,
)


MANUAL_CORRECTIONS = {
    "2002-12-19": "57530",
}


def find_latest_raw_file(raw_dir: Path = RAW_DATA_DIR) -> Path:
    files = sorted(raw_dir.glob("kqxsmb_*.csv"))
    if not files:
        raise FileNotFoundError(f"No kqxsmb_*.csv file found in {raw_dir}")
    return files[-1]


def prepare_raw_data(file_path: Path | None = None) -> pd.DataFrame:
    """Read raw jackpot results and return one clean row per draw date."""

    path = file_path or find_latest_raw_file()
    data = pd.read_csv(
        path,
        dtype={"full_result": str, "last_2_digits": str},
        parse_dates=["date"],
    )

    required = {"date", "full_result"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required raw columns: {sorted(missing)}")

    data = data.dropna(subset=["date", "full_result"]).copy()
    data["full_result"] = data["full_result"].astype("string").str.strip()

    for correction_date, corrected_result in MANUAL_CORRECTIONS.items():
        mask = data["date"].eq(pd.Timestamp(correction_date))
        if not mask.any():
            raise ValueError(f"Manual-correction date is missing: {correction_date}")
        data.loc[mask, "full_result"] = corrected_result

    numeric = data["full_result"].str.fullmatch(r"\d+").fillna(False)
    if not numeric.all():
        invalid = data.loc[~numeric, ["date", "full_result"]]
        raise ValueError(f"Non-numeric results found:\n{invalid}")

    data["full_result"] = data["full_result"].str.zfill(5)
    valid_length = data["full_result"].str.fullmatch(r"\d{5}").fillna(False)
    if not valid_length.all():
        invalid = data.loc[~valid_length, ["date", "full_result"]]
        raise ValueError(f"Invalid five-digit results found:\n{invalid}")

    data = (
        data.sort_values("date")
        .drop_duplicates(subset="date", keep="last")
        .reset_index(drop=True)
    )

    for position in range(5):
        data[f"digit_{position + 1}"] = (
            data["full_result"].str[position].astype("int8")
        )

    data["last_2_digits"] = data["full_result"].str[-2:]
    data["last_2_target"] = data["last_2_digits"].astype("int16")
    data["year"] = data["date"].dt.year.astype("int16")
    data["month"] = data["date"].dt.month.astype("int8")
    data["day"] = data["date"].dt.day.astype("int8")
    data["day_of_week"] = data["date"].dt.dayofweek.astype("int8")

    return data[
        [
            "date",
            "year",
            "month",
            "day",
            "day_of_week",
            "full_result",
            "digit_1",
            "digit_2",
            "digit_3",
            "digit_4",
            "digit_5",
            "last_2_digits",
            "last_2_target",
        ]
    ]


def load_data(
    processed_file: Path = PROCESSED_DATA_FILE,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Load processed data, rebuilding it from tracked raw data when needed."""

    if rebuild or not processed_file.exists():
        data = prepare_raw_data()
        processed_file.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(processed_file, index=False, encoding="utf-8-sig")
    else:
        data = pd.read_csv(
            processed_file,
            dtype={"full_result": str, "last_2_digits": str},
            parse_dates=["date"],
        )
        if "last_2_target" not in data:
            data["last_2_target"] = data["last_2_digits"].astype(int)

    data["full_result"] = data["full_result"].str.zfill(5)
    data["last_2_digits"] = data["last_2_digits"].str.zfill(2)
    return data.sort_values("date").reset_index(drop=True)


def validate_data(data: pd.DataFrame) -> dict[str, object]:
    """Validate data integrity and coverage required by the fixed protocol."""

    required = {
        "date",
        "full_result",
        "last_2_digits",
        "last_2_target",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing processed columns: {sorted(missing)}")

    if data.empty:
        raise ValueError("Dataset is empty")
    if data["date"].duplicated().any():
        raise ValueError("Duplicate draw dates found")
    if not data["date"].is_monotonic_increasing:
        raise ValueError("Draw dates are not sorted")
    if not data["full_result"].str.fullmatch(r"\d{5}").all():
        raise ValueError("full_result must contain exactly five digits")
    if not data["last_2_digits"].str.fullmatch(r"\d{2}").all():
        raise ValueError("last_2_digits must contain exactly two digits")
    if not data["last_2_target"].between(0, 99).all():
        raise ValueError("last_2_target is outside 0..99")
    if (data["full_result"].str[-2:] != data["last_2_digits"]).any():
        raise ValueError("last_2_digits does not match full_result")

    if data["date"].min().year > TRAIN_DEVELOPMENT.start.year:
        raise ValueError("Dataset does not reach the first train/development year")
    if data["date"].max() < FINAL_TEST.start:
        raise ValueError("Dataset does not reach the final-test period")

    counts = {
        TRAIN_DEVELOPMENT.name: int(TRAIN_DEVELOPMENT.mask(data["date"]).sum()),
        VALIDATION.name: int(VALIDATION.mask(data["date"]).sum()),
        FINAL_TEST.name: int(FINAL_TEST.mask(data["date"]).sum()),
    }

    if any(count == 0 for count in counts.values()):
        raise ValueError(f"One or more protocol periods are empty: {counts}")

    return {
        "rows": len(data),
        "start_date": data["date"].min().date().isoformat(),
        "end_date": data["date"].max().date().isoformat(),
        "period_counts": counts,
    }
