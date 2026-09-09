"""Prepare the shared full-prize data for P2A and P2B.

P2A target: digit presence/count in the daily pool of 27 results.
P2B target: exact prize/index streams are retained in a separate long file.

Short prize numbers are right-aligned. Missing leading positions are masked;
they are never converted into artificial zero digits.
"""

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_DIR / "data" / "raw" / "kqxsmb_all_prizes_2007_2026.csv"
OUTPUT_FILE = PROJECT_DIR / "data" / "processed" / "daily_digit_targets.csv"
PRIZE_OUTPUT_FILE = PROJECT_DIR / "data" / "processed" / "prize_targets.csv"

POSITION_NAMES = ("ten_thousands", "thousands", "hundreds", "tens", "units")


def prepare_daily_targets(data: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "number", "prize", "prize_index"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {sorted(missing)}")

    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["number"] = data["number"].astype("string").str.strip()
    valid = data["date"].notna() & data["number"].str.fullmatch(r"\d{2,5}").fillna(False)
    data = data.loc[valid].copy()
    data["number_raw"] = data["number"]
    data["number_5"] = data["number"].str.zfill(5)
    data = data.sort_values(["date", "prize", "prize_index"]).reset_index(drop=True)

    if data.empty:
        raise ValueError("Không còn dữ liệu hợp lệ")

    rows = []
    for date, group in data.groupby("date", sort=True):
        if len(group) != 27:
            raise ValueError(f"Ngày {date.date()} không có đúng 27 dòng kết quả")

        row = {"date": date, "n_results": len(group)}
        numbers = group["number_raw"].tolist()
        for index, position in enumerate(POSITION_NAMES):
            # Position names are left-to-right, while raw prize numbers have
            # different lengths.  A digit is eligible only when the position
            # exists in the original number.
            digits = [int(number[index - (5 - len(number))]) for number in numbers if len(number) >= 5 - index]
            eligible = len(digits)
            row[f"{position}_eligible_count"] = eligible
            for digit in range(10):
                row[f"{position}_d{digit}"] = int(digit in digits)
                row[f"{position}_d{digit}_count"] = int(digits.count(digit))
        row["pool_numbers"] = " ".join(group["number_5"].tolist())
        rows.append(row)

    result = pd.DataFrame(rows)
    result["year"] = result["date"].dt.year.astype("int16")
    result["month"] = result["date"].dt.month.astype("int8")
    result["day_of_week"] = result["date"].dt.dayofweek.astype("int8")
    return result


def main() -> None:
    data = pd.read_csv(RAW_FILE, dtype={"number": str, "prize_index": str})
    result = prepare_daily_targets(data)
    PRIZE_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(PRIZE_OUTPUT_FILE, index=False, encoding="utf-8-sig")
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Đã ghi: {OUTPUT_FILE}")
    print(f"Đã ghi target P2B: {PRIZE_OUTPUT_FILE}")
    print(f"Số ngày: {len(result):,}; số cột: {len(result.columns):,}")
    print(f"Khoảng thời gian: {result.date.min().date()} -> {result.date.max().date()}")
    print("Số chữ số xuất hiện trung bình theo vị trí:")
    for position in POSITION_NAMES:
        columns = [f"{position}_d{digit}" for digit in range(10)]
        print(f"  {position}: {result[columns].sum(axis=1).mean():.3f}")


if __name__ == "__main__":
    main()
