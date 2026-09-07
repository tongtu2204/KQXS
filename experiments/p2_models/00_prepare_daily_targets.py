"""Chuẩn hóa dữ liệu 7 nhóm giải thành target chữ số theo ngày.

Đơn vị quan sát của Phần 2 là một ngày. Với mỗi ngày và mỗi vị trí chữ số,
target là vector 10 phần tử cho biết chữ số 0..9 có xuất hiện ít nhất một
lần trong toàn bộ các giải của ngày đó hay không.
"""

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_DIR / "data" / "raw" / "kqxsmb_all_prizes_2007_2026.csv"
OUTPUT_FILE = PROJECT_DIR / "data" / "processed" / "daily_digit_targets.csv"

POSITION_NAMES = ("ten_thousands", "thousands", "hundreds", "tens", "units")


def prepare_daily_targets(data: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "number", "prize", "prize_index"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {sorted(missing)}")

    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["number"] = data["number"].astype("string").str.strip().str.zfill(5)
    valid = data["date"].notna() & data["number"].str.fullmatch(r"\d{5}").fillna(False)
    data = data.loc[valid].copy()
    data = data.sort_values(["date", "prize", "prize_index"]).reset_index(drop=True)

    if data.empty:
        raise ValueError("Không còn dữ liệu hợp lệ")

    rows = []
    for date, group in data.groupby("date", sort=True):
        if len(group) != 27:
            raise ValueError(f"Ngày {date.date()} không có đúng 27 dòng kết quả")

        row = {"date": date, "n_results": len(group)}
        numbers = group["number"].tolist()
        for index, position in enumerate(POSITION_NAMES):
            digits = [int(number[index]) for number in numbers]
            for digit in range(10):
                row[f"{position}_d{digit}"] = int(digit in digits)
        rows.append(row)

    result = pd.DataFrame(rows)
    result["year"] = result["date"].dt.year.astype("int16")
    result["month"] = result["date"].dt.month.astype("int8")
    result["day_of_week"] = result["date"].dt.dayofweek.astype("int8")
    return result


def main() -> None:
    data = pd.read_csv(RAW_FILE, dtype={"number": str, "prize_index": str})
    result = prepare_daily_targets(data)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Đã ghi: {OUTPUT_FILE}")
    print(f"Số ngày: {len(result):,}; số cột: {len(result.columns):,}")
    print(f"Khoảng thời gian: {result.date.min().date()} -> {result.date.max().date()}")
    print("Số chữ số xuất hiện trung bình theo vị trí:")
    for position in POSITION_NAMES:
        columns = [f"{position}_d{digit}" for digit in range(10)]
        print(f"  {position}: {result[columns].sum(axis=1).mean():.3f}")


if __name__ == "__main__":
    main()

