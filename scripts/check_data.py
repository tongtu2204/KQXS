"""Kiểm tra chất lượng dữ liệu KQXSMB sau khi chuẩn hóa."""

from pathlib import Path

import pandas as pd


# ============================================================
# Đường dẫn
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "kqxsmb_digits.csv"
)


# ============================================================
# Đọc dữ liệu
# ============================================================

def read_data(file_path: Path = DATA_FILE) -> pd.DataFrame:
    """Đọc dữ liệu KQXSMB đã chuẩn hóa."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {file_path}\n"
            "Hãy chạy scripts/prepare_digits.py trước."
        )

    df = pd.read_csv(
        file_path,
        dtype={
            "full_result": str,
            "last_2_digits": str,
        },
        parse_dates=["date"],
    )

    df["full_result"] = df["full_result"].str.zfill(5)
    df["last_2_digits"] = df["last_2_digits"].str.zfill(2)

    return df


# ============================================================
# Các kiểm tra chất lượng
# ============================================================

def check_required_columns(df: pd.DataFrame) -> None:
    """Kiểm tra các cột bắt buộc."""

    required_columns = {
        "date",
        "full_result",
        "digit_1",
        "digit_2",
        "digit_3",
        "digit_4",
        "digit_5",
        "last_2_digits",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Thiếu các cột: {sorted(missing_columns)}"
        )

    print("[OK] Đầy đủ các cột bắt buộc.")


def check_missing_values(df: pd.DataFrame) -> None:
    """Kiểm tra dữ liệu bị thiếu."""

    missing = df.isna().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("[OK] Không có giá trị bị thiếu.")
    else:
        print("[CẢNH BÁO] Các cột có giá trị thiếu:")
        print(missing.to_string())


def check_duplicate_dates(df: pd.DataFrame) -> None:
    """Kiểm tra một ngày xuất hiện nhiều hơn một lần."""

    duplicated = df[
        df.duplicated(
            subset="date",
            keep=False,
        )
    ].sort_values("date")

    if duplicated.empty:
        print("[OK] Không có ngày bị trùng.")
    else:
        print(
            f"[LỖI] Có {len(duplicated):,} "
            "dòng thuộc các ngày bị trùng:"
        )

        print(
            duplicated[
                ["date", "full_result"]
            ].to_string(index=False)
        )


def check_full_result(df: pd.DataFrame) -> None:
    """Kiểm tra full_result có đúng 5 chữ số."""

    valid_format = (
        df["full_result"]
        .str.fullmatch(r"\d{5}")
        .fillna(False)
    )

    invalid = df[~valid_format]

    if invalid.empty:
        print("[OK] Tất cả full_result đều có đúng 5 chữ số.")
    else:
        print(
            f"[LỖI] Có {len(invalid):,} "
            "full_result không hợp lệ:"
        )

        print(
            invalid[
                ["date", "full_result"]
            ].to_string(index=False)
        )


def check_digit_ranges(df: pd.DataFrame) -> None:
    """Kiểm tra digit_1 đến digit_5 nằm trong khoảng 0-9."""

    digit_columns = [
        "digit_1",
        "digit_2",
        "digit_3",
        "digit_4",
        "digit_5",
    ]

    invalid_rows = pd.Series(
        False,
        index=df.index,
    )

    for column in digit_columns:
        invalid_rows |= ~df[column].between(0, 9)

    invalid = df[invalid_rows]

    if invalid.empty:
        print("[OK] Các chữ số đều nằm trong khoảng 0-9.")
    else:
        print(
            f"[LỖI] Có {len(invalid):,} "
            "dòng chứa chữ số ngoài khoảng 0-9:"
        )

        print(
            invalid[
                ["date", "full_result", *digit_columns]
            ].to_string(index=False)
        )


def check_digit_consistency(df: pd.DataFrame) -> None:
    """Kiểm tra 5 cột digit có khớp với full_result."""

    reconstructed = (
        df["digit_1"].astype(str)
        + df["digit_2"].astype(str)
        + df["digit_3"].astype(str)
        + df["digit_4"].astype(str)
        + df["digit_5"].astype(str)
    )

    invalid = df[
        reconstructed.ne(df["full_result"])
    ].copy()

    if invalid.empty:
        print("[OK] Năm cột digit khớp với full_result.")
    else:
        invalid["reconstructed"] = reconstructed.loc[
            invalid.index
        ]

        print(
            f"[LỖI] Có {len(invalid):,} "
            "dòng có các digit không khớp:"
        )

        print(
            invalid[
                [
                    "date",
                    "full_result",
                    "reconstructed",
                ]
            ].to_string(index=False)
        )


def check_last_two_digits(df: pd.DataFrame) -> None:
    """Kiểm tra last_2_digits có khớp với full_result."""

    expected_last_two = df["full_result"].str[-2:]

    invalid = df[
        df["last_2_digits"].ne(expected_last_two)
    ].copy()

    if invalid.empty:
        print("[OK] last_2_digits khớp với full_result.")
    else:
        invalid["expected"] = expected_last_two.loc[
            invalid.index
        ]

        print(
            f"[LỖI] Có {len(invalid):,} "
            "dòng có last_2_digits không khớp:"
        )

        print(
            invalid[
                [
                    "date",
                    "full_result",
                    "last_2_digits",
                    "expected",
                ]
            ].to_string(index=False)
        )


def check_date_order(df: pd.DataFrame) -> None:
    """Kiểm tra dữ liệu đã được sắp xếp tăng dần theo ngày."""

    if df["date"].is_monotonic_increasing:
        print("[OK] Dữ liệu được sắp xếp tăng dần theo ngày.")
    else:
        print("[LỖI] Dữ liệu chưa được sắp xếp theo ngày.")


def find_missing_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Tìm các ngày không xuất hiện trong khoảng dữ liệu."""

    start_date = df["date"].min()
    end_date = df["date"].max()

    expected_dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D",
    )

    actual_dates = pd.DatetimeIndex(
        df["date"].dropna().unique()
    )

    return expected_dates.difference(actual_dates)


def report_missing_dates(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Báo cáo các ngày không có dữ liệu."""

    missing_dates = find_missing_dates(df)

    if len(missing_dates) == 0:
        print("[OK] Không thiếu ngày trong khoảng dữ liệu.")
        return

    print(
        f"[CẢNH BÁO] Có {len(missing_dates):,} "
        "ngày không xuất hiện trong dữ liệu."
    )

    print("10 ngày thiếu đầu tiên:")

    for missing_date in missing_dates[:10]:
        print(f"  - {missing_date.date()}")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = output_dir / "missing_dates.csv"

    pd.DataFrame(
        {"date": missing_dates}
    ).to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Đã lưu danh sách ngày thiếu: {output_file}")


# ============================================================
# Báo cáo tổng quan
# ============================================================

def print_summary(df: pd.DataFrame) -> None:
    """In thông tin tổng quan của dữ liệu."""

    print("=" * 70)
    print("TỔNG QUAN DỮ LIỆU")
    print("=" * 70)

    print(f"Số kỳ              : {len(df):,}")
    print(f"Số cột             : {df.shape[1]:,}")
    print(f"Ngày bắt đầu        : {df['date'].min().date()}")
    print(f"Ngày kết thúc       : {df['date'].max().date()}")
    print(f"Số năm              : {df['date'].dt.year.nunique():,}")
    print(f"Số kết quả khác nhau: {df['full_result'].nunique():,}")

    print("\nSố kỳ theo năm:")

    yearly_count = (
        df.groupby(df["date"].dt.year)
        .size()
        .rename("number_of_draws")
    )

    print(yearly_count.to_string())


# ============================================================
# Chạy chương trình
# ============================================================

def main() -> None:
    print(f"Đang kiểm tra: {DATA_FILE}\n")

    df = read_data()

    print_summary(df)

    print("\n" + "=" * 70)
    print("KIỂM TRA CHẤT LƯỢNG")
    print("=" * 70)

    check_required_columns(df)
    check_missing_values(df)
    check_duplicate_dates(df)
    check_full_result(df)
    check_digit_ranges(df)
    check_digit_consistency(df)
    check_last_two_digits(df)
    check_date_order(df)

    report_missing_dates(
        df=df,
        output_dir=PROJECT_DIR / "artifacts" / "tables",
    )

    print("\nHoàn thành kiểm tra dữ liệu.")


if __name__ == "__main__":
    main()