"""Chuẩn hóa và tách giải đặc biệt miền Bắc thành 5 vị trí chữ số."""

from pathlib import Path

import pandas as pd


# ============================================================
# Đường dẫn
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"


# ============================================================
# Các lỗi dữ liệu đã được đối chiếu thủ công
# ============================================================

MANUAL_CORRECTIONS = {
    "2002-12-19": "57530",
}


# ============================================================
# Đọc dữ liệu
# ============================================================

def find_latest_raw_file() -> Path:
    """Tìm file KQXSMB mới nhất trong thư mục data/raw."""

    files = sorted(RAW_DATA_DIR.glob("kqxsmb_*.csv"))

    if not files:
        raise FileNotFoundError(
            f"Không tìm thấy file kqxsmb_*.csv trong {RAW_DATA_DIR}"
        )

    return files[-1]


def read_raw_data(file_path: Path) -> pd.DataFrame:
    """Đọc dữ liệu và giữ nguyên các số 0 ở đầu."""

    df = pd.read_csv(
        file_path,
        dtype={
            "full_result": str,
            "last_2_digits": str,
        },
        parse_dates=["date"],
    )

    return df


# ============================================================
# Chuẩn hóa dữ liệu
# ============================================================

def apply_manual_corrections(df: pd.DataFrame) -> pd.DataFrame:
    """Áp dụng các chỉnh sửa đã được đối chiếu thủ công."""

    result = df.copy()

    for correction_date, corrected_result in MANUAL_CORRECTIONS.items():
        correction_date = pd.Timestamp(correction_date)

        mask = result["date"].eq(correction_date)

        if not mask.any():
            print(
                f"Cảnh báo: không tìm thấy ngày "
                f"{correction_date.date()} để sửa."
            )
            continue

        old_results = result.loc[mask, "full_result"].unique()

        result.loc[mask, "full_result"] = corrected_result
        result.loc[mask, "last_2_digits"] = corrected_result[-2:]

        print(
            f"Đã sửa {correction_date.date()}: "
            f"{list(old_results)} -> {corrected_result}"
        )

    return result


def prepare_digits(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch dữ liệu và tách full_result thành 5 chữ số."""

    required_columns = {
        "date",
        "full_result",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dữ liệu thiếu các cột bắt buộc: "
            f"{sorted(missing_columns)}"
        )

    result = df.copy()

    # Chuẩn hóa ngày
    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
    )

    invalid_dates = result["date"].isna().sum()

    if invalid_dates > 0:
        print(f"Loại {invalid_dates:,} dòng có ngày không hợp lệ.")

    result = result.dropna(subset=["date"]).copy()

    # Chuẩn hóa kết quả về chuỗi
    result["full_result"] = (
        result["full_result"]
        .astype("string")
        .str.strip()
    )

    # Áp dụng những chỉnh sửa đã được xác minh
    result = apply_manual_corrections(result)

    # Chỉ giữ kết quả gồm các chữ số 0-9
    is_numeric = result["full_result"].str.fullmatch(r"\d+")

    invalid_numeric = (~is_numeric.fillna(False)).sum()

    if invalid_numeric > 0:
        print("\nCác dòng có kết quả không phải số:")

        print(
            result.loc[
                ~is_numeric.fillna(False),
                ["date", "full_result"],
            ].to_string(index=False)
        )

        print(
            f"\nLoại {invalid_numeric:,} dòng "
            f"có kết quả không phải số."
        )

    result = result[is_numeric.fillna(False)].copy()

    # Khôi phục số 0 ở đầu
    # Ví dụ: 1234 -> 01234
    result["full_result"] = result["full_result"].str.zfill(5)

    # Kiểm tra độ dài sau khi chuẩn hóa
    valid_length = result["full_result"].str.len().eq(5)
    invalid_length = (~valid_length).sum()

    if invalid_length > 0:
        print("\nCác dòng có full_result không đúng 5 chữ số:")

        display_columns = ["date", "full_result"]

        if "last_2_digits" in result.columns:
            display_columns.append("last_2_digits")

        print(
            result.loc[
                ~valid_length,
                display_columns,
            ].to_string(index=False)
        )

        print(
            f"\nLoại {invalid_length:,} dòng "
            f"không đúng 5 chữ số."
        )

    result = result[valid_length].copy()

    # Thống kê số ngày trùng trước khi loại
    duplicate_count = result.duplicated(
        subset="date",
        keep=False,
    ).sum()

    if duplicate_count > 0:
        print(
            f"Phát hiện {duplicate_count:,} dòng "
            f"thuộc các ngày bị trùng."
        )

    # Sắp xếp thời gian và loại ngày trùng
    result = (
        result.sort_values("date")
        .drop_duplicates(
            subset="date",
            keep="last",
        )
        .reset_index(drop=True)
    )

    # Tách thành 5 vị trí tương ứng với 5 máy quay
    for position in range(5):
        column_name = f"digit_{position + 1}"

        result[column_name] = (
            result["full_result"]
            .str[position]
            .astype("int8")
        )

    # Tạo lại hai chữ số cuối từ full_result
    result["last_2_digits"] = result["full_result"].str[-2:]

    # Tạo các biến thời gian
    result["year"] = result["date"].dt.year
    result["month"] = result["date"].dt.month
    result["day"] = result["date"].dt.day

    # Thứ Hai = 0 và Chủ nhật = 6
    result["day_of_week"] = result["date"].dt.dayofweek

    output_columns = [
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
    ]

    return result[output_columns]


# ============================================================
# Chạy chương trình
# ============================================================

def main() -> None:
    raw_file = find_latest_raw_file()

    print(f"Đang đọc: {raw_file}")

    raw_df = read_raw_data(raw_file)

    print(f"Số dòng dữ liệu raw: {len(raw_df):,}")

    processed_df = prepare_digits(raw_df)

    if processed_df.empty:
        raise ValueError(
            "Không còn dữ liệu hợp lệ sau khi chuẩn hóa."
        )

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        PROCESSED_DATA_DIR
        / "kqxsmb_digits.csv"
    )

    processed_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n5 dòng đầu:")

    print(
        processed_df.head().to_string(
            index=False
        )
    )

    print(f"\nSố kỳ hợp lệ: {len(processed_df):,}")

    print(
        "Khoảng thời gian: "
        f"{processed_df['date'].min().date()} "
        "đến "
        f"{processed_df['date'].max().date()}"
    )

    print(f"Đã lưu: {output_file}")


if __name__ == "__main__":
    main()