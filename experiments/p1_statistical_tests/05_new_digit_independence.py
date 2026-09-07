"""Thống kê và kiểm định tính ngẫu nhiên cho bài toán KQXS mới.

Phân tích gồm hai phạm vi:
1. Pooled: gộp toàn bộ giải.
2. By-prize: tách theo từng dimension giải.

Tất cả vị trí chữ số được tính từ phải sang trái.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, chisquare


POSITIONS = [
    "unit",
    "tens",
    "hundreds",
    "thousands",
    "ten_thousands",
]

POSITION_LABELS = {
    "unit": "Hàng đơn vị",
    "tens": "Hàng chục",
    "hundreds": "Hàng trăm",
    "thousands": "Hàng nghìn",
    "ten_thousands": "Hàng vạn",
}

PRIZES = [
    "Đặc biệt",
    "Giải nhất",
    "Giải nhì",
    "Giải ba",
    "Giải tư",
    "Giải năm",
    "Giải sáu",
    "Giải bảy",
]

PRIZE_ORDER = {prize: i for i, prize in enumerate(PRIZES, start=1)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/raw/kqxsmb_all_prizes_2007_2026.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/statistics_new",
    )
    return parser.parse_args()


def entropy_from_counts(counts: np.ndarray) -> float:
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def add_fdr_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Điều chỉnh p-value theo Benjamini-Hochberg FDR."""
    result = df.copy()
    p_values = result["p_value"].to_numpy(dtype=float)
    valid = np.isfinite(p_values)
    adjusted = np.full(len(result), np.nan)

    if valid.any():
        valid_indices = np.flatnonzero(valid)
        order = valid_indices[np.argsort(p_values[valid])]
        m = len(order)
        running = 1.0

        for rank in range(m, 0, -1):
            index = order[rank - 1]
            value = p_values[index] * m / rank
            running = min(running, value)
            adjusted[index] = min(running, 1.0)

    result["p_value_fdr"] = adjusted
    result["reject_fdr_0_05"] = result["p_value_fdr"] < 0.05
    return result


def cramers_v(table: np.ndarray, chi2: float) -> float:
    n = table.sum()
    if n == 0:
        return float("nan")
    rows, cols = table.shape
    return float(math.sqrt(chi2 / (n * min(rows - 1, cols - 1))))


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={
        "date": str,
        "province": str,
        "prize": str,
        "prize_index": str,
        "number": str,
        "last_2_digits": str,
    })

    required = {
        "date",
        "province",
        "prize",
        "prize_index",
        "number",
        "last_2_digits",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["number"] = df["number"].str.strip()
    df["last_2_digits"] = df["last_2_digits"].str.strip().str.zfill(2)

    if not df["number"].str.fullmatch(r"\d+").all():
        raise ValueError("number chứa giá trị không phải chữ số.")

    if (df["number"].str[-2:] != df["last_2_digits"]).any():
        raise ValueError("last_2_digits không khớp number.")

    number_length = df["number"].str.len()

    for position, offset in zip(POSITIONS, range(1, 6)):
        digits = pd.to_numeric(
            df["number"].str[-offset],
            errors="coerce",
        )
        # Số 2/3/4 chữ số không có các vị trí bên trái tương ứng.
        df[f"digit_{position}"] = digits.astype("Int8")

    df["suffix_2"] = df["number"].str[-2:]
    df["suffix_3"] = df["number"].str[-3:].where(number_length >= 3)
    df["suffix_4"] = df["number"].str[-4:].where(number_length >= 4)
    df["suffix_5"] = df["number"].str[-5:].where(number_length >= 5)
    return df


def digit_distribution_tests(
    df: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    scopes = [("pooled", "ALL", df)]
    scopes.extend(
        ("by_prize", prize, df[df["prize"] == prize])
        for prize in PRIZES
    )

    for scope, dimension, subset in scopes:
        for position in POSITIONS:
            column = f"digit_{position}"
            values = subset[column].dropna().astype(int)
            if values.empty:
                records.append(
                    {
                        "scope": scope,
                        "dimension": dimension,
                        "position": position,
                        "position_label": POSITION_LABELS[position],
                        "n": 0,
                        "chi2_statistic": np.nan,
                        "p_value": np.nan,
                        "entropy_bits": np.nan,
                        "min_frequency": np.nan,
                        "max_frequency": np.nan,
                        **{f"digit_{digit}_count": 0 for digit in range(10)},
                        **{f"digit_{digit}_rate": np.nan for digit in range(10)},
                    }
                )
                continue
            counts = np.bincount(values, minlength=10)
            expected = np.full(10, len(values) / 10)
            statistic, p_value = chisquare(counts, expected)

            records.append(
                {
                    "scope": scope,
                    "dimension": dimension,
                    "position": position,
                    "position_label": POSITION_LABELS[position],
                    "n": int(len(values)),
                    "chi2_statistic": float(statistic),
                    "p_value": float(p_value),
                    "entropy_bits": entropy_from_counts(counts),
                    "min_frequency": int(counts.min()),
                    "max_frequency": int(counts.max()),
                    **{
                        f"digit_{digit}_count": int(counts[digit])
                        for digit in range(10)
                    },
                    **{
                        f"digit_{digit}_rate": float(counts[digit] / len(values))
                        for digit in range(10)
                    },
                }
            )

    return pd.DataFrame(records)


def suffix_distribution_tests(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    suffix_specs = {
        "suffix_2": 100,
        "suffix_3": 1000,
        "suffix_4": 10000,
        "suffix_5": 100000,
    }

    scopes = [("pooled", "ALL", df)]
    scopes.extend(
        ("by_prize", prize, df[df["prize"] == prize])
        for prize in PRIZES
    )

    for scope, dimension, subset in scopes:
        for suffix, category_count in suffix_specs.items():
            values = subset[suffix].dropna().astype(str).str.zfill(len(suffix))
            if values.empty:
                records.append(
                    {
                        "scope": scope,
                        "dimension": dimension,
                        "suffix": suffix,
                        "n": 0,
                        "category_count": category_count,
                        "unique_values": 0,
                        "chi2_statistic": np.nan,
                        "p_value": np.nan,
                        "entropy_bits": np.nan,
                        "max_frequency": 0,
                        "min_frequency": 0,
                    }
                )
                continue
            counts = values.value_counts()
            observed = np.array(
                [counts.get(f"{i:0{len(suffix)}d}", 0) for i in range(category_count)]
            )
            expected = np.full(category_count, len(values) / category_count)
            statistic, p_value = chisquare(observed, expected)

            records.append(
                {
                    "scope": scope,
                    "dimension": dimension,
                    "suffix": suffix,
                    "n": int(len(values)),
                    "category_count": category_count,
                    "unique_values": int((observed > 0).sum()),
                    "chi2_statistic": float(statistic),
                    "p_value": float(p_value),
                    "entropy_bits": entropy_from_counts(observed),
                    "max_frequency": int(observed.max()),
                    "min_frequency": int(observed.min()),
                }
            )

    return pd.DataFrame(records)


def position_independence_tests(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    position_pairs = []

    for i, left in enumerate(POSITIONS):
        for right in POSITIONS[i + 1:]:
            position_pairs.append((left, right))

    scopes = [("pooled", "ALL", df)]
    scopes.extend(
        ("by_prize", prize, df[df["prize"] == prize])
        for prize in PRIZES
    )

    for scope, dimension, subset in scopes:
        for left, right in position_pairs:
            left_col = f"digit_{left}"
            right_col = f"digit_{right}"
            valid = subset[[left_col, right_col]].dropna()
            if valid.empty:
                records.append(
                    {
                        "scope": scope,
                        "dimension": dimension,
                        "left_position": left,
                        "right_position": right,
                        "n": 0,
                        "chi2_statistic": np.nan,
                        "degrees_of_freedom": np.nan,
                        "p_value": np.nan,
                        "cramers_v": np.nan,
                    }
                )
                continue
            table = pd.crosstab(valid[left_col], valid[right_col])
            table = table.reindex(index=range(10), columns=range(10), fill_value=0)
            statistic, p_value, dof, _ = chi2_contingency(table.to_numpy())

            records.append(
                {
                    "scope": scope,
                    "dimension": dimension,
                    "left_position": left,
                    "right_position": right,
                    "n": int(table.to_numpy().sum()),
                    "chi2_statistic": float(statistic),
                    "degrees_of_freedom": int(dof),
                    "p_value": float(p_value),
                    "cramers_v": cramers_v(table.to_numpy(), statistic),
                }
            )

    return pd.DataFrame(records)


def temporal_dependence_tests(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    grouped = df.groupby(["date"], sort=True)

    daily = grouped[
        [f"digit_{position}" for position in POSITIONS]
    ].mean()

    for scope, dimension, subset in [
        ("pooled", "ALL", df),
        *[
            ("by_prize", prize, df[df["prize"] == prize])
            for prize in PRIZES
        ],
    ]:
        if scope == "by_prize":
            daily_scope = subset.groupby("date")[
                [f"digit_{position}" for position in POSITIONS]
            ].mean()
        else:
            daily_scope = daily

        for position in POSITIONS:
            series = daily_scope[f"digit_{position}"].dropna()

            for lag in [1, 2, 3, 7, 14, 30]:
                if len(series) <= lag:
                    autocorrelation = float("nan")
                else:
                    autocorrelation = float(series.autocorr(lag=lag))

                records.append(
                    {
                        "scope": scope,
                        "dimension": dimension,
                        "position": position,
                        "lag_days": lag,
                        "n_days": int(len(series)),
                        "autocorrelation": autocorrelation,
                    }
                )

    return pd.DataFrame(records)


def build_daily_conditions(day: pd.DataFrame) -> list[tuple[str, str, int]]:
    """Tạo điều kiện multi-win, bỏ qua khuyến khích Đặc biệt.

    Theo quy ước của thống kê này, không tính trường hợp tự động:
    vé trùng Đặc biệt đồng thời trúng Phụ đặc biệt và Khuyến khích ĐB.
    Luật thưởng đầy đủ vẫn được giữ cho phần chiến thuật sau này.
    """
    conditions = []

    for row in day.itertuples(index=False):
        if row.prize == "Đặc biệt":
            conditions.append((row.number, "Phụ đặc biệt", 5))
        elif row.prize in {"Giải nhất", "Giải nhì", "Giải ba"}:
            conditions.append((row.number, row.prize, 5))
        elif row.prize in {"Giải tư", "Giải năm"}:
            conditions.append((row.number[-4:], row.prize, 4))
        elif row.prize == "Giải sáu":
            conditions.append((row.number[-3:], row.prize, 3))
        elif row.prize == "Giải bảy":
            conditions.append((row.number[-2:], row.prize, 2))

    return conditions


def multi_win_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Tính trên kết quả lịch sử thật, không dùng model."""
    payouts = {
        "Phụ đặc biệt": 25_000_000,
        "Giải nhất": 10_000_000,
        "Giải nhì": 5_000_000,
        "Giải ba": 1_000_000,
        "Giải tư": 400_000,
        "Giải năm": 200_000,
        "Giải sáu": 100_000,
        "Giải bảy": 40_000,
    }

    records = []

    for result_date, day in df.groupby("date", sort=True):
        conditions = build_daily_conditions(day)
        def evaluate(condition_list):
            best_count = 0
            best_payout = 0
            multi_pattern_count = 0

            # Mọi điều kiện đều là hậu tố của vé 5 chữ số. Một vé có thể
            # trúng nhiều giải khi hậu tố của một điều kiện chứa hậu tố
            # của điều kiện khác; không cần duyệt 100.000 vé.
            for pattern, _, _ in condition_list:
                matched = [
                    (prize, payouts[prize])
                    for other_pattern, prize, _ in condition_list
                    if pattern.endswith(other_pattern)
                ]
                count = len(matched)
                payout = sum(value for _, value in matched)

                if count >= 2:
                    multi_pattern_count += 1

                best_count = max(best_count, count)
                best_payout = max(best_payout, payout)

            return best_count, best_payout, multi_pattern_count

        best_count, best_payout, multi_pattern_count = evaluate(conditions)

        records.append(
            {
                "date": result_date.date().isoformat(),
                "multi_win_day": bool(best_count >= 2),
                "multi_win_pattern_count": int(multi_pattern_count),
                "max_simultaneous_prizes": int(best_count),
                "max_total_payout": int(best_payout),
                "has_2plus_prizes": bool(best_count >= 2),
                "has_3plus_prizes": bool(best_count >= 3),
                "has_4plus_prizes": bool(best_count >= 4),
            }
        )

    daily = pd.DataFrame(records)
    daily["year"] = pd.to_datetime(daily["date"]).dt.year
    return daily


def make_yearly_multi_win_summary(daily: pd.DataFrame) -> pd.DataFrame:
    summary = (
        daily.groupby("year")
        .agg(
            total_draw_days=("date", "count"),
            days_with_2plus_prizes=("has_2plus_prizes", "sum"),
            days_with_3plus_prizes=("has_3plus_prizes", "sum"),
            days_with_4plus_prizes=("has_4plus_prizes", "sum"),
            average_multi_win_patterns=("multi_win_pattern_count", "mean"),
            max_simultaneous_prizes=("max_simultaneous_prizes", "max"),
            average_max_total_payout=("max_total_payout", "mean"),
        )
        .reset_index()
    )
    for threshold in [2, 3, 4]:
        summary[f"rate_days_with_{threshold}plus_prizes"] = (
            summary[f"days_with_{threshold}plus_prizes"]
            / summary["total_draw_days"]
        )
    return summary


def make_report(
    df: pd.DataFrame,
    digit_tests: pd.DataFrame,
    suffix_tests: pd.DataFrame,
    independence_tests: pd.DataFrame,
    temporal_tests: pd.DataFrame,
    yearly_multi: pd.DataFrame,
) -> str:
    pooled_digit = digit_tests[digit_tests["scope"] == "pooled"]
    pooled_suffix = suffix_tests[suffix_tests["scope"] == "pooled"]
    pooled_independence = independence_tests[
        independence_tests["scope"] == "pooled"
    ]

    return f"""# Báo cáo thống kê tính ngẫu nhiên — bài toán mới

## Phạm vi dữ liệu

- Quy ước multi-win: **không tính** trường hợp tự động vé trùng Giải đặc biệt
  đồng thời trúng Giải phụ đặc biệt và Giải khuyến khích Đặc biệt.
- Khoảng ngày: `{df['date'].min().date()}` đến `{df['date'].max().date()}`
- Số ngày có kết quả: `{df['date'].nunique():,}`
- Số dòng kết quả: `{len(df):,}`
- Số kết quả mỗi ngày: `{df.groupby('date').size().unique().tolist()}`
- Phân tích: gộp toàn bộ giải và tách theo từng giải.
- Vị trí chữ số: tính từ phải sang trái.

## Kiểm định chữ số

Kiểm định Chi-square so với phân phối đều 10 chữ số được lưu trong
`digit_distribution_tests.csv`. Kết quả pooled:

        {pooled_digit[['position', 'n', 'chi2_statistic', 'p_value', 'entropy_bits']].to_string(index=False)}

## Kiểm định hậu tố

Kết quả Chi-square cho hậu tố 2, 3, 4 và 5 chữ số được lưu trong
`suffix_distribution_tests.csv`.

        {pooled_suffix[['suffix', 'n', 'unique_values', 'chi2_statistic', 'p_value', 'entropy_bits']].to_string(index=False)}

## Độc lập giữa các vị trí

Kiểm định Chi-square độc lập và Cramér's V được lưu trong
`position_independence_tests.csv`.

        {pooled_independence[['left_position', 'right_position', 'p_value', 'cramers_v']].to_string(index=False)}

## Phụ thuộc theo thời gian

Tự tương quan theo lag 1, 2, 3, 7, 14 và 30 ngày được lưu trong
`temporal_dependence_tests.csv`.

## Khả năng trúng nhiều giải trong lịch sử thật

Kết quả theo từng ngày được lưu trong `multi_win_daily.csv`, tổng hợp theo năm
trong `multi_win_yearly.csv`. Đây là thống kê trên kết quả thực tế, không dùng
model và không phụ thuộc vào danh sách vé chiến thuật.

        {yearly_multi.to_string(index=False)}
"""


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path)

    digit_tests = digit_distribution_tests(df)
    suffix_tests = suffix_distribution_tests(df)
    independence_tests = position_independence_tests(df)
    temporal_tests = temporal_dependence_tests(df)

    digit_tests = add_fdr_columns(digit_tests)
    suffix_tests = add_fdr_columns(suffix_tests)
    independence_tests = add_fdr_columns(independence_tests)
    multi_daily = multi_win_statistics(df)
    multi_yearly = make_yearly_multi_win_summary(multi_daily)

    digit_tests.to_csv(
        output_dir / "digit_distribution_tests.csv",
        index=False,
        encoding="utf-8-sig",
    )
    suffix_tests.to_csv(
        output_dir / "suffix_distribution_tests.csv",
        index=False,
        encoding="utf-8-sig",
    )
    independence_tests.to_csv(
        output_dir / "position_independence_tests.csv",
        index=False,
        encoding="utf-8-sig",
    )
    temporal_tests.to_csv(
        output_dir / "temporal_dependence_tests.csv",
        index=False,
        encoding="utf-8-sig",
    )
    multi_daily.to_csv(
        output_dir / "multi_win_daily.csv",
        index=False,
        encoding="utf-8-sig",
    )
    multi_yearly.to_csv(
        output_dir / "multi_win_yearly.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "input": str(input_path),
        "date_min": df["date"].min().date().isoformat(),
        "date_max": df["date"].max().date().isoformat(),
        "n_rows": int(len(df)),
        "n_draw_days": int(df["date"].nunique()),
        "analysis_scopes": ["pooled", "by_prize"],
        "positions": POSITIONS,
        "multi_win_rule": (
            "Exclude the automatic Special Prize + Special Encouragement pair."
        ),
        "multiple_testing_note": (
            "Raw p-values are reported; FDR adjustment will be applied in the review step."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = make_report(
        df,
        digit_tests,
        suffix_tests,
        independence_tests,
        temporal_tests,
        multi_yearly,
    )
    (output_dir / "statistical_report.md").write_text(
        report,
        encoding="utf-8",
    )

    print("Hoàn tất thống kê bài toán mới.")
    print(f"Dữ liệu: {input_path}")
    print(f"Số ngày: {df['date'].nunique():,}")
    print(f"Số dòng: {len(df):,}")
    print(f"Kết quả lưu tại: {output_dir}")


if __name__ == "__main__":
    main()
