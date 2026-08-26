"""Kiểm tra phụ thuộc thời gian tại các độ trễ từ 1 đến 30 kỳ."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "kqxsmb_digits.csv"
)

TABLE_DIR = PROJECT_DIR / "artifacts" / "tables"
FIGURE_DIR = PROJECT_DIR / "artifacts" / "figures"

DIGIT_COLUMNS = [
    "digit_1",
    "digit_2",
    "digit_3",
    "digit_4",
    "digit_5",
]

MAX_LAG = 30
ALPHA = 0.05


def read_data() -> pd.DataFrame:
    """Đọc dữ liệu đã chuẩn hóa."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {DATA_FILE}."
        )

    return (
        pd.read_csv(
            DATA_FILE,
            dtype={
                "full_result": str,
                "last_2_digits": str,
            },
            parse_dates=["date"],
        )
        .sort_values("date")
        .reset_index(drop=True)
    )


def benjamini_hochberg(
    p_values: pd.Series,
) -> pd.Series:
    """Hiệu chỉnh p-value theo Benjamini-Hochberg."""

    number_of_tests = len(p_values)

    order = np.argsort(p_values.to_numpy())
    sorted_values = p_values.to_numpy()[order]

    adjusted_sorted = (
        sorted_values
        * number_of_tests
        / np.arange(1, number_of_tests + 1)
    )

    adjusted_sorted = np.minimum.accumulate(
        adjusted_sorted[::-1]
    )[::-1]

    adjusted_sorted = np.clip(
        adjusted_sorted,
        0,
        1,
    )

    adjusted = np.empty(number_of_tests)
    adjusted[order] = adjusted_sorted

    return pd.Series(
        adjusted,
        index=p_values.index,
    )


def calculate_cramers_v(
    chi_square: float,
    sample_size: int,
) -> float:
    """Tính Cramér's V cho bảng 10 × 10."""

    return np.sqrt(
        chi_square
        / (sample_size * 9)
    )


def run_lag_tests(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Kiểm định độc lập ở từng vị trí và độ trễ."""

    records = []

    for position, column in enumerate(
        DIGIT_COLUMNS,
        start=1,
    ):
        for lag in range(1, MAX_LAG + 1):
            previous_digits = df[column].shift(lag)

            valid = previous_digits.notna()

            previous = (
                previous_digits[valid]
                .astype(int)
            )

            current = (
                df.loc[valid, column]
                .astype(int)
            )

            table = pd.crosstab(
                previous,
                current,
            ).reindex(
                index=range(10),
                columns=range(10),
                fill_value=0,
            )

            (
                chi_square,
                p_value,
                degrees_of_freedom,
                expected,
            ) = chi2_contingency(
                table,
                correction=False,
            )

            sample_size = int(
                table.to_numpy().sum()
            )

            records.append(
                {
                    "position": position,
                    "lag": lag,
                    "sample_size": sample_size,
                    "chi_square": chi_square,
                    "degrees_of_freedom": degrees_of_freedom,
                    "p_value": p_value,
                    "cramers_v": calculate_cramers_v(
                        chi_square,
                        sample_size,
                    ),
                    "minimum_expected_count": expected.min(),
                }
            )

    result = pd.DataFrame(records)

    number_of_tests = len(result)

    result["p_bonferroni"] = np.minimum(
        result["p_value"] * number_of_tests,
        1,
    )

    result["p_bh"] = benjamini_hochberg(
        result["p_value"]
    )

    result["reject_raw"] = (
        result["p_value"] < ALPHA
    )

    result["reject_bonferroni"] = (
        result["p_bonferroni"] < ALPHA
    )

    result["reject_bh"] = (
        result["p_bh"] < ALPHA
    )

    return result


def print_results(
    result: pd.DataFrame,
) -> None:
    """In các tín hiệu mạnh nhất."""

    strongest = result.sort_values(
        "p_value"
    ).head(15)

    print("\n" + "=" * 110)
    print("15 QUAN HỆ CÓ P-VALUE THÔ NHỎ NHẤT")
    print("=" * 110)

    print(
        strongest[
            [
                "position",
                "lag",
                "chi_square",
                "p_value",
                "p_bonferroni",
                "p_bh",
                "cramers_v",
            ]
        ].to_string(
            index=False,
            formatters={
                "chi_square": "{:.4f}".format,
                "p_value": "{:.6f}".format,
                "p_bonferroni": "{:.6f}".format,
                "p_bh": "{:.6f}".format,
                "cramers_v": "{:.4f}".format,
            },
        )
    )

    print("\nTổng số kiểm định:", len(result))

    print(
        "Có ý nghĩa trước hiệu chỉnh:",
        int(result["reject_raw"].sum()),
    )

    print(
        "Có ý nghĩa sau Bonferroni:",
        int(result["reject_bonferroni"].sum()),
    )

    print(
        "Có ý nghĩa sau Benjamini-Hochberg:",
        int(result["reject_bh"].sum()),
    )

    print(
        "Cramér's V lớn nhất:",
        f"{result['cramers_v'].max():.4f}",
    )

    print(
        "Expected count nhỏ nhất:",
        f"{result['minimum_expected_count'].min():.2f}",
    )


def plot_temporal_tests(
    result: pd.DataFrame,
) -> None:
    """Vẽ p-value hiệu chỉnh và Cramér's V."""

    p_matrix = result.pivot(
        index="position",
        columns="lag",
        values="p_bh",
    )

    effect_matrix = result.pivot(
        index="position",
        columns="lag",
        values="cramers_v",
    )

    log_p_matrix = -np.log10(
        np.maximum(
            p_matrix,
            1e-300,
        )
    )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(18, 9),
        constrained_layout=True,
    )

    first_image = axes[0].imshow(
        log_p_matrix.values,
        cmap="YlOrRd",
        aspect="auto",
    )

    axes[0].set_title(
        "Mức ý nghĩa sau Benjamini–Hochberg",
        fontsize=14,
        fontweight="bold",
    )

    axes[0].set_ylabel("Vị trí")
    axes[0].set_yticks(range(5))
    axes[0].set_yticklabels(range(1, 6))

    axes[0].set_xticks(range(MAX_LAG))
    axes[0].set_xticklabels(
        range(1, MAX_LAG + 1)
    )

    fig.colorbar(
        first_image,
        ax=axes[0],
        label="-log10(p-value hiệu chỉnh)",
    )

    second_image = axes[1].imshow(
        effect_matrix.values,
        cmap="Blues",
        aspect="auto",
        vmin=0,
    )

    axes[1].set_title(
        "Mức độ phụ thuộc theo Cramér's V",
        fontsize=14,
        fontweight="bold",
    )

    axes[1].set_xlabel("Độ trễ – số kỳ")
    axes[1].set_ylabel("Vị trí")

    axes[1].set_yticks(range(5))
    axes[1].set_yticklabels(range(1, 6))

    axes[1].set_xticks(range(MAX_LAG))
    axes[1].set_xticklabels(
        range(1, MAX_LAG + 1)
    )

    fig.colorbar(
        second_image,
        ax=axes[1],
        label="Cramér's V",
    )

    fig.suptitle(
        "Kiểm tra phụ thuộc thời gian tại độ trễ 1–30 kỳ",
        fontsize=18,
        fontweight="bold",
    )

    output_file = (
        FIGURE_DIR
        / "temporal_dependence_lags.png"
    )

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.show()
    plt.close(fig)

    print(f"Đã lưu: {output_file}")


def save_results(
    result: pd.DataFrame,
) -> None:
    """Lưu kết quả kiểm định."""

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        TABLE_DIR
        / "temporal_dependence_tests.csv"
    )

    result.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Đã lưu: {output_file}")


def main() -> None:
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = read_data()

    print(f"Đã đọc {len(df):,} kỳ.")
    print(
        f"Kiểm tra 5 vị trí × {MAX_LAG} độ trễ."
    )

    result = run_lag_tests(df)

    print_results(result)
    save_results(result)
    plot_temporal_tests(result)


if __name__ == "__main__":
    main()