"""Kiểm định phân phối đồng đều 0-9 tại từng vị trí giải đặc biệt."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chisquare


# ============================================================
# Cấu hình
# ============================================================

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

ALPHA = 0.05


# ============================================================
# Đọc dữ liệu
# ============================================================

def read_data() -> pd.DataFrame:
    """Đọc dữ liệu đã chuẩn hóa."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {DATA_FILE}. "
            "Hãy chạy scripts/prepare_digits.py trước."
        )

    return pd.read_csv(
        DATA_FILE,
        dtype={
            "full_result": str,
            "last_2_digits": str,
        },
        parse_dates=["date"],
    )


# ============================================================
# Hiệu chỉnh nhiều kiểm định
# ============================================================

def benjamini_hochberg(
    p_values: pd.Series,
) -> pd.Series:
    """Hiệu chỉnh p-value theo Benjamini-Hochberg."""

    number_of_tests = len(p_values)

    order = np.argsort(p_values.to_numpy())
    sorted_p_values = p_values.to_numpy()[order]

    adjusted_sorted = (
        sorted_p_values
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


# ============================================================
# Kiểm định Chi-square
# ============================================================

def run_uniformity_tests(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kiểm định Chi-square cho từng vị trí."""

    summary_records = []
    detail_records = []

    number_of_draws = len(df)
    expected_count = number_of_draws / 10

    for position, column in enumerate(
        DIGIT_COLUMNS,
        start=1,
    ):
        observed = (
            df[column]
            .value_counts()
            .reindex(range(10), fill_value=0)
            .sort_index()
        )

        expected = np.repeat(
            expected_count,
            10,
        )

        chi_square, p_value = chisquare(
            f_obs=observed.to_numpy(),
            f_exp=expected,
        )

        observed_proportion = (
            observed.to_numpy()
            / number_of_draws
        )

        expected_proportion = np.repeat(
            0.10,
            10,
        )

        # Cohen's w đo mức độ lệch khỏi phân phối đều
        cohen_w = np.sqrt(
            np.sum(
                (
                    observed_proportion
                    - expected_proportion
                )
                ** 2
                / expected_proportion
            )
        )

        summary_records.append(
            {
                "position": position,
                "sample_size": number_of_draws,
                "chi_square": chi_square,
                "degrees_of_freedom": 9,
                "p_value": p_value,
                "cohen_w": cohen_w,
            }
        )

        standardized_residuals = (
            observed.to_numpy()
            - expected
        ) / np.sqrt(expected)

        for digit in range(10):
            detail_records.append(
                {
                    "position": position,
                    "digit": digit,
                    "observed_count": observed.iloc[digit],
                    "expected_count": expected_count,
                    "observed_proportion": (
                        observed.iloc[digit]
                        / number_of_draws
                    ),
                    "expected_proportion": 0.10,
                    "deviation": (
                        observed.iloc[digit]
                        / number_of_draws
                        - 0.10
                    ),
                    "standardized_residual": (
                        standardized_residuals[digit]
                    ),
                }
            )

    summary = pd.DataFrame(summary_records)
    detail = pd.DataFrame(detail_records)

    # Bonferroni: kiểm soát xác suất có ít nhất một kết luận sai
    summary["p_bonferroni"] = np.minimum(
        summary["p_value"] * len(summary),
        1,
    )

    # Benjamini-Hochberg: kiểm soát tỷ lệ phát hiện sai
    summary["p_bh"] = benjamini_hochberg(
        summary["p_value"]
    )

    summary["reject_raw"] = (
        summary["p_value"] < ALPHA
    )

    summary["reject_bonferroni"] = (
        summary["p_bonferroni"] < ALPHA
    )

    summary["reject_bh"] = (
        summary["p_bh"] < ALPHA
    )

    return summary, detail


# ============================================================
# Diễn giải effect size
# ============================================================

def classify_cohen_w(value: float) -> str:
    """Phân loại Cohen's w."""

    if value < 0.10:
        return "Không đáng kể"
    if value < 0.30:
        return "Nhỏ"
    if value < 0.50:
        return "Trung bình"

    return "Lớn"


def add_interpretation(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Thêm kết luận dễ đọc."""

    result = summary.copy()

    result["effect_size"] = result[
        "cohen_w"
    ].apply(classify_cohen_w)

    result["conclusion"] = np.where(
        result["reject_bonferroni"],
        "Bác bỏ H0",
        "Chưa đủ bằng chứng bác bỏ H0",
    )

    return result


# ============================================================
# Trực quan hóa
# ============================================================

def plot_p_values(
    summary: pd.DataFrame,
) -> None:
    """Vẽ p-value của 5 vị trí."""

    fig, ax = plt.subplots(
        figsize=(11, 6),
        constrained_layout=True,
    )

    positions = summary["position"]
    p_values = summary["p_value"]

    colors = np.where(
        summary["reject_bonferroni"],
        "indianred",
        "steelblue",
    )

    bars = ax.bar(
        positions,
        p_values,
        color=colors,
        width=0.65,
        edgecolor="white",
    )

    bonferroni_threshold = (
        ALPHA / len(summary)
    )

    ax.axhline(
        ALPHA,
        color="darkorange",
        linestyle="--",
        linewidth=1.5,
        label=f"α = {ALPHA}",
    )

    ax.axhline(
        bonferroni_threshold,
        color="red",
        linestyle=":",
        linewidth=2,
        label=(
            "Ngưỡng Bonferroni "
            f"= {bonferroni_threshold:.3f}"
        ),
    )

    ax.bar_label(
        bars,
        labels=[
            f"{value:.4f}"
            for value in p_values
        ],
        padding=4,
        fontsize=10,
    )

    ax.set_title(
        "Kiểm định phân phối đồng đều tại 5 vị trí",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel("Vị trí chữ số")
    ax.set_ylabel("p-value")
    ax.set_xticks(range(1, 6))

    upper_limit = max(
        p_values.max() * 1.2,
        ALPHA * 1.5,
    )

    ax.set_ylim(
        0,
        min(upper_limit, 1),
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output_file = (
        FIGURE_DIR
        / "chi_square_p_values.png"
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


# ============================================================
# Lưu kết quả
# ============================================================

def save_results(
    summary: pd.DataFrame,
    detail: pd.DataFrame,
) -> None:
    """Lưu bảng kết quả kiểm định."""

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_file = (
        TABLE_DIR
        / "digit_uniformity_tests.csv"
    )

    detail_file = (
        TABLE_DIR
        / "digit_uniformity_details.csv"
    )

    summary.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig",
    )

    detail.to_csv(
        detail_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Đã lưu: {summary_file}")
    print(f"Đã lưu: {detail_file}")


# ============================================================
# In kết quả
# ============================================================

def print_results(
    summary: pd.DataFrame,
    detail: pd.DataFrame,
) -> None:
    """In kết quả kiểm định."""

    display_columns = [
        "position",
        "chi_square",
        "p_value",
        "p_bonferroni",
        "p_bh",
        "cohen_w",
        "effect_size",
        "conclusion",
    ]

    print("\n" + "=" * 110)
    print("KIỂM ĐỊNH PHÂN PHỐI ĐỒNG ĐỀU")
    print("=" * 110)

    print(
        summary[display_columns].to_string(
            index=False,
            formatters={
                "chi_square": "{:.4f}".format,
                "p_value": "{:.6f}".format,
                "p_bonferroni": "{:.6f}".format,
                "p_bh": "{:.6f}".format,
                "cohen_w": "{:.4f}".format,
            },
        )
    )

    largest_residuals = (
        detail.assign(
            absolute_residual=lambda data: (
                data["standardized_residual"].abs()
            )
        )
        .sort_values(
            "absolute_residual",
            ascending=False,
        )
        .head(10)
    )

    print("\n10 độ lệch chuẩn hóa lớn nhất:")

    print(
        largest_residuals[
            [
                "position",
                "digit",
                "observed_count",
                "expected_count",
                "deviation",
                "standardized_residual",
            ]
        ].to_string(
            index=False,
            formatters={
                "expected_count": "{:.1f}".format,
                "deviation": "{:+.4%}".format,
                "standardized_residual": "{:+.4f}".format,
            },
        )
    )


# ============================================================
# Chạy chương trình
# ============================================================

def main() -> None:
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = read_data()

    print(f"Đã đọc {len(df):,} kỳ.")

    summary, detail = run_uniformity_tests(df)

    summary = add_interpretation(summary)

    print_results(
        summary,
        detail,
    )

    save_results(
        summary,
        detail,
    )

    plot_p_values(summary)


if __name__ == "__main__":
    main()