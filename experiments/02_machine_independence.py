"""Kiểm định tính độc lập giữa 5 vị trí trong cùng một kỳ quay."""

from itertools import combinations
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

ALPHA = 0.05


def read_data() -> pd.DataFrame:
    """Đọc dữ liệu đã chuẩn hóa."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {DATA_FILE}."
        )

    return pd.read_csv(
        DATA_FILE,
        dtype={
            "full_result": str,
            "last_2_digits": str,
        },
        parse_dates=["date"],
    )


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


def calculate_cramers_v(
    chi_square: float,
    sample_size: int,
    number_of_rows: int,
    number_of_columns: int,
) -> float:
    """Tính Cramér's V."""

    denominator = (
        sample_size
        * min(
            number_of_rows - 1,
            number_of_columns - 1,
        )
    )

    if denominator == 0:
        return 0.0

    return np.sqrt(
        chi_square / denominator
    )


def classify_cramers_v(value: float) -> str:
    """Phân loại mức độ liên hệ."""

    if value < 0.10:
        return "Không đáng kể"
    if value < 0.20:
        return "Nhỏ"
    if value < 0.40:
        return "Trung bình"

    return "Lớn"


def run_independence_tests(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Kiểm định độc lập cho 10 cặp vị trí."""

    records = []

    for first_column, second_column in combinations(
        DIGIT_COLUMNS,
        2,
    ):
        first_position = int(
            first_column.split("_")[1]
        )

        second_position = int(
            second_column.split("_")[1]
        )

        contingency_table = pd.crosstab(
            df[first_column],
            df[second_column],
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
            contingency_table,
            correction=False,
        )

        cramers_v = calculate_cramers_v(
            chi_square=chi_square,
            sample_size=len(df),
            number_of_rows=contingency_table.shape[0],
            number_of_columns=contingency_table.shape[1],
        )

        minimum_expected = expected.min()

        records.append(
            {
                "position_1": first_position,
                "position_2": second_position,
                "sample_size": len(df),
                "chi_square": chi_square,
                "degrees_of_freedom": degrees_of_freedom,
                "p_value": p_value,
                "cramers_v": cramers_v,
                "minimum_expected_count": minimum_expected,
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

    result["effect_size"] = (
        result["cramers_v"]
        .apply(classify_cramers_v)
    )

    result["conclusion"] = np.where(
        result["reject_bonferroni"],
        "Có bằng chứng phụ thuộc",
        "Chưa đủ bằng chứng phụ thuộc",
    )

    return result


def print_results(
    result: pd.DataFrame,
) -> None:
    """In kết quả kiểm định."""

    display_columns = [
        "position_1",
        "position_2",
        "chi_square",
        "p_value",
        "p_bonferroni",
        "p_bh",
        "cramers_v",
        "effect_size",
        "conclusion",
    ]

    print("\n" + "=" * 125)
    print("KIỂM ĐỊNH ĐỘC LẬP GIỮA CÁC VỊ TRÍ")
    print("=" * 125)

    print(
        result[display_columns].to_string(
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

    print(
        "\nSố cặp bác bỏ H0 trước hiệu chỉnh: "
        f"{result['reject_raw'].sum()} / {len(result)}"
    )

    print(
        "Số cặp bác bỏ H0 sau Bonferroni: "
        f"{result['reject_bonferroni'].sum()} / {len(result)}"
    )

    print(
        "Cramér's V lớn nhất: "
        f"{result['cramers_v'].max():.4f}"
    )

    print(
        "Expected count nhỏ nhất: "
        f"{result['minimum_expected_count'].min():.2f}"
    )


def plot_cramers_v_matrix(
    result: pd.DataFrame,
) -> None:
    """Vẽ ma trận Cramér's V giữa 5 vị trí."""

    matrix = np.zeros((5, 5))

    for row in result.itertuples():
        first_index = row.position_1 - 1
        second_index = row.position_2 - 1

        matrix[first_index, second_index] = (
            row.cramers_v
        )

        matrix[second_index, first_index] = (
            row.cramers_v
        )

    fig, ax = plt.subplots(
        figsize=(9, 7),
        constrained_layout=True,
    )

    image = ax.imshow(
        matrix,
        cmap="YlOrRd",
        vmin=0,
        vmax=max(
            matrix.max() * 1.15,
            0.10,
        ),
    )

    labels = [
        f"Vị trí {position}"
        for position in range(1, 6)
    ]

    ax.set_xticks(range(5))
    ax.set_xticklabels(labels)

    ax.set_yticks(range(5))
    ax.set_yticklabels(labels)

    ax.set_title(
        "Mức độ liên hệ giữa 5 vị trí – Cramér's V",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    for row in range(5):
        for column in range(5):
            if row == column:
                label = "—"
            else:
                label = f"{matrix[row, column]:.3f}"

            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=11,
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Cramér's V",
    )

    output_file = (
        FIGURE_DIR
        / "machine_independence_cramers_v.png"
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
    """Lưu kết quả ra CSV."""

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        TABLE_DIR
        / "machine_independence_tests.csv"
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

    result = run_independence_tests(df)

    print_results(result)
    save_results(result)
    plot_cramers_v_matrix(result)


if __name__ == "__main__":
    main()