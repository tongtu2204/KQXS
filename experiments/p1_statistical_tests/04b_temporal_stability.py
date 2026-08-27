"""Kiểm tra tính ổn định của tín hiệu vị trí 4, độ trễ 19 kỳ."""

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

POSITION = 4
LAG = 19
ALPHA = 0.05


def read_data() -> pd.DataFrame:
    """Đọc dữ liệu theo thứ tự thời gian."""

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


def calculate_test(
    df: pd.DataFrame,
    period_name: str,
) -> dict:
    """Kiểm định phụ thuộc tại vị trí 4, lag 19."""

    column = f"digit_{POSITION}"

    current = df[column]
    previous = current.shift(LAG)

    valid = previous.notna()

    current = current[valid].astype(int)
    previous = previous[valid].astype(int)

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

    cramers_v = np.sqrt(
        chi_square
        / (sample_size * 9)
    )

    transition_probability = table.div(
        table.sum(axis=1),
        axis=0,
    )

    maximum_deviation = (
        transition_probability
        .sub(0.10)
        .abs()
        .to_numpy()
        .max()
    )

    return {
        "period": period_name,
        "start_date": df["date"].min(),
        "end_date": df["date"].max(),
        "number_of_draws": len(df),
        "number_of_transitions": sample_size,
        "chi_square": chi_square,
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": p_value,
        "cramers_v": cramers_v,
        "maximum_transition_deviation": maximum_deviation,
        "minimum_expected_count": expected.min(),
    }


def run_stability_tests(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Chia dữ liệu thành các giai đoạn độc lập."""

    periods = {
        "2002-2009": df[
            df["date"].dt.year.between(2002, 2009)
        ],
        "2010-2017": df[
            df["date"].dt.year.between(2010, 2017)
        ],
        "2018-2026": df[
            df["date"].dt.year.between(2018, 2026)
        ],
    }

    records = []

    # Kết quả toàn bộ mẫu để tham chiếu
    records.append(
        calculate_test(
            df,
            "Toàn bộ",
        )
    )

    for period_name, period_df in periods.items():
        records.append(
            calculate_test(
                period_df.reset_index(drop=True),
                period_name,
            )
        )

    result = pd.DataFrame(records)

    # Hiệu chỉnh cho ba giai đoạn con
    period_mask = result["period"].ne("Toàn bộ")

    result["p_bonferroni_periods"] = np.nan

    result.loc[
        period_mask,
        "p_bonferroni_periods",
    ] = np.minimum(
        result.loc[period_mask, "p_value"] * 3,
        1,
    )

    result["significant"] = (
        result["p_value"] < ALPHA
    )

    result["significant_after_correction"] = (
        result["p_bonferroni_periods"] < ALPHA
    ).fillna(False)

    return result


def print_results(
    result: pd.DataFrame,
) -> None:
    """In kết quả ổn định theo thời gian."""

    print("\n" + "=" * 115)
    print(
        f"ỔN ĐỊNH CỦA TÍN HIỆU: "
        f"VỊ TRÍ {POSITION}, LAG {LAG}"
    )
    print("=" * 115)

    print(
        result[
            [
                "period",
                "number_of_transitions",
                "chi_square",
                "p_value",
                "p_bonferroni_periods",
                "cramers_v",
                "maximum_transition_deviation",
            ]
        ].to_string(
            index=False,
            formatters={
                "chi_square": "{:.4f}".format,
                "p_value": "{:.6f}".format,
                "p_bonferroni_periods": (
                    lambda value:
                    ""
                    if pd.isna(value)
                    else f"{value:.6f}"
                ),
                "cramers_v": "{:.4f}".format,
                "maximum_transition_deviation": (
                    "{:.4%}".format
                ),
            },
        )
    )


def plot_results(
    result: pd.DataFrame,
) -> None:
    """So sánh p-value và Cramér's V giữa các giai đoạn."""

    period_result = result[
        result["period"].ne("Toàn bộ")
    ]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(14, 5),
        constrained_layout=True,
    )

    axes[0].bar(
        period_result["period"],
        period_result["p_value"],
        color="steelblue",
    )

    axes[0].axhline(
        ALPHA,
        color="red",
        linestyle="--",
        label="α = 0.05",
    )

    axes[0].set_title("p-value theo giai đoạn")
    axes[0].set_ylabel("p-value")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        period_result["period"],
        period_result["cramers_v"],
        color="darkorange",
    )

    axes[1].axhline(
        0.10,
        color="red",
        linestyle="--",
        label="Ngưỡng hiệu ứng nhỏ = 0.10",
    )

    axes[1].set_title("Cramér's V theo giai đoạn")
    axes[1].set_ylabel("Cramér's V")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"Độ ổn định của tín hiệu vị trí {POSITION}, lag {LAG}",
        fontsize=16,
        fontweight="bold",
    )

    output_file = (
        FIGURE_DIR
        / "temporal_signal_stability.png"
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


def main() -> None:
    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = read_data()

    result = run_stability_tests(df)

    print_results(result)

    output_file = (
        TABLE_DIR
        / "temporal_signal_stability.csv"
    )

    result.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nĐã lưu: {output_file}")

    plot_results(result)


if __name__ == "__main__":
    main()