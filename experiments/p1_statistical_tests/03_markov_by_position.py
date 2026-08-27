"""Kiểm tra phụ thuộc Markov theo thời gian tại từng vị trí chữ số."""

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
    """Đọc và sắp xếp dữ liệu theo ngày."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {DATA_FILE}."
        )

    df = pd.read_csv(
        DATA_FILE,
        dtype={
            "full_result": str,
            "last_2_digits": str,
        },
        parse_dates=["date"],
    )

    return (
        df.sort_values("date")
        .reset_index(drop=True)
    )


def build_lagged_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Tạo trạng thái kỳ trước và kỳ sau."""

    result = df.copy()

    result["previous_date"] = result["date"].shift(1)

    result["day_gap"] = (
        result["date"]
        - result["previous_date"]
    ).dt.days

    for column in DIGIT_COLUMNS:
        result[f"previous_{column}"] = (
            result[column].shift(1)
        )

    return result.dropna(
        subset=["previous_date"]
    ).copy()


def calculate_cramers_v(
    chi_square: float,
    sample_size: int,
    rows: int,
    columns: int,
) -> float:
    """Tính Cramér's V."""

    denominator = (
        sample_size
        * min(rows - 1, columns - 1)
    )

    if denominator == 0:
        return 0.0

    return np.sqrt(
        chi_square / denominator
    )


def classify_effect(value: float) -> str:
    """Phân loại mức độ phụ thuộc."""

    if value < 0.10:
        return "Không đáng kể"
    if value < 0.20:
        return "Nhỏ"
    if value < 0.40:
        return "Trung bình"

    return "Lớn"


def run_markov_tests(
    lagged_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kiểm định trạng thái trước và sau có độc lập hay không."""

    summary_records = []
    transition_records = []

    analysis_scopes = {
        # Xem chuỗi theo thứ tự kỳ quay
        "all_observed_draws": lagged_df,

        # Chỉ giữ hai kỳ ở hai ngày liên tiếp
        "consecutive_days": lagged_df[
            lagged_df["day_gap"].eq(1)
        ],
    }

    for scope_name, scope_df in analysis_scopes.items():
        for position, column in enumerate(
            DIGIT_COLUMNS,
            start=1,
        ):
            previous_column = f"previous_{column}"

            transition_counts = pd.crosstab(
                scope_df[previous_column].astype(int),
                scope_df[column].astype(int),
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
                transition_counts,
                correction=False,
            )

            cramers_v = calculate_cramers_v(
                chi_square=chi_square,
                sample_size=len(scope_df),
                rows=10,
                columns=10,
            )

            summary_records.append(
                {
                    "scope": scope_name,
                    "position": position,
                    "number_of_transitions": len(scope_df),
                    "chi_square": chi_square,
                    "degrees_of_freedom": degrees_of_freedom,
                    "p_value": p_value,
                    "cramers_v": cramers_v,
                    "minimum_expected_count": expected.min(),
                }
            )

            row_totals = transition_counts.sum(
                axis=1
            )

            transition_probabilities = (
                transition_counts.div(
                    row_totals,
                    axis=0,
                )
            )

            for previous_digit in range(10):
                for next_digit in range(10):
                    count = transition_counts.loc[
                        previous_digit,
                        next_digit,
                    ]

                    probability = (
                        transition_probabilities.loc[
                            previous_digit,
                            next_digit,
                        ]
                    )

                    transition_records.append(
                        {
                            "scope": scope_name,
                            "position": position,
                            "previous_digit": previous_digit,
                            "next_digit": next_digit,
                            "count": count,
                            "probability": probability,
                            "deviation_from_uniform": (
                                probability - 0.10
                            ),
                        }
                    )

    summary = pd.DataFrame(summary_records)
    transitions = pd.DataFrame(
        transition_records
    )

    # Hiệu chỉnh 5 kiểm định riêng trong mỗi phạm vi
    summary["p_bonferroni"] = (
        summary.groupby("scope")["p_value"]
        .transform(
            lambda values: np.minimum(
                values * len(values),
                1,
            )
        )
    )

    summary["reject_raw"] = (
        summary["p_value"] < ALPHA
    )

    summary["reject_bonferroni"] = (
        summary["p_bonferroni"] < ALPHA
    )

    summary["effect_size"] = (
        summary["cramers_v"]
        .apply(classify_effect)
    )

    summary["conclusion"] = np.where(
        summary["reject_bonferroni"],
        "Có bằng chứng phụ thuộc thời gian",
        "Chưa đủ bằng chứng phụ thuộc",
    )

    return summary, transitions


def print_results(
    summary: pd.DataFrame,
) -> None:
    """In kết quả kiểm định Markov."""

    display_columns = [
        "scope",
        "position",
        "number_of_transitions",
        "chi_square",
        "p_value",
        "p_bonferroni",
        "cramers_v",
        "effect_size",
        "conclusion",
    ]

    print("\n" + "=" * 145)
    print("KIỂM ĐỊNH PHỤ THUỘC MARKOV THEO VỊ TRÍ")
    print("=" * 145)

    print(
        summary[display_columns].to_string(
            index=False,
            formatters={
                "chi_square": "{:.4f}".format,
                "p_value": "{:.6f}".format,
                "p_bonferroni": "{:.6f}".format,
                "cramers_v": "{:.4f}".format,
            },
        )
    )

    print("\nTóm tắt:")

    for scope, group in summary.groupby(
        "scope",
        sort=False,
    ):
        print(
            f"- {scope}: "
            f"{group['reject_bonferroni'].sum()} / 5 "
            "vị trí có ý nghĩa sau Bonferroni; "
            f"Cramér's V lớn nhất = "
            f"{group['cramers_v'].max():.4f}"
        )


def plot_transition_deviations(
    transitions: pd.DataFrame,
) -> None:
    """Vẽ độ lệch xác suất chuyển so với 10%."""

    plot_data = transitions[
        transitions["scope"].eq(
            "all_observed_draws"
        )
    ]

    maximum_deviation = (
        plot_data[
            "deviation_from_uniform"
        ]
        .abs()
        .max()
    )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(18, 11),
        constrained_layout=True,
    )

    axes = axes.flatten()

    image = None

    for position in range(1, 6):
        ax = axes[position - 1]

        matrix = (
            plot_data[
                plot_data["position"].eq(position)
            ]
            .pivot(
                index="previous_digit",
                columns="next_digit",
                values="deviation_from_uniform",
            )
            * 100
        )

        image = ax.imshow(
            matrix.values,
            cmap="coolwarm",
            aspect="auto",
            vmin=-maximum_deviation * 100,
            vmax=maximum_deviation * 100,
        )

        ax.set_title(
            f"Vị trí {position}",
            fontsize=14,
            fontweight="bold",
        )

        ax.set_xlabel("Chữ số kỳ sau")
        ax.set_ylabel("Chữ số kỳ trước")

        ax.set_xticks(range(10))
        ax.set_yticks(range(10))

        for row in range(10):
            for column in range(10):
                ax.text(
                    column,
                    row,
                    f"{matrix.iloc[row, column]:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    axes[5].axis("off")

    axes[5].text(
        0.5,
        0.55,
        "Giá trị trong ô",
        ha="center",
        fontsize=15,
        fontweight="bold",
        transform=axes[5].transAxes,
    )

    axes[5].text(
        0.5,
        0.40,
        "Xác suất chuyển quan sát\n"
        "trừ mức tham chiếu 10%\n"
        "(đơn vị: điểm phần trăm)",
        ha="center",
        fontsize=12,
        linespacing=1.5,
        transform=axes[5].transAxes,
    )

    fig.colorbar(
        image,
        ax=axes[:5].tolist(),
        label="Chênh lệch điểm phần trăm",
        shrink=0.85,
    )

    fig.suptitle(
        "Ma trận chuyển trạng thái Markov theo từng vị trí",
        fontsize=19,
        fontweight="bold",
    )

    output_file = (
        FIGURE_DIR
        / "markov_transition_deviations.png"
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
    summary: pd.DataFrame,
    transitions: pd.DataFrame,
) -> None:
    """Lưu kết quả."""

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_file = (
        TABLE_DIR
        / "markov_tests_by_position.csv"
    )

    transition_file = (
        TABLE_DIR
        / "markov_transition_probabilities.csv"
    )

    summary.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig",
    )

    transitions.to_csv(
        transition_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Đã lưu: {summary_file}")
    print(f"Đã lưu: {transition_file}")


def main() -> None:
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = read_data()
    lagged_df = build_lagged_data(df)

    print(f"Số kỳ: {len(df):,}")
    print(
        "Số cặp kỳ quan sát liên tiếp: "
        f"{len(lagged_df):,}"
    )
    print(
        "Số cặp cách nhau đúng một ngày: "
        f"{lagged_df['day_gap'].eq(1).sum():,}"
    )

    summary, transitions = run_markov_tests(
        lagged_df
    )

    print_results(summary)

    save_results(
        summary,
        transitions,
    )

    plot_transition_deviations(
        transitions
    )


if __name__ == "__main__":
    main()