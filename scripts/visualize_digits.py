"""Trực quan hóa phân phối 5 vị trí chữ số của giải đặc biệt."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "kqxsmb_digits.csv"
)

FIGURE_DIR = PROJECT_DIR / "artifacts" / "figures"
TABLE_DIR = PROJECT_DIR / "artifacts" / "tables"

DIGIT_COLUMNS = [
    "digit_1",
    "digit_2",
    "digit_3",
    "digit_4",
    "digit_5",
]


def read_data() -> pd.DataFrame:
    """Đọc dữ liệu đã chuẩn hóa."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {DATA_FILE}. "
            "Hãy chạy prepare_digits.py trước."
        )

    return pd.read_csv(
        DATA_FILE,
        dtype={
            "full_result": str,
            "last_2_digits": str,
        },
        parse_dates=["date"],
    )


def build_frequency_table(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo bảng tần suất chữ số 0-9 ở từng vị trí."""

    records = []

    for position, column in enumerate(DIGIT_COLUMNS, start=1):
        counts = (
            df[column]
            .value_counts()
            .reindex(range(10), fill_value=0)
        )

        for digit, count in counts.items():
            records.append(
                {
                    "position": position,
                    "digit": digit,
                    "count": count,
                    "proportion": count / len(df),
                }
            )

    return pd.DataFrame(records)


def plot_frequency_by_position(
    frequency: pd.DataFrame,
    number_of_draws: int,
) -> None:
    """Vẽ phân phối 0-9 riêng cho từng vị trí."""

    expected_count = number_of_draws / 10

    colors = [
        "steelblue",
        "darkorange",
        "seagreen",
        "mediumpurple",
        "indianred",
    ]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(19, 11),
        constrained_layout=True,
    )

    axes = axes.flatten()

    for position in range(1, 6):
        ax = axes[position - 1]

        subset = frequency[
            frequency["position"].eq(position)
        ]

        bars = ax.bar(
            subset["digit"],
            subset["count"],
            color=colors[position - 1],
            alpha=0.85,
            width=0.72,
            edgecolor="white",
        )

        ax.axhline(
            expected_count,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Kỳ vọng = {expected_count:.1f}",
        )

        # Hiển thị số lượng phía trên từng cột
        ax.bar_label(
            bars,
            labels=[
                f"{value:,}"
                for value in subset["count"]
            ],
            padding=3,
            fontsize=9,
        )

        max_count = subset["count"].max()

        ax.set_ylim(
            0,
            max_count * 1.14,
        )

        ax.set_title(
            f"Vị trí {position}",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )

        ax.set_xlabel(
            "Chữ số",
            fontsize=11,
            labelpad=8,
        )

        ax.set_ylabel(
            "Số lần xuất hiện",
            fontsize=11,
            labelpad=8,
        )

        ax.set_xticks(range(10))
        ax.tick_params(
            axis="both",
            labelsize=10,
        )

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.25,
        )

        ax.legend(
            loc="lower right",
            fontsize=9,
            frameon=True,
        )

        # Bỏ đường viền phía trên và bên phải
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Ô thứ sáu dùng làm phần chú thích chung
    axes[5].axis("off")

    axes[5].text(
        0.5,
        0.62,
        "Mốc tham chiếu",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        transform=axes[5].transAxes,
    )

    axes[5].text(
        0.5,
        0.43,
        "Nếu các chữ số phân phối đồng đều,\n"
        f"mỗi chữ số kỳ vọng xuất hiện khoảng\n"
        f"{expected_count:.1f} lần tại mỗi vị trí.",
        ha="center",
        va="center",
        fontsize=13,
        linespacing=1.5,
        transform=axes[5].transAxes,
    )

    fig.suptitle(
        "Phân phối chữ số tại 5 vị trí giải đặc biệt",
        fontsize=20,
        fontweight="bold",
    )

    output_file = (
        FIGURE_DIR
        / "digit_frequency_by_position.png"
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


def plot_frequency_heatmap(
    frequency: pd.DataFrame,
) -> None:
    """Vẽ heatmap tỷ lệ chữ số theo vị trí."""

    matrix = (
        frequency.pivot(
            index="position",
            columns="digit",
            values="proportion",
        )
        * 100
    )

    fig, ax = plt.subplots(
        figsize=(14, 6),
    )

    image = ax.imshow(
        matrix.values,
        cmap="YlOrRd",
        aspect="auto",
    )

    ax.set_title(
        "Tỷ lệ xuất hiện chữ số theo vị trí (%)",
        fontsize=16,
        fontweight="bold",
    )

    ax.set_xlabel("Chữ số")
    ax.set_ylabel("Vị trí máy quay")

    ax.set_xticks(range(10))
    ax.set_xticklabels(range(10))

    ax.set_yticks(range(5))
    ax.set_yticklabels(
        [f"Vị trí {i}" for i in range(1, 6)]
    )

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]

            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Tỷ lệ (%)",
    )

    fig.tight_layout()

    output_file = (
        FIGURE_DIR
        / "digit_frequency_heatmap.png"
    )

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print(f"Đã lưu: {output_file}")


def plot_deviation_from_uniform(
    frequency: pd.DataFrame,
) -> None:
    """Vẽ độ lệch so với xác suất đồng đều 10%."""

    matrix = frequency.pivot(
        index="position",
        columns="digit",
        values="proportion",
    )

    deviation = (matrix - 0.10) * 100

    max_deviation = np.abs(
        deviation.values
    ).max()

    fig, ax = plt.subplots(
        figsize=(14, 6),
    )

    image = ax.imshow(
        deviation.values,
        cmap="coolwarm",
        aspect="auto",
        vmin=-max_deviation,
        vmax=max_deviation,
    )

    ax.set_title(
        "Độ lệch so với phân phối đồng đều 10%",
        fontsize=16,
        fontweight="bold",
    )

    ax.set_xlabel("Chữ số")
    ax.set_ylabel("Vị trí máy quay")

    ax.set_xticks(range(10))
    ax.set_xticklabels(range(10))

    ax.set_yticks(range(5))
    ax.set_yticklabels(
        [f"Vị trí {i}" for i in range(1, 6)]
    )

    for row in range(deviation.shape[0]):
        for column in range(deviation.shape[1]):
            value = deviation.iloc[row, column]

            ax.text(
                column,
                row,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Chênh lệch điểm phần trăm",
    )

    fig.tight_layout()

    output_file = (
        FIGURE_DIR
        / "digit_deviation_from_uniform.png"
    )

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print(f"Đã lưu: {output_file}")


def save_frequency_table(
    frequency: pd.DataFrame,
) -> None:
    """Lưu bảng tần suất để sử dụng trong thí nghiệm."""

    output_file = (
        TABLE_DIR
        / "digit_frequency_by_position.csv"
    )

    frequency.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Đã lưu: {output_file}")


def print_largest_deviations(
    frequency: pd.DataFrame,
) -> None:
    """In những chữ số lệch nhiều nhất so với 10%."""

    result = frequency.copy()

    result["deviation"] = (
        result["proportion"] - 0.10
    )

    result["absolute_deviation"] = (
        result["deviation"].abs()
    )

    largest = (
        result.sort_values(
            "absolute_deviation",
            ascending=False,
        )
        .head(10)
    )

    print("\n10 độ lệch lớn nhất:")

    print(
        largest[
            [
                "position",
                "digit",
                "count",
                "proportion",
                "deviation",
            ]
        ].to_string(
            index=False,
            formatters={
                "proportion": "{:.4%}".format,
                "deviation": "{:+.4%}".format,
            },
        )
    )


def main() -> None:
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = read_data()

    print(f"Đã đọc {len(df):,} kỳ.")

    frequency = build_frequency_table(df)

    save_frequency_table(frequency)
    print_largest_deviations(frequency)

    plot_frequency_by_position(
        frequency,
        number_of_draws=len(df),
    )

    plot_frequency_heatmap(frequency)
    plot_deviation_from_uniform(frequency)


if __name__ == "__main__":
    main()