"""Đọc dữ liệu KQXSMB và tạo dashboard kiểm tra phân phối 00-99."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_latest_csv(data_dir: str = "data") -> pd.DataFrame:
    files = sorted(Path(data_dir).glob("kqxsmb_*.csv"))
    if not files:
        raise FileNotFoundError("Không tìm thấy data/kqxsmb_*.csv")

    file = files[-1]
    df = pd.read_csv(
        file,
        dtype={"full_result": str, "last_2_digits": str},
        parse_dates=["date"],
    )
    df["last_2_digits"] = df["last_2_digits"].str.zfill(2)
    print(f"Đã đọc: {file} | {len(df):,} kỳ")
    return df


def visualize(df: pd.DataFrame, output_dir: str = "artifacts/figures") -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    numbers = [f"{i:02d}" for i in range(100)]
    frequency = df["last_2_digits"].value_counts().reindex(numbers, fill_value=0)
    expected = len(df) / 100

    tens = df["last_2_digits"].str[0].astype(int)
    units = df["last_2_digits"].str[1].astype(int)
    digit_matrix = pd.crosstab(tens, units).reindex(
        index=range(10), columns=range(10), fill_value=0
    )

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    axes[0, 0].bar(range(100), frequency.values, color="steelblue", width=0.85)
    axes[0, 0].axhline(expected, color="red", linestyle="--", label=f"Kỳ vọng: {expected:.1f}")
    axes[0, 0].set_xticks(range(0, 100, 5), [f"{i:02d}" for i in range(0, 100, 5)])
    axes[0, 0].set_title("Tần suất hai chữ số cuối 00–99")
    axes[0, 0].set_xlabel("Hai chữ số cuối")
    axes[0, 0].set_ylabel("Số lần xuất hiện")
    axes[0, 0].legend()

    image = axes[0, 1].imshow(digit_matrix.values, cmap="YlOrRd", aspect="auto")
    axes[0, 1].set_title("Phân phối chữ số hàng chục và hàng đơn vị")
    axes[0, 1].set_xlabel("Hàng đơn vị")
    axes[0, 1].set_ylabel("Hàng chục")
    axes[0, 1].set_xticks(range(10))
    axes[0, 1].set_yticks(range(10))
    for i in range(10):
        for j in range(10):
            axes[0, 1].text(j, i, digit_matrix.iloc[i, j], ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes[0, 1], label="Số lần xuất hiện")

    x = np.arange(10)
    width = 0.4
    axes[1, 0].bar(x - width / 2, tens.value_counts().reindex(range(10), fill_value=0), width, label="Hàng chục")
    axes[1, 0].bar(x + width / 2, units.value_counts().reindex(range(10), fill_value=0), width, label="Hàng đơn vị")
    axes[1, 0].set_title("Tần suất từng chữ số 0–9")
    axes[1, 0].set_xlabel("Chữ số")
    axes[1, 0].set_ylabel("Số lần xuất hiện")
    axes[1, 0].set_xticks(x)
    axes[1, 0].legend()

    yearly = df.groupby(df["date"].dt.year).size()
    axes[1, 1].plot(yearly.index, yearly.values, marker="o", color="darkgreen")
    axes[1, 1].set_title("Số kỳ thu thập theo năm")
    axes[1, 1].set_xlabel("Năm")
    axes[1, 1].set_ylabel("Số kỳ")
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle(f"Tổng quan KQXSMB – {len(df):,} kỳ", fontsize=18, fontweight="bold")
    fig.tight_layout()
    figure_path = output / "kqxsmb_overview.png"
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.show()

    frequency.rename_axis("last_2_digits").reset_index(name="frequency").to_csv(
        output / "number_frequency.csv", index=False, encoding="utf-8-sig"
    )
    print(f"Đã lưu: {figure_path}")


if __name__ == "__main__":
    visualize(read_latest_csv())
