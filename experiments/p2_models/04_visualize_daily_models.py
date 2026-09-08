"""Tạo toàn bộ biểu đồ cho kết quả Phần 2, chỉ dùng validation 2023–2024."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_DIR / "artifacts" / "p2_models"
FIGURE_DIR = RESULT_DIR / "figures"


def main() -> None:
    summary = pd.read_csv(RESULT_DIR / "model_summary.csv")
    boosted = pd.read_csv(RESULT_DIR / "boosted" / "boosted_summary.csv")
    summary = pd.concat([summary, boosted], ignore_index=True)
    daily = pd.read_csv(RESULT_DIR / "daily_scores.csv.gz")
    ranks = pd.read_csv(RESULT_DIR / "actual_digit_ranks.csv.gz")
    calibration = pd.read_csv(RESULT_DIR / "calibration.csv")
    boosted_calibration = pd.read_csv(RESULT_DIR / "boosted" / "boosted_calibration.csv")
    calibration = pd.concat([calibration, boosted_calibration], ignore_index=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.9)

    order = summary.sort_values("brier_score").model.tolist()
    plt.figure(figsize=(11, 5))
    sns.barplot(data=summary, x="model", y="brier_score", order=order, color="#4472C4")
    plt.xticks(rotation=35, ha="right"); plt.ylabel("Daily multi-label Brier score")
    plt.xlabel(""); plt.title("Phần 2 — Brier score theo model (Validation 2023–2024)")
    plt.tight_layout(); plt.savefig(FIGURE_DIR / "01_brier_by_model.svg"); plt.close()

    positions = ["ten_thousands", "thousands", "hundreds", "tens", "units"]
    position_table = summary.set_index("model")[[f"brier_{p}" for p in positions]].rename(columns=lambda x: x.replace("brier_", ""))
    plt.figure(figsize=(10, 5)); sns.heatmap(position_table.loc[order], annot=True, fmt=".3f", cmap="YlOrRd")
    plt.xlabel("Vị trí chữ số"); plt.ylabel(""); plt.title("Brier score theo vị trí và model")
    plt.tight_layout(); plt.savefig(FIGURE_DIR / "02_brier_by_position.svg"); plt.close()

    top_cols = ["recall_top1", "recall_top3", "recall_top5", "recall_top10"]
    top_table = summary.set_index("model")[top_cols].rename(columns={c: c.replace("recall_", "Top ") for c in top_cols})
    plt.figure(figsize=(10, 5)); top_table.loc[order].plot(kind="bar", ax=plt.gca())
    plt.ylabel("Recall chữ số thực tế"); plt.xlabel(""); plt.title("Recall theo Top-k")
    plt.xticks(rotation=35, ha="right"); plt.legend(title="Ranking")
    plt.tight_layout(); plt.savefig(FIGURE_DIR / "03_topk_recall.svg"); plt.close()

    plt.figure(figsize=(8, 5))
    for model in order:
        part = calibration[calibration.model.eq(model)]
        plt.plot(part.mean_predicted, part.observed_rate, marker="o", label=model)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    plt.xlabel("Xác suất dự đoán trung bình"); plt.ylabel("Tỷ lệ thực tế")
    plt.title("Calibration của các model"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIGURE_DIR / "04_calibration.svg"); plt.close()

    plt.figure(figsize=(9, 5))
    for model in order:
        part = ranks[ranks.model.eq(model)].sort_values("rank")
        cdf = part.assign(cdf=lambda x: range(1, len(x) + 1))
        plt.step(cdf["rank"], cdf["cdf"] / len(cdf), where="post", label=model)
    plt.xlabel("Rank của chữ số thực tế"); plt.ylabel("Tỷ lệ tích lũy")
    plt.title("Phân phối rank chữ số thực tế"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIGURE_DIR / "05_actual_digit_rank_cdf.svg"); plt.close()
    print(f"Đã ghi 5 biểu đồ vào {FIGURE_DIR}")


if __name__ == "__main__":
    main()
