"""Build the final fixed-split report and presentation-ready figures."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import (
    FIGURE_ARTIFACT_DIR,
    FINAL_ARTIFACT_DIR,
    REPORT_ARTIFACT_DIR,
    VALIDATION_ARTIFACT_DIR,
)


REPORT_FILE = REPORT_ARTIFACT_DIR / "final_research_summary.md"


def format_percent(value: float) -> str:
    return f"{value:+.2%}"


def format_money(value: float) -> str:
    return f"{value:,.0f} VND"


def markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def build_log_loss_figure(validation: pd.DataFrame, final: pd.DataFrame) -> Path:
    merged = validation[["model", "log_loss"]].merge(
        final[["model", "log_loss"]],
        on="model",
        suffixes=("_validation", "_final"),
    ).sort_values("log_loss_validation")

    positions = np.arange(len(merged))
    width = 0.38
    figure, axis = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    axis.barh(
        positions - width / 2,
        merged["log_loss_validation"],
        height=width,
        label="Validation 2023–2024",
    )
    axis.barh(
        positions + width / 2,
        merged["log_loss_final"],
        height=width,
        label="Final test 2025–2026",
    )
    axis.axvline(np.log(100), color="black", linestyle="--", linewidth=1, label="Uniform")
    axis.set_yticks(positions, merged["model"])
    axis.invert_yaxis()
    axis.set_xlabel("Log loss (thấp hơn là tốt hơn)")
    axis.set_title("Chất lượng xác suất theo split cố định")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(loc="lower right")
    axis.set_xlim(4.59, max(merged[["log_loss_validation", "log_loss_final"]].max()) + 0.01)

    FIGURE_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURE_ARTIFACT_DIR / "model_log_loss_validation_vs_final.png"
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def build_profit_figure(validation_daily: pd.DataFrame, final_daily: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    phases = [
        (axes[0], validation_daily, "Validation 2023–2024"),
        (axes[1], final_daily, "Final test 2025–2026"),
    ]

    for axis, data, title in phases:
        for (strategy, model), subset in data.groupby(["strategy", "model"], sort=True):
            subset = subset.sort_values("date")
            axis.plot(
                subset["date"],
                subset["cumulative_profit"] / 1_000_000,
                linewidth=1.8,
                label=f"{strategy} — {model}",
            )
        axis.axhline(0, color="black", linewidth=0.9)
        axis.set_title(title)
        axis.set_ylabel("Lợi nhuận lũy kế (triệu VND)")
        axis.grid(alpha=0.25)
        axis.legend(loc="best")

    axes[-1].set_xlabel("Ngày")
    FIGURE_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURE_ARTIFACT_DIR / "cumulative_profit_validation_vs_final.png"
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def main() -> None:
    validation_probability = pd.read_csv(
        VALIDATION_ARTIFACT_DIR / "probability_metrics.csv"
    )
    final_probability = pd.read_csv(FINAL_ARTIFACT_DIR / "probability_metrics.csv")
    validation_selected = pd.read_csv(
        VALIDATION_ARTIFACT_DIR / "selected_strategies.csv"
    )
    final_strategy = pd.read_csv(FINAL_ARTIFACT_DIR / "strategy_results.csv")
    validation_daily = pd.read_csv(
        VALIDATION_ARTIFACT_DIR / "selected_strategy_daily.csv.gz",
        parse_dates=["date"],
    )
    final_daily = pd.read_csv(
        FINAL_ARTIFACT_DIR / "strategy_daily.csv.gz",
        parse_dates=["date"],
    )
    frozen = json.loads(
        (VALIDATION_ARTIFACT_DIR / "frozen_strategy.json").read_text(encoding="utf-8")
    )

    log_loss_figure = build_log_loss_figure(validation_probability, final_probability)
    profit_figure = build_profit_figure(validation_daily, final_daily)

    probability_comparison = validation_probability[["model", "log_loss"]].merge(
        final_probability[["model", "log_loss"]],
        on="model",
        suffixes=("_validation", "_final"),
    )
    probability_comparison["gain_validation_vs_uniform"] = (
        np.log(100) - probability_comparison["log_loss_validation"]
    )
    probability_comparison["gain_final_vs_uniform"] = (
        np.log(100) - probability_comparison["log_loss_final"]
    )
    probability_comparison = probability_comparison.sort_values("log_loss_validation")
    probability_display = probability_comparison.copy()
    for column in probability_display.columns[1:]:
        probability_display[column] = probability_display[column].map(lambda x: f"{x:+.6f}")

    strategy_rows = []
    for _, final_row in final_strategy.iterrows():
        validation_row = validation_selected.loc[
            validation_selected["strategy"].eq(final_row["strategy"])
        ].iloc[0]
        p_value = binomtest(
            int(final_row["n_hits"]),
            int(final_row["n_bets"]),
            float(final_row["break_even_hit_rate"]),
            alternative="greater",
        ).pvalue
        strategy_rows.append(
            {
                "strategy": final_row["strategy"],
                "model": final_row["model"],
                "m": int(final_row["m"]),
                "validation ROI": format_percent(validation_row["roi"]),
                "final bets": int(final_row["n_bets"]),
                "final hits": int(final_row["n_hits"]),
                "final profit": format_money(final_row["total_profit"]),
                "final ROI": format_percent(final_row["roi"]),
                "p vs hòa vốn": f"{p_value:.4f}",
            }
        )
    strategy_display = pd.DataFrame(strategy_rows)

    primary = frozen["primary_strategy"]
    best_validation_model = probability_comparison.iloc[0]["model"]
    best_final_model = probability_comparison.sort_values("log_loss_final").iloc[0]["model"]
    always = final_strategy.loc[final_strategy["strategy"].eq("always_top_m")].iloc[0]
    selective = final_strategy.loc[
        final_strategy["strategy"].eq("selective_top_m")
    ].iloc[0]

    report = f"""# Báo cáo cuối — KQXS theo split cố định

## 1. Protocol

- Train/development: **2002–2022** — 7.561 kỳ.
- Validation/strategy selection: **2023–2024** — 723 kỳ.
- Final test khóa: **2025–2026** — 595 kỳ, đến ngày 23/08/2026.
- Mọi prediction được tạo từ history trước ngày dự đoán; model state chỉ cập nhật sau khi nhận kết quả ngày đó.
- Cấu hình chiến lược được commit trước khi chạy final test. Primary strategy đã khóa: **{primary}**.

## 2. Chất lượng xác suất

{markdown_table(probability_display)}

Model đứng đầu validation theo log loss là **{best_validation_model}**. Model đứng đầu final test là **{best_final_model}**. Chênh lệch so với Uniform rất nhỏ; không có bằng chứng về cải thiện xác suất mạnh và ổn định.

![So sánh log loss](../figures/{log_loss_figure.name})

## 3. Chiến lược được khóa trên validation

Quy tắc chọn: tối đa hóa `Wilson lower 95% − hit rate hòa vốn`, tối thiểu 50 lượt cược; tie-break theo tổng lợi nhuận, `m` nhỏ hơn và tên model. Đã xét 1.250 cấu hình từ 10 model.

{markdown_table(strategy_display)}

![Lợi nhuận lũy kế](../figures/{profit_figure.name})

## 4. Kết quả final test

- **Always Top‑1 CDM:** {int(always['n_hits'])}/{int(always['n_bets'])} ngày trúng, lợi nhuận {format_money(always['total_profit'])}, ROI {format_percent(always['roi'])}. Tuy nhiên Wilson lower ({always['wilson_lower']:.4%}) vẫn thấp hơn mức hòa vốn ({always['break_even_hit_rate']:.4%}), nên kết quả dương chưa phải bằng chứng thống kê chắc chắn.
- **Selective Top‑4 Bayesian Markov:** {int(selective['n_hits'])}/{int(selective['n_bets'])} lượt trúng, lợi nhuận {format_money(selective['total_profit'])}, ROI {format_percent(selective['roi'])}. Lợi thế validation không tái lập trên final test.
- CatBoost static trong final bị blend về Uniform theo quyết định từ validation 2023–2024. Periodic retrain cũng không cải thiện log loss final so với Uniform.

## 5. Kết luận

Kết quả refactor xác nhận rằng lợi nhuận quan sát trên validation có thể không tái lập. Always Top‑1 CDM có lợi nhuận dương trong 595 kỳ final, nhưng khoảng tin cậy vẫn chưa vượt hòa vốn. Selective Bayesian Markov — cấu hình tốt nhất trên validation — chuyển sang ROI âm trong final.

Vì vậy, **chưa có bằng chứng đủ mạnh về predictive/economic edge ổn định**. Kết quả Always Top‑1 nên được coi là một quan sát cần thêm dữ liệu tương lai, không phải bằng chứng để khẳng định chiến lược sinh lời bền vững.
"""

    REPORT_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved: {REPORT_FILE}")
    print(f"Saved: {log_loss_figure}")
    print(f"Saved: {profit_figure}")


if __name__ == "__main__":
    main()

