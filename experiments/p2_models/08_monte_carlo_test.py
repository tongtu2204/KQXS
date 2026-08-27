"""Kiểm định ý nghĩa của cải thiện log loss qua các năm backtest."""

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR
    / "artifacts"
    / "tables"
    / "walk_forward_yearly_metrics.csv"
)

TABLE_DIR = PROJECT_DIR / "artifacts" / "tables"
FIGURE_DIR = PROJECT_DIR / "artifacts" / "figures"

NUMBER_OF_BOOTSTRAPS = 20_000
RANDOM_STATE = 42
ALPHA = 0.05


def read_data() -> pd.DataFrame:
    """Đọc kết quả walk-forward theo từng năm."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {INPUT_FILE}. "
            "Hãy chạy 07_walk_forward_backtest.py trước."
        )

    return pd.read_csv(INPUT_FILE)


def benjamini_hochberg(
    p_values: pd.Series,
) -> pd.Series:
    """Hiệu chỉnh p-value Benjamini-Hochberg."""

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


def exact_sign_flip_test(
    improvements: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Kiểm định hoán vị một phía.

    H0: mô hình không cải thiện log loss so với Uniform.
    H1: mô hình có log loss thấp hơn Uniform.
    """

    observed_statistic = np.average(
        improvements,
        weights=weights,
    )

    permutation_statistics = []

    for signs in product(
        [-1, 1],
        repeat=len(improvements),
    ):
        signed_improvements = (
            improvements
            * np.asarray(signs)
        )

        statistic = np.average(
            signed_improvements,
            weights=weights,
        )

        permutation_statistics.append(
            statistic
        )

    permutation_statistics = np.asarray(
        permutation_statistics
    )

    return np.mean(
        permutation_statistics
        >= observed_statistic - 1e-15
    )


def bootstrap_confidence_interval(
    improvements: np.ndarray,
    weights: np.ndarray,
    random_generator: np.random.Generator,
) -> tuple[float, float]:
    """Bootstrap khoảng tin cậy của cải thiện log loss."""

    number_of_years = len(improvements)

    bootstrap_statistics = np.empty(
        NUMBER_OF_BOOTSTRAPS
    )

    for iteration in range(
        NUMBER_OF_BOOTSTRAPS
    ):
        sampled_indices = (
            random_generator.integers(
                0,
                number_of_years,
                size=number_of_years,
            )
        )

        bootstrap_statistics[iteration] = (
            np.average(
                improvements[sampled_indices],
                weights=weights[sampled_indices],
            )
        )

    lower_bound, upper_bound = np.quantile(
        bootstrap_statistics,
        [0.025, 0.975],
    )

    return lower_bound, upper_bound


def run_tests(
    yearly_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """So sánh từng mô hình với Uniform."""

    uniform = (
        yearly_metrics[
            yearly_metrics["model"].eq("Uniform")
        ][
            [
                "test_year",
                "position",
                "log_loss",
            ]
        ]
        .rename(
            columns={
                "log_loss": "uniform_log_loss",
            }
        )
    )

    comparison = yearly_metrics.merge(
        uniform,
        on=[
            "test_year",
            "position",
        ],
        how="left",
    )

    comparison = comparison[
        comparison["model"].ne("Uniform")
    ].copy()

    comparison["improvement"] = (
        comparison["uniform_log_loss"]
        - comparison["log_loss"]
    )

    random_generator = (
        np.random.default_rng(
            RANDOM_STATE
        )
    )

    records = []

    for (
        model_name,
        position,
    ), group in comparison.groupby(
        [
            "model",
            "position",
        ]
    ):
        group = group.sort_values(
            "test_year"
        )

        improvements = (
            group["improvement"]
            .to_numpy()
        )

        weights = (
            group["number_of_test_draws"]
            .to_numpy()
        )

        mean_improvement = np.average(
            improvements,
            weights=weights,
        )

        lower_bound, upper_bound = (
            bootstrap_confidence_interval(
                improvements,
                weights,
                random_generator,
            )
        )

        permutation_p_value = (
            exact_sign_flip_test(
                improvements,
                weights,
            )
        )

        records.append(
            {
                "model": model_name,
                "position": position,
                "number_of_years": len(group),
                "years_better_than_uniform": int(
                    (
                        improvements > 1e-12
                    ).sum()
                ),
                "mean_log_loss_improvement": (
                    mean_improvement
                ),
                "relative_improvement_percent": (
                    mean_improvement
                    / np.log(10)
                    * 100
                ),
                "bootstrap_ci_low": lower_bound,
                "bootstrap_ci_high": upper_bound,
                "permutation_p_value": (
                    permutation_p_value
                ),
            }
        )

    result = pd.DataFrame(records)

    number_of_tests = len(result)

    result["p_bonferroni"] = np.minimum(
        result["permutation_p_value"]
        * number_of_tests,
        1,
    )

    result["p_bh"] = benjamini_hochberg(
        result["permutation_p_value"]
    )

    result["significant_bh"] = (
        result["p_bh"] < ALPHA
    )

    result["ci_excludes_zero"] = (
        result["bootstrap_ci_low"] > 0
    )

    result["conclusion"] = np.where(
        result["significant_bh"]
        & result["ci_excludes_zero"],
        "Cải thiện có ý nghĩa",
        "Chưa đủ bằng chứng cải thiện",
    )

    return result.sort_values(
        "mean_log_loss_improvement",
        ascending=False,
    ).reset_index(drop=True)


def print_results(
    result: pd.DataFrame,
) -> None:
    """In kết quả kiểm định."""

    print("\n" + "=" * 150)
    print("KIỂM ĐỊNH CẢI THIỆN SO VỚI UNIFORM")
    print("=" * 150)

    print(
        result[
            [
                "model",
                "position",
                "years_better_than_uniform",
                "number_of_years",
                "mean_log_loss_improvement",
                "relative_improvement_percent",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "permutation_p_value",
                "p_bonferroni",
                "p_bh",
                "conclusion",
            ]
        ].to_string(
            index=False,
            formatters={
                "mean_log_loss_improvement": (
                    "{:+.8f}".format
                ),
                "relative_improvement_percent": (
                    "{:+.5f}%".format
                ),
                "bootstrap_ci_low": (
                    "{:+.8f}".format
                ),
                "bootstrap_ci_high": (
                    "{:+.8f}".format
                ),
                "permutation_p_value": (
                    "{:.6f}".format
                ),
                "p_bonferroni": (
                    "{:.6f}".format
                ),
                "p_bh": (
                    "{:.6f}".format
                ),
            },
        )
    )


def plot_results(
    result: pd.DataFrame,
) -> None:
    """Vẽ cải thiện và khoảng tin cậy bootstrap."""

    plot_data = result.sort_values(
        "mean_log_loss_improvement"
    ).reset_index(drop=True)

    labels = [
        f"{model} – vị trí {position}"
        for model, position in zip(
            plot_data["model"],
            plot_data["position"],
        )
    ]

    means = (
        plot_data[
            "mean_log_loss_improvement"
        ].to_numpy()
        * 1_000
    )

    lower_errors = (
        plot_data[
            "mean_log_loss_improvement"
        ].to_numpy()
        - plot_data[
            "bootstrap_ci_low"
        ].to_numpy()
    ) * 1_000

    upper_errors = (
        plot_data[
            "bootstrap_ci_high"
        ].to_numpy()
        - plot_data[
            "mean_log_loss_improvement"
        ].to_numpy()
    ) * 1_000

    colors = np.where(
        plot_data["significant_bh"],
        "seagreen",
        "steelblue",
    )

    fig, ax = plt.subplots(
        figsize=(14, 10),
        constrained_layout=True,
    )

    positions = np.arange(
        len(plot_data)
    )

    ax.errorbar(
        means,
        positions,
        xerr=[
            lower_errors,
            upper_errors,
        ],
        fmt="none",
        ecolor="gray",
        capsize=4,
        linewidth=1.5,
    )

    ax.scatter(
        means,
        positions,
        color=colors,
        s=55,
        zorder=3,
    )

    ax.axvline(
        0,
        color="red",
        linestyle="--",
        linewidth=1.5,
    )

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)

    ax.set_xlabel(
        "Cải thiện log loss × 1.000\n"
        "(dương là tốt hơn Uniform)"
    )

    ax.set_title(
        "Permutation test và bootstrap theo năm",
        fontsize=16,
        fontweight="bold",
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    output_file = (
        FIGURE_DIR
        / "monte_carlo_significance.png"
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

    yearly_metrics = read_data()

    result = run_tests(
        yearly_metrics
    )

    print_results(result)

    output_file = (
        TABLE_DIR
        / "monte_carlo_significance.csv"
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