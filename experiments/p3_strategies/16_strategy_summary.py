"""
Strategy 16: Robustness check for Selective Top-m.

Kiểm tra:
1. Best observed selective configuration.
2. Performance theo từng fold.
3. Binomial test so với random hit rate m/100.
4. Exact / Wilson confidence interval.
5. Bonferroni correction do đã search nhiều configurations.
6. So sánh hit rate với break-even.
"""

from pathlib import Path
from math import sqrt

import numpy as np
import pandas as pd

try:
    from scipy.stats import binomtest
except ImportError:
    binomtest = None


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

STRATEGY_DIR = (
    PROJECT_DIR
    / "artifacts"
    / "strategies"
)

INPUT_SUMMARY = (
    STRATEGY_DIR
    / "selective_topm_summary.csv"
)

INPUT_DAILY = (
    STRATEGY_DIR
    / "selective_topm_daily.csv"
)

OUTPUT_BEST = (
    STRATEGY_DIR
    / "selective_robustness_best.csv"
)

OUTPUT_FOLD = (
    STRATEGY_DIR
    / "selective_robustness_by_fold.csv"
)

OUTPUT_TOP = (
    STRATEGY_DIR
    / "selective_robustness_top_configs.csv"
)

NUMBER_OF_CLASSES = 100

COST_PER_NUMBER = 10_000
PAYOUT_IF_HIT = 800_000

ALPHA = 0.05

# số cấu hình đã thử:
# 8 models * 30 m * 4 participation rates
N_CONFIGS = 8 * 30 * 4


# ============================================================
# WILSON CI
# ============================================================

def wilson_interval(
    hits: int,
    n: int,
    z: float = 1.959963984540054,
):
    """
    Wilson 95% CI cho binomial proportion.
    """

    if n == 0:
        return np.nan, np.nan

    p_hat = hits / n

    denom = 1 + z**2 / n

    center = (
        p_hat
        + z**2 / (2 * n)
    ) / denom

    margin = (
        z
        * sqrt(
            p_hat * (1 - p_hat) / n
            + z**2 / (4 * n**2)
        )
        / denom
    )

    return (
        max(0.0, center - margin),
        min(1.0, center + margin),
    )


# ============================================================
# BINOMIAL TEST
# ============================================================

def one_sided_binomial_pvalue(
    hits: int,
    n: int,
    p0: float,
):
    """
    H0: p = p0
    H1: p > p0
    """

    if n == 0:
        return np.nan

    if binomtest is None:
        return np.nan

    result = binomtest(
        k=hits,
        n=n,
        p=p0,
        alternative="greater",
    )

    return float(
        result.pvalue
    )


# ============================================================
# LOAD
# ============================================================

def load_data():

    if not INPUT_SUMMARY.exists():
        raise FileNotFoundError(
            INPUT_SUMMARY
        )

    if not INPUT_DAILY.exists():
        raise FileNotFoundError(
            INPUT_DAILY
        )

    summary = pd.read_csv(
        INPUT_SUMMARY
    )

    daily = pd.read_csv(
        INPUT_DAILY,
        parse_dates=["date"],
    )

    return summary, daily


# ============================================================
# GET BEST OBSERVED CONFIG
# ============================================================

def get_best_observed(
    summary: pd.DataFrame,
):

    valid = (
        summary
        .loc[
            summary[
                "roi"
            ].notna()
        ]
        .copy()
    )

    best = (
        valid
        .sort_values(
            "roi",
            ascending=False,
        )
        .iloc[0]
    )

    return best


# ============================================================
# EVALUATE CONFIG
# ============================================================

def evaluate_configuration(
    row: pd.Series,
):

    model = row["model"]

    m = int(
        row["m"]
    )

    target_rate = float(
        row[
            "target_participation_rate"
        ]
    )

    n_bet_days = int(
        row[
            "n_bet_days"
        ]
    )

    hits = int(
        row[
            "number_hits"
        ]
    )

    hit_rate = (
        hits / n_bet_days
        if n_bet_days > 0
        else np.nan
    )

    random_hit_rate = (
        m
        / NUMBER_OF_CLASSES
    )

    break_even_hit_rate = (
        m
        * COST_PER_NUMBER
        / PAYOUT_IF_HIT
    )

    ci_low, ci_high = (
        wilson_interval(
            hits,
            n_bet_days,
        )
    )

    p_random = (
        one_sided_binomial_pvalue(
            hits=hits,
            n=n_bet_days,
            p0=random_hit_rate,
        )
    )

    if np.isnan(
        p_random
    ):
        p_bonf = np.nan
    else:
        p_bonf = min(
            1.0,
            p_random * N_CONFIGS,
        )

    return {
        "model": model,
        "m": m,

        "target_participation_rate": (
            target_rate
        ),

        "n_bet_days": (
            n_bet_days
        ),

        "number_hits": (
            hits
        ),

        "hit_rate": (
            hit_rate
        ),

        "random_hit_rate": (
            random_hit_rate
        ),

        "break_even_hit_rate": (
            break_even_hit_rate
        ),

        "wilson_ci_low": (
            ci_low
        ),

        "wilson_ci_high": (
            ci_high
        ),

        "pvalue_vs_random": (
            p_random
        ),

        "pvalue_bonferroni": (
            p_bonf
        ),

        "significant_unadjusted_5pct": (
            (
                p_random < ALPHA
            )
            if not np.isnan(
                p_random
            )
            else False
        ),

        "significant_bonferroni_5pct": (
            (
                p_bonf < ALPHA
            )
            if not np.isnan(
                p_bonf
            )
            else False
        ),

        "ci_above_random": (
            ci_low
            > random_hit_rate
        ),

        "ci_above_break_even": (
            ci_low
            > break_even_hit_rate
        ),

        "roi": float(
            row["roi"]
        ),

        "total_profit": float(
            row[
                "total_profit"
            ]
        ),
    }


# ============================================================
# BY FOLD
# ============================================================

def evaluate_best_by_fold(
    daily: pd.DataFrame,
    best: pd.Series,
):

    model = best["model"]

    m = int(
        best["m"]
    )

    target_rate = float(
        best[
            "target_participation_rate"
        ]
    )

    subset = (
        daily.loc[
            daily["model"].eq(
                model
            )
            & daily["m"].eq(
                m
            )
            & np.isclose(
                daily[
                    "target_participation_rate"
                ],
                target_rate,
            )
        ]
        .copy()
    )

    records = []

    for fold, fold_df in (
        subset.groupby(
            "fold"
        )
    ):

        bets = (
            fold_df.loc[
                fold_df[
                    "bet"
                ].eq(1)
            ]
        )

        n_eval = len(
            fold_df
        )

        n_bet = len(
            bets
        )

        hits = int(
            bets[
                "hit_if_bet"
            ].sum()
        )

        hit_rate = (
            hits / n_bet
            if n_bet > 0
            else np.nan
        )

        cost = (
            n_bet
            * m
            * COST_PER_NUMBER
        )

        revenue = (
            hits
            * PAYOUT_IF_HIT
        )

        profit = (
            revenue
            - cost
        )

        roi = (
            profit / cost
            if cost > 0
            else np.nan
        )

        random_hit_rate = (
            m
            / NUMBER_OF_CLASSES
        )

        break_even_hit_rate = (
            m
            * COST_PER_NUMBER
            / PAYOUT_IF_HIT
        )

        ci_low, ci_high = (
            wilson_interval(
                hits,
                n_bet,
            )
        )

        p_random = (
            one_sided_binomial_pvalue(
                hits=hits,
                n=n_bet,
                p0=random_hit_rate,
            )
        )

        records.append(
            {
                "fold": fold,

                "model": model,

                "m": m,

                "target_participation_rate": (
                    target_rate
                ),

                "n_eval_days": (
                    n_eval
                ),

                "n_bet_days": (
                    n_bet
                ),

                "participation_rate": (
                    n_bet / n_eval
                    if n_eval > 0
                    else np.nan
                ),

                "number_hits": (
                    hits
                ),

                "hit_rate": (
                    hit_rate
                ),

                "random_hit_rate": (
                    random_hit_rate
                ),

                "break_even_hit_rate": (
                    break_even_hit_rate
                ),

                "wilson_ci_low": (
                    ci_low
                ),

                "wilson_ci_high": (
                    ci_high
                ),

                "pvalue_vs_random": (
                    p_random
                ),

                "total_cost": (
                    cost
                ),

                "total_revenue": (
                    revenue
                ),

                "total_profit": (
                    profit
                ),

                "roi": (
                    roi
                ),
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# TOP CONFIG ROBUSTNESS
# ============================================================

def evaluate_top_configs(
    summary: pd.DataFrame,
    n_top: int = 30,
):

    top = (
        summary
        .loc[
            summary[
                "roi"
            ].notna()
        ]
        .sort_values(
            "roi",
            ascending=False,
        )
        .head(
            n_top
        )
    )

    records = []

    for _, row in (
        top.iterrows()
    ):

        records.append(
            evaluate_configuration(
                row
            )
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# PRINT
# ============================================================

def print_best(
    result: pd.DataFrame,
):

    print(
        "\n"
        + "=" * 170
    )

    print(
        "ROBUSTNESS - BEST OBSERVED SELECTIVE CONFIG"
    )

    print(
        "=" * 170
    )

    print(
        result.to_string(
            index=False,
            formatters={
                "target_participation_rate": (
                    "{:.0%}".format
                ),

                "hit_rate": (
                    "{:.4%}".format
                ),

                "random_hit_rate": (
                    "{:.2%}".format
                ),

                "break_even_hit_rate": (
                    "{:.2%}".format
                ),

                "wilson_ci_low": (
                    "{:.4%}".format
                ),

                "wilson_ci_high": (
                    "{:.4%}".format
                ),

                "pvalue_vs_random": (
                    "{:.6f}".format
                ),

                "pvalue_bonferroni": (
                    "{:.6f}".format
                ),

                "roi": (
                    "{:.2%}".format
                ),

                "total_profit": (
                    "{:,.0f}".format
                ),
            },
        )
    )


def print_fold(
    fold_df: pd.DataFrame,
):

    print(
        "\n"
        + "=" * 170
    )

    print(
        "BEST OBSERVED CONFIG - PERFORMANCE BY FOLD"
    )

    print(
        "=" * 170
    )

    print(
        fold_df.to_string(
            index=False,
            formatters={
                "target_participation_rate": (
                    "{:.0%}".format
                ),

                "participation_rate": (
                    "{:.2%}".format
                ),

                "hit_rate": (
                    "{:.4%}".format
                ),

                "random_hit_rate": (
                    "{:.2%}".format
                ),

                "break_even_hit_rate": (
                    "{:.2%}".format
                ),

                "wilson_ci_low": (
                    "{:.4%}".format
                ),

                "wilson_ci_high": (
                    "{:.4%}".format
                ),

                "pvalue_vs_random": (
                    "{:.6f}".format
                ),

                "total_profit": (
                    "{:,.0f}".format
                ),

                "roi": (
                    "{:.2%}".format
                ),
            },
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    summary, daily = (
        load_data()
    )

    print(
        f"Selective configurations: "
        f"{len(summary):,}"
    )

    print(
        f"Correction configs: "
        f"{N_CONFIGS:,}"
    )

    best = (
        get_best_observed(
            summary
        )
    )

    best_result = pd.DataFrame(
        [
            evaluate_configuration(
                best
            )
        ]
    )

    fold_result = (
        evaluate_best_by_fold(
            daily,
            best,
        )
    )

    top_result = (
        evaluate_top_configs(
            summary,
            n_top=30,
        )
    )

    print_best(
        best_result
    )

    print_fold(
        fold_result
    )

    print(
        "\n"
        + "=" * 170
    )

    print(
        "TOP CONFIGS ROBUSTNESS"
    )

    print(
        "=" * 170
    )

    print(
        top_result[
            [
                "model",
                "m",
                "target_participation_rate",
                "n_bet_days",
                "number_hits",
                "hit_rate",
                "random_hit_rate",
                "break_even_hit_rate",
                "wilson_ci_low",
                "wilson_ci_high",
                "pvalue_vs_random",
                "pvalue_bonferroni",
                "roi",
            ]
        ]
        .to_string(
            index=False,
            formatters={
                "target_participation_rate": (
                    "{:.0%}".format
                ),

                "hit_rate": (
                    "{:.4%}".format
                ),

                "random_hit_rate": (
                    "{:.2%}".format
                ),

                "break_even_hit_rate": (
                    "{:.2%}".format
                ),

                "wilson_ci_low": (
                    "{:.4%}".format
                ),

                "wilson_ci_high": (
                    "{:.4%}".format
                ),

                "pvalue_vs_random": (
                    "{:.6f}".format
                ),

                "pvalue_bonferroni": (
                    "{:.6f}".format
                ),

                "roi": (
                    "{:.2%}".format
                ),
            },
        )
    )

    best_result.to_csv(
        OUTPUT_BEST,
        index=False,
        encoding="utf-8-sig",
    )

    fold_result.to_csv(
        OUTPUT_FOLD,
        index=False,
        encoding="utf-8-sig",
    )

    top_result.to_csv(
        OUTPUT_TOP,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nĐã lưu:"
    )

    print(
        OUTPUT_BEST
    )

    print(
        OUTPUT_FOLD
    )

    print(
        OUTPUT_TOP
    )


if __name__ == "__main__":
    main()