"""
17 - PROFIT HORIZON ANALYSIS

Phân tích Strategy 15 theo thời gian.

Mục tiêu
--------
1. Sau bao nhiêu evaluation days cumulative P/L lần đầu > 0?
2. Sau bao nhiêu bet days cumulative P/L lần đầu > 0?
3. Sau bao nhiêu calendar days cumulative P/L lần đầu > 0?
4. Từ thời điểm nào cumulative P/L dương bền vững?
5. Hiệu quả trên rolling calendar horizons:
       30, 60, 90, 180, 365 ngày
6. Hiệu quả trên rolling bet horizons:
       10, 20, 30, 50, 100, 180, 365 bets

Quan trọng
----------
evaluation day:
    một row thực sự thuộc test set của Strategy 15.

bet day:
    một evaluation day mà bet = 1.

calendar day:
    chênh lệch ngày lịch.

Do Strategy 15 đánh giá nhiều fold khác nhau,
calendar days có thể chứa gap lớn giữa các fold.

Vì vậy:
    eval_day và bet_day
là hai metric chính để diễn giải time-to-profit.

Rolling windows luôn chạy trong từng fold riêng,
không bao giờ đi xuyên qua boundary fold.

Best horizon chỉ descriptive / post-hoc.
Không phải validated future horizon.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

STRATEGY_DIR = (
    PROJECT_DIR
    / "artifacts"
    / "strategies"
)


INPUT_DAILY = (
    STRATEGY_DIR
    / "selective_topm_daily.csv"
)


# ============================================================
# HORIZONS
# ============================================================

CALENDAR_HORIZONS = [
    30,
    60,
    90,
    180,
    365,
]


BET_HORIZONS = [
    10,
    20,
    30,
    50,
    100,
    180,
    365,
]


MIN_WINDOWS_FOR_BEST = 10


# ------------------------------------------------------------
# Window-level file trước đây có thể >500 MB.
#
# False:
#     chỉ lưu summary / best.
#
# True:
#     lưu cả từng rolling window.
# ------------------------------------------------------------

SAVE_WINDOW_LEVEL = False


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_FIRST_POSITIVE = (
    STRATEGY_DIR
    / "profit_horizon_first_positive.csv"
)


OUTPUT_WINDOWS = (
    STRATEGY_DIR
    / "profit_horizon_windows.csv"
)


OUTPUT_SUMMARY = (
    STRATEGY_DIR
    / "profit_horizon_summary.csv"
)


OUTPUT_BEST_CONFIG = (
    STRATEGY_DIR
    / "profit_horizon_best_by_config.csv"
)


OUTPUT_BEST_OVERALL = (
    STRATEGY_DIR
    / "profit_horizon_best_overall.csv"
)


# ============================================================
# CONFIG COLUMNS
# ============================================================

CONFIG_COLUMNS = [
    "model",
    "m",
    "target_participation_rate",
]


# ============================================================
# LOAD
# ============================================================

def load_data():

    if not INPUT_DAILY.exists():

        raise FileNotFoundError(
            f"Không tìm thấy:\n"
            f"{INPUT_DAILY}\n\n"
            "Hãy chạy Strategy 15 trước."
        )


    daily = pd.read_csv(
        INPUT_DAILY,
        parse_dates=[
            "date"
        ],
    )


    required = [
        "date",
        "fold",
        "model",
        "m",
        "target_participation_rate",
        "bet",
        "hit_if_bet",
        "cost",
        "revenue",
        "profit",
    ]


    missing = [
        column
        for column in required
        if column not in daily.columns
    ]


    if missing:

        raise ValueError(
            "selective_topm_daily.csv thiếu:\n"
            f"{missing}"
        )


    daily[
        "target_participation_rate"
    ] = (
        daily[
            "target_participation_rate"
        ]
        .astype(float)
        .round(10)
    )


    daily[
        "m"
    ] = (
        daily[
            "m"
        ]
        .astype(int)
    )


    daily[
        "bet"
    ] = (
        daily[
            "bet"
        ]
        .astype(int)
    )


    daily[
        "hit_if_bet"
    ] = (
        daily[
            "hit_if_bet"
        ]
        .astype(int)
    )


    daily = (
        daily
        .sort_values(
            CONFIG_COLUMNS
            + [
                "date",
                "fold",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    return daily


# ============================================================
# HELPERS
# ============================================================

def safe_roi(
    profit,
    cost,
):

    if cost <= 0:
        return np.nan

    return (
        profit
        / cost
    )


def safe_hit_rate(
    hits,
    n_bets,
):

    if n_bets <= 0:
        return np.nan

    return (
        hits
        / n_bets
    )


# ============================================================
# FIRST / STABLE POSITIVE
# ============================================================

def first_positive_analysis(
    config_df,
):
    """
    Phân tích toàn bộ observed test timeline
    của một config.

    Metric chính:
        first_positive_eval_day
        stable_positive_eval_day

        first_positive_bet_day
        stable_positive_bet_day

    Calendar metric chỉ là phụ.
    """

    config_df = (
        config_df
        .sort_values(
            [
                "date",
                "fold",
            ]
        )
        .reset_index(
            drop=True
        )
        .copy()
    )


    model = (
        config_df[
            "model"
        ]
        .iloc[0]
    )


    m = int(
        config_df[
            "m"
        ]
        .iloc[0]
    )


    target_rate = float(
        config_df[
            "target_participation_rate"
        ]
        .iloc[0]
    )


    # ========================================================
    # EVALUATION TIMELINE
    # ========================================================

    config_df[
        "eval_day"
    ] = np.arange(
        1,
        len(
            config_df
        ) + 1,
    )


    config_df[
        "cum_profit"
    ] = (
        config_df[
            "profit"
        ]
        .cumsum()
    )


    config_df[
        "cum_bets"
    ] = (
        config_df[
            "bet"
        ]
        .cumsum()
    )


    cumulative_profit = (
        config_df[
            "cum_profit"
        ]
        .to_numpy(
            dtype=float
        )
    )


    positive = (
        cumulative_profit
        > 0
    )


    # ========================================================
    # FIRST POSITIVE EVAL DAY
    # ========================================================

    if positive.any():

        idx = (
            np.flatnonzero(
                positive
            )[0]
        )


        first_positive_eval_day = int(
            config_df.loc[
                idx,
                "eval_day",
            ]
        )


        first_positive_eval_date = (
            config_df.loc[
                idx,
                "date",
            ]
        )


        first_positive_calendar_days = (
            first_positive_eval_date
            - config_df[
                "date"
            ]
            .iloc[0]
        ).days + 1

    else:

        first_positive_eval_day = (
            np.nan
        )

        first_positive_eval_date = (
            pd.NaT
        )

        first_positive_calendar_days = (
            np.nan
        )


    # ========================================================
    # STABLE POSITIVE EVAL DAY
    #
    # earliest t such that:
    #    cumulative_profit[s] > 0
    # for all s >= t
    # ========================================================

    if len(
        cumulative_profit
    ) > 0:

        future_min = (
            np.minimum.accumulate(
                cumulative_profit[
                    ::-1
                ]
            )[
                ::-1
            ]
        )


        stable_mask = (
            future_min
            > 0
        )

    else:

        stable_mask = np.array(
            [],
            dtype=bool,
        )


    if stable_mask.any():

        idx = (
            np.flatnonzero(
                stable_mask
            )[0]
        )


        stable_positive_eval_day = int(
            config_df.loc[
                idx,
                "eval_day",
            ]
        )


        stable_positive_eval_date = (
            config_df.loc[
                idx,
                "date",
            ]
        )


        stable_positive_calendar_days = (
            stable_positive_eval_date
            - config_df[
                "date"
            ]
            .iloc[0]
        ).days + 1

    else:

        stable_positive_eval_day = (
            np.nan
        )

        stable_positive_eval_date = (
            pd.NaT
        )

        stable_positive_calendar_days = (
            np.nan
        )


    # ========================================================
    # BET TIMELINE
    # ========================================================

    bets = (
        config_df.loc[
            config_df[
                "bet"
            ]
            .eq(
                1
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    n_bets = len(
        bets
    )


    if n_bets > 0:

        bets[
            "bet_day"
        ] = np.arange(
            1,
            n_bets + 1,
        )


        bets[
            "cum_profit_bet"
        ] = (
            bets[
                "profit"
            ]
            .cumsum()
        )


        bet_cumulative_profit = (
            bets[
                "cum_profit_bet"
            ]
            .to_numpy(
                dtype=float
            )
        )


        # ====================================================
        # FIRST POSITIVE BET
        # ====================================================

        positive_bet = (
            bet_cumulative_profit
            > 0
        )


        if positive_bet.any():

            idx = (
                np.flatnonzero(
                    positive_bet
                )[0]
            )


            first_positive_bet_day = int(
                bets.loc[
                    idx,
                    "bet_day",
                ]
            )


            first_positive_bet_date = (
                bets.loc[
                    idx,
                    "date",
                ]
            )

        else:

            first_positive_bet_day = (
                np.nan
            )

            first_positive_bet_date = (
                pd.NaT
            )


        # ====================================================
        # STABLE POSITIVE BET
        # ====================================================

        future_min_bet = (
            np.minimum.accumulate(
                bet_cumulative_profit[
                    ::-1
                ]
            )[
                ::-1
            ]
        )


        stable_bet_mask = (
            future_min_bet
            > 0
        )


        if stable_bet_mask.any():

            idx = (
                np.flatnonzero(
                    stable_bet_mask
                )[0]
            )


            stable_positive_bet_day = int(
                bets.loc[
                    idx,
                    "bet_day",
                ]
            )


            stable_positive_bet_date = (
                bets.loc[
                    idx,
                    "date",
                ]
            )

        else:

            stable_positive_bet_day = (
                np.nan
            )

            stable_positive_bet_date = (
                pd.NaT
            )

    else:

        first_positive_bet_day = (
            np.nan
        )

        first_positive_bet_date = (
            pd.NaT
        )

        stable_positive_bet_day = (
            np.nan
        )

        stable_positive_bet_date = (
            pd.NaT
        )


    # ========================================================
    # FINAL PERFORMANCE
    # ========================================================

    total_cost = float(
        config_df[
            "cost"
        ]
        .sum()
    )


    total_profit = float(
        config_df[
            "profit"
        ]
        .sum()
    )


    total_hits = int(
        (
            config_df[
                "bet"
            ]

            * config_df[
                "hit_if_bet"
            ]
        )
        .sum()
    )


    final_roi = (
        safe_roi(
            total_profit,
            total_cost,
        )
    )


    return {
        "model": (
            model
        ),

        "m": (
            m
        ),

        "target_participation_rate": (
            target_rate
        ),

        "first_date": (
            config_df[
                "date"
            ]
            .min()
        ),

        "last_date": (
            config_df[
                "date"
            ]
            .max()
        ),

        "n_eval_days": (
            len(
                config_df
            )
        ),

        "n_bet_days": (
            n_bets
        ),

        "number_hits": (
            total_hits
        ),

        "first_positive_eval_day": (
            first_positive_eval_day
        ),

        "first_positive_eval_date": (
            first_positive_eval_date
        ),

        "stable_positive_eval_day": (
            stable_positive_eval_day
        ),

        "stable_positive_eval_date": (
            stable_positive_eval_date
        ),

        "first_positive_bet_day": (
            first_positive_bet_day
        ),

        "first_positive_bet_date": (
            first_positive_bet_date
        ),

        "stable_positive_bet_day": (
            stable_positive_bet_day
        ),

        "stable_positive_bet_date": (
            stable_positive_bet_date
        ),

        "first_positive_calendar_days": (
            first_positive_calendar_days
        ),

        "stable_positive_calendar_days": (
            stable_positive_calendar_days
        ),

        "total_cost": (
            total_cost
        ),

        "total_profit": (
            total_profit
        ),

        "final_roi": (
            final_roi
        ),
    }


# ============================================================
# PREFIX SUM
# ============================================================

def prefix_sum(
    values,
):

    values = np.asarray(
        values
    )


    return np.concatenate(
        [
            np.array(
                [
                    0
                ],
                dtype=values.dtype,
            ),
            np.cumsum(
                values
            ),
        ]
    )


# ============================================================
# CALENDAR WINDOWS
# ============================================================

def build_calendar_windows(
    fold_df,
    horizon_days,
):
    """
    Rolling calendar windows trong MỘT fold.

    Window:
        [start_date, start_date + H - 1 days]

    Chỉ giữ window có ít nhất một evaluation row.
    """

    fold_df = (
        fold_df
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
        .copy()
    )


    if fold_df.empty:
        return []


    dates = (
        fold_df[
            "date"
        ]
        .to_numpy(
            dtype="datetime64[ns]"
        )
    )


    cost = (
        fold_df[
            "cost"
        ]
        .to_numpy(
            dtype=float
        )
    )


    profit = (
        fold_df[
            "profit"
        ]
        .to_numpy(
            dtype=float
        )
    )


    bet = (
        fold_df[
            "bet"
        ]
        .to_numpy(
            dtype=int
        )
    )


    hit = (
        (
            fold_df[
                "bet"
            ]

            * fold_df[
                "hit_if_bet"
            ]
        )
        .to_numpy(
            dtype=int
        )
    )


    prefix_cost = (
        prefix_sum(
            cost
        )
    )


    prefix_profit = (
        prefix_sum(
            profit
        )
    )


    prefix_bet = (
        prefix_sum(
            bet
        )
    )


    prefix_hit = (
        prefix_sum(
            hit
        )
    )


    records = []


    for start_idx in range(
        len(
            fold_df
        )
    ):

        start_date = (
            fold_df.loc[
                start_idx,
                "date",
            ]
        )


        end_target = (
            start_date
            + pd.Timedelta(
                days=(
                    horizon_days
                    - 1
                )
            )
        )


        end_idx_exclusive = (
            np.searchsorted(
                dates,
                np.datetime64(
                    end_target
                ),
                side="right",
            )
        )


        if (
            end_idx_exclusive
            <= start_idx
        ):

            continue


        n_eval_days = (
            end_idx_exclusive
            - start_idx
        )


        total_cost = float(
            prefix_cost[
                end_idx_exclusive
            ]
            - prefix_cost[
                start_idx
            ]
        )


        total_profit = float(
            prefix_profit[
                end_idx_exclusive
            ]
            - prefix_profit[
                start_idx
            ]
        )


        n_bets = int(
            prefix_bet[
                end_idx_exclusive
            ]
            - prefix_bet[
                start_idx
            ]
        )


        n_hits = int(
            prefix_hit[
                end_idx_exclusive
            ]
            - prefix_hit[
                start_idx
            ]
        )


        roi = (
            safe_roi(
                total_profit,
                total_cost,
            )
        )


        hit_rate = (
            safe_hit_rate(
                n_hits,
                n_bets,
            )
        )


        records.append(
            {
                "window_type": (
                    "calendar"
                ),

                "horizon": (
                    horizon_days
                ),

                "window_start": (
                    start_date
                ),

                "window_end": (
                    fold_df.loc[
                        end_idx_exclusive - 1,
                        "date",
                    ]
                ),

                "n_eval_days": (
                    n_eval_days
                ),

                "n_bets": (
                    n_bets
                ),

                "n_hits": (
                    n_hits
                ),

                "total_cost": (
                    total_cost
                ),

                "total_profit": (
                    total_profit
                ),

                "roi": (
                    roi
                ),

                "hit_rate": (
                    hit_rate
                ),

                "calendar_span_days": (
                    (
                        fold_df.loc[
                            end_idx_exclusive - 1,
                            "date",
                        ]
                        - start_date
                    ).days
                    + 1
                ),
            }
        )


    return records


# ============================================================
# BET WINDOWS
# ============================================================

def build_bet_windows(
    fold_df,
    horizon_bets,
):
    """
    Rolling window gồm đúng H BETS.

    Chỉ chạy trong cùng fold.
    """

    bets = (
        fold_df.loc[
            fold_df[
                "bet"
            ]
            .eq(
                1
            )
        ]
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
        .copy()
    )


    if (
        len(
            bets
        )
        < horizon_bets
    ):

        return []


    cost = (
        bets[
            "cost"
        ]
        .to_numpy(
            dtype=float
        )
    )


    profit = (
        bets[
            "profit"
        ]
        .to_numpy(
            dtype=float
        )
    )


    hits = (
        bets[
            "hit_if_bet"
        ]
        .to_numpy(
            dtype=int
        )
    )


    prefix_cost = (
        prefix_sum(
            cost
        )
    )


    prefix_profit = (
        prefix_sum(
            profit
        )
    )


    prefix_hits = (
        prefix_sum(
            hits
        )
    )


    records = []


    max_start = (
        len(
            bets
        )
        - horizon_bets
        + 1
    )


    for start_idx in range(
        max_start
    ):

        end_idx_exclusive = (
            start_idx
            + horizon_bets
        )


        total_cost = float(
            prefix_cost[
                end_idx_exclusive
            ]
            - prefix_cost[
                start_idx
            ]
        )


        total_profit = float(
            prefix_profit[
                end_idx_exclusive
            ]
            - prefix_profit[
                start_idx
            ]
        )


        n_hits = int(
            prefix_hits[
                end_idx_exclusive
            ]
            - prefix_hits[
                start_idx
            ]
        )


        start_date = (
            bets.loc[
                start_idx,
                "date",
            ]
        )


        end_date = (
            bets.loc[
                end_idx_exclusive - 1,
                "date",
            ]
        )


        records.append(
            {
                "window_type": (
                    "bet"
                ),

                "horizon": (
                    horizon_bets
                ),

                "window_start": (
                    start_date
                ),

                "window_end": (
                    end_date
                ),

                "n_eval_days": (
                    np.nan
                ),

                "n_bets": (
                    horizon_bets
                ),

                "n_hits": (
                    n_hits
                ),

                "total_cost": (
                    total_cost
                ),

                "total_profit": (
                    total_profit
                ),

                "roi": (
                    safe_roi(
                        total_profit,
                        total_cost,
                    )
                ),

                "hit_rate": (
                    n_hits
                    / horizon_bets
                ),

                "calendar_span_days": (
                    (
                        end_date
                        - start_date
                    ).days
                    + 1
                ),
            }
        )


    return records


# ============================================================
# WINDOW SUMMARY
# ============================================================

def summarize_windows(
    windows,
):

    if windows.empty:
        return pd.DataFrame()


    records = []


    grouped = (
        windows
        .groupby(
            CONFIG_COLUMNS
            + [
                "window_type",
                "horizon",
            ],
            sort=False,
        )
    )


    for keys, subset in grouped:

        (
            model,
            m,
            target_rate,
            window_type,
            horizon,
        ) = keys


        valid_roi = (
            subset[
                "roi"
            ]
            .dropna()
        )


        if valid_roi.empty:

            continue


        records.append(
            {
                "model": (
                    model
                ),

                "m": (
                    int(
                        m
                    )
                ),

                "target_participation_rate": (
                    float(
                        target_rate
                    )
                ),

                "window_type": (
                    window_type
                ),

                "horizon": (
                    int(
                        horizon
                    )
                ),

                "n_windows": (
                    len(
                        valid_roi
                    )
                ),

                "mean_roi": float(
                    valid_roi.mean()
                ),

                "median_roi": float(
                    valid_roi.median()
                ),

                "std_roi": float(
                    valid_roi.std(
                        ddof=1
                    )
                )
                if len(
                    valid_roi
                ) > 1
                else 0.0,

                "min_roi": float(
                    valid_roi.min()
                ),

                "max_roi": float(
                    valid_roi.max()
                ),

                "profitable_window_rate": float(
                    (
                        valid_roi
                        > 0
                    )
                    .mean()
                ),

                "nonnegative_window_rate": float(
                    (
                        valid_roi
                        >= 0
                    )
                    .mean()
                ),

                "mean_n_bets": float(
                    subset[
                        "n_bets"
                    ]
                    .mean()
                ),

                "median_n_bets": float(
                    subset[
                        "n_bets"
                    ]
                    .median()
                ),

                "mean_calendar_span_days": float(
                    subset[
                        "calendar_span_days"
                    ]
                    .mean()
                ),

                "median_calendar_span_days": float(
                    subset[
                        "calendar_span_days"
                    ]
                    .median()
                ),
            }
        )


    return pd.DataFrame(
        records
    )


# ============================================================
# BEST HORIZON PER CONFIG
# ============================================================

def build_best_by_config(
    summary,
):
    """
    Best horizon descriptive.

    Ưu tiên:
        profitable_window_rate
        median_roi
        mean_roi

    Chỉ xét horizon có >= MIN_WINDOWS_FOR_BEST.
    """

    eligible = (
        summary.loc[
            summary[
                "n_windows"
            ]
            .ge(
                MIN_WINDOWS_FOR_BEST
            )
        ]
        .copy()
    )


    if eligible.empty:
        return eligible


    best = (
        eligible
        .sort_values(
            CONFIG_COLUMNS
            + [
                "window_type",
                "profitable_window_rate",
                "median_roi",
                "mean_roi",
            ],
            ascending=[
                True,
                True,
                True,
                True,
                False,
                False,
                False,
            ],
        )
        .groupby(
            CONFIG_COLUMNS
            + [
                "window_type"
            ],
            as_index=False,
            sort=False,
        )
        .first()
    )


    return best


# ============================================================
# BEST OVERALL
# ============================================================

def build_best_overall(
    summary,
):
    """
    Tạo bốn dòng descriptive:

    1. calendar max median ROI
    2. calendar max profitable rate
    3. bet max median ROI
    4. bet max profitable rate
    """

    eligible = (
        summary.loc[
            summary[
                "n_windows"
            ]
            .ge(
                MIN_WINDOWS_FOR_BEST
            )
        ]
        .copy()
    )


    if eligible.empty:
        return eligible


    records = []


    for window_type in [
        "calendar",
        "bet",
    ]:

        subset = (
            eligible.loc[
                eligible[
                    "window_type"
                ]
                .eq(
                    window_type
                )
            ]
            .copy()
        )


        if subset.empty:
            continue


        # ====================================================
        # MAX MEDIAN ROI
        # ====================================================

        row = (
            subset
            .sort_values(
                [
                    "median_roi",
                    "profitable_window_rate",
                    "n_windows",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )
            .iloc[0]
            .to_dict()
        )


        row[
            "criterion"
        ] = (
            "max_median_roi"
        )


        records.append(
            row
        )


        # ====================================================
        # MAX PROFITABLE RATE
        # ====================================================

        row = (
            subset
            .sort_values(
                [
                    "profitable_window_rate",
                    "median_roi",
                    "n_windows",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )
            .iloc[0]
            .to_dict()
        )


        row[
            "criterion"
        ] = (
            "max_profitable_window_rate"
        )


        records.append(
            row
        )


    return pd.DataFrame(
        records
    )


# ============================================================
# PRINT FIRST POSITIVE
# ============================================================

def print_first_positive(
    first_positive,
):

    positive = (
        first_positive.loc[
            first_positive[
                "final_roi"
            ]
            .gt(
                0
            )
        ]
        .copy()
    )


    if positive.empty:

        print(
            "\nKhông có config final ROI > 0."
        )

        return


    positive = (
        positive
        .sort_values(
            [
                "stable_positive_bet_day",
                "first_positive_bet_day",
                "final_roi",
            ],
            ascending=[
                True,
                True,
                False,
            ],
            na_position="last",
        )
        .head(
            20
        )
    )


    columns = [
        "model",
        "m",
        "target_participation_rate",
        "n_eval_days",
        "n_bet_days",
        "number_hits",
        "first_positive_eval_day",
        "stable_positive_eval_day",
        "first_positive_bet_day",
        "stable_positive_bet_day",
        "first_positive_calendar_days",
        "stable_positive_calendar_days",
        "final_roi",
    ]


    print(
        "\n"
        + "=" * 220
    )


    print(
        "TIME TO PROFIT - POSITIVE FINAL ROI CONFIGS"
    )


    print(
        "=" * 220
    )


    print(
        positive[
            columns
        ]
        .to_string(
            index=False,

            formatters={
                "target_participation_rate": (
                    "{:.0%}".format
                ),

                "final_roi": (
                    "{:+.2%}".format
                ),
            },
        )
    )


# ============================================================
# PRINT BEST OVERALL
# ============================================================

def print_best_overall(
    best,
):

    if best.empty:
        return


    columns = [
        "criterion",
        "window_type",
        "model",
        "m",
        "target_participation_rate",
        "horizon",
        "n_windows",
        "mean_roi",
        "median_roi",
        "std_roi",
        "profitable_window_rate",
        "mean_n_bets",
        "mean_calendar_span_days",
    ]


    print(
        "\n"
        + "=" * 220
    )


    print(
        "BEST OBSERVED PROFIT HORIZONS"
    )


    print(
        "=" * 220
    )


    print(
        best[
            columns
        ]
        .to_string(
            index=False,

            formatters={
                "target_participation_rate": (
                    "{:.0%}".format
                ),

                "mean_roi": (
                    "{:+.2%}".format
                ),

                "median_roi": (
                    "{:+.2%}".format
                ),

                "std_roi": (
                    "{:.2%}".format
                ),

                "profitable_window_rate": (
                    "{:.2%}".format
                ),

                "mean_n_bets": (
                    "{:.1f}".format
                ),

                "mean_calendar_span_days": (
                    "{:.1f}".format
                ),
            },
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    daily = (
        load_data()
    )


    print(
        f"Daily rows: "
        f"{len(daily):,}"
    )


    print(
        f"Configurations: "
        f"{daily[CONFIG_COLUMNS].drop_duplicates().shape[0]:,}"
    )


    print(
        f"Models: "
        f"{daily['model'].nunique()}"
    )


    # ========================================================
    # FIRST / STABLE POSITIVE
    # ========================================================

    first_positive_records = []


    for _, subset in daily.groupby(
        CONFIG_COLUMNS,
        sort=False,
    ):

        first_positive_records.append(
            first_positive_analysis(
                subset
            )
        )


    first_positive = pd.DataFrame(
        first_positive_records
    )


    # ========================================================
    # WINDOWS
    # ========================================================

    window_records = []


    grouped = (
        daily.groupby(
            CONFIG_COLUMNS
            + [
                "fold"
            ],
            sort=False,
        )
    )


    total_groups = (
        daily[
            CONFIG_COLUMNS
            + [
                "fold"
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )


    print(
        f"Config-fold groups: "
        f"{total_groups:,}"
    )


    for group_number, (
        keys,
        fold_df,
    ) in enumerate(
        grouped,
        start=1,
    ):

        (
            model,
            m,
            target_rate,
            fold,
        ) = keys


        if (
            group_number == 1
            or group_number % 250 == 0
            or group_number == total_groups
        ):

            print(
                f"Processing group "
                f"{group_number:,}/"
                f"{total_groups:,}"
            )


        # ====================================================
        # CALENDAR
        # ====================================================

        for horizon in (
            CALENDAR_HORIZONS
        ):

            records = (
                build_calendar_windows(
                    fold_df,
                    horizon,
                )
            )


            for record in records:

                record.update(
                    {
                        "model": (
                            model
                        ),

                        "m": (
                            int(
                                m
                            )
                        ),

                        "target_participation_rate": (
                            float(
                                target_rate
                            )
                        ),

                        "fold": (
                            fold
                        ),
                    }
                )


            window_records.extend(
                records
            )


        # ====================================================
        # BET
        # ====================================================

        for horizon in (
            BET_HORIZONS
        ):

            records = (
                build_bet_windows(
                    fold_df,
                    horizon,
                )
            )


            for record in records:

                record.update(
                    {
                        "model": (
                            model
                        ),

                        "m": (
                            int(
                                m
                            )
                        ),

                        "target_participation_rate": (
                            float(
                                target_rate
                            )
                        ),

                        "fold": (
                            fold
                        ),
                    }
                )


            window_records.extend(
                records
            )


    windows = pd.DataFrame(
        window_records
    )


    print(
        f"\nWindow rows: "
        f"{len(windows):,}"
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    horizon_summary = (
        summarize_windows(
            windows
        )
    )


    best_by_config = (
        build_best_by_config(
            horizon_summary
        )
    )


    best_overall = (
        build_best_overall(
            horizon_summary
        )
    )


    # ========================================================
    # PRINT
    # ========================================================

    print_first_positive(
        first_positive
    )


    print_best_overall(
        best_overall
    )


    # ========================================================
    # SAVE
    # ========================================================

    first_positive.to_csv(
        OUTPUT_FIRST_POSITIVE,
        index=False,
        encoding="utf-8-sig",
    )


    horizon_summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )


    best_by_config.to_csv(
        OUTPUT_BEST_CONFIG,
        index=False,
        encoding="utf-8-sig",
    )


    best_overall.to_csv(
        OUTPUT_BEST_OVERALL,
        index=False,
        encoding="utf-8-sig",
    )


    if SAVE_WINDOW_LEVEL:

        windows.to_csv(
            OUTPUT_WINDOWS,
            index=False,
            encoding="utf-8-sig",
        )


    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\nĐã lưu:"
    )


    print(
        OUTPUT_FIRST_POSITIVE
    )


    print(
        OUTPUT_SUMMARY
    )


    print(
        OUTPUT_BEST_CONFIG
    )


    print(
        OUTPUT_BEST_OVERALL
    )


    if SAVE_WINDOW_LEVEL:

        print(
            OUTPUT_WINDOWS
        )

    else:

        print(
            "\nWindow-level CSV không lưu "
            "(SAVE_WINDOW_LEVEL=False)."
        )


if __name__ == "__main__":
    main()