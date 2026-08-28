"""
17 - PROFIT HORIZON ANALYSIS

Phân tích Strategy 15 theo thời gian.

Mục tiêu
-------
1. Sau bao nhiêu lần đặt cược thì cumulative P/L lần đầu > 0?
2. Sau bao nhiêu ngày lịch thì cumulative P/L lần đầu > 0?
3. Từ thời điểm nào cumulative P/L dương bền vững?
4. Nếu chơi trong H ngày lịch:
       H = 30, 60, 90, 180, 365
   thì:
       - mean ROI
       - median ROI
       - tỷ lệ window có lãi
       - best / worst ROI
5. Nếu chơi liên tục H lần cược:
       H = 10, 20, 30, 50, 100, 180, 365
   thì hiệu quả như thế nào?
6. Horizon nào cho kết quả ổn định nhất?

Lưu ý
-----
- Phân tích theo từng fold riêng để rolling window không đi xuyên
  qua các giai đoạn test khác nhau.
- "Best horizon" chỉ là descriptive / observed.
  Không được coi là horizon đã xác nhận out-of-sample.
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

INPUT_SUMMARY = (
    STRATEGY_DIR
    / "selective_topm_summary.csv"
)


# ------------------------------------------------------------
# Rolling calendar horizons
# ------------------------------------------------------------

CALENDAR_HORIZONS = [
    30,
    60,
    90,
    180,
    365,
]


# ------------------------------------------------------------
# Rolling bet horizons
# ------------------------------------------------------------

BET_HORIZONS = [
    10,
    20,
    30,
    50,
    100,
    180,
    365,
]


# Ít nhất bao nhiêu rolling windows thì mới xét
# là một horizon đủ dữ liệu để gọi "best".
MIN_WINDOWS_FOR_BEST = 10


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

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

OUTPUT_BEST_HORIZON = (
    STRATEGY_DIR
    / "profit_horizon_best_by_config.csv"
)

OUTPUT_BEST_OVERALL = (
    STRATEGY_DIR
    / "profit_horizon_best_overall.csv"
)


# ============================================================
# LOAD
# ============================================================

def load_data():

    if not INPUT_DAILY.exists():

        raise FileNotFoundError(
            f"Không tìm thấy:\n"
            f"{INPUT_DAILY}"
        )

    if not INPUT_SUMMARY.exists():

        raise FileNotFoundError(
            f"Không tìm thấy:\n"
            f"{INPUT_SUMMARY}"
        )

    daily = pd.read_csv(
        INPUT_DAILY,
        parse_dates=["date"],
    )

    summary = pd.read_csv(
        INPUT_SUMMARY,
    )

    required_daily = [
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
        for column
        in required_daily
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

    summary[
        "target_participation_rate"
    ] = (
        summary[
            "target_participation_rate"
        ]
        .astype(float)
        .round(10)
    )

    daily["m"] = (
        daily["m"]
        .astype(int)
    )

    daily["bet"] = (
        daily["bet"]
        .astype(int)
    )

    daily["hit_if_bet"] = (
        daily["hit_if_bet"]
        .astype(int)
    )

    return (
        daily,
        summary,
    )


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


def get_config_columns():

    return [
        "model",
        "m",
        "target_participation_rate",
    ]


# ============================================================
# FIRST POSITIVE / STABLE POSITIVE
# ============================================================

def first_positive_analysis(
    config_df: pd.DataFrame,
) -> dict:

    """
    Phân tích cumulative P/L của một config
    trên toàn bộ test timeline.

    Tính:
        - first positive bet
        - first positive calendar date
        - stable positive bet
        - stable positive calendar date
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
        ].iloc[0]
    )

    m = int(
        config_df[
            "m"
        ].iloc[0]
    )

    r = float(
        config_df[
            "target_participation_rate"
        ].iloc[0]
    )


    # ========================================================
    # CALENDAR TIMELINE
    # ========================================================

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


    positive_calendar = (
        config_df[
            "cum_profit"
        ]
        > 0
    )


    # --------------------------------------------------------
    # First calendar date positive
    # --------------------------------------------------------

    if positive_calendar.any():

        first_calendar_idx = (
            np.flatnonzero(
                positive_calendar
                .to_numpy()
            )[0]
        )

        first_positive_calendar_date = (
            config_df.loc[
                first_calendar_idx,
                "date",
            ]
        )

        first_positive_calendar_days = (
            first_positive_calendar_date
            - config_df[
                "date"
            ].iloc[0]
        ).days + 1

    else:

        first_positive_calendar_date = (
            pd.NaT
        )

        first_positive_calendar_days = (
            np.nan
        )


    # --------------------------------------------------------
    # Stable positive calendar date
    #
    # earliest t such that:
    # cumulative profit[t] > 0
    # AND all future cumulative profit > 0
    # --------------------------------------------------------

    cumulative_profit = (
        config_df[
            "cum_profit"
        ]
        .to_numpy(
            dtype=float
        )
    )

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

        stable_mask = (
            np.array(
                [],
                dtype=bool,
            )
        )


    if stable_mask.any():

        stable_calendar_idx = (
            np.flatnonzero(
                stable_mask
            )[0]
        )

        stable_positive_calendar_date = (
            config_df.loc[
                stable_calendar_idx,
                "date",
            ]
        )

        stable_positive_calendar_days = (
            stable_positive_calendar_date
            - config_df[
                "date"
            ].iloc[0]
        ).days + 1

    else:

        stable_positive_calendar_date = (
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
            ].eq(1)
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
            "bet_number"
        ] = (
            np.arange(
                1,
                n_bets + 1,
            )
        )

        bets[
            "cum_profit"
        ] = (
            bets[
                "profit"
            ]
            .cumsum()
        )


        # ----------------------------------------------------
        # First positive bet
        # ----------------------------------------------------

        positive_bet = (
            bets[
                "cum_profit"
            ]
            > 0
        )

        if positive_bet.any():

            first_bet_idx = (
                np.flatnonzero(
                    positive_bet
                    .to_numpy()
                )[0]
            )

            first_positive_bet_day = int(
                bets.loc[
                    first_bet_idx,
                    "bet_number",
                ]
            )

            first_positive_bet_date = (
                bets.loc[
                    first_bet_idx,
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


        # ----------------------------------------------------
        # Stable positive bet
        # ----------------------------------------------------

        bet_cum_profit = (
            bets[
                "cum_profit"
            ]
            .to_numpy(
                dtype=float
            )
        )

        future_min_bet = (
            np.minimum.accumulate(
                bet_cum_profit[
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

            stable_bet_idx = (
                np.flatnonzero(
                    stable_bet_mask
                )[0]
            )

            stable_positive_bet_day = int(
                bets.loc[
                    stable_bet_idx,
                    "bet_number",
                ]
            )

            stable_positive_bet_date = (
                bets.loc[
                    stable_bet_idx,
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
        ].sum()
    )

    total_profit = float(
        config_df[
            "profit"
        ].sum()
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
            r
        ),

        "first_date": (
            config_df[
                "date"
            ].min()
        ),

        "last_date": (
            config_df[
                "date"
            ].max()
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

        "first_positive_calendar_date": (
            first_positive_calendar_date
        ),

        "stable_positive_calendar_days": (
            stable_positive_calendar_days
        ),

        "stable_positive_calendar_date": (
            stable_positive_calendar_date
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
# CALENDAR WINDOWS
# ============================================================

def build_calendar_windows(
    fold_df: pd.DataFrame,
    horizon_days: int,
) -> list:

    """
    Rolling H calendar-day windows.

    Chỉ chạy bên trong cùng một fold.
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

    if len(
        fold_df
    ) == 0:

        return []


    dates = (
        fold_df[
            "date"
        ]
        .to_numpy(
            dtype="datetime64[ns]"
        )
    )

    costs = (
        fold_df[
            "cost"
        ]
        .to_numpy(
            dtype=float
        )
    )

    profits = (
        fold_df[
            "profit"
        ]
        .to_numpy(
            dtype=float
        )
    )

    bets = (
        fold_df[
            "bet"
        ]
        .to_numpy(
            dtype=int
        )
    )

    hits = (
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


    # Prefix sums
    prefix_cost = np.concatenate(
        [
            [0.0],
            np.cumsum(
                costs
            ),
        ]
    )

    prefix_profit = np.concatenate(
        [
            [0.0],
            np.cumsum(
                profits
            ),
        ]
    )

    prefix_bets = np.concatenate(
        [
            [0],
            np.cumsum(
                bets
            ),
        ]
    )

    prefix_hits = np.concatenate(
        [
            [0],
            np.cumsum(
                hits
            ),
        ]
    )


    records = []

    start_idx = 0


    for end_idx in range(
        len(
            fold_df
        )
    ):

        end_date = (
            pd.Timestamp(
                dates[
                    end_idx
                ]
            )
        )

        desired_start = (
            end_date
            - pd.Timedelta(
                days=(
                    horizon_days
                    - 1
                )
            )
        )


        while (
            start_idx
            < end_idx
            and pd.Timestamp(
                dates[
                    start_idx
                ]
            )
            < desired_start
        ):

            start_idx += 1


        actual_start_date = (
            pd.Timestamp(
                dates[
                    start_idx
                ]
            )
        )


        actual_span = (
            end_date
            - actual_start_date
        ).days + 1


        # Chỉ giữ khi có đủ một horizon lịch đầy đủ.
        if actual_span < horizon_days:

            continue


        total_cost = (
            prefix_cost[
                end_idx + 1
            ]
            - prefix_cost[
                start_idx
            ]
        )

        total_profit = (
            prefix_profit[
                end_idx + 1
            ]
            - prefix_profit[
                start_idx
            ]
        )

        n_bets = int(
            prefix_bets[
                end_idx + 1
            ]
            - prefix_bets[
                start_idx
            ]
        )

        n_hits = int(
            prefix_hits[
                end_idx + 1
            ]
            - prefix_hits[
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
                    actual_start_date
                ),

                "window_end": (
                    end_date
                ),

                "calendar_span_days": (
                    actual_span
                ),

                "n_bet_days": (
                    n_bets
                ),

                "number_hits": (
                    n_hits
                ),

                "hit_rate": (
                    hit_rate
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

                "profitable": (
                    int(
                        total_profit
                        > 0
                    )
                ),
            }
        )


    return records


# ============================================================
# BET WINDOWS
# ============================================================

def build_bet_windows(
    fold_df: pd.DataFrame,
    horizon_bets: int,
) -> list:

    """
    Rolling windows theo số lần thực sự đặt cược.

    Ví dụ H=30:
        mỗi window gồm đúng 30 lần cược liên tiếp.
    """

    bets_df = (
        fold_df.loc[
            fold_df[
                "bet"
            ].eq(1)
        ]
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
        .copy()
    )


    n = len(
        bets_df
    )


    if n < horizon_bets:

        return []


    profits = (
        bets_df[
            "profit"
        ]
        .to_numpy(
            dtype=float
        )
    )

    costs = (
        bets_df[
            "cost"
        ]
        .to_numpy(
            dtype=float
        )
    )

    hits = (
        bets_df[
            "hit_if_bet"
        ]
        .to_numpy(
            dtype=int
        )
    )


    prefix_profit = np.concatenate(
        [
            [0.0],
            np.cumsum(
                profits
            ),
        ]
    )

    prefix_cost = np.concatenate(
        [
            [0.0],
            np.cumsum(
                costs
            ),
        ]
    )

    prefix_hits = np.concatenate(
        [
            [0],
            np.cumsum(
                hits
            ),
        ]
    )


    records = []


    for end_idx in range(
        horizon_bets - 1,
        n,
    ):

        start_idx = (
            end_idx
            - horizon_bets
            + 1
        )


        total_profit = (
            prefix_profit[
                end_idx + 1
            ]
            - prefix_profit[
                start_idx
            ]
        )

        total_cost = (
            prefix_cost[
                end_idx + 1
            ]
            - prefix_cost[
                start_idx
            ]
        )

        n_hits = int(
            prefix_hits[
                end_idx + 1
            ]
            - prefix_hits[
                start_idx
            ]
        )


        window_start = (
            bets_df.loc[
                start_idx,
                "date",
            ]
        )

        window_end = (
            bets_df.loc[
                end_idx,
                "date",
            ]
        )

        calendar_span = (
            window_end
            - window_start
        ).days + 1


        roi = (
            safe_roi(
                total_profit,
                total_cost,
            )
        )

        hit_rate = (
            n_hits
            / horizon_bets
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
                    window_start
                ),

                "window_end": (
                    window_end
                ),

                "calendar_span_days": (
                    calendar_span
                ),

                "n_bet_days": (
                    horizon_bets
                ),

                "number_hits": (
                    n_hits
                ),

                "hit_rate": (
                    hit_rate
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

                "profitable": (
                    int(
                        total_profit
                        > 0
                    )
                ),
            }
        )


    return records


# ============================================================
# BUILD ALL WINDOWS
# ============================================================

def build_all_windows(
    daily: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    config_columns = (
        get_config_columns()
    )


    grouped = (
        daily
        .groupby(
            config_columns
            + ["fold"],
            sort=False,
        )
    )


    total_groups = (
        grouped.ngroups
    )


    for group_number, (
        key,
        fold_df,
    ) in enumerate(
        grouped,
        start=1,
    ):

        (
            model,
            m,
            r,
            fold,
        ) = key


        if (
            group_number == 1
            or group_number % 100 == 0
            or group_number
            == total_groups
        ):

            print(
                f"Window analysis: "
                f"{group_number:,}"
                f"/{total_groups:,}"
            )


        # ----------------------------------------------------
        # Calendar horizons
        # ----------------------------------------------------

        for horizon in (
            CALENDAR_HORIZONS
        ):

            window_records = (
                build_calendar_windows(
                    fold_df,
                    horizon,
                )
            )

            for record in (
                window_records
            ):

                record.update(
                    {
                        "model": (
                            model
                        ),

                        "m": (
                            int(m)
                        ),

                        "target_participation_rate": (
                            float(r)
                        ),

                        "fold": (
                            fold
                        ),
                    }
                )

            records.extend(
                window_records
            )


        # ----------------------------------------------------
        # Bet horizons
        # ----------------------------------------------------

        for horizon in (
            BET_HORIZONS
        ):

            window_records = (
                build_bet_windows(
                    fold_df,
                    horizon,
                )
            )

            for record in (
                window_records
            ):

                record.update(
                    {
                        "model": (
                            model
                        ),

                        "m": (
                            int(m)
                        ),

                        "target_participation_rate": (
                            float(r)
                        ),

                        "fold": (
                            fold
                        ),
                    }
                )

            records.extend(
                window_records
            )


    if len(
        records
    ) == 0:

        return pd.DataFrame()


    return pd.DataFrame(
        records
    )


# ============================================================
# WINDOW SUMMARY
# ============================================================

def summarize_windows(
    windows: pd.DataFrame,
) -> pd.DataFrame:

    if windows.empty:

        return pd.DataFrame()


    # Windows không có bet thì ROI = NaN.
    valid = (
        windows.loc[
            windows[
                "roi"
            ].notna()
        ]
        .copy()
    )


    group_columns = [
        "model",
        "m",
        "target_participation_rate",
        "window_type",
        "horizon",
    ]


    records = []


    for key, subset in (
        valid.groupby(
            group_columns,
            sort=False,
        )
    ):

        (
            model,
            m,
            r,
            window_type,
            horizon,
        ) = key


        roi_values = (
            subset[
                "roi"
            ]
            .to_numpy(
                dtype=float
            )
        )

        profit_values = (
            subset[
                "total_profit"
            ]
            .to_numpy(
                dtype=float
            )
        )


        records.append(
            {
                "model": (
                    model
                ),

                "m": (
                    int(m)
                ),

                "target_participation_rate": (
                    float(r)
                ),

                "window_type": (
                    window_type
                ),

                "horizon": (
                    int(horizon)
                ),

                "n_windows": (
                    len(
                        subset
                    )
                ),

                "profitable_window_rate": float(
                    subset[
                        "profitable"
                    ].mean()
                ),

                "mean_roi": float(
                    np.mean(
                        roi_values
                    )
                ),

                "median_roi": float(
                    np.median(
                        roi_values
                    )
                ),

                "std_roi": float(
                    np.std(
                        roi_values,
                        ddof=1,
                    )
                    if len(
                        roi_values
                    ) > 1
                    else 0.0
                ),

                "min_roi": float(
                    np.min(
                        roi_values
                    )
                ),

                "max_roi": float(
                    np.max(
                        roi_values
                    )
                ),

                "mean_profit": float(
                    np.mean(
                        profit_values
                    )
                ),

                "median_profit": float(
                    np.median(
                        profit_values
                    )
                ),

                "min_profit": float(
                    np.min(
                        profit_values
                    )
                ),

                "max_profit": float(
                    np.max(
                        profit_values
                    )
                ),

                "mean_bet_days": float(
                    subset[
                        "n_bet_days"
                    ].mean()
                ),

                "median_bet_days": float(
                    subset[
                        "n_bet_days"
                    ].median()
                ),

                "mean_calendar_span_days": float(
                    subset[
                        "calendar_span_days"
                    ].mean()
                ),

                "mean_hit_rate": float(
                    subset[
                        "hit_rate"
                    ].mean()
                ),
            }
        )


    return pd.DataFrame(
        records
    )


# ============================================================
# BEST HORIZON PER CONFIG
# ============================================================

def find_best_horizon_by_config(
    horizon_summary: pd.DataFrame,
) -> pd.DataFrame:

    if horizon_summary.empty:

        return pd.DataFrame()


    eligible = (
        horizon_summary.loc[
            horizon_summary[
                "n_windows"
            ]
            .ge(
                MIN_WINDOWS_FOR_BEST
            )
        ]
        .copy()
    )


    if eligible.empty:

        return pd.DataFrame()


    records = []


    config_columns = (
        get_config_columns()
    )


    for key, subset in (
        eligible.groupby(
            config_columns
            + ["window_type"],
            sort=False,
        )
    ):

        (
            model,
            m,
            r,
            window_type,
        ) = key


        # -----------------------------------------------
        # Primary:
        #     median ROI cao nhất.
        #
        # Tie-break:
        #     profitable-window rate cao hơn,
        #     std ROI thấp hơn.
        # -----------------------------------------------

        best = (
            subset
            .sort_values(
                [
                    "median_roi",
                    "profitable_window_rate",
                    "std_roi",
                ],
                ascending=[
                    False,
                    False,
                    True,
                ],
            )
            .iloc[0]
        )


        records.append(
            best.to_dict()
        )


    return pd.DataFrame(
        records
    )


# ============================================================
# BEST OVERALL HORIZONS
# ============================================================

def find_best_overall(
    best_by_config: pd.DataFrame,
) -> pd.DataFrame:

    if best_by_config.empty:

        return pd.DataFrame()


    rows = []


    for window_type in [
        "calendar",
        "bet",
    ]:

        subset = (
            best_by_config.loc[
                best_by_config[
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


        # ----------------------------------------------------
        # Best median ROI
        # ----------------------------------------------------

        best_roi = (
            subset
            .sort_values(
                [
                    "median_roi",
                    "profitable_window_rate",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .iloc[0]
            .copy()
        )

        best_roi[
            "selection_metric"
        ] = (
            "max_median_roi"
        )

        rows.append(
            best_roi
        )


        # ----------------------------------------------------
        # Best profitable window rate
        # ----------------------------------------------------

        best_stability = (
            subset
            .sort_values(
                [
                    "profitable_window_rate",
                    "median_roi",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .iloc[0]
            .copy()
        )

        best_stability[
            "selection_metric"
        ] = (
            "max_profitable_window_rate"
        )

        rows.append(
            best_stability
        )


    return pd.DataFrame(
        rows
    )


# ============================================================
# PRINT FIRST POSITIVE
# ============================================================

def print_first_positive(
    first_positive: pd.DataFrame,
    summary: pd.DataFrame,
):

    merged = (
        first_positive
        .merge(
            summary[
                [
                    "model",
                    "m",
                    "target_participation_rate",
                    "roi",
                ]
            ],
            on=[
                "model",
                "m",
                "target_participation_rate",
            ],
            how="left",
            suffixes=(
                "",
                "_strategy15",
            ),
        )
    )


    top = (
        merged
        .sort_values(
            "roi",
            ascending=False,
            na_position="last",
        )
        .head(
            30
        )
    )


    columns = [
        "model",
        "m",
        "target_participation_rate",
        "n_bet_days",
        "number_hits",
        "first_positive_bet_day",
        "stable_positive_bet_day",
        "first_positive_calendar_days",
        "stable_positive_calendar_days",
        "final_roi",
    ]


    print(
        "\n"
        + "=" * 190
    )

    print(
        "FIRST / STABLE POSITIVE - "
        "TOP STRATEGY 15 CONFIGURATIONS"
    )

    print(
        "=" * 190
    )


    print(
        top[
            columns
        ]
        .to_string(
            index=False,
            formatters={
                "target_participation_rate": (
                    "{:.0%}".format
                ),

                "final_roi": (
                    "{:.2%}".format
                ),
            },
        )
    )


# ============================================================
# PRINT BEST HORIZONS
# ============================================================

def print_best_horizons(
    best_overall: pd.DataFrame,
):

    if best_overall.empty:

        print(
            "\nKhông có đủ horizon "
            "để phân tích."
        )

        return


    columns = [
        "selection_metric",
        "window_type",
        "model",
        "m",
        "target_participation_rate",
        "horizon",
        "n_windows",
        "median_roi",
        "mean_roi",
        "profitable_window_rate",
        "std_roi",
        "mean_bet_days",
        "mean_calendar_span_days",
    ]


    print(
        "\n"
        + "=" * 200
    )

    print(
        "BEST OBSERVED PROFIT HORIZONS"
    )

    print(
        "=" * 200
    )


    print(
        best_overall[
            columns
        ]
        .to_string(
            index=False,
            formatters={
                "target_participation_rate": (
                    "{:.0%}".format
                ),

                "median_roi": (
                    "{:.2%}".format
                ),

                "mean_roi": (
                    "{:.2%}".format
                ),

                "profitable_window_rate": (
                    "{:.2%}".format
                ),

                "std_roi": (
                    "{:.2%}".format
                ),

                "mean_bet_days": (
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

    (
        daily,
        strategy_summary,
    ) = load_data()


    print(
        f"Daily rows: "
        f"{len(daily):,}"
    )

    print(
        f"Strategy configs: "
        f"{len(strategy_summary):,}"
    )

    print(
        "Calendar horizons: "
        + ", ".join(
            str(x)
            for x
            in CALENDAR_HORIZONS
        )
    )

    print(
        "Bet horizons: "
        + ", ".join(
            str(x)
            for x
            in BET_HORIZONS
        )
    )


    # ========================================================
    # FIRST POSITIVE
    # ========================================================

    print(
        "\nCalculating "
        "first/stable positive..."
    )


    first_positive_records = []


    grouped_configs = (
        daily
        .groupby(
            get_config_columns(),
            sort=False,
        )
    )


    for key, config_df in (
        grouped_configs
    ):

        first_positive_records.append(
            first_positive_analysis(
                config_df
            )
        )


    first_positive = (
        pd.DataFrame(
            first_positive_records
        )
    )


    # ========================================================
    # ROLLING WINDOWS
    # ========================================================

    print(
        "\nBuilding rolling windows..."
    )


    windows = (
        build_all_windows(
            daily
        )
    )


    print(
        f"Window rows: "
        f"{len(windows):,}"
    )


    # ========================================================
    # SUMMARIZE WINDOWS
    # ========================================================

    horizon_summary = (
        summarize_windows(
            windows
        )
    )


    print(
        f"Horizon summary rows: "
        f"{len(horizon_summary):,}"
    )


    # ========================================================
    # BEST HORIZON PER CONFIG
    # ========================================================

    best_by_config = (
        find_best_horizon_by_config(
            horizon_summary
        )
    )


    best_overall = (
        find_best_overall(
            best_by_config
        )
    )


    # ========================================================
    # PRINT
    # ========================================================

    print_first_positive(
        first_positive,
        strategy_summary,
    )


    print_best_horizons(
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

    windows.to_csv(
        OUTPUT_WINDOWS,
        index=False,
        encoding="utf-8-sig",
    )

    horizon_summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    best_by_config.to_csv(
        OUTPUT_BEST_HORIZON,
        index=False,
        encoding="utf-8-sig",
    )

    best_overall.to_csv(
        OUTPUT_BEST_OVERALL,
        index=False,
        encoding="utf-8-sig",
    )


    print(
        "\nĐã lưu:"
    )

    print(
        OUTPUT_FIRST_POSITIVE
    )

    print(
        OUTPUT_WINDOWS
    )

    print(
        OUTPUT_SUMMARY
    )

    print(
        OUTPUT_BEST_HORIZON
    )

    print(
        OUTPUT_BEST_OVERALL
    )


if __name__ == "__main__":
    main()