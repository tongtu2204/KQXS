"""
09b - CATBOOST PERIODIC RETRAIN - OPTIMIZED

CatBoost 100 lớp với walk-forward periodic retraining.

Tối ưu runtime:
    - provisional model:
          max 300 iterations
          early stopping 20
    - final model:
          chỉ fit đúng số iterations đã chọn
    - thread_count = -1
    - feature build một lần
    - không verbose CatBoost từng 100 vòng

Logic:
    Past
      -> provisional train/validation
      -> chọn best_iteration + blend weight
      -> refit trên toàn bộ history
      -> predict future block
      -> actual block trở thành history
      -> retrain block tiếp theo

Không leakage:
    actual ngày t chỉ được dùng từ ngày t+1 trở đi.
"""

from pathlib import Path
import importlib.util
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


# ============================================================
# PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

BASE_SCRIPT = (
    PROJECT_DIR
    / "experiments"
    / "p2_models"
    / "09_modern_ml_last2.py"
)

TABLE_DIR = (
    PROJECT_DIR
    / "artifacts"
    / "tables"
)


STATIC_PREDICTION_FILE = (
    TABLE_DIR
    / "modern_ml_last2_predictions.csv"
)

OUTPUT_PREDICTIONS = (
    TABLE_DIR
    / "catboost_retrain_predictions.csv"
)

OUTPUT_BLOCKS = (
    TABLE_DIR
    / "catboost_retrain_blocks.csv"
)

OUTPUT_SUMMARY = (
    TABLE_DIR
    / "catboost_retrain_summary.csv"
)

OUTPUT_COMPARISON = (
    TABLE_DIR
    / "catboost_static_vs_retrain.csv"
)


# ============================================================
# CONFIG
# ============================================================

NUMBER_OF_CLASSES = 100
RANDOM_STATE = 42


# ------------------------------------------------------------
# Retrain cadence
# ------------------------------------------------------------

RETRAIN_DAYS = 30


# ------------------------------------------------------------
# Validation history
# ------------------------------------------------------------

VALIDATION_YEARS = 2


# ------------------------------------------------------------
# Minimum train rows
# ------------------------------------------------------------

MIN_TRAIN_ROWS = 1_000


# ------------------------------------------------------------
# PROVISIONAL model
#
# Chỉ dùng để:
#     - chọn best_iteration
#     - chọn blend weight
#
# Log hiện tại best iteration ~5-6,
# nên patience 20 là dư an toàn.
# ------------------------------------------------------------

PROVISIONAL_MAX_ITERATIONS = 300

PROVISIONAL_EARLY_STOPPING = 20


# ------------------------------------------------------------
# Final model
# ------------------------------------------------------------

MIN_FINAL_ITERATIONS = 1


# ------------------------------------------------------------
# GPU
#
# False:
#     CPU multithread
#
# True:
#     CatBoost GPU
#
# Nếu máy không có CUDA thì để False.
# ------------------------------------------------------------

USE_GPU = False

GPU_DEVICE = "0"


# ------------------------------------------------------------
# Test folds
# ------------------------------------------------------------

FOLDS = [
    {
        "name": "2020-2021",
        "test_start": 2020,
        "test_end": 2021,
    },
    {
        "name": "2022-2023",
        "test_start": 2022,
        "test_end": 2023,
    },
    {
        "name": "2024-2026",
        "test_start": 2024,
        "test_end": 2026,
    },
]


PROBABILITY_COLUMNS = [
    f"p_{number:02d}"
    for number in range(
        NUMBER_OF_CLASSES
    )
]


# ============================================================
# LOAD BASE 09
# ============================================================

def load_base_module():

    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Không tìm thấy:\n{BASE_SCRIPT}"
        )

    spec = (
        importlib.util
        .spec_from_file_location(
            "modern_ml_last2",
            BASE_SCRIPT,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise ImportError(
            f"Không load được:\n{BASE_SCRIPT}"
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


BASE = load_base_module()


# ============================================================
# DATA
# ============================================================

def prepare_full_data():

    raw = (
        BASE.read_data()
    )

    features = (
        BASE.build_features(
            raw
        )
    )

    valid_mask = (
        features
        .notna()
        .all(axis=1)
    )

    features = (
        features.loc[
            valid_mask
        ]
        .astype(float)
        .reset_index(drop=True)
    )

    target = (
        raw.loc[
            valid_mask,
            "last_2_target",
        ]
        .astype(int)
        .reset_index(drop=True)
    )

    dates = (
        raw.loc[
            valid_mask,
            "date",
        ]
        .reset_index(drop=True)
    )

    return (
        features,
        target,
        dates,
    )


# ============================================================
# MODEL FACTORIES
# ============================================================

def common_model_params():

    params = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "learning_rate": 0.04,
        "depth": 7,
        "l2_leaf_reg": 15,
        "random_strength": 1,
        "random_seed": RANDOM_STATE,
        "allow_writing_files": False,
        "verbose": False,
    }

    if USE_GPU:

        params.update(
            {
                "task_type": "GPU",
                "devices": GPU_DEVICE,
            }
        )

    else:

        params.update(
            {
                "thread_count": -1,
            }
        )

    return params


def create_provisional_model():

    params = (
        common_model_params()
    )

    params.update(
        {
            "iterations": (
                PROVISIONAL_MAX_ITERATIONS
            ),

            "early_stopping_rounds": (
                PROVISIONAL_EARLY_STOPPING
            ),
        }
    )

    return CatBoostClassifier(
        **params
    )


def create_final_model(
    iterations: int,
):

    iterations = max(
        int(iterations),
        MIN_FINAL_ITERATIONS,
    )

    params = (
        common_model_params()
    )

    params.update(
        {
            "iterations": (
                iterations
            ),
        }
    )

    return CatBoostClassifier(
        **params
    )


# ============================================================
# BLOCKS
# ============================================================

def build_test_blocks(
    dates,
    fold,
):

    fold_mask = (
        dates
        .dt.year
        .between(
            fold["test_start"],
            fold["test_end"],
        )
    )

    fold_dates = (
        dates.loc[
            fold_mask
        ]
        .sort_values()
    )

    if fold_dates.empty:
        return []

    first_date = (
        fold_dates.iloc[0]
    )

    last_date = (
        fold_dates.iloc[-1]
    )

    blocks = []

    block_start = (
        first_date
    )

    block_id = 1


    while (
        block_start
        <= last_date
    ):

        block_end = min(
            block_start
            + pd.Timedelta(
                days=(
                    RETRAIN_DAYS
                    - 1
                )
            ),
            last_date,
        )

        blocks.append(
            {
                "block_id": (
                    block_id
                ),

                "block_start": (
                    block_start
                ),

                "block_end": (
                    block_end
                ),
            }
        )

        block_start = (
            block_end
            + pd.Timedelta(
                days=1
            )
        )

        block_id += 1


    return blocks


# ============================================================
# HISTORY SPLIT
# ============================================================

def get_history_masks(
    dates,
    prediction_start,
):

    history_mask = (
        dates
        < prediction_start
    )

    validation_start = (
        prediction_start
        - pd.DateOffset(
            years=VALIDATION_YEARS
        )
    )

    validation_mask = (
        history_mask
        & dates.ge(
            validation_start
        )
    )

    train_old_mask = (
        history_mask
        & dates.lt(
            validation_start
        )
    )

    return (
        train_old_mask,
        validation_mask,
        history_mask,
    )


# ============================================================
# RUN ONE BLOCK
# ============================================================

def run_block(
    features,
    target,
    dates,
    fold,
    block,
):

    block_start = (
        block[
            "block_start"
        ]
    )

    block_end = (
        block[
            "block_end"
        ]
    )


    (
        train_old_mask,
        validation_mask,
        full_history_mask,
    ) = (
        get_history_masks(
            dates,
            block_start,
        )
    )


    # --------------------------------------------------------
    # OLD TRAIN
    # --------------------------------------------------------

    x_train_old = (
        features.loc[
            train_old_mask
        ]
    )

    y_train_old = (
        target.loc[
            train_old_mask
        ]
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    x_validation = (
        features.loc[
            validation_mask
        ]
    )

    y_validation = (
        target.loc[
            validation_mask
        ]
    )


    # --------------------------------------------------------
    # FULL HISTORY
    # --------------------------------------------------------

    x_history = (
        features.loc[
            full_history_mask
        ]
    )

    y_history = (
        target.loc[
            full_history_mask
        ]
    )


    # --------------------------------------------------------
    # FUTURE BLOCK
    # --------------------------------------------------------

    prediction_mask = (
        dates.between(
            block_start,
            block_end,
        )
        & dates
        .dt.year
        .between(
            fold["test_start"],
            fold["test_end"],
        )
    )


    x_test = (
        features.loc[
            prediction_mask
        ]
    )

    y_test = (
        target.loc[
            prediction_mask
        ]
    )

    test_dates = (
        dates.loc[
            prediction_mask
        ]
    )


    # --------------------------------------------------------
    # CHECK
    # --------------------------------------------------------

    if (
        len(x_train_old)
        < MIN_TRAIN_ROWS
    ):

        return (
            None,
            None,
        )


    if (
        len(x_validation) == 0
        or len(x_test) == 0
    ):

        return (
            None,
            None,
        )


    # ========================================================
    # TIMER
    # ========================================================

    block_timer = (
        time.perf_counter()
    )


    # ========================================================
    # STEP 1
    # provisional train
    # ========================================================

    provisional_timer = (
        time.perf_counter()
    )


    provisional_model = (
        create_provisional_model()
    )


    provisional_model.fit(
        x_train_old,
        y_train_old,

        eval_set=(
            x_validation,
            y_validation,
        ),

        use_best_model=True,
    )


    provisional_seconds = (
        time.perf_counter()
        - provisional_timer
    )


    # --------------------------------------------------------
    # best iteration
    # --------------------------------------------------------

    best_iteration_zero_based = (
        provisional_model
        .get_best_iteration()
    )


    if (
        best_iteration_zero_based
        is None
        or best_iteration_zero_based
        < 0
    ):

        selected_iterations = (
            MIN_FINAL_ITERATIONS
        )

    else:

        selected_iterations = (
            int(
                best_iteration_zero_based
            )
            + 1
        )


    # ========================================================
    # STEP 2
    # validation blend
    # ========================================================

    validation_probability_raw = (
        BASE.align_probabilities(
            provisional_model,

            provisional_model
            .predict_proba(
                x_validation
            ),
        )
    )


    (
        selected_weight,
        validation_log_loss,
    ) = (
        BASE.select_blend_weight(
            y_validation
            .to_numpy(
                dtype=int
            ),

            validation_probability_raw,
        )
    )


    # ========================================================
    # STEP 3
    # final refit full history
    # ========================================================

    final_timer = (
        time.perf_counter()
    )


    final_model = (
        create_final_model(
            selected_iterations
        )
    )


    final_model.fit(
        x_history,
        y_history,
    )


    final_seconds = (
        time.perf_counter()
        - final_timer
    )


    # ========================================================
    # STEP 4
    # future predict
    # ========================================================

    raw_test_probabilities = (
        BASE.align_probabilities(
            final_model,

            final_model
            .predict_proba(
                x_test
            ),
        )
    )


    test_probabilities = (
        BASE.blend_with_uniform(
            raw_test_probabilities,
            selected_weight,
        )
    )


    y_test_array = (
        y_test
        .to_numpy(
            dtype=int
        )
    )


    metrics = (
        BASE.evaluate(
            y_test_array,
            test_probabilities,
        )
    )


    total_seconds = (
        time.perf_counter()
        - block_timer
    )


    # ========================================================
    # PRINT PROGRESS
    # ========================================================

    print(
        f"{fold['name']} | "
        f"block {block['block_id']:02d} | "
        f"{block_start.date()} -> "
        f"{block_end.date()} | "
        f"hist={len(x_history):,} | "
        f"pred={len(x_test):,} | "
        f"iter={selected_iterations:3d} | "
        f"w={selected_weight:.2f} | "
        f"prov={provisional_seconds:.1f}s | "
        f"final={final_seconds:.1f}s | "
        f"total={total_seconds:.1f}s"
    )


    # ========================================================
    # DAILY OUTPUT
    # ========================================================

    predicted = (
        test_probabilities
        .argmax(
            axis=1
        )
    )


    prediction_df = pd.DataFrame(
        {
            "date": (
                test_dates
                .to_numpy()
            ),

            "fold": (
                fold[
                    "name"
                ]
            ),

            "actual": (
                y_test_array
            ),

            "predicted": (
                predicted
            ),

            "retrain_block": (
                block[
                    "block_id"
                ]
            ),

            "retrain_date": (
                block_start
            ),

            "history_end": (
                block_start
                - pd.Timedelta(
                    days=1
                )
            ),

            "selected_iterations": (
                selected_iterations
            ),

            "model_weight": (
                selected_weight
            ),
        }
    )


    probability_df = pd.DataFrame(
        test_probabilities,

        columns=(
            PROBABILITY_COLUMNS
        ),
    )


    prediction_df = pd.concat(
        [
            prediction_df
            .reset_index(
                drop=True
            ),

            probability_df,
        ],

        axis=1,
    )


    # ========================================================
    # BLOCK SUMMARY
    # ========================================================

    block_summary = {
        "fold": (
            fold[
                "name"
            ]
        ),

        "block_id": (
            block[
                "block_id"
            ]
        ),

        "block_start": (
            block_start
        ),

        "block_end": (
            block_end
        ),

        "history_end": (
            block_start
            - pd.Timedelta(
                days=1
            )
        ),

        "train_old_size": (
            len(
                x_train_old
            )
        ),

        "validation_size": (
            len(
                x_validation
            )
        ),

        "full_history_size": (
            len(
                x_history
            )
        ),

        "test_size": (
            len(
                x_test
            )
        ),

        "selected_iterations": (
            selected_iterations
        ),

        "model_weight": (
            selected_weight
        ),

        "validation_log_loss": (
            validation_log_loss
        ),

        "provisional_seconds": (
            provisional_seconds
        ),

        "final_seconds": (
            final_seconds
        ),

        "total_seconds": (
            total_seconds
        ),

        **metrics,
    }


    return (
        prediction_df,
        block_summary,
    )


# ============================================================
# SUMMARY
# ============================================================

def summarize_predictions(
    predictions,
):

    records = []


    for (
        fold_name,
        subset,
    ) in predictions.groupby(
        "fold",
        sort=False,
    ):

        probabilities = (
            subset[
                PROBABILITY_COLUMNS
            ]
            .to_numpy(
                dtype=float
            )
        )

        y_true = (
            subset[
                "actual"
            ]
            .to_numpy(
                dtype=int
            )
        )


        metrics = (
            BASE.evaluate(
                y_true,
                probabilities,
            )
        )


        records.append(
            {
                "fold": (
                    fold_name
                ),

                "n_predictions": (
                    len(
                        subset
                    )
                ),

                "n_retrains": (
                    subset[
                        "retrain_block"
                    ]
                    .nunique()
                ),

                "mean_selected_iterations": float(
                    subset[
                        "selected_iterations"
                    ]
                    .mean()
                ),

                "mean_model_weight": float(
                    subset[
                        "model_weight"
                    ]
                    .mean()
                ),

                **metrics,
            }
        )


    # --------------------------------------------------------
    # ALL
    # --------------------------------------------------------

    probabilities = (
        predictions[
            PROBABILITY_COLUMNS
        ]
        .to_numpy(
            dtype=float
        )
    )

    y_true = (
        predictions[
            "actual"
        ]
        .to_numpy(
            dtype=int
        )
    )


    metrics = (
        BASE.evaluate(
            y_true,
            probabilities,
        )
    )


    records.append(
        {
            "fold": "ALL",

            "n_predictions": (
                len(
                    predictions
                )
            ),

            "n_retrains": (
                predictions[
                    [
                        "fold",
                        "retrain_block",
                    ]
                ]
                .drop_duplicates()
                .shape[0]
            ),

            "mean_selected_iterations": float(
                predictions[
                    "selected_iterations"
                ]
                .mean()
            ),

            "mean_model_weight": float(
                predictions[
                    "model_weight"
                ]
                .mean()
            ),

            **metrics,
        }
    )


    return pd.DataFrame(
        records
    )


# ============================================================
# STATIC COMPARISON
# ============================================================

def build_static_comparison(
    retrain_predictions,
):

    records = []


    # --------------------------------------------------------
    # RETRAIN
    # --------------------------------------------------------

    for (
        fold_name,
        subset,
    ) in retrain_predictions.groupby(
        "fold",
        sort=False,
    ):

        probabilities = (
            subset[
                PROBABILITY_COLUMNS
            ]
            .to_numpy(
                dtype=float
            )
        )

        y_true = (
            subset[
                "actual"
            ]
            .to_numpy(
                dtype=int
            )
        )


        metrics = (
            BASE.evaluate(
                y_true,
                probabilities,
            )
        )


        records.append(
            {
                "model": (
                    "catboost_retrain"
                ),

                "fold": (
                    fold_name
                ),

                "n": (
                    len(
                        subset
                    )
                ),

                **metrics,
            }
        )


    # --------------------------------------------------------
    # STATIC
    # --------------------------------------------------------

    if (
        STATIC_PREDICTION_FILE
        .exists()
    ):

        static_df = pd.read_csv(
            STATIC_PREDICTION_FILE,
            parse_dates=[
                "date"
            ],
        )


        for (
            fold_name,
            subset,
        ) in static_df.groupby(
            "fold",
            sort=False,
        ):

            probabilities = (
                subset[
                    PROBABILITY_COLUMNS
                ]
                .to_numpy(
                    dtype=float
                )
            )

            y_true = (
                subset[
                    "actual"
                ]
                .to_numpy(
                    dtype=int
                )
            )


            metrics = (
                BASE.evaluate(
                    y_true,
                    probabilities,
                )
            )


            records.append(
                {
                    "model": (
                        "catboost_static"
                    ),

                    "fold": (
                        fold_name
                    ),

                    "n": (
                        len(
                            subset
                        )
                    ),

                    **metrics,
                }
            )


    comparison = pd.DataFrame(
        records
    )


    # ========================================================
    # ALL
    # ========================================================

    all_records = []


    # retrain all
    probabilities = (
        retrain_predictions[
            PROBABILITY_COLUMNS
        ]
        .to_numpy(
            dtype=float
        )
    )

    y_true = (
        retrain_predictions[
            "actual"
        ]
        .to_numpy(
            dtype=int
        )
    )

    all_records.append(
        {
            "model": (
                "catboost_retrain"
            ),

            "fold": "ALL",

            "n": (
                len(
                    retrain_predictions
                )
            ),

            **BASE.evaluate(
                y_true,
                probabilities,
            ),
        }
    )


    # static all
    if (
        STATIC_PREDICTION_FILE
        .exists()
    ):

        static_df = pd.read_csv(
            STATIC_PREDICTION_FILE
        )

        probabilities = (
            static_df[
                PROBABILITY_COLUMNS
            ]
            .to_numpy(
                dtype=float
            )
        )

        y_true = (
            static_df[
                "actual"
            ]
            .to_numpy(
                dtype=int
            )
        )

        all_records.append(
            {
                "model": (
                    "catboost_static"
                ),

                "fold": "ALL",

                "n": (
                    len(
                        static_df
                    )
                ),

                **BASE.evaluate(
                    y_true,
                    probabilities,
                ),
            }
        )


    comparison = pd.concat(
        [
            comparison,
            pd.DataFrame(
                all_records
            ),
        ],

        ignore_index=True,
    )


    return comparison


# ============================================================
# PRINT
# ============================================================

def print_summary(
    summary,
):

    columns = [
        "fold",
        "n_predictions",
        "n_retrains",
        "mean_selected_iterations",
        "mean_model_weight",
        "top_1_accuracy",
        "top_5_accuracy",
        "top_10_accuracy",
        "top_20_accuracy",
        "log_loss",
        "mean_true_rank",
    ]


    print(
        "\n"
        + "=" * 180
    )

    print(
        "CATBOOST PERIODIC RETRAIN SUMMARY"
    )

    print(
        "=" * 180
    )


    print(
        summary[
            columns
        ]
        .to_string(
            index=False,

            formatters={
                "mean_selected_iterations": (
                    "{:.1f}".format
                ),

                "mean_model_weight": (
                    "{:.4f}".format
                ),

                "top_1_accuracy": (
                    "{:.4%}".format
                ),

                "top_5_accuracy": (
                    "{:.4%}".format
                ),

                "top_10_accuracy": (
                    "{:.4%}".format
                ),

                "top_20_accuracy": (
                    "{:.4%}".format
                ),

                "log_loss": (
                    "{:.6f}".format
                ),

                "mean_true_rank": (
                    "{:.2f}".format
                ),
            },
        )
    )


def print_comparison(
    comparison,
):

    if comparison.empty:
        return


    columns = [
        "model",
        "fold",
        "n",
        "top_1_accuracy",
        "top_5_accuracy",
        "top_10_accuracy",
        "top_20_accuracy",
        "log_loss",
        "mean_true_rank",
    ]


    print(
        "\n"
        + "=" * 180
    )

    print(
        "STATIC CATBOOST VS PERIODIC RETRAIN"
    )

    print(
        "=" * 180
    )


    print(
        comparison[
            columns
        ]
        .sort_values(
            [
                "fold",
                "model",
            ]
        )
        .to_string(
            index=False,

            formatters={
                "top_1_accuracy": (
                    "{:.4%}".format
                ),

                "top_5_accuracy": (
                    "{:.4%}".format
                ),

                "top_10_accuracy": (
                    "{:.4%}".format
                ),

                "top_20_accuracy": (
                    "{:.4%}".format
                ),

                "log_loss": (
                    "{:.6f}".format
                ),

                "mean_true_rank": (
                    "{:.2f}".format
                ),
            },
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    start_time = (
        time.perf_counter()
    )


    (
        features,
        target,
        dates,
    ) = (
        prepare_full_data()
    )


    print(
        f"Rows after feature preparation: "
        f"{len(features):,}"
    )

    print(
        f"Number of features: "
        f"{features.shape[1]:,}"
    )

    print(
        f"Retrain every: "
        f"{RETRAIN_DAYS} calendar days"
    )

    print(
        f"Validation window: "
        f"{VALIDATION_YEARS} years"
    )

    print(
        f"Provisional max iterations: "
        f"{PROVISIONAL_MAX_ITERATIONS}"
    )

    print(
        f"Early stopping patience: "
        f"{PROVISIONAL_EARLY_STOPPING}"
    )

    print(
        f"GPU: "
        f"{USE_GPU}"
    )


    prediction_frames = []

    block_records = []


    # ========================================================
    # FOLDS
    # ========================================================

    for fold in FOLDS:

        blocks = (
            build_test_blocks(
                dates,
                fold,
            )
        )


        print(
            "\n"
            + "=" * 110
        )

        print(
            f"FOLD {fold['name']} "
            f"| {len(blocks)} blocks"
        )

        print(
            "=" * 110
        )


        for block in blocks:

            (
                prediction_df,
                block_summary,
            ) = (
                run_block(
                    features,
                    target,
                    dates,
                    fold,
                    block,
                )
            )


            if (
                prediction_df
                is not None
            ):

                prediction_frames.append(
                    prediction_df
                )


            if (
                block_summary
                is not None
            ):

                block_records.append(
                    block_summary
                )


    # ========================================================
    # CHECK
    # ========================================================

    if not prediction_frames:

        raise RuntimeError(
            "Không tạo được prediction."
        )


    # ========================================================
    # COMBINE
    # ========================================================

    predictions = (
        pd.concat(
            prediction_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "date",
                "fold",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    block_summary = (
        pd.DataFrame(
            block_records
        )
    )


    summary = (
        summarize_predictions(
            predictions
        )
    )


    comparison = (
        build_static_comparison(
            predictions
        )
    )


    # ========================================================
    # PRINT
    # ========================================================

    print_summary(
        summary
    )


    print_comparison(
        comparison
    )


    # ========================================================
    # SAVE
    # ========================================================

    predictions.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
        encoding="utf-8-sig",
    )


    block_summary.to_csv(
        OUTPUT_BLOCKS,
        index=False,
        encoding="utf-8-sig",
    )


    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )


    comparison.to_csv(
        OUTPUT_COMPARISON,
        index=False,
        encoding="utf-8-sig",
    )


    total_minutes = (
        (
            time.perf_counter()
            - start_time
        )
        / 60
    )


    print(
        "\n"
        + "=" * 100
    )

    print(
        f"TOTAL RUNTIME: "
        f"{total_minutes:.2f} minutes"
    )

    print(
        "=" * 100
    )

    print(
        "\nĐã lưu:"
    )

    print(
        OUTPUT_PREDICTIONS
    )

    print(
        OUTPUT_BLOCKS
    )

    print(
        OUTPUT_SUMMARY
    )

    print(
        OUTPUT_COMPARISON
    )


if __name__ == "__main__":
    main()