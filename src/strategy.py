"""Leakage-safe Top-m strategy selection and evaluation utilities."""

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.config import NUMBER_OF_CLASSES, PROBABILITY_COLUMNS, RANDOM_STATE


COST_PER_NUMBER = 10_000
PAYOUT_IF_HIT = 800_000
M_VALUES = tuple(range(1, 26))
TARGET_PARTICIPATION_RATES = (0.10, 0.20, 0.30, 0.50)
MIN_VALIDATION_BETS = 50


@dataclass(frozen=True)
class FrozenStrategy:
    strategy: str
    model: str
    m: int
    target_participation_rate: float | None
    confidence_threshold: float | None
    tie_acceptance_rate: float | None
    validation_selection_score: float
    validation_n_bets: int
    validation_roi: float


def deterministic_priority(seed: int = RANDOM_STATE) -> np.ndarray:
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(NUMBER_OF_CLASSES)
    priority = np.empty(NUMBER_OF_CLASSES, dtype=int)
    priority[permutation] = np.arange(NUMBER_OF_CLASSES)
    return priority


TIE_BREAK_PRIORITY = deterministic_priority()


def normalize_probabilities(frame: pd.DataFrame) -> np.ndarray:
    probabilities = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("Probability matrix contains non-finite values")
    probabilities = np.clip(probabilities, 0.0, 1.0)
    row_sum = probabilities.sum(axis=1, keepdims=True)
    if (row_sum <= 0).any():
        raise ValueError("Probability row has non-positive mass")
    return probabilities / row_sum


def rank_probabilities(probabilities: np.ndarray) -> np.ndarray:
    priorities = np.broadcast_to(TIE_BREAK_PRIORITY, probabilities.shape)
    return np.lexsort((priorities, -probabilities), axis=1)


def model_metrics(frame: pd.DataFrame) -> dict[str, float | int | str]:
    probabilities = normalize_probabilities(frame)
    actual = frame["actual"].to_numpy(dtype=int)
    rows = np.arange(len(frame))
    order = rank_probabilities(probabilities)
    inverse = np.empty_like(order)
    inverse[rows[:, None], order] = np.arange(NUMBER_OF_CLASSES)
    actual_rank = inverse[rows, actual] + 1
    actual_probability = probabilities[rows, actual]
    one_hot = np.eye(NUMBER_OF_CLASSES)[actual]

    result: dict[str, float | int | str] = {
        "model": str(frame["model_key"].iloc[0]),
        "n_days": len(frame),
        "log_loss": float(-np.log(np.clip(actual_probability, 1e-15, 1.0)).mean()),
        "brier_score": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "mean_true_rank": float(actual_rank.mean()),
    }
    for top_m in (1, 3, 5, 10, 20):
        result[f"top_{top_m}_accuracy"] = float((actual_rank <= top_m).mean())
    return result


def prepare_ranking(frame: pd.DataFrame) -> dict[str, object]:
    probabilities = normalize_probabilities(frame)
    actual = frame["actual"].to_numpy(dtype=int)
    order = rank_probabilities(probabilities)
    rows = np.arange(len(frame))
    inverse = np.empty_like(order)
    inverse[rows[:, None], order] = np.arange(NUMBER_OF_CLASSES)
    actual_rank = inverse[rows, actual] + 1
    sorted_probability = np.take_along_axis(probabilities, order, axis=1)
    cumulative_probability = sorted_probability.cumsum(axis=1)
    return {
        "frame": frame.reset_index(drop=True),
        "order": order,
        "actual_rank": actual_rank,
        "cumulative_probability": cumulative_probability,
    }


def wilson_lower_bound(successes: int, trials: int) -> float:
    if trials <= 0:
        return float("nan")
    z = 1.959963984540054
    p_hat = successes / trials
    z2 = z * z
    denominator = 1 + z2 / trials
    center = p_hat + z2 / (2 * trials)
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z2 / (4 * trials)) / trials)
    return float((center - margin) / denominator)


def _hash_fraction(date: pd.Timestamp, label: str) -> float:
    value = f"{pd.Timestamp(date).date().isoformat()}|{label}|{RANDOM_STATE}"
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return integer / 2**64


def fit_selective_threshold(
    confidence: np.ndarray,
    target_rate: float,
) -> tuple[float, float]:
    if not 0 < target_rate <= 1:
        raise ValueError("target_rate must be in (0, 1]")
    target_count = int(round(len(confidence) * target_rate))
    target_count = min(max(target_count, 1), len(confidence))
    descending = np.sort(confidence)[::-1]
    threshold = float(descending[target_count - 1])
    greater_count = int((confidence > threshold).sum())
    equal_count = int(np.isclose(confidence, threshold, rtol=1e-12, atol=1e-15).sum())
    needed_from_ties = min(max(target_count - greater_count, 0), equal_count)
    tie_rate = needed_from_ties / equal_count if equal_count else 0.0
    return threshold, float(tie_rate)


def apply_selective_threshold(
    dates: pd.Series,
    confidence: np.ndarray,
    threshold: float,
    tie_acceptance_rate: float,
    label: str,
) -> np.ndarray:
    greater = confidence > threshold
    equal = np.isclose(confidence, threshold, rtol=1e-12, atol=1e-15)
    accepted_ties = np.fromiter(
        (
            _hash_fraction(date, label) < tie_acceptance_rate
            for date in dates
        ),
        dtype=bool,
        count=len(dates),
    )
    return greater | (equal & accepted_ties)


def evaluate_bets(
    prepared: dict[str, object],
    *,
    m: int,
    bet: np.ndarray,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    frame = prepared["frame"].copy()
    actual_rank = np.asarray(prepared["actual_rank"])
    order = np.asarray(prepared["order"])
    confidence = np.asarray(prepared["cumulative_probability"])[:, m - 1]
    hit = actual_rank <= m
    bet = np.asarray(bet, dtype=bool)
    cost = bet.astype(np.int64) * m * COST_PER_NUMBER
    revenue = (bet & hit).astype(np.int64) * PAYOUT_IF_HIT
    profit = revenue - cost
    cumulative_profit = profit.cumsum()
    equity = np.concatenate(([0], cumulative_profit))
    running_max = np.maximum.accumulate(equity)
    max_drawdown = float(np.max(running_max - equity))
    n_bets = int(bet.sum())
    n_hits = int((bet & hit).sum())
    total_cost = float(cost.sum())
    total_revenue = float(revenue.sum())
    total_profit = total_revenue - total_cost
    hit_rate = n_hits / n_bets if n_bets else float("nan")
    roi = total_profit / total_cost if total_cost else float("nan")
    break_even = m * COST_PER_NUMBER / PAYOUT_IF_HIT
    lower = wilson_lower_bound(n_hits, n_bets)

    metrics: dict[str, float | int] = {
        "m": m,
        "n_days": len(frame),
        "n_bets": n_bets,
        "n_hits": n_hits,
        "participation_rate": n_bets / len(frame),
        "hit_rate": hit_rate,
        "break_even_hit_rate": break_even,
        "wilson_lower": lower,
        "selection_score": lower - break_even if n_bets else float("-inf"),
        "total_cost": total_cost,
        "total_revenue": total_revenue,
        "total_profit": total_profit,
        "roi": roi,
        "max_drawdown": max_drawdown,
    }

    daily = pd.DataFrame(
        {
            "date": frame["date"].to_numpy(),
            "actual": frame["actual"].to_numpy(),
            "m": m,
            "confidence": confidence,
            "selected_numbers": [
                " ".join(f"{number:02d}" for number in row[:m]) for row in order
            ],
            "actual_rank": actual_rank,
            "bet": bet,
            "hit": hit,
            "cost": cost,
            "revenue": revenue,
            "profit": profit,
            "cumulative_profit": cumulative_profit,
        }
    )
    return metrics, daily


def frozen_strategy_from_row(row: pd.Series) -> FrozenStrategy:
    return FrozenStrategy(
        strategy=str(row["strategy"]),
        model=str(row["model"]),
        m=int(row["m"]),
        target_participation_rate=(
            None if pd.isna(row["target_participation_rate"])
            else float(row["target_participation_rate"])
        ),
        confidence_threshold=(
            None if pd.isna(row["confidence_threshold"])
            else float(row["confidence_threshold"])
        ),
        tie_acceptance_rate=(
            None if pd.isna(row["tie_acceptance_rate"])
            else float(row["tie_acceptance_rate"])
        ),
        validation_selection_score=float(row["selection_score"]),
        validation_n_bets=int(row["n_bets"]),
        validation_roi=float(row["roi"]),
    )


def frozen_strategy_to_dict(strategy: FrozenStrategy) -> dict[str, object]:
    return asdict(strategy)
