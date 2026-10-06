"""
Walk-forward fold generation for forecast backtesting.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Tuple

SUPPORTED_WINDOWS = ("expanding",)


@dataclass(frozen=True)
class Fold:
    """A single walk-forward evaluation fold.

    Attributes:
        origin: Last year available for training (the forecast origin).
        train_years: Years used to fit the model, in chronological order.
        test_years: Held-out years to forecast, in chronological order.
    """

    origin: int
    train_years: Tuple[int, ...]
    test_years: Tuple[int, ...]

    @property
    def horizon(self) -> int:
        return len(self.test_years)


def _require_positive_int(name: str, value) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")


def _normalize_years(years: Iterable) -> List[int]:
    normalized = []
    for year in years:
        if isinstance(year, bool):
            raise TypeError("years must be integers, got bool")
        try:
            as_int = int(year)
        except (TypeError, ValueError):
            raise TypeError(f"years must be integers, got {year!r}") from None
        if as_int != year:
            raise TypeError(f"years must be whole numbers, got {year!r}")
        normalized.append(as_int)
    duplicates = sorted(year for year, count in Counter(normalized).items() if count > 1)
    if duplicates:
        raise ValueError(f"years contains duplicates: {duplicates}")
    return sorted(normalized)


def walk_forward_folds(
    years: Iterable[int],
    min_train_years: int,
    horizon: int,
    step: int = 1,
    window: str = "expanding",
) -> List[Fold]:
    """Generate deterministic walk-forward folds over annual observations.

    Each fold trains on every year up to and including its origin and tests on
    the next ``horizon`` years. Only folds with a complete test horizon are
    produced, so every fold contributes the same horizons to the evaluation.

    Args:
        years: Available observation years, in any order. Years must be unique
            and consecutive because the baseline models forecast positionally.
        min_train_years: Number of years in the first training window.
        horizon: Number of years forecast per fold.
        step: Number of years the origin advances between folds. Folds start
            at the earliest valid origin, so with ``step > 1`` the final
            year(s) of the series may not appear in any test window.
        window: Training window type. Only ``"expanding"`` is supported.

    Returns:
        Folds ordered by origin.

    Raises:
        TypeError: If a parameter or year has the wrong type.
        ValueError: If the configuration is invalid, the years contain
            duplicates or gaps, or no folds can be produced.
    """
    _require_positive_int("min_train_years", min_train_years)
    _require_positive_int("horizon", horizon)
    _require_positive_int("step", step)
    if window not in SUPPORTED_WINDOWS:
        raise ValueError(
            f"Unsupported window {window!r}. Supported: {list(SUPPORTED_WINDOWS)}"
        )

    sorted_years = _normalize_years(years)
    if not sorted_years:
        raise ValueError("years is empty")

    gaps = [
        (prev, curr)
        for prev, curr in zip(sorted_years, sorted_years[1:])
        if curr - prev != 1
    ]
    if gaps:
        raise ValueError(f"years must be consecutive; gaps found between {gaps}")

    required = min_train_years + horizon
    if len(sorted_years) < required:
        raise ValueError(
            f"Need at least {required} years (min_train_years={min_train_years} + "
            f"horizon={horizon}), got {len(sorted_years)}"
        )

    folds = []
    last_origin_index = len(sorted_years) - horizon - 1
    for origin_index in range(min_train_years - 1, last_origin_index + 1, step):
        folds.append(
            Fold(
                origin=sorted_years[origin_index],
                train_years=tuple(sorted_years[: origin_index + 1]),
                test_years=tuple(sorted_years[origin_index + 1: origin_index + 1 + horizon]),
            )
        )
    return folds
