"""
Out-of-sample evaluation of forecasting models.
"""

from .splits import Fold, walk_forward_folds
from .backtest import (
    BacktestResult,
    NaiveLastValueForecaster,
    NAIVE_MODEL_NAME,
    run_backtest,
)
from .report import summarize_backtest

__all__ = [
    'Fold',
    'walk_forward_folds',
    'BacktestResult',
    'NaiveLastValueForecaster',
    'NAIVE_MODEL_NAME',
    'run_backtest',
    'summarize_backtest',
]
