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
from .experiment import (
    ExperimentConfig,
    ExperimentResult,
    run_comparative_experiment,
)
from .comparison import (
    ComparisonReport,
    ImprovementResult,
    MetricSummary,
    RankedEntry,
    adjustment_effects,
    benchmark_improvements,
    build_comparison_report,
    rank_models,
    summarize_experiment,
)
from .export import experiment_to_dict, export_csv, export_json

__all__ = [
    'Fold',
    'walk_forward_folds',
    'BacktestResult',
    'NaiveLastValueForecaster',
    'NAIVE_MODEL_NAME',
    'run_backtest',
    'summarize_backtest',
    'ExperimentConfig',
    'ExperimentResult',
    'run_comparative_experiment',
    'ComparisonReport',
    'ImprovementResult',
    'MetricSummary',
    'RankedEntry',
    'adjustment_effects',
    'benchmark_improvements',
    'build_comparison_report',
    'rank_models',
    'summarize_experiment',
    'experiment_to_dict',
    'export_csv',
    'export_json',
]
