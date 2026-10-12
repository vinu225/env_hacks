"""
Baseline Models for Evaluation and Benchmark Comparison.

Provides non-learning, rule-free, and naive benchmark heuristics (e.g. Persistence)
to scientifically establish whether future trained ML models add predictive value over simple assumptions.
"""

from ml.baselines.persistence import PersistenceHeatBaseline

__all__ = ["PersistenceHeatBaseline"]
