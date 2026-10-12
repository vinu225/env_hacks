"""
Temporal Chronological Dataset Splitting for Environmental Time Series.

CRITICAL ARCHITECTURAL RULE:
Random splitting (e.g. sklearn.model_selection.train_test_split with shuffle=True)
is STRICTLY PROHIBITED for environmental time series.

Why Random Splitting Causes Catastrophic Leakage:
------------------------------------------------
1. Temporal Autocorrelation: Meteorological variables (temperature, relative humidity, pressure)
   exhibit strong temporal persistence across hours and days. A test observation from hour t+1
   is strongly correlated with hour t. If hour t is randomly placed in training and hour t+1 in test,
   the model memorizes the localized weather state rather than generalizing predictive patterns,
   producing artificially inflated benchmark metrics that collapse during live deployment.

2. Lookahead Bias: Training on future data to predict past events reverses physical causality.

3. Seasonal and Synoptic Contamination: Heatwaves and cyclones operate over multi-day synoptic regimes.
   A single heatwave episode must belong entirely to train, validation, or test, rather than having its
   hours scattered randomly across all partitions.

Standard Temporal Strategy:
---------------------------
Oldest data (e.g. 70%)  -> Training Set
Subsequent data (e.g. 15%) -> Validation Set (hyperparameter tuning & early stopping)
Most recent data (e.g. 15%) -> Test Set (uncontaminated out-of-time evaluation)
"""

from typing import Tuple
import pandas as pd


def temporal_train_val_test_split(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into chronologically ordered train, validation, and test subsets.

    Guarantees:
    max(train[time_col]) < min(val[time_col]) <= max(val[time_col]) < min(test[time_col])
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-5:
        raise ValueError(f"Split ratios must sum to 1.0 (got {train_ratio} + {val_ratio} + {test_ratio})")

    if time_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{time_col}' column for temporal ordering.")

    # Sort strictly by timestamp
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    unique_times = df_sorted[time_col].drop_duplicates().tolist()
    n_unique = len(unique_times)

    if n_unique < 3:
        raise ValueError(f"Cannot split small dataset with only {n_unique} unique timestamps into train/val/test partitions.")

    train_end_idx = int(n_unique * train_ratio)
    val_end_idx = int(n_unique * (train_ratio + val_ratio))

    # Ensure each split has at least 1 timestamp partition if n_unique >= 3
    train_end_idx = max(1, min(train_end_idx, n_unique - 2))
    val_end_idx = max(train_end_idx + 1, min(val_end_idx, n_unique - 1))

    train_cutoff = unique_times[train_end_idx - 1]
    val_cutoff = unique_times[val_end_idx - 1]

    train_df = df_sorted[df_sorted[time_col] <= train_cutoff].copy().reset_index(drop=True)
    val_df = df_sorted[(df_sorted[time_col] > train_cutoff) & (df_sorted[time_col] <= val_cutoff)].copy().reset_index(drop=True)
    test_df = df_sorted[df_sorted[time_col] > val_cutoff].copy().reset_index(drop=True)

    return train_df, val_df, test_df
