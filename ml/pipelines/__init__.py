"""
Offline ML Data Pipelines Package.
"""

from ml.pipelines.dataset_generator import MLDatasetGenerator
from ml.pipelines.provider import (
    HistoricalWeatherProvider,
    LocalFileHistoricalProvider,
    OpenMeteoArchiveHistoricalProvider,
    SyntheticHistoricalWeatherProvider,
)
from ml.pipelines.schema import (
    DatasetMetadata,
    DatasetQualityReport,
    HistoricalObservation,
)
from ml.pipelines.splitting import temporal_train_val_test_split
from ml.pipelines.target import (
    HeatRiskTargetGenerator,
    ThresholdBasedHeatTargetGenerator,
)
from ml.pipelines.validation import (
    detect_target_leakage,
    generate_dataset_quality_report,
    validate_chronological_integrity,
    validate_numeric_boundaries,
)

__all__ = [
    "HistoricalObservation",
    "DatasetMetadata",
    "DatasetQualityReport",
    "HistoricalWeatherProvider",
    "SyntheticHistoricalWeatherProvider",
    "LocalFileHistoricalProvider",
    "OpenMeteoArchiveHistoricalProvider",
    "HeatRiskTargetGenerator",
    "ThresholdBasedHeatTargetGenerator",
    "temporal_train_val_test_split",
    "validate_chronological_integrity",
    "detect_target_leakage",
    "validate_numeric_boundaries",
    "generate_dataset_quality_report",
    "MLDatasetGenerator",
]
