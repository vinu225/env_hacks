"""
Pydantic Schemas Package.
"""

from app.schemas.alert import AlertRead
from app.schemas.environmental import (
    DailyForecastObservation,
    EnvironmentalObservation,
    HourlyForecastObservation,
    NormalizedWeatherData,
    NormalizedWeatherObservation,
    WeatherSnapshot,
)
from app.schemas.features import (
    DATA_VERSION,
    FEATURE_VERSION,
    EnvironmentalFeatureVector,
    FeatureValue,
)
from app.schemas.heat import HeatFeatureInput, HeatRiskAssessment
from app.schemas.location import Coordinates, LocationBase, LocationCreate, LocationRead
from app.schemas.rainfall import RainfallFeatureInput, RainfallRiskAssessment
from app.schemas.report import (
    IncidentReportBase,
    IncidentReportCreate,
    IncidentReportRead,
    IncidentType,
    ReportSeverity,
    VerificationStatus,
)
from app.schemas.risk import (
    CompositeRiskSummary,
    RiskCategory,
    RiskDriver,
    RiskLevel,
    RiskPrediction,
)
from app.schemas.water import WaterFeatureInput, WaterRiskAssessment

__all__ = [
    "AlertRead",
    "EnvironmentalObservation",
    "WeatherSnapshot",
    "NormalizedWeatherObservation",
    "HourlyForecastObservation",
    "DailyForecastObservation",
    "NormalizedWeatherData",
    "FeatureValue",
    "EnvironmentalFeatureVector",
    "DATA_VERSION",
    "FEATURE_VERSION",
    "HeatFeatureInput",
    "HeatRiskAssessment",
    "Coordinates",
    "LocationBase",
    "LocationCreate",
    "LocationRead",
    "RainfallFeatureInput",
    "RainfallRiskAssessment",
    "IncidentReportBase",
    "IncidentReportCreate",
    "IncidentReportRead",
    "IncidentType",
    "ReportSeverity",
    "VerificationStatus",
    "CompositeRiskSummary",
    "RiskCategory",
    "RiskDriver",
    "RiskLevel",
    "RiskPrediction",
    "WaterFeatureInput",
    "WaterRiskAssessment",
]
