"""
Environmental Observation Validation Module.

Enforces physical plausibility bounds, required fields, and temporal consistency
prior to feature engineering and risk scoring.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class ValidationResult:
    """Result of validating an environmental observation."""
    is_valid: bool
    is_usable: bool
    issues: List[str] = field(default_factory=list)
    quality_score: float = 1.0


class EnvironmentalValidator:
    """
    Validates meteorological observations against physical reality and schema boundaries.
    Never silently coerces corrupt or impossible data into valid-looking numbers.
    """

    # Physical plausibility limits
    MIN_LATITUDE = -90.0
    MAX_LATITUDE = 90.0
    MIN_LONGITUDE = -180.0
    MAX_LONGITUDE = 180.0

    MIN_TEMPERATURE_C = -60.0  # Lowest recorded Indian/global inhabited ground temp
    MAX_TEMPERATURE_C = 65.0   # Highest theoretical surface ambient threshold
    MIN_HUMIDITY_PCT = 0.0
    MAX_HUMIDITY_PCT = 100.0
    MIN_PRECIPITATION_MM = 0.0
    MAX_HOURLY_PRECIPITATION_MM = 350.0  # Max recorded cloudburst hourly rate
    MIN_WIND_SPEED_KMH = 0.0
    MAX_WIND_SPEED_KMH = 350.0  # Super-cyclonic boundary

    @classmethod
    def validate_coordinates(cls, latitude: Optional[float], longitude: Optional[float]) -> List[str]:
        issues = []
        if latitude is None:
            issues.append("Latitude is required.")
        elif not (cls.MIN_LATITUDE <= latitude <= cls.MAX_LATITUDE):
            issues.append(f"Latitude {latitude} is outside valid range [{cls.MIN_LATITUDE}, {cls.MAX_LATITUDE}].")

        if longitude is None:
            issues.append("Longitude is required.")
        elif not (cls.MIN_LONGITUDE <= longitude <= cls.MAX_LONGITUDE):
            issues.append(f"Longitude {longitude} is outside valid range [{cls.MIN_LONGITUDE}, {cls.MAX_LONGITUDE}].")

        return issues

    @classmethod
    def validate_timestamp(cls, dt: Optional[datetime], field_name: str = "observed_at") -> List[str]:
        issues = []
        if dt is None:
            issues.append(f"Timestamp '{field_name}' is required.")
        else:
            if dt.tzinfo is None:
                issues.append(f"Timestamp '{field_name}' must be timezone-aware (UTC).")
        return issues

    @classmethod
    def validate_raw_weather_payload(cls, data: Dict[str, Any]) -> ValidationResult:
        """
        Validates raw payload fields before domain conversion.
        Does not discard records outright; computes usability and quality score.
        """
        issues: List[str] = []
        fatal_issues: List[str] = []

        # 1. Coordinates
        coord_issues = cls.validate_coordinates(data.get("latitude"), data.get("longitude"))
        fatal_issues.extend(coord_issues)

        # 2. Timestamp
        obs_at = data.get("observed_at")
        ts_issues = cls.validate_timestamp(obs_at, "observed_at")
        if ts_issues:
            issues.extend(ts_issues)

        # 3. Temperature plausibility
        temp = data.get("temperature_c")
        if temp is None:
            issues.append("Temperature is missing.")
            fatal_issues.append("Missing essential parameter: temperature_c")
        elif not (cls.MIN_TEMPERATURE_C <= temp <= cls.MAX_TEMPERATURE_C):
            issues.append(f"Temperature {temp}°C is physically implausible [{cls.MIN_TEMPERATURE_C}, {cls.MAX_TEMPERATURE_C}].")
            fatal_issues.append(f"Implausible temperature: {temp}")

        # 4. Humidity
        rh = data.get("relative_humidity_pct")
        if rh is not None and not (cls.MIN_HUMIDITY_PCT <= rh <= cls.MAX_HUMIDITY_PCT):
            issues.append(f"Relative humidity {rh}% outside valid range [0, 100].")

        # 5. Precipitation
        precip = data.get("precipitation_mm")
        if precip is not None:
            if precip < cls.MIN_PRECIPITATION_MM:
                issues.append(f"Precipitation cannot be negative (got {precip} mm).")
            elif precip > cls.MAX_HOURLY_PRECIPITATION_MM:
                issues.append(f"Precipitation {precip} mm exceeds maximum cloudburst plausibility.")

        # 6. Precipitation Probability
        precip_prob = data.get("precipitation_probability_pct")
        if precip_prob is not None and not (0.0 <= precip_prob <= 100.0):
            issues.append(f"Precipitation probability {precip_prob}% outside [0, 100].")

        # 7. Wind Speed
        wind = data.get("wind_speed_kmh")
        if wind is not None:
            if wind < cls.MIN_WIND_SPEED_KMH:
                issues.append(f"Wind speed cannot be negative (got {wind} km/h).")
            elif wind > cls.MAX_WIND_SPEED_KMH:
                issues.append(f"Wind speed {wind} km/h exceeds plausible limit.")

        # Calculate quality penalty based on issues
        quality = 1.0 - (0.2 * len(issues))
        quality = max(0.0, min(1.0, quality))

        is_valid = len(issues) == 0
        is_usable = len(fatal_issues) == 0

        return ValidationResult(
            is_valid=is_valid,
            is_usable=is_usable,
            issues=issues,
            quality_score=quality if is_usable else 0.0,
        )
