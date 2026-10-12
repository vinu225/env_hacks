"""
Weather Feature Engineering Module.

Computes meteorological derived indices:
- Dew Point Temperature (°C) using Magnus-Tetens approximation
- Heat Index (°C) using Rothfusz regression
- Wet-Bulb Temperature (°C) using Stull's empirical approximation
- Vapor Pressure Deficit (VPD) in kPa
"""

import math
from typing import Optional


def calculate_dew_point(
    temperature_c: Optional[float], relative_humidity_pct: Optional[float]
) -> Optional[float]:
    """
    Compute dew point temperature in Celsius using the Magnus-Tetens approximation.
    Valid for temperatures between 0°C and 60°C.
    Returns None if temperature or humidity is missing.
    """
    if temperature_c is None or relative_humidity_pct is None:
        return None

    # Constrain humidity to avoid log(0)
    rh = max(1.0, min(100.0, relative_humidity_pct))
    a = 17.27
    b = 237.7
    alpha = ((a * temperature_c) / (b + temperature_c)) + math.log(rh / 100.0)
    dew_point = (b * alpha) / (a - alpha)
    return round(dew_point, 2)


def calculate_heat_index(
    temperature_c: Optional[float], relative_humidity_pct: Optional[float]
) -> Optional[float]:
    """
    Compute Heat Index (apparent temperature) in Celsius.
    Uses NOAA / Rothfusz regression converted to Celsius.
    Returns None if temperature or humidity is missing.
    """
    if temperature_c is None or relative_humidity_pct is None:
        return None

    # For temperatures below 27°C (80°F), heat index is approximately equal to dry-bulb temperature
    if temperature_c < 27.0:
        return round(temperature_c, 2)

    # Convert Celsius to Fahrenheit for standard NOAA formula
    t_f = (temperature_c * 9.0 / 5.0) + 32.0
    rh = max(0.0, min(100.0, relative_humidity_pct))

    # Rothfusz regression
    hi_f = (
        -42.379
        + 2.04901523 * t_f
        + 10.14333127 * rh
        - 0.22475541 * t_f * rh
        - 0.00683783 * t_f * t_f
        - 0.05481717 * rh * rh
        + 0.00122874 * t_f * t_f * rh
        + 0.00085282 * t_f * rh * rh
        - 0.00000199 * t_f * t_f * rh * rh
    )

    # Convert back to Celsius
    hi_c = (hi_f - 32.0) * 5.0 / 9.0
    return round(hi_c, 2)


def calculate_wet_bulb_temperature(
    temperature_c: Optional[float], relative_humidity_pct: Optional[float]
) -> Optional[float]:
    """
    Compute wet-bulb temperature in Celsius using Stull's equation.
    T_w = T * atan(0.151977 * (RH% + 8.313659)^(1/2))
          + atan(T + RH%) - atan(RH% - 1.676331)
          + 0.00391838 * (RH%)^(3/2) * atan(0.023101 * RH%) - 4.686035
    Returns None if temperature or humidity is missing.
    """
    if temperature_c is None or relative_humidity_pct is None:
        return None

    t = temperature_c
    rh = max(1.0, min(100.0, relative_humidity_pct))

    tw = (
        t * math.atan(0.151977 * math.sqrt(rh + 8.313659))
        + math.atan(t + rh)
        - math.atan(rh - 1.676331)
        + 0.00391838 * (rh ** 1.5) * math.atan(0.023101 * rh)
        - 4.686035
    )
    return round(tw, 2)
