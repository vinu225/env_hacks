# Problem Formulation: Hyperlocal Heat Risk Prediction

**Target Subsystem**: Machine Learning Subsystem (`ml/`)  
**Domain**: Meteorological & Physiological Heat Stress  
**Target Variable**: Forward-Looking Maximum Heat Risk (`target_heat_risk_category` / `target_heat_risk_score`)  
**Forecast Horizon**: $H = 6$ Hours  
**Status**: Real Historical Dataset Generated & Audited (`READY_FOR_MODEL_TRAINING = TRUE`)

---

## 1. Canonical Forecasting Contract

The canonical ML problem is formally defined as:

> **Given multi-variable environmental information available at prediction time $t$, predict the maximum heat-risk severity occurring during the following 6 hours $(t, t+6\text{h}]$.**

```text
INPUT (Features X):
Multi-variable environmental information known strictly at or before prediction time t

TARGET (Label y):
Maximum heat-stress severity occurring strictly in the forward window (t, t + 6h]

FORECAST HORIZON:
6 hours (configurable, H)

TASK TYPE:
Ordinal multi-class classification

CLASSES:
LOW (0), MODERATE (1), HIGH (2), SEVERE (3), CRITICAL (4)

LEAKAGE CONTRACT:
The target never uses observations from t or earlier as target observations.
Future observations are allowed ONLY for constructing the training label.
Features at time t strictly ingest lagging observations <= t.
```

---

## 2. Target Formulations

### Formulation A: Ordinal Classification (`target_heat_risk_category`)
Predicting one of five scientifically grounded thermal hazard tiers:

* **`LOW` (0)**:
  * Forward maximum Heat Index $< 32.0^\circ\text{C}$ AND Forward maximum Wet-Bulb $< 25.0^\circ\text{C}$.
  * Represents baseline, safe environmental conditions for typical outdoor exertion.
* **`MODERATE` (1)**:
  * Forward maximum Heat Index in $[32.0^\circ\text{C}, 41.0^\circ\text{C})$ (NOAA Extreme Caution) OR Wet-Bulb $\ge 25.0^\circ\text{C}$.
  * Fatigue and heat cramps possible with prolonged exposure and physical activity.
* **`HIGH` (2)**:
  * Forward maximum Heat Index in $[41.0^\circ\text{C}, 46.0^\circ\text{C})$ (NOAA Danger zone) OR Wet-Bulb $\ge 27.5^\circ\text{C}$.
  * Significant reduction in evaporative cooling efficiency; vulnerable populations experience acute thermal distress.
* **`SEVERE` (3)**:
  * Forward maximum Heat Index in $[46.0^\circ\text{C}, 54.0^\circ\text{C})$ OR Wet-Bulb $\ge 29.0^\circ\text{C}$.
  * High risk of heat exhaustion and progression to exertional heat stroke across outdoor workers without frequent shade and hydration breaks.
* **`CRITICAL` (4)**:
  * Forward maximum Heat Index $\ge 54.0^\circ\text{C}$ (NOAA Extreme Danger) OR Wet-Bulb $\ge 31.0^\circ\text{C}$.
  * Extreme population-level heat stress hazard triggering municipal emergency protocols.

### Formulation B: Continuous Regression (`target_heat_risk_score`)
Continuous hazard severity score in $[0.0, 100.0]$:
$$\text{Score} = \text{clip}\left(\frac{HI_{\max, 6h} - 25.0}{55.0 - 25.0} \times 80.0 + \max(0, (Tw_{\max, 6h} - 27.0) \times 5.0), 0.0, 100.0\right)$$

---

## 3. Scientific Formulation & Physiological Calibration of Wet-Bulb ($T_w$)

### Anti-Pattern Avoided: Never Imitate Programmatic Rules
The ML target is **never** generated from the outputs of the heuristic `RuleBasedHeatModel`. That anti-pattern forces the model to memorize programmatic rules rather than learning real atmospheric dynamics. The target represents verifiable, future atmospheric and physiological heat stress.

### Rigorous Scientific Framing of Wet-Bulb Thresholds
The wet-bulb temperature ($T_w$) indicates the lowest temperature to which air can be cooled by the evaporation of water. In human biometeorology, sweat evaporation is the primary mechanism of thermoregulation in hot environments.

1. **NOAA National Weather Service (NWS) Heat Index**:
   * Evaluates apparent temperature based on Rothfusz multivariate regression from dry-bulb temperature and relative humidity.
2. **Wet-Bulb Escalation Indicators (Vecellio et al. 2022, Sherwood & Huber 2010)**:
   * **$T_w \ge 25.0^\circ\text{C}$**: Initial impairment of evaporative sweat efficiency.
   * **$T_w \ge 27.5^\circ\text{C}$**: High thermal strain during physical labor.
   * **$T_w \ge 29.0^\circ\text{C}$**: Severe heat stress; substantial risk of occupational heat illness.
   * **$T_w \ge 31.0^\circ\text{C}$**: Critical environmental heat-stress benchmark.
     * Empirical climatic chamber trials (*Vecellio et al., 2022, Penn State University*) demonstrated that under sustained light-to-moderate physical exertion (metabolic rate $\approx 150-200\text{ W/m}^2$), continuous heat exchange becomes uncompensable in unacclimatized subjects even in well-ventilated environments.
     * **Critical Boundary Delineation**: This threshold is an **environmental hazard benchmark** triggering municipal emergency action (curtailment of outdoor labor, opening of emergency cooling shelters, hospital triage preparation). It is **not** an absolute, universal personal physiological collapse boundary claiming instantaneous mortality for every individual regardless of acclimatization, hydration, and shade.
     * **Role in Target Formulation**: Wet-bulb acts as a non-compensable escalation mechanism. Even when dry-bulb temperature is moderate (e.g. 33°C), extreme humidity driving $T_w \ge 31^\circ\text{C}$ escalates the classification to `CRITICAL` because sweat cannot evaporate.

---

## 4. Operational Utility of 6-Hour Forward Horizon

1. **Intra-Day Heat Escalation Window**: Morning observations at 08:00–10:00 UTC must predict afternoon peak heating at 13:00–15:00 UTC. A 6-hour forward horizon directly spans the diurnal ramp-up.
2. **Pre-Emptive Municipal Action**: 6 hours provides actionable lead-time for urban local bodies to dispatch mobile water tankers, issue citizen advisory broadcasts, and shift outdoor labor shifts.
3. **Beating Persistence**: The diurnal temperature swing makes static persistence systematically underestimate afternoon peaks. The forward 6-hour target tests whether models actually learn thermal inertia and synoptic heating.

---

## 5. Input Data Contract & Inference Availability

| Input Feature | Source | Availability at $t$ | Type | Leakage Prevention |
| :--- | :--- | :--- | :--- | :--- |
| `temperature_c` | ERA5 / Station gauge | Yes (at $t$) | Observed | Current observation only |
| `apparent_temperature_c` | ERA5 / Station | Yes (at $t$) | Observed | Current observation only |
| `relative_humidity_pct` | ERA5 / Station | Yes (at $t$) | Observed | Current observation only |
| `wind_speed_kmh` | ERA5 / Station | Yes (at $t$) | Observed | Current observation only |
| `dew_point_c` | Magnus-Tetens formula | Yes (at $t$) | Derived | Computed strictly from $T(t)$ and $RH(t)$ |
| `heat_index_c` | NOAA Rothfusz regression | Yes (at $t$) | Derived | Computed strictly from $T(t)$ and $RH(t)$ |
| `wet_bulb_temperature_c`| Stull equation | Yes (at $t$) | Derived | Computed strictly from $T(t)$ and $RH(t)$ |
| `consecutive_hot_days` | Daily max temperature | Yes (at $t$) | Derived | Uses strictly lagging daily max $\le t$ |
| `rainfall_accumulation_24h_mm` | Rolling sum | Yes (at $t$) | Derived | Backward rolling sum in $[t - 24\text{h}, t]$ |
| `rainfall_accumulation_7d_mm` | Rolling sum | Yes (at $t$) | Derived | Backward rolling sum in $[t - 168\text{h}, t]$ |
| `forecast_precipitation_6h_mm` | NWP Numerical Model | Yes (at $t$) | Forecast | Weather forecast produced at $\le t$ |

---

## 6. Assumptions and Limitations

1. **Standard Metabolic Rate**: Thresholds assume light-to-moderate outdoor physical exertion ($\approx 150-200\text{ W/m}^2$) under standard lightweight clothing.
2. **Reanalysis vs Direct Station Data**: ERA5 reanalysis represents $\approx 9-31\text{ km}$ spatial grids blending physical numerical models with satellite and station observations. It does not replace building-level micro-sensor telemetry in dense informal urban slums.
3. **Exclusion of Surface Pressure**: Surface barometric pressure is excluded from the V1 heat feature set (Option A). Missing barometric measurements are never imputed with 1013.25 hPa.
