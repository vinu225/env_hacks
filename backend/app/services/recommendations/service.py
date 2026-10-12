"""
Deterministic Recommendations Service.

Provides proven, deterministic public health and municipal guidance based on
risk categories and identified drivers.
"""

from typing import List
from app.schemas.risk import RiskCategory, RiskLevel, RiskPrediction


class RecommendationService:
    """Produces actionable, deterministic recommendations without relying on hallucinations."""

    def generate_recommendations(self, prediction: RiskPrediction) -> List[str]:
        recommendations: List[str] = []

        if prediction.category == RiskCategory.HEAT:
            if prediction.risk_level in [RiskLevel.CRITICAL, RiskLevel.SEVERE]:
                recommendations.extend([
                    "Suspend outdoor manual labor between 12:00 PM and 4:00 PM.",
                    "Open air-conditioned municipal cooling centers and shaded public hydration hubs.",
                    "Issue red alerts to emergency clinics for heat exhaustion and dehydration triage.",
                    "Direct vulnerable demographics (elderly, infants, outdoor workers) to remain indoors.",
                ])
            elif prediction.risk_level == RiskLevel.HIGH:
                recommendations.extend([
                    "Increase oral hydration intake; avoid direct sun exposure during peak afternoon hours.",
                    "Ensure municipal water tankers replenish public drinking fountains.",
                    "Deploy cool roofs or temporary shade canopies in open vegetable and street markets.",
                ])
            else:
                recommendations.append("Normal heat precautions: stay adequately hydrated.")

        elif prediction.category == RiskCategory.WATER_STRESS:
            if prediction.risk_level in [RiskLevel.CRITICAL, RiskLevel.SEVERE]:
                recommendations.extend([
                    "Enact emergency water rationing and prioritize essential hospitals and residential drinking lines.",
                    "Deploy GPS-tracked municipal water tankers to deficient wards.",
                    "Prohibit non-essential commercial water usage (car washing, decorative fountains, turf watering).",
                ])
            elif prediction.risk_level == RiskLevel.HIGH:
                recommendations.extend([
                    "Request voluntary 20% household conservation.",
                    "Activate groundwater recharge monitoring and prevent illegal borewell extractions.",
                ])
            else:
                recommendations.append("Maintain baseline water conservation practices.")

        elif prediction.category == RiskCategory.RAINFALL:
            if prediction.risk_level in [RiskLevel.CRITICAL, RiskLevel.SEVERE]:
                recommendations.extend([
                    "Deploy high-capacity dewatering suction pumps to known low-lying waterlogging underpasses.",
                    "Issue flash flood transit warnings and barricade submerged roads.",
                    "Clear silt grates and stormwater outfalls immediately.",
                ])
            elif prediction.risk_level == RiskLevel.HIGH:
                recommendations.extend([
                    "Alert emergency storm drain rapid response teams.",
                    "Advise motorists to avoid subway underpasses and low-lying arterial corridors.",
                ])
            else:
                recommendations.append("Monitor localized drainage updates during sporadic showers.")

        return recommendations
