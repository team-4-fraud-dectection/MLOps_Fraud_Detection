import numpy as np
import logging

logger = logging.getLogger(__name__)


class RiskScoringEngine:

    def __init__(self):
        """
        Risk scoring based on ML fraud probability
        """
        self.low_threshold = 30
        self.medium_threshold = 60
        self.high_threshold = 85

    def probability_to_score(self, fraud_probability: float) -> float:
        """
        Convert ML probability to risk score (0-100)
        """

        score = fraud_probability * 100

        score = np.clip(score, 0, 100)

        return round(float(score), 2)

    def risk_level(self, score: float) -> str:

        if score <= self.low_threshold:
            return "LOW"

        elif score <= self.medium_threshold:
            return "MEDIUM"

        elif score <= self.high_threshold:
            return "HIGH"

        else:
            return "VERY_HIGH"

    def recommended_action(self, score: float) -> dict:

        if score <= self.low_threshold:

            return {
                "verification": "None",
                "action": "Approve transaction"
            }

        elif score <= self.medium_threshold:

            return {
                "verification": "OTP",
                "action": "Send OTP verification"
            }

        elif score <= self.high_threshold:

            return {
                "verification": "OTP + Biometric",
                "action": "Require OTP + FaceID/Fingerprint"
            }

        else:

            return {
                "verification": "Blocked",
                "action": "Block transaction and trigger fraud alert"
            }

    def generate(self, fraud_probability: float) -> dict:

        score = self.probability_to_score(fraud_probability)

        level = self.risk_level(score)

        action_info = self.recommended_action(score)

        result = {

            "fraud_probability": round(float(fraud_probability), 4),

            "risk_score": score,

            "risk_level": level,

            "verification_required": action_info["verification"],

            "recommended_action": action_info["action"]
        }

        logger.info(
            f"Fraud Prob={fraud_probability:.4f} | Score={score} | Level={level}"
        )

        return result
