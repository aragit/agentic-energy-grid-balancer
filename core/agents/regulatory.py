import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class RegulatoryAgent:
    def __init__(self, carbon_cap_kg: float = 10000.0):
        self.carbon_cap = carbon_cap_kg
        self.violations: List[Dict] = []

    def check_frequency(self, frequency: float, step: int) -> bool:
        if frequency < 49.0 or frequency > 51.0:
            violation = {
                "step": step,
                "type": "frequency",
                "value": frequency,
                "limit": "49.0-51.0 Hz",
                "severity": "CRITICAL",
            }
            self.violations.append(violation)
            logger.error(
                f"[REGULATOR] CRITICAL: Frequency {frequency:.3f} Hz out of bounds!"
            )
            return False
        return True

    def check_carbon(self, agent, step: int) -> bool:
        if agent.state.total_carbon_emitted > self.carbon_cap:
            violation = {
                "step": step,
                "type": "carbon",
                "agent": agent.name,
                "emitted": agent.state.total_carbon_emitted,
                "cap": self.carbon_cap,
                "severity": "VIOLATION",
            }
            self.violations.append(violation)
            logger.warning(f"[REGULATOR] {agent.name} exceeded carbon cap!")
            return False
        return True

    def check_market_fairness(self, bids: List[Dict], step: int) -> bool:
        if len(bids) < 2:
            return True
        prices = [b["bid_price"] for b in bids]
        if max(prices) - min(prices) < 0.01:
            violation = {
                "step": step,
                "type": "market_manipulation",
                "detail": "All bids identical",
                "severity": "SUSPICIOUS",
            }
            self.violations.append(violation)
            logger.warning("[REGULATOR] Suspicious: all agents bid identical prices")
            return False
        return True

    def get_summary(self) -> Dict:
        critical = [v for v in self.violations if v["severity"] == "CRITICAL"]
        violations = [v for v in self.violations if v["severity"] == "VIOLATION"]
        return {
            "total_violations": len(self.violations),
            "critical": len(critical),
            "violations": len(violations),
            "details": self.violations,
        }
