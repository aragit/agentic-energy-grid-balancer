"""Agent memory system for episodic learning and pattern recognition."""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    step: int
    market_price: float
    bid_price: float
    output_mw: float
    revenue: float
    carbon_cost: float
    net_profit: float
    frequency: float
    weather: Dict[str, Any]
    decision: Dict[str, Any]
    outcome: str


class AgentMemory:
    def __init__(self, max_history: int = 100):
        self.experiences: deque = deque(maxlen=max_history)
        self.pattern_cache: Dict[str, Any] = {}

    def record(self, experience: Experience):
        self.experiences.append(experience)
        self._update_patterns()

    def _update_patterns(self):
        if len(self.experiences) < 5:
            return

        prices = [e.market_price for e in self.experiences]
        profits = [e.net_profit for e in self.experiences]

        self.pattern_cache = {
            "avg_market_price": round(np.mean(prices), 2),
            "price_volatility": round(np.std(prices), 2),
            "avg_profit": round(np.mean(profits), 2),
            "profit_trend": "improving" if profits[-1] > profits[0] else "declining",
            "best_price_range": self._find_best_price_range(),
            "storm_frequency": self._storm_frequency(),
            "peak_demand_hours": self._peak_demand_hours(),
        }

    def _find_best_price_range(self) -> str:
        profitable = [
            (e.market_price, e.net_profit) for e in self.experiences if e.net_profit > 0
        ]
        if not profitable:
            return "unknown"
        prices = [p for p, _ in profitable]
        return f"${min(prices):.0f}-${max(prices):.0f}"

    def _storm_frequency(self) -> str:
        recent = list(self.experiences)[-20:]
        storms = sum(1 for e in recent if e.weather.get("is_storm", False))
        if storms > 5:
            return "high"
        elif storms > 2:
            return "moderate"
        return "low"

    def _peak_demand_hours(self) -> List[int]:
        demand_by_hour: Dict[int, List[float]] = {}
        for e in self.experiences:
            hour = e.step % 24
            demand_by_hour.setdefault(hour, []).append(e.output_mw)
        avg_demand = {h: np.mean(v) for h, v in demand_by_hour.items()}
        sorted_hours = sorted(avg_demand.items(), key=lambda x: x[1], reverse=True)
        return [h for h, _ in sorted_hours[:3]]

    def get_context(self, current_step: int) -> str:
        if len(self.experiences) < 3:
            return "No prior experience. This is a fresh start."

        recent = list(self.experiences)[-5:]
        patterns = self.pattern_cache

        context = f"""
Memory Summary (last {len(self.experiences)} steps):
- Average market price: ${patterns.get('avg_market_price', 'N/A')}/MWh
- Price volatility: ${patterns.get('price_volatility', 'N/A')}
- Average net profit: ${patterns.get('avg_profit', 'N/A')}
- Profit trend: {patterns.get('profit_trend', 'unknown')}
- Most profitable price range: {patterns.get('best_price_range', 'unknown')}
- Storm frequency (recent): {patterns.get('storm_frequency', 'unknown')}
- Peak demand hours: {patterns.get('peak_demand_hours', [])}

Recent experiences (last 5):
"""
        for exp in recent:
            context += (
                f"  Step {exp.step}: Market=${exp.market_price}, "
                f"Bid=${exp.bid_price}, Output={exp.output_mw}MW, "
                f"Profit=${exp.net_profit}, Outcome={exp.outcome}\n"
            )
        return context

    def get_strategy_advice(self) -> str:
        if len(self.experiences) < 10:
            return "Insufficient experience for strategic advice."

        recent_losses = [e for e in list(self.experiences)[-10:] if e.net_profit < 0]
        if len(recent_losses) > 5:
            return "WARNING: Recent losses are high. Consider conservative bidding."

        profitable_bids = [e.bid_price for e in self.experiences if e.net_profit > 0]
        if profitable_bids:
            avg_profitable = np.mean(profitable_bids)
            return f"Historical insight: Bids around ${avg_profitable:.2f} have been most profitable."

        return "No clear pattern detected. Explore different strategies."


class SimulationMemory:
    def __init__(self, db_session=None):
        self.db = db_session
        self.agent_profiles: Dict[str, Dict] = {}
        self.market_patterns: List[Dict] = []
        self.system_learnings: List[str] = []

    def record_simulation(self, simulation_id: int, agent_records: List[Dict]):
        for record in agent_records:
            name = record["agent_name"]
            if name not in self.agent_profiles:
                self.agent_profiles[name] = {
                    "runs": 0,
                    "total_profit": 0.0,
                    "avg_profit": 0.0,
                }
            profile = self.agent_profiles[name]
            profile["runs"] += 1
            profile["total_profit"] += record.get("final_balance", 0)
            profile["avg_profit"] = profile["total_profit"] / profile["runs"]

    def get_agent_profile(self, agent_name: str) -> str:
        profile = self.agent_profiles.get(agent_name)
        if not profile:
            return f"No historical data for {agent_name}."
        return (
            f"Long-term Profile for {agent_name}:\n- Runs: {profile['runs']}\n"
            f"- Avg profit: ${profile['avg_profit']:.2f}\n"
            f"- Total: ${profile['total_profit']:.2f}"
        )

    def add_system_learning(self, insight: str):
        self.system_learnings.append(insight)
        logger.info(f"[SYSTEM MEMORY] {insight}")

    def get_system_context(self) -> str:
        if not self.system_learnings:
            return "No system-level learnings yet."
        return "System Learnings:\n" + "\n".join(
            f"- {item}" for item in self.system_learnings[-5:]
        )
