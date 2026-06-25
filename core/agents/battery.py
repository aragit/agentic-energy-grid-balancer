"""Battery storage agent with arbitrage strategy."""

import json
import logging
from typing import Dict, Any
from core.agents.base import BaseAgent
from core.memory import Experience
from core.schemas import BidStrategy, ValidatedBid

logger = logging.getLogger(__name__)


class BatteryAgent(BaseAgent):
    """Battery energy storage system with charge/discharge arbitrage."""

    def __init__(self, name: str, capacity_mwh: float, max_charge_mw: float, llm):
        super().__init__(
            name, "battery", max_charge_mw, carbon_intensity_g_kwh=0.0, llm=llm
        )
        self.capacity_mwh = capacity_mwh
        self.charge_level = 0.5 * capacity_mwh  # Start at 50%
        self.max_charge_mw = max_charge_mw
        self.round_trip_efficiency = 0.9

    def compute_output(self, weather, market_price: float) -> float:
        """Battery output depends on market strategy, not weather."""
        self.record_output(0.0)
        return 0.0

    def get_available_discharge(self) -> float:
        """Max MWh we can discharge this hour."""
        return min(self.max_charge_mw, self.charge_level)

    def get_available_charge(self) -> float:
        """Max MWh we can charge this hour."""
        return min(self.max_charge_mw, self.capacity_mwh - self.charge_level)

    def apply_charge(self, mwh: float):
        """Positive = charge in, negative = discharge out."""
        if mwh > 0:
            self.charge_level += mwh * self.round_trip_efficiency
        else:
            self.charge_level += mwh  # mwh is negative
        self.charge_level = max(0, min(self.capacity_mwh, self.charge_level))
        self.record_output(-mwh)  # Positive output = discharging

    def update_after_trade(
        self,
        energy_mwh: float,
        price_per_mwh: float,
        carbon_cost: float = 0.0,
        step: int = 0,
        weather: Dict = None,
    ):
        """Override to track battery-specific profit and record memory."""
        revenue = energy_mwh * price_per_mwh
        net_profit = revenue - carbon_cost

        if energy_mwh > 0:
            # Discharging: we sold energy
            self.state.balance += net_profit
            self.state.total_revenue += revenue
        else:
            # Charging: we bought energy (cost)
            self.state.balance -= abs(revenue)
            self.state.total_cost += abs(revenue)

        self.state.total_carbon_emitted += (
            energy_mwh * self.state.carbon_intensity_g_kwh / 1000
        )

        # RECORD MEMORY: essential for learning
        experience = Experience(
            step=step,
            market_price=price_per_mwh,
            bid_price=(
                self.state.strategy_history[-1]["bid_price"]
                if self.state.strategy_history
                else price_per_mwh
            ),
            output_mw=abs(energy_mwh),
            revenue=revenue if energy_mwh > 0 else -abs(revenue),
            carbon_cost=carbon_cost,
            net_profit=net_profit if energy_mwh > 0 else -abs(revenue),
            frequency=50.0,
            weather=weather or {},
            decision=(
                self.state.strategy_history[-1] if self.state.strategy_history else {}
            ),
            outcome=(
                "profitable"
                if (energy_mwh > 0 and net_profit > 0)
                or (energy_mwh < 0 and abs(revenue) < 500)
                else "loss"
            ),
        )
        self.memory.record(experience)

    def _build_prompt(
        self,
        market_price: float,
        demand: float,
        frequency: float,
        carbon_price: float,
        memory_context: str,
        strategy_advice: str,
        weather: Dict[str, Any] = None,
    ) -> str:
        weather_str = json.dumps(weather) if weather else "N/A"

        # FIX: deque does not support slicing — convert to list first
        experiences_list = list(self.memory.experiences)
        recent_prices = [e.market_price for e in experiences_list[-10:]]
        price_trend = "stable"
        if len(recent_prices) >= 2:
            if recent_prices[-1] > recent_prices[-2] * 1.05:
                price_trend = "rising"
            elif recent_prices[-1] < recent_prices[-2] * 0.95:
                price_trend = "falling"

        # FIX: No $ symbols before numbers — ReasoningEngine parser fails on $
        return f"""You are an autonomous battery arbitrage agent with memory and learning.

Agent: {self.name}
Type: {self.agent_type}
Capacity: {self.state.capacity_mw} MW / {self.capacity_mwh} MWh
Current Charge Level: {self.charge_level:.1f} MWh ({100*self.charge_level/self.capacity_mwh:.0f}%)
Max Charge Rate: {self.max_charge_mw} MW
Round-Trip Efficiency: {self.round_trip_efficiency*100:.0f}%

Current Market Context:
- Market Price: {market_price:.2f} USD per MWh
- Price Trend: {price_trend}
- Total Demand: {demand:.2f} MWh
- Grid Frequency: {frequency:.3f} Hz
- Carbon Price: {carbon_price:.2f} USD per ton
- Weather: {weather_str}

{memory_context}

Strategic Advice:
{strategy_advice}

ARBITRAGE STRATEGY:
- Charge (buy) when prices are LOW and you have capacity
- Discharge (sell) when prices are HIGH and you have charge
- HOLD when prices are uncertain or you are near empty/full
- Consider price trends: if prices are rising, hold; if falling, sell now

Respond with VALID JSON matching this exact schema:
{{
    "bid_price": float,        // USD per MWh, must be between 1.0 and 200.0
    "output_adjustment": string,  // One of: "charge", "discharge", "hold", "sell", "buy", "maintain", "ramp_up", "ramp_down", "reduce_demand"
    "carbon_trade": float,     // kg CO2, optional (default 0)
    "reasoning": string,       // Explain your arbitrage decision
    "confidence": float        // 0.0 to 1.0
}}
"""

    def get_validated_bid(
        self,
        market_price: float,
        demand: float,
        frequency: float,
        carbon_price: float,
        weather: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Get LLM strategy and apply physical guardrails (clamp, don't replace).

        Type 2 (Symbolic[Neuro]) principle: The neural subroutine proposes;
        the symbolic layer validates and only overrides when physically
        impossible or safety-critical.
        """
        # Step 1: Invoke neural subroutine (LLM) — returns already Pydantic-validated dict
        raw_decision = self.decide_bid(
            market_price=market_price,
            demand=demand,
            frequency=frequency,
            carbon_price=carbon_price,
            weather=weather,
        )

        # Step 2: Extract LLM's proposed values (already validated by BidSchema)
        llm_price = raw_decision["bid_price"]
        llm_action_raw = raw_decision["output_adjustment"]
        confidence = raw_decision.get("confidence", 0.5)
        reasoning = raw_decision.get("reasoning", "")
        validation_errors = raw_decision.get("validation_errors", [])

        # Normalize action vocabulary to battery-specific canonical forms
        if llm_action_raw in ("sell", "discharge"):
            llm_action = "discharge"
        elif llm_action_raw in ("buy", "charge", "ramp_up"):
            llm_action = "charge"
        elif llm_action_raw in ("maintain", "hold", "reduce_demand"):
            llm_action = "hold"
        else:
            llm_action = "hold"

        # Step 3: GUARDRAIL — Physical/safety override only when necessary
        charge_ratio = self.charge_level / self.capacity_mwh
        final_action = llm_action
        final_price = llm_price
        guardrail_triggered = False

        # CRITICAL: Force charge if critically empty (< 5%)
        if charge_ratio < 0.05:
            final_action = "charge"
            final_price = min(llm_price, market_price - 2)
            guardrail_triggered = True
            logger.warning(
                f"[BATTERY GUARDRAIL] SoC critically low ({charge_ratio:.1%}) — "
                f"LLM wanted {llm_action}, forced charge"
            )

        # CRITICAL: Force discharge if critically full (> 95%)
        elif charge_ratio > 0.95:
            final_action = "discharge"
            final_price = max(llm_price, market_price + 2)
            guardrail_triggered = True
            logger.warning(
                f"[BATTERY GUARDRAIL] SoC critically high ({charge_ratio:.1%}) — "
                f"LLM wanted {llm_action}, forced discharge"
            )

        # WARNING: Override discharge if nearly empty (< 15%)
        elif llm_action == "discharge" and charge_ratio < 0.15:
            final_action = "hold"
            guardrail_triggered = True
            logger.warning(
                f"[BATTERY GUARDRAIL] LLM wanted discharge at {charge_ratio:.1%} SoC — "
                f"overridden to hold (safety)"
            )

        # WARNING: Override charge if nearly full (> 85%)
        elif llm_action == "charge" and charge_ratio > 0.85:
            final_action = "hold"
            guardrail_triggered = True
            logger.warning(
                f"[BATTERY GUARDRAIL] LLM wanted charge at {charge_ratio:.1%} SoC — "
                f"overridden to hold (safety)"
            )

        return {
            "bid_price": round(final_price, 2),
            "output_adjustment": final_action,
            "llm_bid_price": round(llm_price, 2),
            "llm_action": llm_action,
            "reasoning": reasoning,
            "confidence": confidence,
            "guardrail_triggered": guardrail_triggered,
            "validation_errors": validation_errors,
            "is_fallback": raw_decision.get("is_fallback", False),
        }
