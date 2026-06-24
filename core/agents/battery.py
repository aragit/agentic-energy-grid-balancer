"""Battery storage agent with arbitrage strategy."""

import json
from typing import Dict, Any
from core.agents.base import BaseAgent
from core.memory import Experience


class BatteryAgent(BaseAgent):
    """Battery energy storage system with charge/discharge arbitrage."""

    def __init__(self, name: str, capacity_mwh: float, max_charge_mw: float, llm):
        super().__init__(name, "battery", max_charge_mw, carbon_intensity_g_kwh=0.0, llm=llm)
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

    def update_after_trade(self, energy_mwh: float, price_per_mwh: float,
                           carbon_cost: float = 0.0, step: int = 0,
                           weather: Dict = None):
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

        self.state.total_carbon_emitted += energy_mwh * self.state.carbon_intensity_g_kwh / 1000

        # RECORD MEMORY: essential for learning
        experience = Experience(
            step=step,
            market_price=price_per_mwh,
            bid_price=self.state.strategy_history[-1]["bid_price"] if self.state.strategy_history else price_per_mwh,
            output_mw=abs(energy_mwh),
            revenue=revenue if energy_mwh > 0 else -abs(revenue),
            carbon_cost=carbon_cost,
            net_profit=net_profit if energy_mwh > 0 else -abs(revenue),
            frequency=50.0,
            weather=weather or {},
            decision=self.state.strategy_history[-1] if self.state.strategy_history else {},
            outcome="profitable" if (energy_mwh > 0 and net_profit > 0) or (energy_mwh < 0 and abs(revenue) < 500) else "loss",
        )
        self.memory.record(experience)

    def _build_prompt(self, market_price: float, demand: float,
                      frequency: float, carbon_price: float,
                      memory_context: str, strategy_advice: str,
                      weather: Dict[str, Any] = None) -> str:
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
        
        # FIX: No $ symbols before numbers — MockLLM parser fails on $
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

Respond with JSON:
{{
    "bid_price": float,
    "output_adjustment": "charge" | "discharge" | "hold",
    "carbon_trade": float,
    "reasoning": "string explaining your arbitrage decision based on price trend and charge level",
    "confidence": 0.0-1.0
}}
"""