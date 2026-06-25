"""Base agent class with state management, memory, and LLM integration."""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass, field

from core.llm_engine import BaseLLMEngine, LLMResponse
from core.memory import AgentMemory, Experience

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    name: str
    agent_type: str
    capacity_mw: float
    current_output_mw: float = 0.0
    balance: float = 0.0
    total_revenue: float = 0.0
    total_cost: float = 0.0
    total_carbon_emitted: float = 0.0
    carbon_intensity_g_kwh: float = 0.0
    strategy_history: List[Dict] = field(default_factory=list)
    is_active: bool = True


class BaseAgent(ABC):
    def __init__(
        self,
        name: str,
        agent_type: str,
        capacity_mw: float,
        carbon_intensity_g_kwh: float,
        llm: BaseLLMEngine,
    ):
        self.name = name
        self.agent_type = agent_type
        self.llm = llm
        self.state = AgentState(
            name=name,
            agent_type=agent_type,
            capacity_mw=capacity_mw,
            carbon_intensity_g_kwh=carbon_intensity_g_kwh,
        )
        self.memory = AgentMemory(max_history=100)

    @abstractmethod
    def compute_output(self, weather: Any, market_price: float) -> float:
        pass

    def decide_bid(
        self,
        market_price: float,
        demand: float,
        frequency: float,
        carbon_price: float,
        weather: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        memory_context = self.memory.get_context(current_step=0)
        strategy_advice = self.memory.get_strategy_advice()

        prompt = self._build_prompt(
            market_price,
            demand,
            frequency,
            carbon_price,
            memory_context,
            strategy_advice,
            weather,
        )
        messages = [{"role": "system", "content": prompt}]

        try:
            response: LLMResponse = self.llm.chat_completion(
                messages=messages, temperature=0.7, max_tokens=256
            )
            strategy = json.loads(response.content)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"[{self.name}] LLM parse failed: {e}. Using fallback.")
            strategy = self._fallback_strategy(market_price)

        bid_price = float(strategy.get("bid_price", market_price))
        bid_price = max(bid_price, 1.0)

        decision = {
            "agent_name": self.name,
            "agent_type": self.agent_type,
            "bid_price": round(bid_price, 2),
            "output_adjustment": strategy.get("output_adjustment", "maintain"),
            "carbon_trade": round(float(strategy.get("carbon_trade", 0.0)), 2),
            "reasoning": strategy.get("reasoning", "Fallback strategy"),
            "confidence": float(strategy.get("confidence", 0.5)),
            "latency_ms": getattr(response, "latency_ms", 0),
        }

        self.state.strategy_history.append(decision)
        return decision

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
        return f"""You are an autonomous energy market agent with memory and learning.

Agent: {self.name}
Type: {self.agent_type}
Capacity: {self.state.capacity_mw} MW
Current Output: {self.state.current_output_mw} MW
Balance: ${self.state.balance:.2f}
Carbon Intensity: {self.state.carbon_intensity_g_kwh} gCO2/kWh

Current Market Context:
- Market Price: ${market_price:.2f}/MWh
- Total Demand: {demand:.2f} MWh
- Grid Frequency: {frequency:.3f} Hz
- Carbon Price: ${carbon_price:.2f}/ton
- Weather: {weather_str}

{memory_context}

Strategic Advice:
{strategy_advice}

Respond with JSON:
{{
    "bid_price": float,
    "output_adjustment": "sell" | "ramp_down" | "ramp_up"
    | "maintain" | "charge" | "discharge" | "reduce_demand" | "hold",
    "carbon_trade": float,
    "reasoning": "string",
    "confidence": 0.0-1.0
}}
"""

    def _fallback_strategy(self, market_price: float) -> Dict[str, Any]:
        return {
            "bid_price": round(market_price * 0.95, 2),
            "output_adjustment": "maintain",
            "carbon_trade": 0.0,
            "reasoning": "Fallback: conservative bid at 95% of market price",
            "confidence": 0.5,
        }

    def update_after_trade(
        self,
        energy_mwh: float,
        price_per_mwh: float,
        carbon_cost: float = 0.0,
        step: int = 0,
        weather: Dict[str, Any] = None,
    ):
        revenue = energy_mwh * price_per_mwh
        net_profit = revenue - carbon_cost

        experience = Experience(
            step=step,
            market_price=price_per_mwh,
            bid_price=(
                self.state.strategy_history[-1]["bid_price"]
                if self.state.strategy_history
                else price_per_mwh
            ),
            output_mw=abs(energy_mwh),
            revenue=revenue,
            carbon_cost=carbon_cost,
            net_profit=net_profit,
            frequency=50.0,
            weather=weather or {},
            decision=(
                self.state.strategy_history[-1] if self.state.strategy_history else {}
            ),
            outcome="profitable" if net_profit > 0 else "loss",
        )
        self.memory.record(experience)

        self.state.balance += net_profit
        self.state.total_revenue += revenue
        self.state.total_cost += carbon_cost
        self.state.total_carbon_emitted += (
            energy_mwh * self.state.carbon_intensity_g_kwh / 1000
        )

    def record_output(self, output_mw: float):
        self.state.current_output_mw = output_mw
