"""Base agent class with state management, memory, and LLM integration."""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass, field

from core.llm_engine import BaseLLMEngine, LLMResponse
from core.memory import AgentMemory, Experience
from core.schemas import BidStrategy

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
        """Invoke neural subroutine and validate output via Pydantic schema.

        Type 2 (Symbolic[Neuro]): The LLM produces stochastic tokens;
        BidStrategy.model_validate_json() compresses them into a structured,
        bounded, typed object before any symbolic computation touches the data.
        """
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

        validated_strategy = None
        parse_errors = []

        try:
            response: LLMResponse = self.llm.chat_completion(
                messages=messages, temperature=0.7, max_tokens=256
            )
            # TYPE 2 BOUNDARY: Pydantic validation compresses stochastic tokens → exact params
            validated_strategy = BidStrategy.model_validate_json(response.content)
            logger.info(
                f"[{self.name}] LLM output validated: price={validated_strategy.bid_price}, "
                f"action={validated_strategy.output_adjustment}, confidence={validated_strategy.confidence}"
            )

        except json.JSONDecodeError as e:
            parse_errors.append(f"JSON decode failed: {e}")
            logger.warning(f"[{self.name}] LLM returned invalid JSON: {e}")
        except Exception as e:
            # Pydantic ValidationError or any other exception
            parse_errors.append(f"Validation failed: {e}")
            logger.warning(f"[{self.name}] LLM output failed Pydantic validation: {e}")

        # If validation failed, use fallback but RECORD the failure
        if validated_strategy is None:
            fallback = self._fallback_strategy(market_price)
            # Convert fallback dict to BidStrategy for uniform interface
            try:
                validated_strategy = BidStrategy.model_validate(fallback)
                logger.info(f"[{self.name}] Using fallback strategy due to validation failure")
            except Exception:
                # Ultimate fallback: hardcoded safe values
                validated_strategy = BidStrategy(
                    bid_price=market_price,
                    output_adjustment="hold",
                    confidence=0.5,
                    reasoning="Ultimate fallback: validation and fallback both failed",
                )

        # Build decision dict with full audit trail
        decision = {
            "agent_name": self.name,
            "agent_type": self.agent_type,
            "bid_price": round(validated_strategy.bid_price, 2),
            "output_adjustment": validated_strategy.output_adjustment,
            "carbon_trade": round(validated_strategy.carbon_trade, 2),
            "reasoning": validated_strategy.reasoning,
            "confidence": round(validated_strategy.confidence, 2),
            "latency_ms": getattr(response, "latency_ms", 0) if 'response' in dir() else 0,
            "validation_errors": parse_errors if parse_errors else None,
            "is_fallback": len(parse_errors) > 0,
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

Respond with VALID JSON matching this exact schema:
{{
    "bid_price": float,        // USD per MWh, must be between 1.0 and 200.0
    "output_adjustment": string,  // One of: "charge", "discharge", "hold", "sell", "buy", "maintain", "ramp_up", "ramp_down", "reduce_demand"
    "carbon_trade": float,     // kg CO2, optional (default 0)
    "reasoning": string,       // Explain your decision in 1-2 sentences
    "confidence": float        // 0.0 to 1.0
}}
"""

    def _fallback_strategy(self, market_price: float) -> Dict[str, Any]:
        return {
            "bid_price": round(max(1.0, min(200.0, market_price * 0.95)), 2),
            "output_adjustment": "hold",
            "carbon_trade": 0.0,
            "reasoning": "Fallback: conservative hold due to LLM parse/validation failure",
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
