"""Simulation orchestrator tying weather, agents, auction, and grid physics together."""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass

from core.grid_physics import WeatherEngine, GridPhysics
from core.auction import DoubleSidedAuction, Bid
from core.agents.base import BaseAgent
from core.agents.solar import SolarAgent
from core.agents.wind import WindAgent
from core.agents.coal import CoalAgent
from core.agents.nuclear import NuclearAgent
from core.agents.battery import BatteryAgent
from core.agents.consumer import ConsumerAgent
from core.agents.orchestrator import GridOrchestratorAgent
from core.agents.regulatory import RegulatoryAgent

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    simulation_id: int
    total_steps: int
    final_frequency: float
    total_carbon_emitted: float
    total_demand_mwh: float
    total_supply_mwh: float
    agent_balances: Dict[str, float]
    price_history: List[float]
    violations: List[Dict]
    status: str


class GridSimulation:
    def __init__(self, llm, db_session=None, steps: int = 24):
        self.llm = llm
        self.steps = steps
        self.db = db_session
        self.weather = WeatherEngine(seed=42)
        self.physics = GridPhysics(base_demand=500.0)
        self.auction = DoubleSidedAuction(carbon_price_per_ton=25.0)
        self.orchestrator = GridOrchestratorAgent(reserve_capacity_mw=100.0)
        self.regulator = RegulatoryAgent(carbon_cap_kg=50000.0)

        self.agents: List[BaseAgent] = [
            SolarAgent("SolarFarm-A", 100.0, llm),
            WindAgent("WindFarm-B", 80.0, llm),
            CoalAgent("CoalPlant-C", 200.0, llm),
            NuclearAgent("NuclearPlant-D", 300.0, llm),
            BatteryAgent("GridBattery-E", 50.0, 25.0, llm),
            ConsumerAgent("MetroCity", 500.0, llm),
        ]

        self.generators = [a for a in self.agents if a.agent_type in ["solar", "wind", "coal", "nuclear"]]
        self.battery = next((a for a in self.agents if a.agent_type == "battery"), None)
        self.consumer = next((a for a in self.agents if a.agent_type == "consumer"), None)

    def run(self) -> SimulationResult:
        logger.info(f"[SIMULATION] Starting {self.steps}-step grid simulation")

        total_demand = 0.0
        total_supply = 0.0

        for step in range(self.steps):
            step_demand, step_supply = self._run_step(step)
            total_demand += step_demand
            total_supply += step_supply

        result = SimulationResult(
            simulation_id=0,
            total_steps=self.steps,
            final_frequency=round(self.physics.frequency, 3),
            total_carbon_emitted=sum(a.state.total_carbon_emitted for a in self.agents),
            total_demand_mwh=round(total_demand, 2),
            total_supply_mwh=round(total_supply, 2),
            agent_balances={a.name: round(a.state.balance, 2) for a in self.agents},
            price_history=self.auction.price_history,
            violations=self.regulator.violations,
            status="completed",
        )

        logger.info(f"[SIMULATION] Complete. Final frequency: {result.final_frequency} Hz")
        return result

    def _run_step(self, step: int) -> tuple[float, float]:
        weather = self.weather.step()

        # Compute generation outputs
        raw_generation = {}
        for gen in self.generators:
            output = gen.compute_output(weather, self.auction.price_history[-1] if self.auction.price_history else 50.0)
            raw_generation[gen.name] = output

        # Compute demand
        current_price = self.auction.price_history[-1] if self.auction.price_history else 50.0
        if self.consumer:
            self.consumer.set_hour(self.weather.hour)
            demand = self.consumer.compute_output(weather, current_price)
        else:
            demand = 500.0

        # Build bids
        bids = []
        
        # Consumer bids based on willingness-to-pay (rule-based — demand curve)
        if self.consumer and demand > 0:
            wtp = self._compute_wtp(demand)
            bids.append(Bid(
                agent_name=self.consumer.name,
                agent_type=self.consumer.agent_type,
                bid_price=wtp,
                quantity_mw=demand,
                is_buy=True,
            ))

        # Generators bid at marginal cost + competitive margin (rule-based — economic reality)
        for gen in self.generators:
            if raw_generation[gen.name] > 0.01:
                marginal_cost = self._compute_marginal_cost(gen)
                scarcity_premium = 0.0
                total_gen_capacity = sum(raw_generation.values())
                if demand > total_gen_capacity * 0.9:
                    scarcity_premium = 10.0
                bid_price = marginal_cost + 5.0 + scarcity_premium
                
                bids.append(Bid(
                    agent_name=gen.name,
                    agent_type=gen.agent_type,
                    bid_price=bid_price,
                    quantity_mw=raw_generation[gen.name],
                    is_buy=False,
                ))

        # Battery: AGENTIC — uses LLM reasoning for arbitrage strategy
        if self.battery:
            battery_strategy = self.battery.decide_bid(
                market_price=current_price,
                demand=demand,
                frequency=self.physics.frequency,
                carbon_price=self.auction.carbon_price,
                weather=weather.__dict__,
            )
            
            bid_price = float(battery_strategy.get("bid_price", current_price))
            output_adj = battery_strategy.get("output_adjustment", "hold")
            
            # MOCKLLM GUARDRAIL: Override broken parser with physically correct logic
            charge_ratio = self.battery.charge_level / self.battery.capacity_mwh
            
            if bid_price <= 0.01 or bid_price > 200:
                bid_price = current_price
            
            # Normalize action: MockLLM returns "sell"/"maintain"/etc
            if output_adj in ("sell", "discharge"):
                output_adj = "discharge"
            elif output_adj in ("buy", "charge", "ramp_up"):
                output_adj = "charge"
            elif output_adj in ("maintain", "hold", "reduce_demand"):
                output_adj = "hold"
            
            # PHYSICAL ARBITRAGE LOGIC: Buy low, sell high, with safety bounds
            if charge_ratio > 0.85:
                output_adj = "discharge"
                bid_price = max(bid_price, current_price + 2)
            elif charge_ratio < 0.15:
                output_adj = "charge"
                bid_price = min(bid_price, current_price - 2)
            elif current_price < 40 and charge_ratio < 0.8:
                output_adj = "charge"
                bid_price = current_price - 2
            elif current_price > 55 and charge_ratio > 0.2:
                output_adj = "discharge"
                bid_price = current_price + 2
            else:
                output_adj = "hold"
            
            logger.info(f"[BATTERY] step={step}, llm_price={float(battery_strategy.get('bid_price', 0)):.2f}, "
                       f"final_action={output_adj}, final_price={bid_price:.2f}, "
                       f"charge={self.battery.charge_level:.1f}/{self.battery.capacity_mwh} ({charge_ratio*100:.0f}%)")
            
            if output_adj == "charge" and self.battery.get_available_charge() > 0.01:
                bids.append(Bid(
                    agent_name=self.battery.name,
                    agent_type=self.battery.agent_type,
                    bid_price=bid_price,
                    quantity_mw=self.battery.get_available_charge(),
                    is_buy=True,
                ))
                logger.info(f"[BATTERY BID] BUY {self.battery.get_available_charge():.1f} MW at {bid_price:.2f}")
            elif output_adj == "discharge" and self.battery.get_available_discharge() > 0.01:
                bids.append(Bid(
                    agent_name=self.battery.name,
                    agent_type=self.battery.agent_type,
                    bid_price=bid_price,
                    quantity_mw=self.battery.get_available_discharge(),
                    is_buy=False,
                ))
                logger.info(f"[BATTERY BID] SELL {self.battery.get_available_discharge():.1f} MW at {bid_price:.2f}")
            else:
                logger.info(f"[BATTERY BID] HOLD — no bid submitted")

        # Clear market
        clearing = self.auction.clear_market(bids)

        # Calculate actual supply vs demand
        actual_supply = 0.0
        actual_demand_met = 0.0
        
        for tx in clearing.transactions:
            seller = next((a for a in self.agents if a.name == tx["seller"]), None)
            buyer = next((a for a in self.agents if a.name == tx["buyer"]), None)

            if seller:
                seller.update_after_trade(tx["quantity_mwh"], tx["price_per_mwh"], tx["carbon_cost"], step, weather.__dict__)
                if seller.agent_type == "battery":
                    seller.apply_charge(-tx["quantity_mwh"])  # Discharging reduces charge
                actual_supply += tx["quantity_mwh"]

            if buyer:
                if buyer.agent_type == "battery":
                    # Battery buying (charging): track cost and increase charge level
                    buyer.update_after_trade(-tx["quantity_mwh"], tx["price_per_mwh"], 0, step, weather.__dict__)
                    buyer.apply_charge(tx["quantity_mwh"])  # Charging increases charge
                    actual_demand_met += tx["quantity_mwh"]  # Battery charging is grid demand
                elif buyer.agent_type == "consumer":
                    buyer.update_after_trade(-tx["quantity_mwh"], tx["price_per_mwh"], 0, step, weather.__dict__)
                    actual_demand_met += tx["quantity_mwh"]

        demand_met = min(actual_demand_met, demand)
        imbalance = actual_supply - demand_met
        frequency = self.physics.compute_frequency(imbalance)

        if not self.physics.is_stable(frequency):
            action = self.orchestrator.stabilize(frequency, imbalance, self.agents)
            if action["action"] == "dispatch":
                frequency = self.physics.compute_frequency(imbalance + action["dispatch_mw"])

        self.regulator.check_frequency(frequency, step)
        for agent in self.agents:
            self.regulator.check_carbon(agent, step)

        logger.info(f"[STEP {step}] RawGen={sum(raw_generation.values()):.1f}MW, Demand={demand:.1f}MW, "
                   f"Traded={clearing.total_traded_mwh:.1f}MW, Supply={actual_supply:.1f}MW, "
                   f"Imbalance={imbalance:.1f}MW, Freq={frequency:.3f}Hz, Price=${clearing.clearing_price}")

        return demand, actual_supply

    def _compute_wtp(self, demand: float) -> float:
        """Compute consumer willingness-to-pay based on demand level."""
        base_wtp = 60.0
        if demand > 500:
            base_wtp = 75.0
        if demand > 700:
            base_wtp = 85.0
        if demand < 300:
            base_wtp = 45.0
        return min(90.0, max(35.0, base_wtp))

    def _compute_marginal_cost(self, gen: BaseAgent) -> float:
        """Compute marginal cost for a generator."""
        carbon_cost = (gen.state.carbon_intensity_g_kwh / 1000) * self.auction.carbon_price
        if gen.agent_type == "solar":
            return 0.0
        elif gen.agent_type == "wind":
            return 0.0
        elif gen.agent_type == "nuclear":
            return 5.0
        elif gen.agent_type == "coal":
            return 20.0 + carbon_cost
        return 10.0