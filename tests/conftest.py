"""Shared fixtures for all tests."""

import pytest
from core.llm_engine import MockLLMEngine
from core.grid_physics import WeatherEngine, GridPhysics, DemandModel
from core.auction import DoubleSidedAuction, Bid
from core.agents.solar import SolarAgent
from core.agents.wind import WindAgent
from core.agents.coal import CoalAgent
from core.agents.nuclear import NuclearAgent
from core.agents.battery import BatteryAgent
from core.agents.consumer import ConsumerAgent
from core.agents.orchestrator import GridOrchestratorAgent
from core.agents.regulatory import RegulatoryAgent
from core.simulation import GridSimulation
from core.memory import AgentMemory, Experience


@pytest.fixture
def llm():
    return MockLLMEngine(seed=42)


@pytest.fixture
def weather():
    return WeatherEngine(seed=42)


@pytest.fixture
def physics():
    return GridPhysics(base_demand=500.0)


@pytest.fixture
def auction():
    return DoubleSidedAuction(carbon_price_per_ton=25.0)


@pytest.fixture
def solar_agent(llm):
    return SolarAgent("SolarTest", 100.0, llm)


@pytest.fixture
def wind_agent(llm):
    return WindAgent("WindTest", 80.0, llm)


@pytest.fixture
def coal_agent(llm):
    return CoalAgent("CoalTest", 200.0, llm)


@pytest.fixture
def nuclear_agent(llm):
    return NuclearAgent("NuclearTest", 300.0, llm)


@pytest.fixture
def battery_agent(llm):
    return BatteryAgent("BatteryTest", 50.0, 25.0, llm)


@pytest.fixture
def consumer_agent(llm):
    return ConsumerAgent("ConsumerTest", 500.0, llm)


@pytest.fixture
def orchestrator():
    return GridOrchestratorAgent(reserve_capacity_mw=100.0)


@pytest.fixture
def regulator():
    return RegulatoryAgent(carbon_cap_kg=50000.0)


@pytest.fixture
def simulation(llm):
    return GridSimulation(llm=llm, steps=5)


@pytest.fixture
def sample_experience():
    return Experience(
        step=1,
        market_price=50.0,
        bid_price=45.0,
        output_mw=100.0,
        revenue=4500.0,
        carbon_cost=0.0,
        net_profit=4500.0,
        frequency=50.0,
        weather={"temperature": 20.0, "is_storm": False},
        decision={"bid_price": 45.0, "reasoning": "test"},
        outcome="profitable",
    )