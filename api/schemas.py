"""Pydantic request/response models."""

from pydantic import BaseModel
from typing import List, Dict, Optional


class HealthResponse(BaseModel):
    status: str
    llm_mode: str


class SimulationRequest(BaseModel):
    steps: int = 24
    llm_backend: Optional[str] = "mock"
    ollama_model: Optional[str] = "tinyllama"


class SimulationResponse(BaseModel):
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


class AgentPerformanceResponse(BaseModel):
    agent_name: str
    agent_type: str
    balance: float
    total_revenue: float
    total_cost: float
    total_carbon_emitted: float
    strategy_count: int


class GridStateResponse(BaseModel):
    step: int
    frequency: float
    total_demand: float
    total_supply: float
    clearing_price: float
    carbon_price: float


class CarbonReportResponse(BaseModel):
    total_carbon_kg: float
    carbon_cost_total: float
    agent_breakdown: List[Dict]


class MarketHistoryResponse(BaseModel):
    transactions: List[Dict]
    price_trend: str
