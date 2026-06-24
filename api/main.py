"""FastAPI application for Agentic Energy Grid Balancer."""

import os
import logging
from contextlib import asynccontextmanager
from typing import List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from api.schemas import (
    HealthResponse, SimulationRequest, SimulationResponse,
    AgentPerformanceResponse, GridStateResponse,
    CarbonReportResponse, MarketHistoryResponse,
)
from core.llm_engine import LLMEngineFactory
from core.simulation import GridSimulation
from database.models import init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

current_simulation = None
current_llm = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[API] Starting up...")
    init_db()
    global current_llm
    backend = os.getenv("LLM_BACKEND", "mock")
    current_llm = LLMEngineFactory.create(backend=backend)
    logger.info(f"[API] LLM backend: {backend}")
    yield
    logger.info("[API] Shutting down...")
    if current_llm:
        current_llm.shutdown()


app = FastAPI(
    title="Agentic Energy Grid Balancer",
    description="Autonomous multi-agent energy market simulation with LLM reasoning",
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/health", response_model=HealthResponse)
async def health():
    backend = os.getenv("LLM_BACKEND", "mock")
    return HealthResponse(status="ok", llm_mode=backend)


@app.post("/simulation/run", response_model=SimulationResponse)
async def run_simulation(request: SimulationRequest):
    global current_simulation, current_llm

    if request.llm_backend:
        current_llm = LLMEngineFactory.create(backend=request.llm_backend)

    current_simulation = GridSimulation(llm=current_llm, steps=request.steps)
    result = current_simulation.run()

    return SimulationResponse(
        simulation_id=result.simulation_id,
        total_steps=result.total_steps,
        final_frequency=result.final_frequency,
        total_carbon_emitted=result.total_carbon_emitted,
        total_demand_mwh=result.total_demand_mwh,
        total_supply_mwh=result.total_supply_mwh,
        agent_balances=result.agent_balances,
        price_history=result.price_history,
        violations=result.violations,
        status=result.status,
    )


@app.get("/simulation/status")
async def simulation_status():
    if not current_simulation:
        raise HTTPException(status_code=404, detail="No simulation running")
    return {
        "steps_completed": len(current_simulation.auction.price_history),
        "current_frequency": round(current_simulation.physics.frequency, 3),
        "current_price": current_simulation.auction.price_history[-1] if current_simulation.auction.price_history else 50.0,
    }


@app.get("/agents/performance", response_model=List[AgentPerformanceResponse])
async def agent_performance():
    if not current_simulation:
        raise HTTPException(status_code=404, detail="No simulation running")

    results = []
    for agent in current_simulation.agents:
        results.append(AgentPerformanceResponse(
            agent_name=agent.name,
            agent_type=agent.agent_type,
            balance=round(agent.state.balance, 2),
            total_revenue=round(agent.state.total_revenue, 2),
            total_cost=round(agent.state.total_cost, 2),
            total_carbon_emitted=round(agent.state.total_carbon_emitted, 2),
            strategy_count=len(agent.state.strategy_history),
        ))
    return results


@app.get("/market/history", response_model=MarketHistoryResponse)
async def market_history():
    if not current_simulation:
        raise HTTPException(status_code=404, detail="No simulation running")

    return MarketHistoryResponse(
        transactions=current_simulation.auction.transaction_history,
        price_trend=current_simulation.auction.get_price_trend(),
    )


@app.get("/carbon/report", response_model=CarbonReportResponse)
async def carbon_report():
    if not current_simulation:
        raise HTTPException(status_code=404, detail="No simulation running")

    total_carbon = sum(a.state.total_carbon_emitted for a in current_simulation.agents)
    breakdown = []
    for agent in current_simulation.agents:
        breakdown.append({
            "agent_name": agent.name,
            "agent_type": agent.agent_type,
            "carbon_kg": round(agent.state.total_carbon_emitted, 2),
            "carbon_intensity": agent.state.carbon_intensity_g_kwh,
        })

    return CarbonReportResponse(
        total_carbon_kg=round(total_carbon, 2),
        carbon_cost_total=round(current_simulation.auction.carbon_cost_total, 2),
        agent_breakdown=breakdown,
    )


@app.get("/")
async def root():
    return {"message": "Agentic Energy Grid Balancer API", "docs": "/docs", "dashboard": "/static/index.html"}