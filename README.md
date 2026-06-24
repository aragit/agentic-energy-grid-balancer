# Agentic Energy Grid Balancer

Autonomous multi-agent energy market simulation with LLM-driven bidding, double-sided auction clearing, and real-time grid physics.

## Overview

A production-grade simulation where autonomous agents (solar, wind, coal, nuclear, battery storage, consumer) bid into a double-sided energy market. The system balances supply/demand, tracks carbon emissions, and stabilizes grid frequency — all observable via a REST API.

## Architecture

| Component | Technology |
|-----------|------------|
| API | FastAPI |
| Physics | Custom weather + demand models, grid frequency simulation |
| Market | Continuous double-sided auction with carbon pricing |
| Agents | 6 agent types with LLM-powered strategy (Mock/Ollama) |
| Memory | Episodic learning with pattern recognition |
| Tests | 93 pytest cases (physics, auction, agents, API, E2E) |

## Agents

- **SolarFarm / WindFarm** — Renewable, zero marginal cost, weather-dependent
- **CoalPlant / NuclearPlant** — Baseload with carbon costs
- **GridBattery** — Arbitrage: charges when cheap, discharges when expensive
- **MetroCity** — Consumer with price-elastic demand
- **GridOrchestrator** — Emergency frequency stabilization
- **RegulatoryAgent** — Carbon cap + frequency violation monitoring

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run API
python -m uvicorn api.main:app --host 0.0.0.0 --port 8001

# Run simulation
curl -X POST http://localhost:8001/simulation/run \
  -H "Content-Type: application/json" \
  -d '{"steps": 24, "llm_backend": "mock"}'

# View results
curl http://localhost:8001/agents/performance
curl http://localhost:8001/market/history
curl http://localhost:8001/carbon/report