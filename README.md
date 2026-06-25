<h1 align="center">🔋 Agentic Energy Grid Balancer</h1>
<p align="center"><b>Multi-Agent Energy Market Simulation with LLM Battery Bidding</b></p>

<p align="center"><sub>FastAPI · NumPy · Docker · pytest · Ollama</sub></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.111+-teal?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Architecture-Type%202%20(Symbolic[Neural])-violet" alt="Neural-Symbolic">
  <img src="https://img.shields.io/badge/Tests-120%20passing-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT">
</p>

---

Six energy agents trade in a double-sided auction. Generators and consumers use
deterministic marginal-cost bidding; the battery agent uses an LLM (or rule-based
ReasoningEngine) for arbitrage strategy. A symbolic orchestrator validates all
bids, grid physics checks frequency stability, and a regulator enforces carbon caps.

**Architecture: Type 2 (Symbolic[Neuro]) — symbolic outer loop governs a neural
bidding subroutine. Only the battery agent uses LLM reasoning; the rest use
hardcoded economic rules.**

---

## Quick Start

```bash
git clone https://github.com/aragit/agentic-energy-grid-balancer.git
cd agentic-energy-grid-balancer
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Start API in background
python -m uvicorn api.main:app --host 0.0.0.0 --port 8001 &
sleep 3

# Run a 24-step simulation
curl -X POST http://localhost:8001/simulation/run \
  -H "Content-Type: application/json" \
  -d '{"steps": 24, "llm_backend": "mock"}'
```

Open [http://localhost:8001/docs](http://localhost:8001/docs) for the interactive
API documentation (Swagger UI). If port 8001 is in use, kill the old process
first: `fuser -k 8001/tcp`.

**Docker:**

```bash
docker compose up --build
```

The CI pipeline (GitHub Actions) builds Docker and runs health + simulation
endpoint tests automatically on push. Local Docker builds may hit Docker Hub
rate limits on shared IPs.

---

## How It Works

| Step | Component | What Happens |
|:-----|:----------|:-------------|
| 1 | Weather Engine | Generates hour-by-hour solar irradiance, wind speed, temperature, cloud cover, storms (seasonal sinusoidal + noise) |
| 2 | Generator Agents | Solar, wind, coal, nuclear compute physical output from weather. All bid at marginal cost + $5 markup (hardcoded — no LLM) |
| 3 | Consumer Agent | Submits a buy bid based on a piecewise willingness-to-pay function (hardcoded — no LLM) |
| 4 | Battery Agent | Calls LLM (Ollama) or ReasoningEngine for charge/discharge/hold decision. Physical guardrails (SoC thresholds) clamp extreme outputs but preserve the LLM's direction in the mid-range |
| 5 | Orchestrator | Validates all bids (price ± bounds, min quantity). Rejects out-of-range bids before auction |
| 6 | Auction Engine | Matches buy/sell orders at per-trade midpoint prices, adds carbon cost ($25/ton, coal only). Reports weighted-average clearing price |
| 7 | Grid Physics | Computes frequency from supply-demand imbalance (damped model, clamped to 47–53 Hz) |
| 8 | Regulatory Agent | Logs frequency violations (±1 Hz threshold) and cumulative carbon cap (50,000 kg default) |
| 9 | Results | JSON response with agent balances, price history, carbon totals, violations |

### LLM Usage

| Agent | LLM Called? | Bid Source |
|:------|:-----------|:-----------|
| SolarFarm | No | Marginal cost ($0) + $5 |
| WindFarm | No | Marginal cost ($0) + $5 |
| CoalPlant | No | Marginal cost ($20 + carbon) + $5 |
| NuclearPlant | No | Marginal cost ($5) + $5 |
| GridBattery | **Yes** | `decide_bid()` → LLM → Pydantic validation → guardrails |
| MetroCity | No | Piecewise WTP ($45–$85) |

The battery agent has two backends:

| Backend | Speed | Behavior |
|:--------|:------|:---------|
| **ReasoningEngine** (default) | Instant | Rule-based: buy if price < $40, sell if > $70, hold otherwise. Deterministic with seed |
| **Ollama** | ~10s/call | Real local LLM (tinyllama/qwen2.5). Structured output validated via Pydantic before use |

Battery guardrails preserve the LLM's decision when SoC is between 5% and 95%.
Below 5% → force charge. Above 95% → force discharge. Between 15%–85% the
LLM's choice passes through unchanged.

---

## Architecture Notes

This is a **Type 2 (Symbolic[Neuro])** system per Kautz taxonomy — the
symbolic simulation loop is the primary controller and calls a neural
subroutine (battery LLM) as a bounded, replaceable component.

**What works:**
- Pydantic `BidStrategy` validates LLM output at the neural/symbolic boundary
- Orchestrator rejects out-of-range bids before auction clearing
- Generator economic dispatch follows rational marginal-cost curves
- Battery guardrails clamp only extreme SoC, preserving LLM output in mid-range
- Price history and carbon accounting exposed via REST API

**Known gaps:**
- **Only battery uses LLM** — other agents use hardcoded rules (intentional:
  generators follow marginal-cost economics, not LLM strategy)
- **Auction uses per-trade midpoint pricing**, not true uniform-price clearing
  — the reported "clearing price" is a weighted average, not an intersection price
- **Orchestrator validation is basic** — price bounds [1, 200] and min quantity
  only. No physics-aware or cross-agent validation
- **No database persistence** — `db_session` is accepted but never written to
- **Frequency model is simple damping** — not a full harmonic oscillator
- **No real-time capability** — each step takes ~10s with Ollama on CPU; the
  system is a research/educational simulation, not production grid software

---

## API Endpoints

| Method | Path | Description |
|:-------|:-----|:------------|
| `GET` | `/` | Root message with docs link |
| `GET` | `/health` | System health + LLM backend mode |
| `POST` | `/simulation/run` | Run simulation. Body: `{steps, llm_backend, ollama_model}` |
| `GET` | `/simulation/status` | Current state (steps, frequency, price) |
| `GET` | `/agents/performance` | Per-agent financials |
| `GET` | `/market/history` | All transactions + price trend |
| `GET` | `/market/prices` | Price history as numeric array |
| `GET` | `/carbon/report` | Total carbon + per-agent breakdown |

---

## Testing

```bash
pytest tests/ -v
```

~120 tests across 9 modules:

| Module | Count | What's Verified |
|:-------|:-----|:----------------|
| `test_grid_physics.py` | 17 | Weather generation, solar/wind physics, frequency, demand |
| `test_auction.py` | 14 | Matching, carbon cost, price clamping, surplus |
| `test_agents.py` | 9 | Agent output computation |
| `test_api.py` | 14 | Health, run, status, data endpoints |
| `test_simulation.py` | 12 | Full run, frequency, carbon, determinism |
| `test_orchestrator.py` | 8 | Frequency stabilization, reserve dispatch |
| `test_orchestrator_validation.py` | 15 | Bid validation rules |
| `test_battery_guardrails.py` | 12 | LLM output honored, guardrail boundary conditions |
| `test_pydantic_boundary.py` | ~18 | BidStrategy validation, missing fields, coercion |

---

## Tech Stack

| Layer | Technology |
|:------|:-----------|
| **LLM Backends** | ReasoningEngine (rule-based, instant) / Ollama (tinyllama, qwen2.5) |
| **Math** | NumPy (grid physics, statistics) |
| **API** | FastAPI + Pydantic v2 |
| **ORM** | SQLAlchemy 2.0 (models defined, not written to at runtime) |
| **Container** | Docker + docker compose |
| **Testing** | pytest |
| **CI** | GitHub Actions (test, lint, docker) |

---

## Neuro-Symbolic Architecture Trace

```bash
python scripts/trace_neuro_symbolic.py
```

Prints a 3-step trace showing L1 weather → L3 battery LLM reasoning → auction
clearing → L6 regulatory oversight with data-flow diagram.

---

## License

MIT — see [LICENSE](LICENSE) for details.
