<h1 align="center">🔋 Agentic Energy Grid Balancer</h1>
<p align="center"><b>Multi-Agent Energy Market Simulation — Type 2 (Symbolic[Neuro])</b></p>

<p align="center"><sub>FastAPI · Pydantic v2 · SQLAlchemy · NumPy · httpx · Docker · pytest · Ollama</sub></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.111+-teal?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Architecture-Type%202%20(Symbolic[Neural])-violet" alt="Neural-Symbolic">
  <img src="https://img.shields.io/badge/Tests-120%20passing-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT">
</p>

---

Six energy agents (solar, wind, coal, nuclear, battery, consumer) trade in a
double-sided auction. A symbolic orchestrator manages the simulation loop,
validates bids, clears the market, computes grid physics, and enforces
regulatory constraints. A neural subroutine (battery LLM) proposes arbitrage
strategies — bounded by Pydantic validation and physical guardrails.

**Architecture: Type 2 (Symbolic[Neuro]) per Kautz taxonomy (2020).**

> **ChatGPT prompt for architecture diagram:**
>
> *"Create a clean system architecture diagram (SVG/PNG) for a Type 2
> (Symbolic[Neuro]) energy grid simulation. The diagram has 4 colored zones:
>
> Zone 1 — User/Client (curl, Swagger) on the left, arrows pointing right.
>
> Zone 2 — API Layer (FastAPI with /simulation/run, /health, /market/* endpoints).
>
> Zone 3 — Symbolic Outer Loop (blue background), the primary controller,
> containing: GridSimulation.run() → GridOrchestratorAgent.validate_bids()
> → DoubleSidedAuction (midpoint pricing) → GridPhysics (weather + frequency)
> → RegulatoryAgent (carbon + frequency checks). These flow left-to-right.
>
> Zone 4 — below or beside the loop, two sub-rows:
>   (a) Deterministic Agents (green background): SolarFarm ($0+$5), WindFarm
>   ($0+$5), CoalPlant ($20+$25/ton carbon+$5), NuclearPlant ($5+$5),
>   MetroCity consumer (piecewise WTP $45-$85). All arrow into Orchestrator.
>   (b) Neural Subroutine (purple background) showing: LLMEngineFactory
>   branches to ReasoningEngine (rule-based) and Ollama (tinyllama/qwen2.5);
>   both output JSON to BidStrategy Pydantic boundary (red border); then to
>   BatteryGuardrails (orange, <5% force charge, >95% force discharge); then
>   arrow into Orchestrator.
>
> Also show: AgentMemory (in-memory context) feeding LLMEngineFactory; SQLAlchemy models (defined, unused) with a dashed line; SimulationResult JSON as final output.
>
> Use a technical color scheme with clean boxes, rounded corners, thin borders,
> no clipart. Label all arrows with brief text like 'POST /simulation/run' or
> 'validated bid'. No title block needed."*

**How the neural subroutine integrates:**

---

## Type 2 Architecture — Why This Qualifies

Per Kautz's AAAI 2020 taxonomy, Type 2 Symbolic[Neuro] means a symbolic system
is the primary controller, owns the execution loop, and calls neural components
as bounded, replaceable subroutines. The symbolic layer makes final decisions.

| Type 2 Requirement | This System |
|:-------------------|:------------|
| **Symbolic primary controller** | `GridSimulation._run_step()` owns the hour-by-hour loop — weather → bidding → auction → physics → governance |
| **Neural as bounded subroutine** | Battery LLM proposes a bid; `BidStrategy` Pydantic model validates JSON schema, types, and ranges at the boundary |
| **Symbolic validates before execution** | `GridOrchestrator.validate_bids()` rejects out-of-range bids (price, quantity) before auction clearing |
| **Symbolic can override neural** | Battery guardrails force charge at SoC < 5% and force discharge at SoC > 95%. Between 15%–85% the LLM output passes through unchanged |
| **Loose coupling** | LLM backend is swappable via `LLMEngineFactory` — ReasoningEngine (rule-based), Ollama (local LLM), or any `BaseLLMEngine` subclass |
| **Deterministic replay** | Same seed → identical weather, generator bids, and reasoning engine output. Only Ollama introduces variance |

**How the neural subroutine integrates:**

```text
BatteryAgent.decide_bid()
  → LLM (ReasoningEngine or Ollama)
  → raw JSON string
  → BidStrategy.model_validate_json()    ← Pydantic boundary
  → guardrails (SoC thresholds)          ← symbolic clamps extreme cases
  → validated bid to orchestrator
  → auction → physics → governance
```

This is not a neural-only system. The LLM proposes; the symbolic core disposes.
Generators and consumer use deterministic economic rules (marginal cost +
markup) — not because the architecture forbids LLM there, but because
marginal-cost bidding is the economically rational baseline for those agent
types in a simulation context.

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
API documentation (Swagger UI). If port 8001 is in use: `fuser -k 8001/tcp`.

**Docker:**

```bash
docker compose up --build
```

CI (GitHub Actions) builds Docker and runs health + simulation endpoint tests on
every push. Local Docker builds may hit Docker Hub rate limits on shared IPs.

---

## How It Works

| Step | Component | What Happens |
|:-----|:----------|:-------------|
| 1 | Weather Engine | Hour-by-hour solar irradiance, wind speed, temperature, storms (seasonal sinusoidal + Perlin noise) |
| 2 | Generators | Solar, wind, coal, nuclear compute output from weather. Bid at marginal cost + $5 markup |
| 3 | Consumer | Buy bid from piecewise willingness-to-pay function ($45–$85 based on demand) |
| 4 | Battery Agent | LLM proposes charge/discharge/hold. Pydantic validates JSON. Guardrails clamp extreme SoC |
| 5 | Orchestrator | Validates all bids (price bounds, min quantity). Rejects invalid bids pre-auction |
| 6 | Auction Engine | Matches buys/sells at per-trade midpoint prices. Adds carbon cost ($25/ton, coal only) |
| 7 | Grid Physics | Frequency from supply-demand imbalance (damped model, clamped 47–53 Hz) |
| 8 | Regulatory Agent | Logs frequency violations (±1 Hz) and cumulative carbon cap (50,000 kg) |
| 9 | Results | JSON: agent balances, price history, carbon totals, violations |

### Battery LLM Backends

| Backend | Speed | Behavior |
|:--------|:------|:---------|
| **ReasoningEngine** (default) | Instant | Rule-based: buy if price < $40, sell if > $70, hold otherwise. Deterministic with seed |
| **Ollama** | ~10s/call | Real local LLM (tinyllama, qwen2.5). Structured output → Pydantic validation → guardrails |

Guardrails preserve the LLM's decision when SoC is 15%–85%. Below 5% → force
charge. Above 95% → force discharge. The LLM's reasoning influences the auction
for all intermediate states.

---

## Directory Structure

```
.
├── api/                  # FastAPI REST layer
│   ├── main.py           #   Server, lifespan, 8 endpoints
│   └── schemas.py        #   Pydantic request/response models
├── core/                 # Simulation engine
│   ├── agents/           #   Agent implementations (6 types)
│   │   ├── base.py       #     BaseAgent — LLM calling + Pydantic boundary
│   │   ├── battery.py    #     BatteryAgent — guardrails + get_validated_bid()
│   │   ├── solar.py      #     SolarFarm — compute_output from irradiance
│   │   ├── wind.py       #     WindFarm — compute_output from wind speed
│   │   ├── coal.py       #     CoalPlant — baseload + carbon cost
│   │   ├── nuclear.py    #     NuclearPlant — baseload must-run
│   │   ├── consumer.py   #     MetroCity — price-elastic demand
│   │   ├── regulatory.py #     RegulatoryAgent — frequency + carbon checks
│   │   └── orchestrator.py#    GridOrchestrator — bid validation + stabilization
│   ├── auction.py        #   Double-sided auction engine
│   ├── grid_physics.py   #   Weather engine + frequency model + demand
│   ├── llm_engine.py     #   ReasoningEngine + OllamaEngine + factory
│   ├── memory.py         #   AgentMemory — episodic storage
│   ├── schemas.py        #   BidStrategy Pydantic model (neural boundary)
│   └── simulation.py     #   GridSimulation — the symbolic outer loop
├── database/             # SQLAlchemy models (defined, not written at runtime)
│   └── models.py
├── scripts/              # Tooling
│   └── trace_neuro_symbolic.py  # 3-step architecture trace
├── tests/                # ~120 tests across 9 modules
│   ├── test_grid_physics.py
│   ├── test_auction.py
│   ├── test_agents.py
│   ├── test_api.py
│   ├── test_simulation.py
│   ├── test_orchestrator.py
│   ├── test_orchestrator_validation.py
│   ├── test_battery_guardrails.py
│   └── test_pydantic_boundary.py
├── api/main.py → uvicorn
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

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
| `test_grid_physics.py` | 17 | Weather, solar/wind physics, frequency, demand |
| `test_auction.py` | 14 | Matching, carbon cost, price clamping, surplus |
| `test_agents.py` | 9 | Agent output computation |
| `test_api.py` | 14 | Health, run, status, data endpoints |
| `test_simulation.py` | 12 | Full run, frequency, carbon, determinism |
| `test_orchestrator.py` | 8 | Frequency stabilization, reserve dispatch |
| `test_orchestrator_validation.py` | 15 | Bid validation rules |
| `test_battery_guardrails.py` | 12 | LLM output honored, guardrail boundaries |
| `test_pydantic_boundary.py` | ~18 | BidStrategy validation, missing fields, coercion |

---

## Tech Stack

| Layer | Technology | Role |
|:------|:-----------|:-----|
| **Language** | Python 3.12 | Runtime |
| **Web Framework** | FastAPI 0.111 + Uvicorn 0.30 | REST API server |
| **Data Validation** | Pydantic v2.7 | Request/response models + `BidStrategy` neural boundary |
| **ORM** | SQLAlchemy 2.0 | Database models (defined, not written at runtime) |
| **HTTP Client** | httpx 0.27 | Ollama API calls |
| **Math** | NumPy 1.26 | Grid physics, statistics |
| **Scripting** | Python scripts | `trace_neuro_symbolic.py` architecture trace |
| **Container** | Docker + docker compose | Deployment |
| **CI/CD** | GitHub Actions | 3-job matrix (test, lint, docker) |
| **Testing** | pytest 8.2 + pytest-cov 4.1 | Unit tests + coverage |
| **Linting** | black 24.4 + flake8 7.0 | Code formatting |
| **LLM (external)** | Ollama (tinyllama, qwen2.5) | Local neural inference |
| **Listed, unused** | SciPy 1.13, Jinja2 3.1 | Present in requirements.txt, not imported in code |

---

## Neuro-Symbolic Architecture Trace

```bash
python scripts/trace_neuro_symbolic.py
```

Prints a 3-step trace showing L1 weather → L3 battery LLM reasoning → auction
clearing → L6 regulatory oversight with data-flow diagram.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make changes and run tests: `pytest tests/ -v`
4. Format: `black . && flake8 core/ api/ tests/ --max-line-length=120`
5. Commit with a descriptive message
6. Push: `git push origin feat/your-feature`
7. Open a Pull Request against `main`

Please ensure all ~120 tests pass and both `black` and `flake8` are clean.

---

## License

MIT — see [LICENSE](LICENSE) for details.
