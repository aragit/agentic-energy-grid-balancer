#!/usr/bin/env python3
"""Neuro-Symbolic Architecture Trace for Agentic Energy Grid Balancer.

Demonstrates the Type 2 (Symbolic[Neural]) architecture by stepping
through GridSimulation and printing L1 (Perception), L3 (LLM Reasoning),
Symbolic Auction, and L6 (Governance) layers.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm_engine import ReasoningEngine  # noqa: E402
from core.simulation import GridSimulation  # noqa: E402
from core.grid_physics import WeatherEngine  # noqa: E402


def trace():
    """Run a 3-step trace and print all architecture layers."""
    llm = ReasoningEngine(seed=42)
    sim = GridSimulation(llm=llm, steps=3)

    # Patch WeatherEngine to capture states for trace display
    captured_weather = []
    original_step = WeatherEngine.step

    def capturing_step(self):
        state = original_step(self)
        captured_weather.append(state)
        return state

    WeatherEngine.step = capturing_step

    BANNER = [
        "  ╔══════════════════════════════════════════════════════════════╗",
        "  ║       NEURO-SYMBOLIC ARCHITECTURE TRACE                     ║",
        "  ║    Type 2: Symbolic[Neural]  ·  Agentic Energy Grid        ║",
        "  ╚══════════════════════════════════════════════════════════════╝",
    ]
    for line in BANNER:
        print(line)
    print()

    for step in range(3):
        # Capture pre-step state
        pre_price = sim.auction.price_history[-1] if sim.auction.price_history else 50.0
        pre_freq = sim.physics.frequency
        pre_tx_count = len(sim.auction.transaction_history)

        # Run one full simulation step
        demand, actual_supply = sim._run_step(step)

        ws = captured_weather[-1]

        # ————————————————————————————————————————————
        #  STEP HEADER
        # ————————————————————————————————————————————
        sep = "─" * 76
        print(f"  ┌{sep}┐")
        print(f"  │  STEP {step}" + " " * 70 + "│")
        print(f"  └{sep}┘")
        print()

        # ————————————————————————————————————————————
        #  L1 — SYMBOLIC PERCEPTION
        # ————————————————————————————————————————————
        print("  ╭── L1 PERCEPTION  ·  Symbolic Environment Layer")
        print("  │")
        print(
            f"  │  Weather @ Hour {sim.weather.hour:02d}:00"
            f"  (Day {sim.weather.day_of_year})"
        )
        print(f"  │    Solar Irradiance  {ws.solar_irradiance:>8.1f}  W/m²")
        print(f"  │    Wind Speed        {ws.wind_speed:>8.1f}  m/s")
        print(f"  │    Temperature       {ws.temperature:>8.1f}  °C")
        print(f"  │    Cloud Cover       {ws.cloud_cover:>7.1%}")
        print(f"  │    Storm             {'⚠ YES' if ws.is_storm else 'No'}")
        print("  │")
        print("  │  Generator Outputs  (rule-based, weather-driven):")
        for gen in sim.generators:
            print(f"  │    {gen.name:<20s}  " f"{gen.state.current_output_mw:>7.1f} MW")
        print("  │")
        print(
            f"  │  Consumer Demand    {demand:>10.1f} MWh" f"  (price-sensitive curve)"
        )
        print("  │")
        print("  │  Grid Snapshot (before step):")
        print(f"  │    Last Clearing Price  ${pre_price:>6.2f} / MWh")
        print(f"  │    Grid Frequency       {pre_freq:>6.3f} Hz")
        print()

        # ————————————————————————————————————————————
        #  L3 — NEURAL SUBROUTINE
        # ————————————————————————————————————————————
        print("  ╭── L3 REASONING  ·  Neural Subroutine  (BatteryAgent)")
        print("  │")
        bat = sim.battery
        pct = 100 * bat.charge_level / max(bat.capacity_mwh, 0.01)
        print(f"  │  Agent    {bat.name}  ({bat.agent_type})")
        print(
            f"  │  Charge   {bat.charge_level:.1f} / {bat.capacity_mwh} MWh"
            f"  ({pct:.0f}%)"
        )
        print(f"  │  Memory   {len(bat.memory.experiences)} experiences stored")
        print(
            f"  │  Max Chg  {bat.max_charge_mw} MW  ·  "
            f"RTE  {bat.round_trip_efficiency * 100:.0f}%"
        )
        print("  │")
        print(
            "  │  LLM Input Context  (built dynamically from" " environment + memory):"
        )
        print(f"  │    Market Price    ${pre_price:>6.2f} / MWh")
        print(f"  │    Grid Demand     {demand:>10.1f} MWh")
        print(f"  │    Grid Frequency  {pre_freq:>6.3f} Hz")
        print("  │    Price Trend     " + sim.auction.get_price_trend())
        print("  │")
        if bat.state.strategy_history:
            s = bat.state.strategy_history[-1]
            print("  │  LLM Output  (decoded strategy JSON):")
            print(f"  │    ├─ Bid Price     ${s['bid_price']:>6.2f}")
            print(f"  │    ├─ Action        {s['output_adjustment']}")
            print(f"  │    ├─ Reasoning     {s['reasoning']}")
            print(f"  │    ├─ Confidence    {s['confidence']:.0%}")
            print(f"  │    └─ Latency       {s['latency_ms']:.1f} ms")
        print("  │")
        # Show physical guardrail that overrode LLM output
        charge_ratio = bat.charge_level / max(bat.capacity_mwh, 0.01)
        if charge_ratio > 0.85:
            guardrail = "Forced discharge  (SoC > 85%)"
        elif charge_ratio < 0.15:
            guardrail = "Forced charge  (SoC < 15%)"
        elif pre_price < 40 and charge_ratio < 0.8:
            guardrail = "Buy signal  (price < $40, capacity available)"
        elif pre_price > 55 and charge_ratio > 0.2:
            guardrail = "Sell signal  (price > $55, charge available)"
        else:
            guardrail = "Hold  (price within neutral band)"
        print(f"  │  ├─ Safety Guardrail  {guardrail}")
        print()

        # ————————————————————————————————————————————
        #  SYMBOLIC ENGINE — DoubleSidedAuction
        # ————————————————————————————————————————————
        print("  ╭── SYMBOLIC ENGINE  ·  DoubleSidedAuction" "  (Uniform Price)")
        print("  │")
        cp = sim.auction.price_history[-1] if sim.auction.price_history else 0.0
        print(f"  │  Market Clearing Price  ${cp:.2f} / MWh")
        print(f"  │  Total Supply Traded    {actual_supply:>8.2f} MWh")
        print(f"  │  Demand Met             {demand:>8.2f} MWh")
        imbalance = actual_supply - demand
        print(f"  │  Supply-Demand Δ        {imbalance:>+8.2f} MWh")
        print("  │")
        # Show transactions from this step only
        new_tx = sim.auction.transaction_history[pre_tx_count:]
        if new_tx:
            print("  │  ╭─ Matched Orders" "  ───────────────────────────────────────╮")
            for tx in new_tx:
                print(
                    f"  │  │ {tx['buyer']:<18s}"
                    f"  ←  {tx['seller']:<18s}"
                    f"  {tx['quantity_mwh']:>6.2f} MW"
                    f"  @ ${tx['price_per_mwh']:<6.2f}  │"
                )
            print(
                "  │  ╰──────────────────────────────────────" "────────────────────╯"
            )
        print("  │")
        print("  │  Clearing Math:" "  clearing_price = weighted_avg(matched_prices)")
        print("  │                 buyer_surplus" "  = Σ qty × (wtp − clearing_price)")
        print(
            "  │                 seller_surplus"
            "  = Σ qty × (clearing_price − marginal_cost)"
        )
        print()

        # ————————————————————————————————————————————
        #  L6 — GOVERNANCE
        # ————————————————————————————————————————————
        print("  ╭── L6 GOVERNANCE  ·  RegulatoryAgent Oversight")
        print("  │")
        freq = sim.physics.frequency
        freq_ok = 49.0 <= freq <= 51.0
        print("  │  ├── Frequency Guard")
        print("  │  │     Threshold  49.0 – 51.0 Hz")
        print(f"  │  │     Actual     {freq:.3f} Hz")
        print(f"  │  │     Status     {'✅  PASS' if freq_ok else '❌  CRITICAL'}")
        print("  │")
        print(f"  │  ├── Carbon Cap  {sim.regulator.carbon_cap:.1f} kg")
        for agent in sim.agents:
            emitted = agent.state.total_carbon_emitted
            ok = emitted <= sim.regulator.carbon_cap
            if emitted > 0 or agent.agent_type == "coal":
                print(
                    f"  │  │     {agent.name:<20s}  {emitted:>8.1f} kg"
                    f"  {'✅' if ok else '❌'}"
                )
        print("  │")
        step_violations = [v for v in sim.regulator.violations if v["step"] == step]
        if step_violations:
            print(f"  │  ├── ⚠  {len(step_violations)} violation(s)" f" this step:")
            for v in step_violations:
                print(
                    f"  │  │     [{v['severity']:^10s}]"
                    f"  {v['type'].upper()}"
                    f"  —  {v.get('agent', 'grid')}"
                )
        else:
            print("  │  ├── ✅  No violations this step")
        print()

    # ————————————————————————————————————————————
    #  FINAL SUMMARY — Data Flow Network Graph
    # ————————————————————————————————————————————
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║  DATA FLOW  ·  Type 2 (Symbolic[Neural]) Architecture      ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print()
    print("  ┌───────┐")
    print("  │L1     │  WeatherEngine  ──  GridPhysics  ──  DemandModel")
    print("  │SYMBOLIC│       │                 │                │")
    print("  │ENVIRON-│       ▼                 ▼                ▼")
    print("  │MENT   │  Solar  Wind  Coal  Nuclear  Consumer  Battery")
    print("  │       │  (rule-based, weather-driven)          │")
    print("  └───────┘                                        │")
    print("                                                    ▼")
    print("  ┌───────┐")
    print("  │L3     │  BatteryAgent._build_prompt()")
    print("  │NEURAL │       │")
    print("  │SUB-   │       ▼")
    print("  │ROUTINE│  ReasoningEngine.chat_completion()")
    print("  │       │       │")
    print("  │       │       ▼")
    print("  │       │  JSON strategy: {bid_price, action, reasoning}")
    print("  │       │       │")
    print("  │       │       ▼")
    print("  │       │  sim._run_step() guardrails + override")
    print("  └───────┘       │")
    print("                    ▼")
    print("  ┌───────┐")
    print("  │SYMBOLIC│  DoubleSidedAuction.clear_market()")
    print("  │ENGINE │       │")
    print("  │       │       ▼")
    print("  │       │  Sort buys  ↓price, sells ↑price")
    print("  │       │  Match at uniform clearing price")
    print("  │       │  Compute surplus, carbon cost")
    print("  └───────┘       │")
    print("                    ▼")
    print("  ┌───────┐")
    print("  │L6     │  RegulatoryAgent")
    print("  │GOVERN-│       │")
    print("  │ANCE   │       ▼")
    print("  │       │  check_frequency(49.0–51.0 Hz)")
    print("  │       │  check_carbon(cap threshold)")
    print("  │       │  check_market_fairness()")
    print("  │       │       │")
    print("  │       │       ▼")
    print("  │       │  Violations → GridOrchestrator")
    print("  │       │  Orchestrator.stabilize()")
    print("  │       │       │")
    print("  │       │       ▼")
    print("  │       │  Frequency correction → next step")
    print("  └───────┘")
    print()
    print("  Result Summary:")
    print(f"    Steps executed    : {sim.steps}")
    print("    Final frequency   :" f"  {sim.physics.frequency:.3f} Hz")
    total_violations = len(sim.regulator.violations)
    print(f"    Total violations  :  {total_violations}")
    print("    Agent balances    :")
    for agent in sim.agents:
        print(f"      {agent.name:<20s}" f"  ${agent.state.balance:>8.2f}")
    print(
        "    Price history     :"
        f"  {[f'${p:.1f}' for p in sim.auction.price_history]}"
    )


if __name__ == "__main__":
    trace()
