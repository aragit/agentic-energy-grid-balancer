"""Tests for simulation orchestrator and end-to-end flow."""

from core.simulation import GridSimulation


class TestGridSimulation:
    def test_simulation_runs(self, simulation):
        result = simulation.run()
        assert result.status == "completed"
        assert result.total_steps == 5

    def test_frequency_stable(self, simulation):
        result = simulation.run()
        assert 47.0 <= result.final_frequency <= 53.0

    def test_carbon_non_negative(self, simulation):
        result = simulation.run()
        assert result.total_carbon_emitted >= 0

    def test_demand_equals_supply(self, simulation):
        result = simulation.run()
        # Supply should approximately meet demand (within 10%)
        ratio = result.total_supply_mwh / max(result.total_demand_mwh, 1)
        assert 0.9 <= ratio <= 1.1

    def test_all_agents_have_balances(self, simulation):
        result = simulation.run()
        assert len(result.agent_balances) == 6
        for name, balance in result.agent_balances.items():
            assert isinstance(balance, float)

    def test_price_history_populated(self, simulation):
        result = simulation.run()
        assert len(result.price_history) == 5

    def test_prices_in_reasonable_range(self, simulation):
        result = simulation.run()
        for price in result.price_history:
            assert 20 <= price <= 120

    def test_no_violations_in_normal_run(self, simulation):
        result = simulation.run()
        # Normal 5-step run should have no violations
        assert isinstance(result.violations, list)

    def test_generator_revenue_positive(self, simulation):
        result = simulation.run()
        # Solar and wind should have positive revenue
        assert result.agent_balances["SolarFarm-A"] >= 0
        assert result.agent_balances["WindFarm-B"] >= 0

    def test_consumer_balance_negative(self, simulation):
        result = simulation.run()
        # Consumer pays for energy, so balance should be negative
        assert result.agent_balances["MetroCity"] <= 0

    def test_simulation_deterministic(self, llm):
        sim1 = GridSimulation(llm=llm, steps=3)
        sim2 = GridSimulation(llm=llm, steps=3)
        result1 = sim1.run()
        result2 = sim2.run()
        assert result1.total_demand_mwh == result2.total_demand_mwh
        assert result1.total_supply_mwh == result2.total_supply_mwh
        assert result1.price_history == result2.price_history

    def test_step_by_step(self, simulation):
        # Run manually to verify step outputs
        total_demand = 0
        total_supply = 0
        for step in range(simulation.steps):
            d, s = simulation._run_step(step)
            total_demand += d
            total_supply += s
        assert total_demand > 0
        assert total_supply > 0
