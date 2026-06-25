"""Tests for grid orchestrator and regulatory agents."""


class TestGridOrchestrator:
    def test_normal_no_action(self, orchestrator):
        action = orchestrator.stabilize(50.0, 0.0, [])
        assert action["action"] == "none"
        assert action["dispatch"] == 0.0

    def test_warning_small_dispatch(self, orchestrator):
        action = orchestrator.stabilize(49.6, -50.0, [])
        assert action["action"] == "dispatch"
        assert action["dispatch_mw"] != 0.0

    def test_emergency_large_dispatch(self, orchestrator):
        action = orchestrator.stabilize(48.0, -200.0, [])
        assert action["action"] == "dispatch"
        assert abs(action["dispatch_mw"]) <= 100.0  # Reserve limit

    def test_reserve_tracked(self, orchestrator):
        initial = orchestrator.reserve_used
        orchestrator.stabilize(48.0, -50.0, [])
        assert orchestrator.reserve_used > initial

    def test_emergency_actions_recorded(self, orchestrator):
        orchestrator.stabilize(48.0, -50.0, [])
        assert len(orchestrator.emergency_actions) == 1

    def test_dispatch_clamped(self, orchestrator):
        action = orchestrator.stabilize(48.0, -500.0, [])
        assert abs(action["dispatch_mw"]) <= 100.0


class TestRegulatoryAgent:
    def test_frequency_check_normal(self, regulator):
        regulator.check_frequency(50.0, 1)
        assert len(regulator.violations) == 0

    def test_frequency_check_violation(self, regulator):
        regulator.check_frequency(48.0, 1)
        assert len(regulator.violations) > 0

    def test_carbon_check_within_cap(self, regulator, solar_agent):
        solar_agent.state.total_carbon_emitted = 100.0
        regulator.check_carbon(solar_agent, 1)
        assert len(regulator.violations) == 0

    def test_carbon_check_exceeds_cap(self, regulator, coal_agent):
        coal_agent.state.total_carbon_emitted = 60000.0
        regulator.check_carbon(coal_agent, 1)
        assert len(regulator.violations) > 0

    def test_violation_details(self, regulator):
        regulator.check_frequency(47.5, 5)
        violation = regulator.violations[0]
        assert "step" in violation
        assert "type" in violation
        assert "value" in violation
