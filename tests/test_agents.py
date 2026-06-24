"""Tests for agent behavior, state, and memory."""

import pytest
from core.agents.base import AgentState
from core.memory import Experience


class TestSolarAgent:
    def test_compute_output_day(self, solar_agent, weather):
        weather.hour = 12
        weather.day_of_year = 172  # Summer solstice
        state = weather.step()
        output = solar_agent.compute_output(state, 50.0)
        assert output > 0
        assert output <= solar_agent.state.capacity_mw

    def test_compute_output_night(self, solar_agent, weather):
        weather.hour = 0
        weather.day_of_year = 172
        state = weather.step()
        output = solar_agent.compute_output(state, 50.0)
        assert output <= 20  # Low at night, summer twilight can be ~15

    def test_carbon_intensity(self, solar_agent):
        assert solar_agent.state.carbon_intensity_g_kwh == 0

    def test_state_initialization(self, solar_agent):
        assert solar_agent.state.balance == 0.0
        assert solar_agent.state.total_revenue == 0.0
        assert solar_agent.state.is_active is True


class TestWindAgent:
    def test_compute_output(self, wind_agent, weather):
        state = weather.step()
        output = wind_agent.compute_output(state, 50.0)
        assert output >= 0
        assert output <= wind_agent.state.capacity_mw

    def test_high_wind_output(self, wind_agent, weather):
        class MockRNG:
            def normal(self, *a): return 0
            def random(self): return 0.5
            def exponential(self, *a): return 15
            def uniform(self, *a): return 0
        weather.rng = MockRNG()
        state = weather.step()
        output = wind_agent.compute_output(state, 50.0)
        assert output > 0


class TestCoalAgent:
    def test_compute_output(self, coal_agent, weather):
        state = weather.step()
        output = coal_agent.compute_output(state, 50.0)
        assert output > 0
        assert output <= coal_agent.state.capacity_mw