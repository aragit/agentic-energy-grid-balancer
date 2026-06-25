"""Tests for grid physics, weather, and demand models."""

from core.grid_physics import WeatherEngine, DemandModel


class TestWeatherEngine:
    def test_weather_step_returns_state(self, weather):
        state = weather.step()
        assert state.solar_irradiance >= 0
        assert 0 <= state.wind_speed <= 25
        assert -10 <= state.temperature <= 40
        assert 0 <= state.cloud_cover <= 1

    def test_weather_hour_cycles(self, weather):
        initial_hour = weather.hour
        for _ in range(24):
            weather.step()
        assert weather.hour == initial_hour

    def test_storm_probability(self, weather):
        storms = 0
        for _ in range(200):
            state = weather.step()
            if state.is_storm:
                storms += 1
        assert 0 <= storms <= 50  # ~5% daylight + 2% night

    def test_solar_irradiance_day_night(self, weather):
        # Step until hour 0 (midnight)
        while weather.hour != 0:
            weather.step()
        night_state = weather.step()
        # At night irradiance is very low, not necessarily exactly 0
        assert night_state.solar_irradiance <= 100

    def test_weather_deterministic_with_seed(self):
        w1 = WeatherEngine(seed=42)
        w2 = WeatherEngine(seed=42)
        for _ in range(10):
            s1 = w1.step()
            s2 = w2.step()
            assert s1.solar_irradiance == s2.solar_irradiance
            assert s1.wind_speed == s2.wind_speed


class TestGridPhysics:
    def test_frequency_stable_when_balanced(self, physics):
        freq = physics.compute_frequency(0.0)
        assert 49.9 <= freq <= 50.1

    def test_frequency_rises_with_excess_supply(self, physics):
        freq = physics.compute_frequency(100.0)
        assert freq > 50.0

    def test_frequency_falls_with_shortage(self, physics):
        freq = physics.compute_frequency(-100.0)
        assert freq < 50.0

    def test_frequency_clamped(self, physics):
        freq_high = physics.compute_frequency(1000.0)
        freq_low = physics.compute_frequency(-1000.0)
        assert freq_high <= 53.0
        assert freq_low >= 47.0

    def test_is_stable_normal(self, physics):
        assert physics.is_stable(50.0) is True
        assert physics.is_stable(49.6) is True
        assert physics.is_stable(49.4) is False

    def test_severity_levels(self, physics):
        assert physics.severity(50.0) == "NORMAL"
        assert physics.severity(49.6) == "WARNING"
        assert physics.severity(49.3) == "ALERT"
        assert physics.severity(48.5) == "EMERGENCY"

    def test_frequency_damping(self, physics):
        # Large imbalance should not cause immediate 53Hz
        physics.compute_frequency(500.0)
        f2 = physics.compute_frequency(500.0)
        assert f2 <= 53.0  # Damping prevents exceeding 53


class TestDemandModel:
    def test_base_demand(self):
        model = DemandModel(base_demand_mw=500.0)
        demand = model.compute(hour=12, temperature=20.0, price=50.0)
        assert 400 <= demand <= 700

    def test_peak_demand_evening(self):
        model = DemandModel(base_demand_mw=500.0)
        evening = model.compute(hour=19, temperature=20.0, price=50.0)
        noon = model.compute(hour=12, temperature=20.0, price=50.0)
        assert evening > noon

    def test_cold_weather_boost(self):
        model = DemandModel(base_demand_mw=500.0)
        cold = model.compute(hour=12, temperature=5.0, price=50.0)
        mild = model.compute(hour=12, temperature=20.0, price=50.0)
        assert cold > mild

    def test_high_price_reduces_demand(self):
        model = DemandModel(base_demand_mw=500.0)
        cheap = model.compute(hour=12, temperature=20.0, price=30.0)
        expensive = model.compute(hour=12, temperature=20.0, price=100.0)
        assert cheap > expensive

    def test_night_low_demand(self):
        model = DemandModel(base_demand_mw=500.0)
        night = model.compute(hour=2, temperature=20.0, price=50.0)
        day = model.compute(hour=12, temperature=20.0, price=50.0)
        assert night < day
