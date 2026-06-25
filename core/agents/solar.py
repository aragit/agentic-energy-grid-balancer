from core.agents.base import BaseAgent


class SolarAgent(BaseAgent):
    def __init__(self, name: str, capacity_mw: float, llm):
        super().__init__(
            name, "solar", capacity_mw, carbon_intensity_g_kwh=0.0, llm=llm
        )

    def compute_output(self, weather, market_price: float) -> float:
        irradiance = max(0, weather.solar_irradiance)
        output = self.state.capacity_mw * (irradiance / 1000.0)
        output = min(output, self.state.capacity_mw)
        self.record_output(output)
        return round(output, 2)
