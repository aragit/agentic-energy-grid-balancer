from core.agents.base import BaseAgent


class WindAgent(BaseAgent):
    def __init__(self, name: str, capacity_mw: float, llm):
        super().__init__(name, "wind", capacity_mw, carbon_intensity_g_kwh=0.0, llm=llm)

    def compute_output(self, weather, market_price: float) -> float:
        v = weather.wind_speed
        if v < 3 or v > 25:
            output = 0.0
        elif v < 12:
            output = self.state.capacity_mw * ((v - 3) / 9.0) ** 3
        else:
            output = self.state.capacity_mw
        self.record_output(output)
        return round(output, 2)
