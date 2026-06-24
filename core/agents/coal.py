from core.agents.base import BaseAgent

class CoalAgent(BaseAgent):
    def __init__(self, name: str, capacity_mw: float, llm):
        super().__init__(name, "coal", capacity_mw, carbon_intensity_g_kwh=820.0, llm=llm)

    def compute_output(self, weather, market_price: float) -> float:
        target = self.state.capacity_mw * 0.85
        current = self.state.current_output_mw or target
        max_ramp = self.state.capacity_mw * 0.2
        if target > current + max_ramp:
            output = current + max_ramp
        elif target < current - max_ramp:
            output = current - max_ramp
        else:
            output = target
        output = min(output, self.state.capacity_mw)
        self.record_output(output)
        return round(output, 2)
