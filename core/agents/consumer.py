from core.agents.base import BaseAgent
from core.grid_physics import DemandModel


class ConsumerAgent(BaseAgent):
    def __init__(self, name: str, base_demand_mw: float, llm):
        super().__init__(
            name, "consumer", base_demand_mw, carbon_intensity_g_kwh=0.0, llm=llm
        )
        self.demand_model = DemandModel(base_demand_mw=base_demand_mw)
        self.base_demand = base_demand_mw

    def compute_output(self, weather, market_price: float) -> float:
        demand = self.demand_model.compute(
            hour=0, temperature=weather.temperature, price=market_price
        )
        self.record_output(demand)
        return round(demand, 2)

    def set_hour(self, hour: int):
        self.demand_model.hour = hour
