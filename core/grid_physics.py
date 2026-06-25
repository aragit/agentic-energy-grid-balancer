"""Grid physics, weather simulation, and demand modeling."""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class WeatherState:
    solar_irradiance: float
    wind_speed: float
    temperature: float
    cloud_cover: float
    is_storm: bool


@dataclass
class GridState:
    step: int
    weather: WeatherState
    frequency: float
    total_demand: float
    total_supply: float
    imbalance: float
    generation: Dict[str, float] = field(default_factory=dict)
    prices: Dict[str, float] = field(default_factory=dict)


class WeatherEngine:
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.hour = 0
        self.day_of_year = 0

    def step(self) -> WeatherState:
        self.hour = (self.hour + 1) % 24
        if self.hour == 0:
            self.day_of_year = (self.day_of_year + 1) % 365

        seasonal = 1 + 0.5 * np.sin(2 * np.pi * (self.day_of_year - 80) / 365)
        daylight = max(0, np.sin(np.pi * self.hour / 24))
        base_irradiance = 1000 * seasonal * daylight

        cloud_noise = self.rng.normal(0, 0.1)
        cloud_cover = np.clip(0.3 + cloud_noise, 0, 1)

        is_storm = False
        if daylight > 0.1 and self.rng.random() < 0.05:
            is_storm = True
            cloud_cover = 0.9
        elif daylight <= 0.1 and self.rng.random() < 0.02:
            is_storm = True

        irradiance = base_irradiance * (1 - 0.7 * cloud_cover)
        irradiance = max(0, irradiance)

        base_wind = 5 + 3 * np.sin(2 * np.pi * self.day_of_year / 365)
        gust = self.rng.exponential(2) if self.rng.random() < 0.3 else 0
        wind_speed = base_wind + gust
        if is_storm:
            wind_speed += self.rng.uniform(10, 20)
        wind_speed = np.clip(wind_speed, 0, 25)

        temp_base = 15 + 10 * np.sin(2 * np.pi * (self.day_of_year - 80) / 365)
        temp_diurnal = (
            5 * np.sin(np.pi * (self.hour - 6) / 12) if 6 <= self.hour <= 18 else -3
        )
        temperature = temp_base + temp_diurnal + self.rng.normal(0, 2)

        return WeatherState(
            solar_irradiance=round(irradiance, 2),
            wind_speed=round(wind_speed, 2),
            temperature=round(temperature, 2),
            cloud_cover=round(cloud_cover, 2),
            is_storm=is_storm,
        )


class DemandModel:
    def __init__(self, base_demand_mw: float = 500.0):
        self.base_demand = base_demand_mw

    def compute(self, hour: int, temperature: float, price: float = 50.0) -> float:
        if 0 <= hour < 6:
            factor = 0.6
        elif 6 <= hour < 9:
            factor = 0.9
        elif 9 <= hour < 17:
            factor = 1.0
        elif 17 <= hour < 21:
            factor = 1.3
        else:
            factor = 0.8

        temp_effect = 1.0
        if temperature < 10:
            temp_effect = 1 + (10 - temperature) * 0.03
        elif temperature > 28:
            temp_effect = 1 + (temperature - 28) * 0.04

        price_effect = (50.0 / max(price, 1.0)) ** 0.2
        demand = self.base_demand * factor * temp_effect * price_effect
        return round(demand, 2)


class GridPhysics:
    def __init__(self, base_demand: float = 500.0):
        self.base_demand = base_demand
        self.frequency = 50.0
        self.inertia_constant = 5.0
        self.droop = 0.05

    def compute_frequency(self, imbalance_mw: float, dt_hours: float = 1.0) -> float:
        imbalance_pu = imbalance_mw / self.base_demand if self.base_demand > 0 else 0
        df = (imbalance_pu / self.inertia_constant) * dt_hours * 50.0
        new_freq = self.frequency + df
        new_freq = 50.0 + (new_freq - 50.0) * 0.9
        self.frequency = np.clip(new_freq, 47.0, 53.0)
        return round(self.frequency, 3)

    def is_stable(self, frequency: float) -> bool:
        return 49.5 <= frequency <= 50.5

    def severity(self, frequency: float) -> str:
        if 49.8 <= frequency <= 50.2:
            return "NORMAL"
        elif 49.5 <= frequency <= 50.5:
            return "WARNING"
        elif 49.0 <= frequency <= 51.0:
            return "ALERT"
        else:
            return "EMERGENCY"
