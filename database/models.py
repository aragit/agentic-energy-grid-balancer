from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

Base = declarative_base()

class Simulation(Base):
    __tablename__ = "simulations"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    name = Column(String, default="unnamed")
    total_steps = Column(Integer, default=0)
    status = Column(String, default="running")
    final_frequency = Column(Float, default=50.0)
    total_carbon_emitted = Column(Float, default=0.0)
    total_demand_mwh = Column(Float, default=0.0)
    total_supply_mwh = Column(Float, default=0.0)
    steps = relationship("GridStep", back_populates="simulation", cascade="all, delete")
    transactions = relationship("MarketTransaction", back_populates="simulation", cascade="all, delete")

class GridStep(Base):
    __tablename__ = "grid_steps"
    id = Column(Integer, primary_key=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id"))
    step_number = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    solar_irradiance = Column(Float, default=0.0)
    wind_speed = Column(Float, default=0.0)
    temperature = Column(Float, default=20.0)
    frequency = Column(Float, default=50.0)
    total_demand = Column(Float, default=0.0)
    total_supply = Column(Float, default=0.0)
    imbalance = Column(Float, default=0.0)
    solar_output = Column(Float, default=0.0)
    wind_output = Column(Float, default=0.0)
    coal_output = Column(Float, default=0.0)
    nuclear_output = Column(Float, default=0.0)
    battery_output = Column(Float, default=0.0)
    clearing_price = Column(Float, default=0.0)
    carbon_price = Column(Float, default=0.0)
    carbon_emitted_kg = Column(Float, default=0.0)
    simulation = relationship("Simulation", back_populates="steps")

class AgentRecord(Base):
    __tablename__ = "agents"
    id = Column(Integer, primary_key=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id"))
    agent_type = Column(String)
    agent_name = Column(String)
    initial_balance = Column(Float, default=0.0)
    current_balance = Column(Float, default=0.0)
    total_revenue = Column(Float, default=0.0)
    total_cost = Column(Float, default=0.0)
    capacity_mw = Column(Float, default=0.0)
    current_output = Column(Float, default=0.0)
    carbon_intensity_g_kwh = Column(Float, default=0.0)
    total_carbon_emitted = Column(Float, default=0.0)
    strategy_config = Column(JSON, default=dict)
    decision_history = Column(JSON, default=list)

class MarketTransaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id"))
    step_number = Column(Integer)
    buyer_name = Column(String)
    seller_name = Column(String)
    amount_mwh = Column(Float, default=0.0)
    price_per_mwh = Column(Float, default=0.0)
    total_cost = Column(Float, default=0.0)
    carbon_cost = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=datetime.utcnow)
    simulation = relationship("Simulation", back_populates="transactions")

class CarbonRecord(Base):
    __tablename__ = "carbon_records"
    id = Column(Integer, primary_key=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id"))
    step_number = Column(Integer)
    agent_name = Column(String)
    energy_mwh = Column(Float, default=0.0)
    carbon_kg = Column(Float, default=0.0)
    carbon_price = Column(Float, default=0.0)
    carbon_cost = Column(Float, default=0.0)

engine = create_engine("sqlite:///./energy_grid.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
