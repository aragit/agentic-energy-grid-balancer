"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "llm_mode" in data

    def test_health_method_not_allowed(self, client):
        response = client.post("/health")
        assert response.status_code == 405


class TestSimulationEndpoints:
    def test_run_simulation(self, client):
        response = client.post(
            "/simulation/run",
            json={"steps": 5, "llm_backend": "mock"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["total_steps"] == 5
        assert "agent_balances" in data
        assert "price_history" in data

    def test_run_simulation_custom_steps(self, client):
        response = client.post(
            "/simulation/run",
            json={"steps": 3, "llm_backend": "mock"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_steps"] == 3

    def test_simulation_status_after_run(self, client):
        client.post("/simulation/run", json={"steps": 5, "llm_backend": "mock"})
        response = client.get("/simulation/status")
        assert response.status_code == 200
        data = response.json()
        assert "steps_completed" in data
        assert "current_frequency" in data

    def test_simulation_status_no_run(self, client):
        response = client.get("/simulation/status")
        # API returns 200 with default data, not 404
        assert response.status_code == 200
        data = response.json()
        assert "steps_completed" in data

    def test_agents_performance(self, client):
        client.post("/simulation/run", json={"steps": 5, "llm_backend": "mock"})
        response = client.get("/agents/performance")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 6
        for agent in data:
            assert "agent_name" in agent
            assert "balance" in agent
            assert "strategy_count" in agent

    def test_agents_performance_no_run(self, client):
        response = client.get("/agents/performance")
        # API returns 200 with default data, not 404
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_market_history(self, client):
        client.post("/simulation/run", json={"steps": 5, "llm_backend": "mock"})
        response = client.get("/market/history")
        assert response.status_code == 200
        data = response.json()
        assert "transactions" in data
        assert "price_trend" in data

    def test_carbon_report(self, client):
        client.post("/simulation/run", json={"steps": 5, "llm_backend": "mock"})
        response = client.get("/carbon/report")
        assert response.status_code == 200
        data = response.json()
        assert "total_carbon_kg" in data
        assert "agent_breakdown" in data
        assert len(data["agent_breakdown"]) == 6

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "Agentic Energy Grid Balancer" in data["message"]


class TestSimulationValidation:
    def test_run_simulation_negative_steps(self, client):
        response = client.post(
            "/simulation/run",
            json={"steps": -1, "llm_backend": "mock"}
        )
        # Should either reject or handle gracefully
        assert response.status_code in [200, 422]

    def test_run_simulation_zero_steps(self, client):
        response = client.post(
            "/simulation/run",
            json={"steps": 0, "llm_backend": "mock"}
        )
        assert response.status_code in [200, 422]