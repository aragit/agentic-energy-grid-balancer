"""Tests that LLM output is used as primary input and guardrails only clamp/reject
physically impossible actions — Critical Fix #1 for Type 2 (Symbolic[Neuro]) validity."""

import pytest
import sys
import os
import json

# Ensure core is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agents.battery import BatteryAgent
from core.llm_engine import LLMEngineFactory, LLMResponse


class TestBatteryGuardrails:
    @pytest.fixture
    def battery(self):
        llm = LLMEngineFactory.create(backend="mock")
        return BatteryAgent("TestBattery", capacity_mwh=50.0, max_charge_mw=25.0, llm=llm)

    def test_llm_charge_strategy_is_used_when_safe(self, battery):
        """When SoC is at 50% and LLM says charge, the LLM decision must be honored."""
        battery.charge_level = 25.0  # 50% — safe middle range

        def mock_decide_bid(*args, **kwargs):
            return {
                "bid_price": 30.0,
                "output_adjustment": "charge",
                "reasoning": "Buy low",
                "confidence": 0.9,
            }

        battery.decide_bid = mock_decide_bid

        result = battery.get_validated_bid(50.0, 100.0, 50.0, 25.0, {})

        assert result["output_adjustment"] == "charge"
        assert result["bid_price"] == 30.0
        assert result["guardrail_triggered"] is False
        assert result["llm_action"] == "charge"

    def test_llm_discharge_strategy_is_used_when_safe(self, battery):
        """When SoC is at 50% and LLM says discharge, the LLM decision must be honored."""
        battery.charge_level = 25.0  # 50%

        def mock_decide_bid(*args, **kwargs):
            return {
                "bid_price": 70.0,
                "output_adjustment": "discharge",
                "reasoning": "Sell high",
                "confidence": 0.9,
            }

        battery.decide_bid = mock_decide_bid

        result = battery.get_validated_bid(50.0, 100.0, 50.0, 25.0, {})

        assert result["output_adjustment"] == "discharge"
        assert result["bid_price"] == 70.0
        assert result["guardrail_triggered"] is False
        assert result["llm_action"] == "discharge"

    def test_llm_hold_strategy_is_used_when_safe(self, battery):
        """When SoC is at 50% and LLM says hold, it should hold."""
        battery.charge_level = 25.0

        def mock_decide_bid(*args, **kwargs):
            return {
                "bid_price": 50.0,
                "output_adjustment": "hold",
                "reasoning": "Neutral",
                "confidence": 0.8,
            }

        battery.decide_bid = mock_decide_bid

        result = battery.get_validated_bid(50.0, 100.0, 50.0, 25.0, {})

        assert result["output_adjustment"] == "hold"
        assert result["guardrail_triggered"] is False

    def test_guardrail_overrides_discharge_when_nearly_empty(self, battery):
        """When SoC is < 15%, LLM discharge must be overridden to hold."""
        battery.charge_level = 5.0  # 10% — below 15% warning threshold

        def mock_decide_bid(*args, **kwargs):
            return {
                "bid_price": 70.0,
                "output_adjustment": "discharge",
                "reasoning": "Sell high",
                "confidence": 0.9,
            }

        battery.decide_bid = mock_decide_bid

        result = battery.get_validated_bid(50.0, 100.0, 50.0, 25.0, {})

        assert result["output_adjustment"] == "hold"
        assert result["guardrail_triggered"] is True
        assert result["llm_action"] == "discharge"

    def test_guardrail_overrides_charge_when_nearly_full(self, battery):
        """When SoC is > 85%, LLM charge must be overridden to hold."""
        battery.charge_level = 45.0  # 90% — above 85% warning threshold

        def mock_decide_bid(*args, **kwargs):
            return {
                "bid_price": 30.0,
                "output_adjustment": "charge",
                "reasoning": "Buy low",
                "confidence": 0.9,
            }

        battery.decide_bid = mock_decide_bid

        result = battery.get_validated_bid(50.0, 100.0, 50.0, 25.0, {})

        assert result["output_adjustment"] == "hold"
        assert result["guardrail_triggered"] is True
        assert result["llm_action"] == "charge"

    def test_guardrail_forces_charge_when_critically_empty(self, battery):
        """When SoC is < 5%, ANY LLM action must be overridden to charge."""
        battery.charge_level = 1.5  # 3% — critically low

        def mock_decide_bid(*args, **kwargs):
            return {
                "bid_price": 70.0,
                "output_adjustment": "discharge",
                "reasoning": "Sell high",
                "confidence": 0.9,
            }

        battery.decide_bid = mock_decide_bid

        result = battery.get_validated_bid(50.0, 100.0, 50.0, 25.0, {})

        assert result["output_adjustment"] == "charge"
        assert result["guardrail_triggered"] is True

    def test_guardrail_forces_discharge_when_critically_full(self, battery):
        """When SoC is > 95%, ANY LLM action must be overridden to discharge."""
        battery.charge_level = 48.5  # 97% — critically high

        def mock_decide_bid(*args, **kwargs):
            return {
                "bid_price": 30.0,
                "output_adjustment": "charge",
                "reasoning": "Buy low",
                "confidence": 0.9,
            }

        battery.decide_bid = mock_decide_bid

        result = battery.get_validated_bid(50.0, 100.0, 50.0, 25.0, {})

        assert result["output_adjustment"] == "discharge"
        assert result["guardrail_triggered"] is True

    def test_pydantic_rejects_invalid_price_at_boundary(self, battery):
        """Pydantic rejects prices outside [1.0, 200.0] at the neural/symbolic boundary.
        The decide_bid() method falls back to safe defaults, so get_validated_bid()
        never sees invalid prices."""
        battery.charge_level = 25.0

        # Mock the LLM to return invalid JSON that will fail Pydantic validation
        class FakeLLM:
            def chat_completion(self, messages, temperature=0.7, max_tokens=512):
                return LLMResponse(
                    content='{"bid_price": 999.0, "output_adjustment": "hold", "confidence": 0.5, "reasoning": "test"}',
                    tokens_in=10,
                    tokens_out=10,
                    latency_ms=1.0,
                    model="fake"
                )
            def shutdown(self):
                pass

        battery.llm = FakeLLM()

        # decide_bid() should reject 999.0 and fall back
        decision = battery.decide_bid(50.0, 100.0, 50.0, 25.0, {})
        
        # The fallback strategy should have a valid price (market_price * 0.95, clamped)
        assert decision["bid_price"] <= 200.0
        assert decision["bid_price"] >= 1.0
        assert decision["is_fallback"] is True
        assert decision["validation_errors"] is not None

    def test_pydantic_rejects_invalid_price_low_at_boundary(self, battery):
        """Pydantic rejects negative prices at the boundary."""
        battery.charge_level = 25.0

        class FakeLLM:
            def chat_completion(self, messages, temperature=0.7, max_tokens=512):
                return LLMResponse(
                    content='{"bid_price": -50.0, "output_adjustment": "hold", "confidence": 0.5, "reasoning": "test"}',
                    tokens_in=10,
                    tokens_out=10,
                    latency_ms=1.0,
                    model="fake"
                )
            def shutdown(self):
                pass

        battery.llm = FakeLLM()

        decision = battery.decide_bid(50.0, 100.0, 50.0, 25.0, {})
        
        # Fallback should produce valid price
        assert decision["bid_price"] <= 200.0
        assert decision["bid_price"] >= 1.0
        assert decision["is_fallback"] is True

    def test_price_clamping_does_not_override_valid_prices(self, battery):
        """Valid prices within [1, 200] must pass through unchanged."""
        battery.charge_level = 25.0

        def mock_decide_bid(*args, **kwargs):
            return {
                "bid_price": 42.5,
                "output_adjustment": "charge",
                "reasoning": "Test",
                "confidence": 0.5,
            }

        battery.decide_bid = mock_decide_bid

        result = battery.get_validated_bid(50.0, 100.0, 50.0, 25.0, {})

        assert result["bid_price"] == 42.5
        assert result["guardrail_triggered"] is False

    def test_action_vocabulary_normalization(self, battery):
        """LLM actions like 'sell', 'buy', 'maintain' must be normalized to canonical forms."""
        battery.charge_level = 25.0

        def mock_decide_bid(*args, **kwargs):
            return {
                "bid_price": 50.0,
                "output_adjustment": "sell",
                "reasoning": "Test",
                "confidence": 0.5,
            }

        battery.decide_bid = mock_decide_bid

        result = battery.get_validated_bid(50.0, 100.0, 50.0, 25.0, {})

        assert result["llm_action"] == "discharge"
        assert result["output_adjustment"] == "discharge"
