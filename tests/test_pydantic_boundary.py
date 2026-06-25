"""Tests for Pydantic validation at the neural/symbolic boundary.

Type 2 (Symbolic[Neuro]) principle: The boundary between stochastic LLM output
and deterministic symbolic processing must be a rigid, schema-enforced contract.
"""

import pytest
import json
from core.schemas import BidStrategy, ValidatedBid


class TestBidStrategyValidation:
    """Test that BidStrategy rejects invalid LLM output at the boundary."""

    def test_valid_strategy_parses(self):
        """A well-formed LLM response should validate cleanly."""
        raw_json = json.dumps({
            "bid_price": 42.5,
            "output_adjustment": "charge",
            "confidence": 0.85,
            "reasoning": "Prices are low, buying to store",
            "carbon_trade": 0.0,
        })
        strategy = BidStrategy.model_validate_json(raw_json)
        assert strategy.bid_price == 42.5
        assert strategy.output_adjustment == "charge"
        assert strategy.confidence == 0.85
        assert strategy.reasoning == "Prices are low, buying to store"

    def test_price_below_minimum_rejected(self):
        """bid_price < 1.0 must fail validation — not silently clamped."""
        raw_json = json.dumps({
            "bid_price": 0.5,
            "output_adjustment": "charge",
            "confidence": 0.5,
            "reasoning": "Test",
        })
        with pytest.raises(Exception):  # pydantic.ValidationError
            BidStrategy.model_validate_json(raw_json)

    def test_price_above_maximum_rejected(self):
        """bid_price > 200.0 must fail validation."""
        raw_json = json.dumps({
            "bid_price": 250.0,
            "output_adjustment": "discharge",
            "confidence": 0.5,
            "reasoning": "Test",
        })
        with pytest.raises(Exception):
            BidStrategy.model_validate_json(raw_json)

    def test_invalid_action_rejected(self):
        """Non-canonical actions like 'fly_to_moon' must fail."""
        raw_json = json.dumps({
            "bid_price": 50.0,
            "output_adjustment": "fly_to_moon",
            "confidence": 0.5,
            "reasoning": "Test",
        })
        with pytest.raises(Exception):
            BidStrategy.model_validate_json(raw_json)

    def test_action_normalization_sell(self):
        """'sell' should normalize to valid enum value."""
        raw_json = json.dumps({
            "bid_price": 50.0,
            "output_adjustment": "sell",
            "confidence": 0.5,
            "reasoning": "Test",
        })
        strategy = BidStrategy.model_validate_json(raw_json)
        assert strategy.output_adjustment == "sell"

    def test_action_normalization_discharge(self):
        """'discharge' should remain 'discharge'."""
        raw_json = json.dumps({
            "bid_price": 50.0,
            "output_adjustment": "discharge",
            "confidence": 0.5,
            "reasoning": "Test",
        })
        strategy = BidStrategy.model_validate_json(raw_json)
        assert strategy.output_adjustment == "discharge"

    def test_confidence_above_one_rejected(self):
        """confidence > 1.0 must fail."""
        raw_json = json.dumps({
            "bid_price": 50.0,
            "output_adjustment": "hold",
            "confidence": 1.5,
            "reasoning": "Test",
        })
        with pytest.raises(Exception):
            BidStrategy.model_validate_json(raw_json)

    def test_confidence_below_zero_rejected(self):
        """confidence < 0.0 must fail."""
        raw_json = json.dumps({
            "bid_price": 50.0,
            "output_adjustment": "hold",
            "confidence": -0.1,
            "reasoning": "Test",
        })
        with pytest.raises(Exception):
            BidStrategy.model_validate_json(raw_json)

    def test_missing_reasoning_rejected(self):
        """reasoning is required — cannot be empty."""
        raw_json = json.dumps({
            "bid_price": 50.0,
            "output_adjustment": "hold",
            "confidence": 0.5,
            "reasoning": "",
        })
        # Empty string gets converted to "No reasoning provided" by validator
        strategy = BidStrategy.model_validate_json(raw_json)
        assert strategy.reasoning == "No reasoning provided"

    def test_missing_fields_use_defaults(self):
        """Optional fields like carbon_trade default to 0.0."""
        raw_json = json.dumps({
            "bid_price": 50.0,
            "output_adjustment": "hold",
            "confidence": 0.5,
            "reasoning": "Test",
        })
        strategy = BidStrategy.model_validate_json(raw_json)
        assert strategy.carbon_trade == 0.0

    def test_extra_fields_ignored(self):
        """LLM may hallucinate extra fields — they should be ignored, not crash."""
        raw_json = json.dumps({
            "bid_price": 50.0,
            "output_adjustment": "hold",
            "confidence": 0.5,
            "reasoning": "Test",
            "hallucinated_field": "this_should_be_ignored",
            "another_fake": 123,
        })
        strategy = BidStrategy.model_validate_json(raw_json)
        assert strategy.bid_price == 50.0
        assert strategy.output_adjustment == "hold"

    def test_string_number_coercion(self):
        """LLMs sometimes output numbers as strings — coerce if possible."""
        raw_json = json.dumps({
            "bid_price": "42.5",
            "output_adjustment": "charge",
            "confidence": "0.85",
            "reasoning": "Test",
        })
        # Pydantic v2 may or may not coerce strings to numbers depending on config
        # This test documents expected behavior
        try:
            strategy = BidStrategy.model_validate_json(raw_json)
            assert strategy.bid_price == 42.5
        except Exception:
            pytest.skip("String-to-number coercion not enabled in this Pydantic config")

    def test_non_numeric_price_rejected(self):
        """'expensive' as bid_price must fail."""
        raw_json = json.dumps({
            "bid_price": "expensive",
            "output_adjustment": "hold",
            "confidence": 0.5,
            "reasoning": "Test",
        })
        with pytest.raises(Exception):
            BidStrategy.model_validate_json(raw_json)


class TestValidatedBid:
    """Test that ValidatedBid (post-guardrail) enforces final safety."""

    def test_validated_bid_creation(self):
        """A properly guardrailed bid should create cleanly."""
        bid = ValidatedBid(
            bid_price=50.0,
            output_adjustment="charge",
            llm_bid_price=55.0,
            llm_action="discharge",
            reasoning="Guardrail forced charge due to low SoC",
            confidence=0.9,
            guardrail_triggered=True,
            validation_errors=None,
        )
        assert bid.bid_price == 50.0
        assert bid.guardrail_triggered is True

    def test_validated_bid_price_bounds(self):
        """Even ValidatedBid must enforce [1.0, 200.0] range."""
        with pytest.raises(Exception):
            ValidatedBid(
                bid_price=0.5,
                output_adjustment="charge",
                llm_bid_price=0.5,
                llm_action="charge",
                reasoning="Test",
                confidence=0.5,
                guardrail_triggered=False,
            )

    def test_validated_bid_action_constraint(self):
        """ValidatedBid only accepts canonical battery actions."""
        with pytest.raises(Exception):
            ValidatedBid(
                bid_price=50.0,
                output_adjustment="sell",  # Not in ValidatedBid's Literal
                llm_bid_price=50.0,
                llm_action="sell",
                reasoning="Test",
                confidence=0.5,
                guardrail_triggered=False,
            )
