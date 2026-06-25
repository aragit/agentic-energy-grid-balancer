"""Pydantic schemas for structured validation at the neural/symbolic boundary.

Type 2 (Symbolic[Neuro]) principle: The neural subroutine outputs stochastic
tokens; the symbolic layer compresses them into exact parametric arrays via
rigorous structural validation before any physical computation.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


class BidStrategy(BaseModel):
    """Structured output from neural subroutine, validated before symbolic use.

    This schema acts as the epistemic tether: it guarantees that whatever
    stochastic reasoning the LLM produced, the symbolic controller receives
    only well-typed, bounded, semantically valid parameters.
    """

    model_config = ConfigDict(extra="ignore")

    bid_price: float = Field(
        ...,
        ge=1.0,
        le=200.0,
        description="Bid price in USD per MWh, clamped to physically valid range",
    )

    output_adjustment: Literal[
        "charge",
        "discharge",
        "hold",
        "sell",
        "buy",
        "maintain",
        "ramp_up",
        "ramp_down",
        "reduce_demand",
    ] = Field(
        ...,
        description="Canonical action the agent proposes to take",
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in its own reasoning (0.0 = uncertain, 1.0 = certain)",
    )

    reasoning: str = Field(
        ...,
        min_length=1,
        description="Human-readable explanation for the proposed action",
    )

    carbon_trade: float = Field(
        default=0.0,
        ge=-1000.0,
        le=1000.0,
        description="Carbon offset trade quantity (kg), optional",
    )

    @field_validator("output_adjustment", mode="before")
    @classmethod
    def normalize_action(cls, v):
        """Normalize non-canonical LLM action strings to valid enum values."""
        if not isinstance(v, str):
            raise ValueError(f"output_adjustment must be string, got {type(v).__name__}")
        v_lower = v.lower().strip()
        # Map synonyms to canonical forms
        synonym_map = {
            "sell": "sell",
            "discharge": "discharge",
            "buy": "buy",
            "charge": "charge",
            "ramp_up": "ramp_up",
            "ramp_down": "ramp_down",
            "maintain": "maintain",
            "hold": "hold",
            "reduce_demand": "reduce_demand",
            "reduce": "reduce_demand",
            "ramp up": "ramp_up",
            "ramp down": "ramp_down",
        }
        if v_lower not in synonym_map:
            raise ValueError(
                f"Invalid output_adjustment '{v}'. Must be one of: charge, discharge, hold, "
                f"sell, buy, maintain, ramp_up, ramp_down, reduce_demand"
            )
        return synonym_map[v_lower]

    @field_validator("reasoning", mode="before")
    @classmethod
    def ensure_reasoning(cls, v):
        """Ensure reasoning is a non-empty string."""
        if not isinstance(v, str) or not v.strip():
            return "No reasoning provided"
        return v.strip()


class ValidatedBid(BaseModel):
    """Output from symbolic validation layer, ready for physical execution.

    This is the contract between the guardrail layer and the simulation loop:
    every field is guaranteed to have passed both schema validation AND
    physical safety checks.
    """

    model_config = ConfigDict(extra="ignore")

    bid_price: float = Field(..., ge=1.0, le=200.0)
    output_adjustment: Literal["charge", "discharge", "hold"]
    llm_bid_price: float
    llm_action: str
    reasoning: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    guardrail_triggered: bool
    validation_errors: Optional[list[str]] = Field(default=None)
