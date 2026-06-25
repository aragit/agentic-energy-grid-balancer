"""Tests for GridOrchestratorAgent bid validation — Critical Fix #3.

Type 2 (Symbolic[Neuro]) principle: The symbolic orchestrator must validate
ALL bids before auction clearing, acting as a centralized gatekeeper.
"""

import pytest
from core.agents.orchestrator import GridOrchestratorAgent, BidValidationResult
from core.auction import Bid


class TestOrchestratorBidValidation:
    @pytest.fixture
    def orchestrator(self):
        return GridOrchestratorAgent(reserve_capacity_mw=100.0)

    def test_valid_bid_accepted(self, orchestrator):
        """A normal, valid bid should pass validation."""
        bid = Bid(
            agent_name="SolarFarm-A",
            agent_type="solar",
            bid_price=50.0,
            quantity_mw=80.0,
            is_buy=False,
        )
        bids = [bid]
        validated = orchestrator.validate_bids(
            bids=bids,
            current_frequency=50.0,
            current_demand=500.0,
            carbon_cap_remaining=10000.0,
        )
        assert len(validated) == 1
        assert validated[0].agent_name == "SolarFarm-A"

    def test_price_too_high_rejected(self, orchestrator):
        """Bid price > 200.0 must be rejected."""
        bid = Bid(
            agent_name="CoalPlant-C",
            agent_type="coal",
            bid_price=250.0,
            quantity_mw=100.0,
            is_buy=False,
        )
        validated = orchestrator.validate_bids(
            bids=[bid],
            current_frequency=50.0,
            current_demand=500.0,
            carbon_cap_remaining=10000.0,
        )
        assert len(validated) == 0
        assert orchestrator.total_rejected_bids == 1

    def test_price_too_low_rejected(self, orchestrator):
        """Bid price < 1.0 must be rejected."""
        bid = Bid(
            agent_name="Battery-E",
            agent_type="battery",
            bid_price=0.5,
            quantity_mw=10.0,
            is_buy=True,
        )
        validated = orchestrator.validate_bids(
            bids=[bid],
            current_frequency=50.0,
            current_demand=500.0,
            carbon_cap_remaining=10000.0,
        )
        assert len(validated) == 0

    def test_negative_quantity_rejected(self, orchestrator):
        """Negative quantity bids must be rejected."""
        bid = Bid(
            agent_name="WindFarm-B",
            agent_type="wind",
            bid_price=30.0,
            quantity_mw=-10.0,
            is_buy=False,
        )
        validated = orchestrator.validate_bids(
            bids=[bid],
            current_frequency=50.0,
            current_demand=500.0,
            carbon_cap_remaining=10000.0,
        )
        assert len(validated) == 0

    def test_zero_quantity_rejected(self, orchestrator):
        """Zero quantity bids must be rejected."""
        bid = Bid(
            agent_name="NuclearPlant-D",
            agent_type="nuclear",
            bid_price=20.0,
            quantity_mw=0.0,
            is_buy=False,
        )
        validated = orchestrator.validate_bids(
            bids=[bid],
            current_frequency=50.0,
            current_demand=500.0,
            carbon_cap_remaining=10000.0,
        )
        assert len(validated) == 0

    def test_carbon_cap_not_checked_pre_clearing(self, orchestrator):
        """Carbon cap is checked POST-CLEARING by RegulatoryAgent, not pre-clearing.
        
        Pre-clearing carbon checks would be too restrictive (coal emits ~164t CO2/hr
        but cap is 50t total). Carbon is a regulatory concern, not a bid validity
        concern."""
        bid = Bid(
            agent_name="CoalPlant-C",
            agent_type="coal",
            bid_price=60.0,
            quantity_mw=200.0,  # Would emit 164,000 kg CO2
            is_buy=False,
        )
        validated = orchestrator.validate_bids(
            bids=[bid],
            current_frequency=50.0,
            current_demand=500.0,
            carbon_cap_remaining=100.0,  # Very small remaining cap
        )
        # Carbon cap is NOT checked pre-clearing — bid is accepted
        assert len(validated) == 1

    def test_mixed_valid_and_invalid_bids(self, orchestrator):
        """Some bids accepted, some rejected."""
        bids = [
            Bid("Solar-A", "solar", 40.0, 100.0, False),      # Valid
            Bid("Coal-B", "coal", 300.0, 50.0, False),         # Invalid: price too high
            Bid("Wind-C", "wind", 35.0, 80.0, False),           # Valid
            Bid("Bad-D", "coal", 50.0, -10.0, False),           # Invalid: negative qty
        ]
        validated = orchestrator.validate_bids(
            bids=bids,
            current_frequency=50.0,
            current_demand=500.0,
            carbon_cap_remaining=10000.0,
        )
        assert len(validated) == 2
        assert validated[0].agent_name == "Solar-A"
        assert validated[1].agent_name == "Wind-C"
        assert orchestrator.total_rejected_bids == 2

    def test_validation_log_populated(self, orchestrator):
        """Validation log should record all decisions."""
        bids = [
            Bid("Solar-A", "solar", 40.0, 100.0, False),
            Bid("Coal-B", "coal", 300.0, 50.0, False),
        ]
        orchestrator.validate_bids(
            bids=bids,
            current_frequency=50.0,
            current_demand=500.0,
            carbon_cap_remaining=10000.0,
        )
        summary = orchestrator.get_validation_summary()
        assert summary["total_bids_validated"] == 2
        assert summary["total_bids_rejected"] == 1
        assert summary["rejection_rate"] == 0.5

    def test_stabilize_still_works_post_clearing(self, orchestrator):
        """Post-clearing frequency stabilization should still function."""
        action = orchestrator.stabilize(
            frequency=48.0,  # Emergency — below 49.0
            imbalance=-100.0,
            agents=[],
        )
        assert action["action"] == "dispatch"
        assert action["severity"] == "EMERGENCY"

    def test_orchestrator_has_pre_and_post_methods(self, orchestrator):
        """Orchestrator must have both validate_bids (pre) and stabilize (post)."""
        assert hasattr(orchestrator, "validate_bids")
        assert hasattr(orchestrator, "stabilize")
        assert callable(orchestrator.validate_bids)
        assert callable(orchestrator.stabilize)
