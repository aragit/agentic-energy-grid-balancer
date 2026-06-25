"""Grid orchestrator with pre-clearing bid validation and post-clearing frequency stabilization."""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from core.grid_physics import GridPhysics
from core.auction import Bid

logger = logging.getLogger(__name__)


@dataclass
class BidValidationResult:
    """Result of bid validation by the orchestrator."""
    is_valid: bool
    bid: Bid
    rejection_reason: Optional[str] = None


class GridOrchestratorAgent:
    """Symbolic gatekeeper for the energy grid.

    Type 2 (Symbolic[Neuro]) principle: The orchestrator is the deterministic
    primary controller. It validates ALL bids before market clearing, ensuring
    no neural or rule-based agent can submit physically dangerous parameters.
    """

    def __init__(self, reserve_capacity_mw: float = 100.0):
        self.reserve_capacity = reserve_capacity_mw
        self.reserve_used = 0.0
        self.emergency_actions: List[Dict[str, Any]] = []
        self.validation_log: List[Dict[str, Any]] = []
        self.total_rejected_bids = 0

    # =====================================================================
    # PRE-CLEARING BID VALIDATION (Type 2: Symbolic gatekeeper)
    # =====================================================================

    def validate_bids(
        self,
        bids: List[Bid],
        current_frequency: float,
        current_demand: float,
        carbon_cap_remaining: float,
    ) -> List[Bid]:
        """Validate all bids before auction clearing. Reject dangerous ones.

        This is the core Type 2 mechanism: the symbolic orchestrator acts as
        a gatekeeper, preventing invalid neural/rule-based output from ever
        reaching the physical grid simulation.

        Validation rules:
        1. Price bounds: All bids must be in [1.0, 200.0] USD/MWh
        2. Quantity bounds: Must be positive and non-negligible
        3. Economic sanity: Flag extreme bids for monitoring

        NOTE: Carbon cap is checked POST-CLEARING by RegulatoryAgent, not
        pre-clearing. Pre-clearing carbon checks would be too restrictive
        and prevent valid market operation.
        """
        validated_bids = []

        for bid in bids:
            result = self._validate_single_bid(bid)

            if result.is_valid:
                validated_bids.append(bid)
                logger.info(
                    f"[ORCHESTRATOR] ACCEPTED bid from {bid.agent_name}: "
                    f"${bid.bid_price:.2f}/MWh for {bid.quantity_mw:.1f} MW"
                )
            else:
                self.total_rejected_bids += 1
                logger.warning(
                    f"[ORCHESTRATOR] REJECTED bid from {bid.agent_name}: "
                    f"${bid.bid_price:.2f}/MWh for {bid.quantity_mw:.1f} MW — {result.rejection_reason}"
                )

            self.validation_log.append({
                "agent_name": bid.agent_name,
                "agent_type": bid.agent_type,
                "bid_price": bid.bid_price,
                "quantity_mw": bid.quantity_mw,
                "is_buy": bid.is_buy,
                "is_valid": result.is_valid,
                "rejection_reason": result.rejection_reason,
            })

        logger.info(
            f"[ORCHESTRATOR] Validation complete: {len(validated_bids)}/{len(bids)} bids accepted, "
            f"{len(bids) - len(validated_bids)} rejected"
        )
        return validated_bids

    def _validate_single_bid(
        self,
        bid: Bid,
    ) -> BidValidationResult:
        """Validate a single bid against physical and economic constraints."""

        # RULE 1: Price bounds — must be in economically valid range
        if not (1.0 <= bid.bid_price <= 200.0):
            return BidValidationResult(
                is_valid=False,
                bid=bid,
                rejection_reason=f"Price ${bid.bid_price:.2f} outside valid range [1.0, 200.0]",
            )

        # RULE 2: Quantity must be positive and non-negligible
        if bid.quantity_mw <= 0.01:
            return BidValidationResult(
                is_valid=False,
                bid=bid,
                rejection_reason=f"Quantity {bid.quantity_mw:.3f} MW too small or negative",
            )

        # RULE 3: Economic sanity — flag extreme bids for monitoring
        if bid.is_buy and bid.bid_price > 150.0:
            logger.warning(
                f"[ORCHESTRATOR] WARNING: High buy bid ${bid.bid_price:.2f} from {bid.agent_name} — "
                f"accepting but monitoring"
            )

        return BidValidationResult(
            is_valid=True,
            bid=bid,
        )

    # =====================================================================
    # POST-CLEARING FREQUENCY STABILIZATION (reactive)
    # =====================================================================

    def stabilize(
        self, frequency: float, imbalance: float, agents: List[Any]
    ) -> Dict[str, Any]:
        """Dispatch reserve capacity to stabilize grid frequency after clearing."""
        physics = GridPhysics()
        severity = physics.severity(frequency)

        if severity == "NORMAL":
            return {"action": "none", "dispatch": 0.0}

        dispatch_needed = -imbalance
        if severity == "EMERGENCY":
            dispatch = max(
                -self.reserve_capacity, min(self.reserve_capacity, dispatch_needed)
            )
            logger.warning(
                f"[ORCHESTRATOR] EMERGENCY: dispatching {dispatch:.2f} MW reserve"
            )
        elif severity == "ALERT":
            dispatch = dispatch_needed * 0.5
            dispatch = max(
                -self.reserve_capacity * 0.5, min(self.reserve_capacity * 0.5, dispatch)
            )
            logger.warning(
                f"[ORCHESTRATOR] ALERT: dispatching {dispatch:.2f} MW reserve"
            )
        else:
            dispatch = dispatch_needed * 0.2
            logger.info(
                f"[ORCHESTRATOR] WARNING: signaling {dispatch:.2f} MW correction"
            )

        self.reserve_used += abs(dispatch)
        action = {
            "action": "dispatch",
            "dispatch_mw": round(dispatch, 2),
            "severity": severity,
            "frequency_before": round(frequency, 3),
        }
        self.emergency_actions.append(action)
        return action

    def get_validation_summary(self) -> Dict[str, Any]:
        """Return summary of bid validation activity."""
        total_validated = len(self.validation_log)
        total_rejected = sum(1 for v in self.validation_log if not v["is_valid"])
        return {
            "total_bids_validated": total_validated,
            "total_bids_rejected": total_rejected,
            "rejection_rate": round(total_rejected / max(total_validated, 1), 3),
            "validation_log": self.validation_log[-10:],  # Last 10 entries
        }
