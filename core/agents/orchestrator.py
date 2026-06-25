import logging
from typing import List, Dict, Any
from core.grid_physics import GridPhysics

logger = logging.getLogger(__name__)


class GridOrchestratorAgent:
    def __init__(self, reserve_capacity_mw: float = 100.0):
        self.reserve_capacity = reserve_capacity_mw
        self.reserve_used = 0.0
        self.emergency_actions: List[Dict[str, Any]] = []

    def stabilize(
        self, frequency: float, imbalance: float, agents: List[Any]
    ) -> Dict[str, Any]:
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
