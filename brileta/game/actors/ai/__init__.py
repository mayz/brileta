"""
AI system for autonomous actor behavior.

This package contains the utility-based decision system that drives NPC
behavior. The system works by scoring candidate actions every tick and
executing the highest-scoring one.

Package structure:
    utility     - Scoring framework: UtilityAction, UtilityBrain, Consideration,
                  ResponseCurve.
    goals       - Goal framework: Goal ABC, GoalState, ContinueGoalAction.
    actions     - Simple stateless actions: Attack, Avoid, Watch, Idle.
    behaviors/  - Goal-backed behaviors: Wander, Flee, Patrol.
    perception  - PerceptionComponent: range + LOS awareness gating.
    component   - AIComponent that wires it all together.
"""

# Re-export the public API that was previously available from
# ``brileta.game.actors.ai`` when it was a single module.
from .component import (
    DISPOSITION_BANDS,
    HOSTILE_UPPER,
    AIComponent,
    FleeCandidate,
    disposition_label,
    disposition_to_normalized,
    escalate_hostility,
)
from .perception import PerceivedActor, PerceptionComponent

__all__ = [
    "DISPOSITION_BANDS",
    "HOSTILE_UPPER",
    "AIComponent",
    "FleeCandidate",
    "PerceivedActor",
    "PerceptionComponent",
    "disposition_label",
    "disposition_to_normalized",
    "escalate_hostility",
]
