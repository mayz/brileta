"""
ActionPlan system for composable, timed action sequences.

This module defines the core data structures for the ActionPlan system, which
provides a unified way to define player actions as multi-step sequences.

Core Concepts:
- ActionPlan: A declarative definition of a player-selectable action with
  metadata and an ordered list of steps.
- Step: An abstract base for units within a plan. Concrete types are IntentStep
  (execute a single GameIntent) and ApproachStep (generate MoveIntents until
  reaching a destination).
- PlanContext: Captured state when a plan starts executing. Immutable snapshot
  of the action's parameters.
- ActivePlan: Runtime state for an executing plan. Lives on the Character and
  tracks current step index and cached path state.

The ActionPlan system replaces the ad-hoc PathfindingGoal.final_intent pattern
with a unified, declarative approach to multi-step actions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.actions.base import GameIntent
    from catley.game.actors import Actor, Character
    from catley.game.items.item_core import Item
    from catley.util.coordinates import WorldTilePos


@dataclass
class PlanContext:
    """Captured state when a plan starts executing.

    An immutable snapshot of the action's parameters, captured at plan creation
    time. This context is passed to step callbacks to provide information about
    the actor, target, and other relevant state.

    Attributes:
        actor: The character executing the plan.
        controller: The game controller for accessing game state.
        target_actor: The target actor for targeted actions (e.g., attack, push).
        target_position: The target position for position-based actions (e.g., move).
        weapon: The weapon being used, if any.
    """

    actor: Character
    controller: Controller
    target_actor: Character | Actor | None = None
    target_position: WorldTilePos | None = None
    weapon: Item | None = None


@dataclass
class Step:
    """Base class for steps within an ActionPlan.

    Steps are the building blocks of action plans. Each step represents a unit
    of work that can be executed as part of a larger action sequence.

    This is an abstract base - do not instantiate directly. Use subclasses:
        - IntentStep: Execute a single GameIntent (one turn).
        - ApproachStep: Generate MoveIntents until reaching a destination (N turns).
    """

    skip_if: Callable[[PlanContext], bool] | None = None
    """Optional predicate to skip this step if conditions are met."""


@dataclass
class IntentStep(Step):
    """Execute a single GameIntent as part of a plan.

    IntentStep wraps a GameIntent class and provides dynamic parameter generation
    through a callback. When the plan executor reaches this step, it instantiates
    the intent with parameters from the current context and executes it.

    Note: IntentStep has no presentation_ms field. The executor owns timing and
    returns it in GameActionResult.presentation_ms. This keeps timing logic with
    the action logic (e.g., critical hits might take longer).

    Attributes:
        intent_class: The GameIntent subclass to instantiate.
        params: A callable that takes PlanContext and returns a dict of parameters
            to pass to the intent constructor (in addition to controller).
    """

    intent_class: type[GameIntent] = field(default=None)  # type: ignore[assignment]
    """The GameIntent subclass to instantiate and execute."""

    params: Callable[[PlanContext], dict] = field(default=lambda ctx: {})
    """Callable returning intent constructor parameters from context."""


@dataclass
class ApproachStep(Step):
    """Generate MoveIntents until reaching a destination.

    ApproachStep handles pathfinding toward a target position or actor. It
    generates one MoveIntent per turn until the actor reaches the desired
    distance from the target.

    Unlike IntentStep, ApproachStep spans multiple turns. The ActivePlan tracks
    the cached path and advances it as moves succeed.

    Attributes:
        stop_distance: Distance at which to stop approaching.
            0 = at destination, 1 = adjacent to target.
    """

    stop_distance: int = 0
    """Distance at which to stop. 0 = at destination, 1 = adjacent."""


@dataclass
class ActionPlan:
    """A declarative definition of a player-selectable action.

    ActionPlan defines what steps comprise an action and metadata about how
    the action should be presented and validated. Plans are immutable templates
    that can be instantiated into ActivePlans for execution.

    Example:
        PunchPlan = ActionPlan(
            name="Punch",
            requires_target=True,
            requires_adjacency=True,
            steps=[
                ApproachStep(stop_distance=1),
                IntentStep(
                    intent_class=HolsterWeaponIntent,
                    params=lambda ctx: {"actor": ctx.actor},
                    skip_if=lambda ctx: ctx.actor.inventory.get_active_item() is None,
                ),
                IntentStep(
                    intent_class=PunchIntent,
                    params=lambda ctx: {
                        "attacker": ctx.actor,
                        "defender": ctx.target_actor,
                    },
                ),
            ],
        )

    Attributes:
        name: Human-readable name for the action.
        steps: Ordered list of steps to execute.
        requires_target: Whether this action requires a target actor.
        requires_adjacency: Whether the final step requires adjacency to target.
    """

    name: str
    steps: list[Step]
    requires_target: bool = False
    requires_adjacency: bool = False


@dataclass
class ActivePlan:
    """Runtime state for an executing ActionPlan.

    ActivePlan tracks the execution state of a plan in progress. It lives on
    the Character and maintains the current step index plus any cached state
    needed for multi-turn steps like ApproachStep.

    The TurnManager checks for an active plan and advances it each turn,
    generating the appropriate intent for the current step.

    Attributes:
        plan: The ActionPlan being executed.
        context: The PlanContext captured when the plan started.
        current_step_index: Index of the step currently being executed.
        cached_path: For ApproachStep, the current path to the target.
            Peek at [0] for next position. Pop only after move succeeds.
    """

    plan: ActionPlan
    context: PlanContext
    current_step_index: int = 0
    cached_path: list[WorldTilePos] | None = None

    def get_current_step(self) -> Step | None:
        """Return the current step, or None if the plan is complete.

        Returns:
            The Step at current_step_index, or None if all steps are done.
        """
        if self.current_step_index >= len(self.plan.steps):
            return None
        return self.plan.steps[self.current_step_index]

    def advance(self) -> None:
        """Advance to the next step in the plan.

        This should be called after an IntentStep executes successfully,
        or when an ApproachStep reaches its destination.
        """
        self.current_step_index += 1
        # Clear cached path when advancing - new step may need different path
        self.cached_path = None

    def is_complete(self) -> bool:
        """Return True if all steps have been executed.

        Returns:
            True if current_step_index >= len(plan.steps).
        """
        return self.current_step_index >= len(self.plan.steps)


# =============================================================================
# Pre-defined Action Plans
# =============================================================================

WalkToPlan = ActionPlan(
    name="Walk",
    steps=[ApproachStep(stop_distance=0)],
)


def get_punch_plan() -> ActionPlan:
    """Create PunchPlan lazily to avoid circular imports.

    PunchPlan needs to import intent classes from stunts.py, which can cause
    circular imports if done at module level. This factory function defers
    the import until the plan is actually needed.
    """
    from catley.game.actions.stunts import HolsterWeaponIntent, PunchIntent

    return ActionPlan(
        name="Punch",
        requires_target=True,
        requires_adjacency=True,
        steps=[
            ApproachStep(stop_distance=1),
            IntentStep(
                intent_class=HolsterWeaponIntent,
                params=lambda ctx: {"actor": ctx.actor},
                skip_if=lambda ctx: ctx.actor.inventory.get_active_item() is None,
            ),
            IntentStep(
                intent_class=PunchIntent,
                params=lambda ctx: {
                    "attacker": ctx.actor,
                    "defender": ctx.target_actor,
                },
            ),
        ],
    )
