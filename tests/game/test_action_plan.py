"""Tests for the ActionPlan system data structures.

These tests validate that the ActionPlan infrastructure can be instantiated
and behaves correctly. This is the foundation for composable, timed action
sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from brileta import colors
from brileta.controller import Controller
from brileta.game.action_plan import (
    ActionPlan,
    ActivePlan,
    ApproachStep,
    IntentStep,
    PlanContext,
)
from brileta.game.actions.base import GameIntent
from brileta.game.actors import Character
from brileta.game.game_world import GameWorld
from tests.helpers import DummyGameWorld


@dataclass
class DummyController:
    """Minimal controller for testing."""

    gw: DummyGameWorld
    frame_manager: object | None = None


class DummyIntent(GameIntent):
    """A minimal intent for testing IntentStep."""

    def __init__(self, controller: Controller, actor, custom_value: str = "default"):
        super().__init__(controller, actor)
        self.custom_value = custom_value


def make_test_context() -> tuple[DummyController, Character, PlanContext]:
    """Create a test context with a player and controller."""
    gw = DummyGameWorld()
    player = Character(
        5,
        5,
        "@",
        colors.WHITE,
        "Player",
        game_world=cast(GameWorld, gw),
    )
    gw.player = player
    gw.add_actor(player)
    controller = DummyController(gw=gw, frame_manager=object())

    context = PlanContext(
        actor=player,
        controller=cast(Controller, controller),
    )

    return controller, player, context


# -----------------------------------------------------------------------------
# PlanContext tests
# -----------------------------------------------------------------------------


def test_plan_context_basic_instantiation() -> None:
    """PlanContext can be instantiated with required fields."""
    _, player, context = make_test_context()

    assert context.actor is player
    assert context.controller is not None
    assert context.target_actor is None
    assert context.target_position is None
    assert context.weapon is None


def test_plan_context_with_target_actor() -> None:
    """PlanContext can hold a target actor."""
    controller, player, _ = make_test_context()

    target = Character(
        7, 5, "r", colors.RED, "Raider", game_world=cast(GameWorld, controller.gw)
    )
    controller.gw.add_actor(target)

    context = PlanContext(
        actor=player,
        controller=cast(Controller, controller),
        target_actor=target,
    )

    assert context.target_actor is target


def test_plan_context_with_target_position() -> None:
    """PlanContext can hold a target position."""
    _, player, _ = make_test_context()
    controller, _, _ = make_test_context()

    context = PlanContext(
        actor=player,
        controller=cast(Controller, controller),
        target_position=(10, 15),
    )

    assert context.target_position == (10, 15)


# -----------------------------------------------------------------------------
# Step tests
# -----------------------------------------------------------------------------


def test_intent_step_basic_instantiation() -> None:
    """IntentStep can be instantiated with an intent class."""
    step = IntentStep(
        intent_class=DummyIntent,
        params=lambda ctx: {"custom_value": "test_value"},
    )

    assert step.intent_class is DummyIntent
    assert step.skip_if is None


def test_intent_step_with_skip_condition() -> None:
    """IntentStep can have a skip_if predicate."""
    step = IntentStep(
        intent_class=DummyIntent,
        params=lambda ctx: {},
        skip_if=lambda ctx: True,
    )

    assert step.skip_if is not None
    # Verify the skip_if callable works
    _, _, context = make_test_context()
    assert step.skip_if(context) is True


def test_intent_step_params_callable() -> None:
    """IntentStep params callable returns correct parameters from context."""
    _, _, context = make_test_context()

    step = IntentStep(
        intent_class=DummyIntent,
        params=lambda ctx: {"custom_value": f"actor_{ctx.actor.name}"},
    )

    params = step.params(context)
    assert params == {"custom_value": "actor_Player"}


def test_approach_step_basic_instantiation() -> None:
    """ApproachStep can be instantiated with default stop_distance."""
    step = ApproachStep()

    assert step.stop_distance == 0
    assert step.skip_if is None


def test_approach_step_with_adjacency() -> None:
    """ApproachStep can be configured to stop adjacent to target."""
    step = ApproachStep(stop_distance=1)

    assert step.stop_distance == 1


def test_approach_step_with_skip_condition() -> None:
    """ApproachStep can have a skip_if predicate."""
    step = ApproachStep(
        stop_distance=1,
        skip_if=lambda ctx: ctx.target_position is None,
    )

    _, _, context = make_test_context()
    assert step.skip_if is not None
    assert step.skip_if(context) is True  # No target_position set


# -----------------------------------------------------------------------------
# ActionPlan tests
# -----------------------------------------------------------------------------


def test_action_plan_basic_instantiation() -> None:
    """ActionPlan can be instantiated with name and steps."""
    plan = ActionPlan(
        name="Walk",
        steps=[ApproachStep(stop_distance=0)],
    )

    assert plan.name == "Walk"
    assert len(plan.steps) == 1
    assert plan.requires_target is False
    assert plan.requires_adjacency is False


def test_action_plan_with_requirements() -> None:
    """ActionPlan can specify target and adjacency requirements."""
    plan = ActionPlan(
        name="Punch",
        steps=[
            ApproachStep(stop_distance=1),
            IntentStep(intent_class=DummyIntent, params=lambda ctx: {}),
        ],
        requires_target=True,
        requires_adjacency=True,
    )

    assert plan.requires_target is True
    assert plan.requires_adjacency is True
    assert len(plan.steps) == 2


def test_action_plan_multi_step() -> None:
    """ActionPlan can contain multiple steps of different types."""
    plan = ActionPlan(
        name="Complex Action",
        steps=[
            ApproachStep(stop_distance=1),
            IntentStep(
                intent_class=DummyIntent,
                params=lambda ctx: {"custom_value": "step1"},
                skip_if=lambda ctx: False,
            ),
            IntentStep(
                intent_class=DummyIntent,
                params=lambda ctx: {"custom_value": "step2"},
            ),
        ],
        requires_target=True,
    )

    assert len(plan.steps) == 3
    assert isinstance(plan.steps[0], ApproachStep)
    assert isinstance(plan.steps[1], IntentStep)
    assert isinstance(plan.steps[2], IntentStep)


# -----------------------------------------------------------------------------
# ActivePlan tests
# -----------------------------------------------------------------------------


def test_active_plan_basic_instantiation() -> None:
    """ActivePlan can be instantiated with a plan and context."""
    _, _, context = make_test_context()

    plan = ActionPlan(name="Test", steps=[ApproachStep()])
    active = ActivePlan(plan=plan, context=context)

    assert active.plan is plan
    assert active.context is context
    assert active.current_step_index == 0
    assert active.cached_path is None


def test_active_plan_get_current_step() -> None:
    """ActivePlan.get_current_step returns the correct step."""
    _, _, context = make_test_context()

    step1 = ApproachStep(stop_distance=1)
    step2 = IntentStep(intent_class=DummyIntent, params=lambda ctx: {})
    plan = ActionPlan(name="Test", steps=[step1, step2])
    active = ActivePlan(plan=plan, context=context)

    assert active.get_current_step() is step1


def test_active_plan_advance() -> None:
    """ActivePlan.advance increments the step index and clears cached path."""
    _, _, context = make_test_context()

    step1 = ApproachStep()
    step2 = IntentStep(intent_class=DummyIntent, params=lambda ctx: {})
    plan = ActionPlan(name="Test", steps=[step1, step2])
    active = ActivePlan(plan=plan, context=context)
    active.cached_path = [(1, 1), (2, 2)]

    active.advance()

    assert active.current_step_index == 1
    assert active.get_current_step() is step2
    assert active.cached_path is None  # Cleared on advance


def test_active_plan_get_current_step_when_complete() -> None:
    """ActivePlan.get_current_step returns None when all steps are done."""
    _, _, context = make_test_context()

    plan = ActionPlan(name="Test", steps=[ApproachStep()])
    active = ActivePlan(plan=plan, context=context)

    active.advance()  # Move past the only step

    assert active.get_current_step() is None


def test_active_plan_is_complete() -> None:
    """ActivePlan.is_complete returns True when all steps are done."""
    _, _, context = make_test_context()

    step1 = ApproachStep()
    step2 = IntentStep(intent_class=DummyIntent, params=lambda ctx: {})
    plan = ActionPlan(name="Test", steps=[step1, step2])
    active = ActivePlan(plan=plan, context=context)

    assert active.is_complete() is False

    active.advance()
    assert active.is_complete() is False

    active.advance()
    assert active.is_complete() is True


def test_active_plan_cached_path() -> None:
    """ActivePlan can store and access a cached path."""
    _, _, context = make_test_context()

    plan = ActionPlan(name="Test", steps=[ApproachStep()])
    active = ActivePlan(plan=plan, context=context)

    path = [(1, 1), (2, 2), (3, 3)]
    active.cached_path = path

    assert active.cached_path is path
    assert active.cached_path[0] == (1, 1)


def test_active_plan_rewind_to_previous_approach_step_resets_step_and_caches() -> None:
    """rewind_to_previous_approach_step rewinds and clears path caches."""
    _, _, context = make_test_context()

    plan = ActionPlan(
        name="Rewind Test",
        steps=[
            ApproachStep(stop_distance=1),
            IntentStep(intent_class=DummyIntent, params=lambda ctx: {}),
        ],
    )
    active = ActivePlan(plan=plan, context=context, current_step_index=2)
    active.cached_path = [(1, 1), (2, 2)]
    active.cached_hierarchical_path = [1, 2]

    rewound = active.rewind_to_previous_approach_step()

    assert rewound is True
    assert active.current_step_index == 0
    assert isinstance(active.get_current_step(), ApproachStep)
    assert active.cached_path is None
    assert active.cached_hierarchical_path is None


def test_active_plan_rewind_to_previous_approach_step_returns_false_without_approach_step() -> (
    None
):
    """rewind_to_previous_approach_step fails without an ApproachStep."""
    _, _, context = make_test_context()

    plan = ActionPlan(
        name="No Approach",
        steps=[
            IntentStep(intent_class=DummyIntent, params=lambda ctx: {}),
            IntentStep(intent_class=DummyIntent, params=lambda ctx: {}),
        ],
    )
    active = ActivePlan(plan=plan, context=context, current_step_index=1)

    assert active.rewind_to_previous_approach_step() is False


def test_punch_plan_rewind_from_post_punch_returns_to_approach() -> None:
    """PunchPlan rewinds from post-intent index back to its ApproachStep."""
    from brileta.game.actions.stunts import PunchPlan

    _, _, context = make_test_context()
    active = ActivePlan(plan=PunchPlan, context=context, current_step_index=3)

    rewound = active.rewind_to_previous_approach_step()

    assert rewound is True
    assert active.current_step_index == 0
    assert isinstance(active.get_current_step(), ApproachStep)


# -----------------------------------------------------------------------------
# Character.active_plan integration
# -----------------------------------------------------------------------------


def test_character_has_active_plan_field() -> None:
    """Character has an active_plan field that defaults to None."""
    gw = DummyGameWorld()
    player = Character(
        0,
        0,
        "@",
        colors.WHITE,
        "Player",
        game_world=cast(GameWorld, gw),
    )

    assert player.active_plan is None


def test_character_can_set_active_plan() -> None:
    """Character.active_plan can be set to an ActivePlan."""
    _controller, player, context = make_test_context()

    plan = ActionPlan(name="Walk", steps=[ApproachStep()])
    active = ActivePlan(plan=plan, context=context)

    player.active_plan = active

    assert player.active_plan is active
    assert player.active_plan.plan.name == "Walk"


# -----------------------------------------------------------------------------
# ApproachStep pathfinding fallback tests
# -----------------------------------------------------------------------------


def test_approach_step_finds_path_to_adjacent_tile_when_target_occupied() -> None:
    """ApproachStep should pathfind to adjacent tile when target is blocked.

    When stop_distance > 0, the actor doesn't need to reach the exact target
    position - just get within range. This tests the fallback logic in
    TurnManager._handle_approach_step: when pathfinding to an occupied tile
    fails, it should try adjacent tiles.
    """
    from brileta.events import reset_event_bus_for_testing
    from brileta.game.actions.movement import MoveIntent
    from tests.helpers import get_controller_with_player_and_map

    reset_event_bus_for_testing()
    controller = get_controller_with_player_and_map()
    player = controller.gw.player
    gm = controller.gw.game_map

    # Find a walkable tile 2 squares away from the player for the enemy
    # Search in the cardinal directions first, then diagonals
    enemy_pos: tuple[int, int] | None = None
    for dx, dy in [(2, 0), (-2, 0), (0, 2), (0, -2), (2, 2), (-2, -2)]:
        tx, ty = player.x + dx, player.y + dy
        in_bounds = 0 <= tx < gm.width and 0 <= ty < gm.height
        if not in_bounds or not gm.walkable[tx, ty]:
            continue
        # Also verify intermediate tile is walkable (for pathfinding)
        mid_x, mid_y = player.x + dx // 2, player.y + dy // 2
        if gm.walkable[mid_x, mid_y]:
            enemy_pos = (tx, ty)
            break

    assert enemy_pos is not None, "Test setup failed: no walkable tile found for enemy"

    # Create enemy at the found position - 2 tiles away from player
    enemy = Character(
        enemy_pos[0],
        enemy_pos[1],
        "r",
        colors.RED,
        "Raider",
        game_world=controller.gw,
    )
    controller.gw.add_actor(enemy)

    # Create an ActivePlan with ApproachStep targeting the enemy
    # stop_distance=1 means we want to end up adjacent
    context = PlanContext(
        actor=player,
        controller=controller,
        target_actor=enemy,
        target_position=(enemy.x, enemy.y),
    )
    plan = ActionPlan(
        name="Approach Enemy",
        steps=[ApproachStep(stop_distance=1)],
        requires_target=True,
    )
    player.active_plan = ActivePlan(plan=plan, context=context)

    # Get the next intent from the plan
    # This should succeed despite the target tile being occupied by the enemy
    intent = controller.turn_manager._get_intent_from_plan(player)

    # Should get a valid MoveIntent (not None - the adjacent tile fallback worked)
    assert intent is not None, (
        "ApproachStep should find path to adjacent tile when target is occupied"
    )
    assert isinstance(intent, MoveIntent)

    # The plan should still be active (not cancelled)
    assert player.active_plan is not None
