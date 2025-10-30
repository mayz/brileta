from dataclasses import dataclass
from typing import cast

from catley import colors
from catley.controller import Controller
from catley.game.actions.movement import MoveIntent
from catley.game.actors import PC, conditions, status_effects
from catley.game.enums import InjuryLocation
from catley.game.game_world import GameWorld
from catley.game.turn_manager import TurnManager
from tests.helpers import DummyGameWorld


@dataclass
class DummyController:
    gw: DummyGameWorld

    def __post_init__(self) -> None:
        self.action_cost = 100
        self.turn_manager = TurnManager(cast(Controller, self))

    def update_fov(self) -> None:
        pass

    def run_one_turn(self) -> None:
        # Start of Turn phase: All actors regenerate energy and process status effects
        for actor in self.gw.actors:
            actor.update_turn(cast(Controller, self))
            actor.energy.regenerate()

        # Player action (check if there's a queued action)
        player_action = self.turn_manager.dequeue_player_action()
        if player_action and self.gw.player:
            self.turn_manager.execute_intent(player_action)
            self.gw.player.energy.spend(self.action_cost)

        # NPC Action Resolution: Process all NPCs with sufficient energy
        for actor in list(self.gw.actors):
            if actor is self.gw.player:
                continue
            if hasattr(actor, "energy") and actor.energy.can_afford(self.action_cost):
                action = actor.get_next_action(cast(Controller, self))
                if action is not None:
                    self.turn_manager.execute_intent(action)
                    actor.energy.spend(self.action_cost)


def make_world() -> tuple[DummyController, PC]:
    gw = DummyGameWorld()
    actor = PC(
        0, 0, "A", colors.WHITE, "Act", game_world=cast(GameWorld, gw), strength=5
    )
    gw.player = actor
    gw.add_actor(actor)
    controller = DummyController(gw=gw)
    return controller, actor


def test_offbalance_effect_expires() -> None:
    controller, actor = make_world()
    actor.status_effects.apply_status_effect(status_effects.OffBalanceEffect())
    assert actor.status_effects.has_status_effect(status_effects.OffBalanceEffect)
    actor.update_turn(cast(Controller, controller))
    assert not actor.status_effects.has_status_effect(status_effects.OffBalanceEffect)


def test_focused_effect_expires() -> None:
    controller, actor = make_world()
    actor.status_effects.apply_status_effect(status_effects.FocusedEffect())
    assert actor.status_effects.has_status_effect(status_effects.FocusedEffect)
    actor.update_turn(cast(Controller, controller))
    assert not actor.status_effects.has_status_effect(status_effects.FocusedEffect)


def test_tripped_effect_expires() -> None:
    controller, actor = make_world()
    actor.status_effects.apply_status_effect(status_effects.TrippedEffect())
    assert actor.status_effects.has_status_effect(status_effects.TrippedEffect)
    actor.update_turn(cast(Controller, controller))
    assert not actor.status_effects.has_status_effect(status_effects.TrippedEffect)


def test_strength_boost_duration() -> None:
    controller, actor = make_world()
    actor.status_effects.apply_status_effect(
        status_effects.StrengthBoostEffect(duration=2)
    )
    assert actor.stats.strength == 7
    actor.update_turn(cast(Controller, controller))
    assert actor.stats.strength == 7
    actor.update_turn(cast(Controller, controller))
    assert actor.stats.strength == 5
    assert not actor.status_effects.has_status_effect(
        status_effects.StrengthBoostEffect
    )


def test_multiple_effects_coexist() -> None:
    controller, actor = make_world()
    actor.status_effects.apply_status_effect(status_effects.OffBalanceEffect())
    actor.status_effects.apply_status_effect(
        status_effects.StrengthBoostEffect(duration=2)
    )
    assert actor.status_effects.has_status_effect(status_effects.OffBalanceEffect)
    assert actor.status_effects.has_status_effect(status_effects.StrengthBoostEffect)
    actor.update_turn(cast(Controller, controller))
    assert not actor.status_effects.has_status_effect(status_effects.OffBalanceEffect)
    assert actor.status_effects.has_status_effect(status_effects.StrengthBoostEffect)


def test_offbalance_persists_until_next_round() -> None:
    controller, actor = make_world()
    tm = controller.turn_manager

    # Simulate an action applying OffBalanceEffect at the end of the round
    tm.queue_action(MoveIntent(cast(Controller, controller), actor, 1, 0))
    controller.run_one_turn()
    actor.status_effects.apply_status_effect(status_effects.OffBalanceEffect())
    assert actor.status_effects.has_status_effect(status_effects.OffBalanceEffect)

    # Next round should remove it before the actor acts again
    tm.queue_action(MoveIntent(cast(Controller, controller), actor, 1, 0))
    controller.run_one_turn()
    assert not actor.status_effects.has_status_effect(status_effects.OffBalanceEffect)


def test_condition_management_methods() -> None:
    _controller, actor = make_world()
    injury = conditions.Injury(InjuryLocation.LEFT_ARM, "Test")
    assert actor.conditions.add_condition(injury)[0]
    assert actor.conditions.has_condition(conditions.Injury)
    assert injury in actor.conditions.get_all_conditions()
    assert actor.conditions.get_conditions_by_type(conditions.Injury) == [injury]
    assert actor.conditions.remove_condition(injury)
    assert not actor.conditions.has_condition(conditions.Injury)
