from brileta import colors
from brileta.controller import Controller
from brileta.game.actions.movement import MoveIntent
from brileta.game.actors import NPC
from brileta.game.actors.status_effects import OffBalanceEffect


def test_npc_status_effect_duration_is_independent_of_player_actions(
    controller: Controller,
):
    """
    Asserts that an NPC's status effects do not tick down when the player acts.

    This test verifies that the duration of a status effect is tied to the
    affected actor's own turn, not the turn of other actors.

    Scenario:
    1. An NPC is given a status effect with a 1-turn duration (OffBalanceEffect).
    2. The player, who is faster than the NPC, takes an action.
    3. The test asserts that the NPC's status effect is still active, as the
       NPC has not yet taken its own turn to consume the effect's duration.
    """
    # 1. SETUP
    # Create a controller and a faster-than-normal player.
    player = controller.gw.player
    player.energy.speed = 200  # Player is faster than the NPC

    # Create an NPC with default speed (100).
    npc = NPC(
        x=1, y=1, ch="N", color=colors.WHITE, name="Test NPC", game_world=controller.gw
    )
    controller.gw.add_actor(npc)
    # Ensure the new NPC is added to the TurnManager cache.
    controller.turn_manager.invalidate_cache()

    # Apply a 1-turn status effect to the NPC.
    effect_to_apply = OffBalanceEffect()
    npc.status_effects.apply_status_effect(effect_to_apply)
    assert npc.status_effects.has_status_effect(OffBalanceEffect), (
        "Pre-condition failed: Effect was not applied correctly."
    )
    assert effect_to_apply.duration == 1

    # 2. ACTION
    # The player takes one action.
    player_action = MoveIntent(controller, player, 1, 0)
    controller._execute_player_action_immediately(player_action)

    # All NPCs that can act now do so. Since the NPC's speed is 100 and the
    # player's is 200, the NPC will not have enough energy to act yet.
    controller._process_all_available_npc_actions()

    # 3. ASSERTION
    # The NPC has not taken a turn, so the status effect's duration should
    # not have changed. It should still be active.
    # THIS ASSERTION WILL FAIL WITH THE BUGGY CODE.
    assert npc.status_effects.has_status_effect(OffBalanceEffect), (
        "Bug detected: NPC status effect expired prematurely due to player's action."
    )
