from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from brileta import colors
from brileta.constants.combat import CombatConstants as Combat
from brileta.environment import tile_types
from brileta.events import (
    ActorDeathEvent,
    CombatInitiatedEvent,
    EffectEvent,
    FloatingTextEvent,
    FloatingTextValence,
    MessageEvent,
    ScreenShakeEvent,
    SoundEvent,
    publish_event,
)
from brileta.game import ranges
from brileta.game.actions.base import GameActionResult
from brileta.game.actions.executors.base import ActionExecutor
from brileta.game.actors import Character, status_effects
from brileta.game.actors.ai import escalate_hostility
from brileta.game.consequences import (
    AttackConsequenceGenerator,
    ConsequenceHandler,
)
from brileta.game.enums import OutcomeTier
from brileta.game.items.capabilities import Attack
from brileta.game.items.item_core import Item
from brileta.game.items.item_types import FISTS_TYPE
from brileta.game.items.properties import TacticalProperty, WeaponProperty
from brileta.game.resolution import combat_arbiter
from brileta.game.resolution.base import ResolutionResult
from brileta.game.resolution.outcomes import CombatOutcome
from brileta.sound.materials import AudioMaterialResolver, get_impact_sound_id
from brileta.sound.weapon_sounds import get_reload_sound_id, get_weapon_sound_id
from brileta.types import DeltaTime
from brileta.view.presentation import PresentationEvent

if TYPE_CHECKING:
    from brileta.game.actions.combat import AttackIntent, ReloadIntent


class AttackExecutor(ActionExecutor):
    """Executes attack intents by applying all combat logic."""

    def execute(self, intent: AttackIntent) -> GameActionResult | None:  # type: ignore[override]
        # Check for tile shot (no defender, but target coordinates set)
        if (
            intent.defender is None
            and intent.target_x is not None
            and intent.target_y is not None
        ):
            return self._execute_tile_shot(intent)

        # For actor attacks, defender must be set
        assert intent.defender is not None, "Actor attack requires a defender"

        # 1. Determine what attack method to use
        attack, weapon = self._determine_attack_method(intent)
        if not attack:
            return GameActionResult(succeeded=False)

        # 2. Validate the attack can be performed
        range_modifiers = self._validate_attack(intent, attack, weapon)
        if range_modifiers is None:
            return GameActionResult(
                succeeded=False
            )  # Validation failed, error messages already logged

        # Auto-enter combat mode when NPC attacks player (hit or miss)
        # Intent alone warrants combat mode - we don't wait for the outcome.
        if (
            intent.defender == intent.controller.gw.player
            and intent.attacker != intent.controller.gw.player
        ):
            publish_event(
                CombatInitiatedEvent(
                    attacker=intent.attacker,
                    defender=intent.defender,
                )
            )

        # 3. Perform the attack roll and immediate effects
        attack_result = self._execute_attack_roll(
            intent, attack, weapon, range_modifiers
        )
        if attack_result is None:
            return GameActionResult(succeeded=False)

        # 4. Determine combat consequences based on the resolution result
        outcome = combat_arbiter.determine_outcome(
            attack_result, intent.attacker, intent.defender, weapon
        )

        # 5. Apply the outcome and post-attack effects
        damage = self._apply_combat_outcome(
            intent, attack_result, outcome, attack, weapon
        )
        self._handle_post_attack_effects(intent, attack_result, attack, weapon, damage)

        # 6. Generate and apply additional consequences
        generator = AttackConsequenceGenerator()
        consequences = generator.generate(
            attacker=intent.attacker,
            weapon=weapon,
            outcome_tier=attack_result.outcome_tier,
        )
        handler = ConsequenceHandler()
        for consequence in consequences:
            handler.apply_consequence(consequence)

        # Switch active weapon to the one that was just used
        if intent.weapon is not None and intent.attacker.inventory is not None:
            for slot_index, equipped_weapon in enumerate(
                intent.attacker.inventory.ready_slots
            ):
                if equipped_weapon == intent.weapon:
                    if slot_index != intent.attacker.inventory.active_slot:
                        intent.attacker.inventory.switch_to_slot(slot_index)
                    break

        # Determine presentation timing based on attack type.
        duration_ms = (
            Combat.RANGED_DURATION_MS
            if attack == weapon.ranged_attack
            else Combat.MELEE_DURATION_MS
        )

        return GameActionResult(consequences=consequences, duration_ms=duration_ms)

    def _fire_weapon(
        self,
        intent: AttackIntent,
        weapon: Item,
        target_x: int,
        target_y: int,
    ) -> bool:
        """Consume ammo and emit common firing effects for ranged attacks.

        This consolidates the firing mechanics shared between actor attacks and
        tile shots: ammo consumption, inventory revision, muzzle flash, gunfire
        sound, and dry-fire click when empty.

        Args:
            intent: The attack intent with attacker and controller info.
            weapon: The weapon being fired.
            target_x: X coordinate of the target (actor or tile).
            target_y: Y coordinate of the target (actor or tile).

        Returns:
            True if the weapon fired successfully, False if it couldn't fire
            (no ammo or no ranged capability).
        """
        ranged_attack = weapon.ranged_attack
        if not ranged_attack or ranged_attack.current_ammo <= 0:
            publish_event(MessageEvent(f"{weapon.name} is empty!", colors.RED))
            return False

        # Consume ammo and notify inventory for UI cache invalidation
        ranged_attack.current_ammo -= 1
        intent.attacker.inventory._increment_revision()

        # Direction for muzzle flash
        direction_x = target_x - intent.attacker.x
        direction_y = target_y - intent.attacker.y

        # Build gunfire sound event
        sound_events: list[SoundEvent] = []
        sound_id = get_weapon_sound_id(ranged_attack.ammo_type)
        if sound_id:
            sound_events.append(
                SoundEvent(
                    sound_id=sound_id,
                    x=intent.attacker.x,
                    y=intent.attacker.y,
                    pitch_jitter=(0.95, 1.05),
                )
            )

        # Emit muzzle flash and gunfire sound
        is_player = intent.attacker == intent.controller.gw.player
        publish_event(
            PresentationEvent(
                effect_events=[
                    EffectEvent(
                        "muzzle_flash",
                        intent.attacker.x,
                        intent.attacker.y,
                        direction_x=direction_x,
                        direction_y=direction_y,
                    )
                ],
                sound_events=sound_events,
                source_x=intent.attacker.x,
                source_y=intent.attacker.y,
                is_player_action=is_player,
            )
        )

        # Emit dry fire click if weapon is now empty
        if ranged_attack.current_ammo == 0:
            self._emit_dry_fire(intent)

        return True

    def _execute_tile_shot(self, intent: AttackIntent) -> GameActionResult:
        """Execute a shot at an environmental tile (wall, door, etc.).

        Uses _fire_weapon for common firing mechanics, then plays impact sound
        at the target tile based on its material.
        """
        weapon = intent.weapon
        if weapon is None:
            publish_event(MessageEvent("No weapon selected!", colors.RED))
            return GameActionResult(succeeded=False)

        target_x = intent.target_x
        target_y = intent.target_y
        assert target_x is not None and target_y is not None  # Type narrowing

        # Fire the weapon (handles ammo, muzzle flash, sounds, dry fire)
        if not self._fire_weapon(intent, weapon, target_x, target_y):
            return GameActionResult(succeeded=False)

        # Get target tile info for impact effects
        game_map = intent.controller.gw.game_map
        tile_type_id = int(game_map.tiles[target_x, target_y])
        tile_name = tile_types.get_tile_type_name_by_id(tile_type_id)
        is_player = intent.attacker == intent.controller.gw.player

        # Emit impact sound at target tile
        material = AudioMaterialResolver.resolve_tile_material(tile_type_id)
        impact_sound_id = get_impact_sound_id(material)
        publish_event(
            PresentationEvent(
                sound_events=[
                    SoundEvent(
                        sound_id=impact_sound_id,
                        x=target_x,
                        y=target_y,
                        pitch_jitter=(0.92, 1.08),
                    )
                ],
                source_x=target_x,
                source_y=target_y,
                is_player_action=is_player,
            )
        )

        # Log message
        publish_event(
            MessageEvent(
                f"{intent.attacker.name} shoots at the {tile_name.lower()}.",
                colors.LIGHT_GREY,
            )
        )

        # Generate consequences (fumbles can happen even when shooting at walls)
        # Simple d20 roll: 1 = critical failure, otherwise success
        from brileta.util.dice import roll_d

        fumble_roll = roll_d(20)
        outcome_tier = (
            OutcomeTier.CRITICAL_FAILURE if fumble_roll == 1 else OutcomeTier.SUCCESS
        )

        generator = AttackConsequenceGenerator()
        consequences = generator.generate(
            attacker=intent.attacker,
            weapon=weapon,
            outcome_tier=outcome_tier,
        )
        handler = ConsequenceHandler()
        for consequence in consequences:
            handler.apply_consequence(consequence)

        return GameActionResult(
            consequences=consequences, duration_ms=Combat.RANGED_DURATION_MS
        )

    def _determine_attack_method(
        self, intent: AttackIntent
    ) -> tuple[Attack | None, Item]:
        """Determine which attack method and weapon to use."""
        assert intent.defender is not None  # Tile shots handled separately
        weapon = intent.weapon or self._select_appropriate_weapon(
            intent.attacker,
            force_melee=intent.attack_mode == "melee",
        )

        distance = ranges.calculate_distance(
            intent.attacker.x,
            intent.attacker.y,
            intent.defender.x,
            intent.defender.y,
        )

        # Highest priority: explicit attack mode selection
        if intent.attack_mode == "melee":
            return weapon.melee_attack, weapon
        if intent.attack_mode == "ranged":
            return weapon.ranged_attack, weapon

        # Next priority: weapon's preferred attack for this distance
        preferred = weapon.get_preferred_attack_mode(distance)
        if (
            isinstance(preferred, Attack)
            and WeaponProperty.PREFERRED in preferred.properties
        ):
            return preferred, weapon

        # Fallback heuristic based on distance if no preference exists
        if distance == 1 and weapon.melee_attack:
            return weapon.melee_attack, weapon

        if weapon.ranged_attack:
            return weapon.ranged_attack, weapon

        if weapon.melee_attack:
            return weapon.melee_attack, weapon

        publish_event(
            MessageEvent(f"{intent.attacker.name} has no way to attack!", colors.RED)
        )
        return None, weapon

    def _select_appropriate_weapon(
        self, actor: Character, *, force_melee: bool = False
    ) -> Item:
        """Select the most appropriate weapon for the attack.

        When ``force_melee`` is True (used for ramming), only consider weapons
        that are viable for melee combat. If none are available, return fists.
        Otherwise, follow the normal heuristic which may fall back to the active
        weapon even if it is ranged-only.
        """

        active_weapon = actor.inventory.get_active_item()

        # If we have an active weapon and it's suitable for melee, use it
        if active_weapon and self._is_suitable_melee_for_ramming(active_weapon):
            return active_weapon

        # Look for other suitable melee weapons in equipped slots
        candidates: list[Item] = []
        for weapon, _ in actor.inventory.get_equipped_items():
            if weapon is not active_weapon and self._is_suitable_melee_for_ramming(
                weapon
            ):
                candidates.append(weapon)

        if candidates:
            # Prefer designed weapons over improvised ones
            non_improvised = [
                w
                for w in candidates
                if WeaponProperty.IMPROVISED not in w.melee_attack.properties  # type: ignore[union-attr]
            ]
            if non_improvised:
                return non_improvised[0]
            return candidates[0]

        # If we specifically require a melee option and none are found,
        # fall back to fists instead of an unsuitable active weapon.
        if force_melee:
            return FISTS_TYPE.create()

        # Otherwise, fall back to the active weapon even if it's ranged-only.
        if active_weapon:
            return active_weapon

        # Fall back to fists
        return FISTS_TYPE.create()

    def _is_suitable_melee_for_ramming(self, weapon: Item) -> bool:
        """Return True if the weapon can be used effectively in melee combat."""
        melee = weapon.melee_attack
        if melee is None:
            return False

        ranged = weapon.ranged_attack
        return not (
            ranged is not None and WeaponProperty.PREFERRED in ranged.properties
        )

    def _validate_attack(
        self, intent: AttackIntent, attack: Attack, weapon: Item
    ) -> dict | None:
        """Validate the attack can be performed and return range modifiers."""
        assert intent.defender is not None  # Tile shots handled separately
        if attack is None:
            publish_event(
                MessageEvent(f"{weapon.name} cannot perform this attack!", colors.RED)
            )
            return None

        distance = ranges.calculate_distance(
            intent.attacker.x, intent.attacker.y, intent.defender.x, intent.defender.y
        )

        range_modifiers: dict[str, bool] = {}

        if attack == weapon.melee_attack and distance > 1:
            publish_event(
                MessageEvent(
                    f"Too far away for melee attack with {weapon.name}!", colors.RED
                )
            )
            return None

        ranged_attack = weapon.ranged_attack
        if attack == ranged_attack and ranged_attack is not None:
            if ranged_attack.current_ammo <= 0:
                publish_event(
                    MessageEvent(f"{weapon.name} is out of ammo!", colors.RED)
                )
                return None

            range_category = ranges.get_range_category(distance, weapon)
            if range_category == "out_of_range":
                publish_event(
                    MessageEvent(
                        f"{intent.defender.name} is too far away for {weapon.name}!",
                        colors.RED,
                    )
                )
                return None

            if not ranges.has_line_of_sight(
                intent.controller.gw.game_map,
                intent.attacker.x,
                intent.attacker.y,
                intent.defender.x,
                intent.defender.y,
            ):
                publish_event(
                    MessageEvent(
                        f"No clear shot to {intent.defender.name}!", colors.RED
                    )
                )
                return None

            range_modifiers = ranges.get_range_modifier(weapon, range_category) or {}

        return range_modifiers

    def _adjacent_cover_bonus(self, intent: AttackIntent) -> int:
        """Return the highest cover bonus adjacent to the defender."""
        assert intent.defender is not None  # Tile shots handled separately
        game_map = intent.controller.gw.game_map
        max_bonus = 0
        x, y = intent.defender.x, intent.defender.y

        # Cover checks are infrequent, so we prioritize memory usage over
        # lookup speed by querying tile types directly rather than caching a
        # full map of bonuses.
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < game_map.width and 0 <= ny < game_map.height:
                    tile_id = game_map.tiles[nx, ny]
                    tile_data = tile_types.get_tile_type_data_by_id(int(tile_id))
                    bonus = int(tile_data["cover_bonus"])
                    max_bonus = max(max_bonus, bonus)

        return max_bonus

    def _execute_attack_roll(
        self, intent: AttackIntent, attack: Attack, weapon: Item, range_modifiers: dict
    ) -> ResolutionResult | None:
        """Perform the attack roll and handle immediate effects like ammo use."""
        assert intent.defender is not None  # Tile shots handled separately
        stat_name = attack.stat_name
        attacker_score = getattr(intent.attacker.stats, stat_name)
        defender_score = intent.defender.stats.agility + self._adjacent_cover_bonus(
            intent
        )

        # Get all resolution modifiers from the unified facade
        resolution_args: dict[str, Any] = (
            intent.attacker.modifiers.get_resolution_modifiers(stat_name)
        )

        final_advantage = range_modifiers.get(
            "has_advantage", False
        ) or resolution_args.get("has_advantage", False)
        final_disadvantage = range_modifiers.get(
            "has_disadvantage", False
        ) or resolution_args.get("has_disadvantage", False)

        resolver = intent.controller.create_resolver(
            ability_score=attacker_score,
            roll_to_exceed=defender_score + Combat.D20_DC_BASE,
            has_advantage=final_advantage,
            has_disadvantage=final_disadvantage,
        )
        attack_result = resolver.resolve(intent.attacker, intent.defender, weapon)

        # Fire the weapon for ranged attacks (handles ammo, muzzle flash, sounds)
        ranged_attack = weapon.ranged_attack
        if attack == ranged_attack and ranged_attack is not None:
            assert intent.defender is not None
            self._fire_weapon(intent, weapon, intent.defender.x, intent.defender.y)

            # Handle thrown weapons when empty - remove from inventory and spawn
            # at target location for potential retrieval.
            # Skip if critical failure - weapon_drop consequence handles that case.
            if (
                ranged_attack.current_ammo == 0
                and WeaponProperty.THROWN in ranged_attack.properties
                and attack_result.outcome_tier != OutcomeTier.CRITICAL_FAILURE
            ):
                inv = intent.attacker.inventory
                inv.try_remove_item(weapon)

                # Spawn at target location for potential retrieval
                if weapon.can_materialize and intent.defender:
                    gw = intent.attacker.gw
                    if gw:
                        gw.spawn_ground_item(
                            weapon, intent.defender.x, intent.defender.y
                        )

        return attack_result

    def _emit_dry_fire(self, intent: AttackIntent) -> None:
        """Emit dry fire click sound when weapon empties.

        Uses a 150ms delay so the click plays after the gunshot finishes.
        """
        is_player = intent.attacker == intent.controller.gw.player
        publish_event(
            PresentationEvent(
                sound_events=[
                    SoundEvent(
                        sound_id="gun_dry_fire",
                        x=intent.attacker.x,
                        y=intent.attacker.y,
                        delay=0.15,  # 150ms after gunshot
                    )
                ],
                source_x=intent.attacker.x,
                source_y=intent.attacker.y,
                is_player_action=is_player,
            )
        )

    def _apply_combat_outcome(
        self,
        intent: AttackIntent,
        attack_result: ResolutionResult,
        outcome: CombatOutcome,
        attack: Attack,
        weapon: Item,
    ) -> int:
        """Apply the combat outcome and return the damage dealt.

        Note: Armor penetration and AP loss are handled in combat_arbiter.
        The outcome.damage_dealt is already the final damage after armor reduction.
        """
        assert intent.defender is not None  # Tile shots handled separately
        if attack_result.outcome_tier in (
            OutcomeTier.SUCCESS,
            OutcomeTier.CRITICAL_SUCCESS,
            OutcomeTier.PARTIAL_SUCCESS,
        ):
            damage = outcome.damage_dealt
            if outcome.injury_inflicted is not None:
                intent.defender.inventory.add_to_inventory(outcome.injury_inflicted)

            # Note: Armor reduction is handled in combat_arbiter.
            # outcome.damage_dealt is already the final damage after armor reduction.
            # Radiation damage type is still needed to add Rads conditions.
            damage_type = "normal"
            if weapon:
                props = weapon.get_weapon_properties()
                if TacticalProperty.RADIATION in props:
                    damage_type = "radiation"

            if damage > 0:
                intent.defender.take_damage(damage, damage_type=damage_type)
                # Use presentation layer for staggered combat feedback
                is_player = intent.attacker == intent.controller.gw.player

                # Resolve impact material and get corresponding sound
                material = AudioMaterialResolver.resolve_actor_material(intent.defender)
                impact_sound_id = get_impact_sound_id(material)

                # Calculate direction from attacker to defender for blood splatter cone
                dx = intent.defender.x - intent.attacker.x
                dy = intent.defender.y - intent.attacker.y
                dist = math.sqrt(dx * dx + dy * dy) or 1.0
                dir_x = dx / dist
                dir_y = dy / dist

                # Determine ray count based on weapon type
                # Shotgun: 3 rays (pellet spread), everything else: 1 ray
                ray_count = 1
                if (
                    weapon
                    and weapon.ranged_attack
                    and weapon.ranged_attack.ammo_type == "shotgun"
                ):
                    ray_count = 3

                publish_event(
                    PresentationEvent(
                        effect_events=[
                            EffectEvent(
                                "blood_splatter",
                                intent.defender.x,
                                intent.defender.y,
                                intensity=damage / Combat.DAMAGE_INTENSITY_DIVISOR,
                                direction_x=dir_x,
                                direction_y=dir_y,
                                ray_count=ray_count,
                            )
                        ],
                        sound_events=[
                            SoundEvent(
                                sound_id=impact_sound_id,
                                x=intent.defender.x,
                                y=intent.defender.y,
                                pitch_jitter=(0.95, 1.05),
                            )
                        ],
                        source_x=intent.defender.x,
                        source_y=intent.defender.y,
                        is_player_action=is_player,
                    )
                )
                self._log_hit_message(intent, attack_result, weapon, damage)

                if not intent.defender.health.is_alive():
                    publish_event(
                        MessageEvent(
                            f"{intent.defender.name} has been killed!", colors.RED
                        )
                    )
                    publish_event(ActorDeathEvent(intent.defender))
            return damage

        self._handle_attack_miss(intent, attack_result, attack, weapon)
        return 0

    def _handle_attack_miss(
        self,
        intent: AttackIntent,
        attack_result: ResolutionResult,
        attack: Attack,
        weapon: Item,
    ) -> None:
        """Handle an attack miss - messages and weapon property effects."""
        assert intent.defender is not None  # Tile shots handled separately
        # Miss
        if attack_result.outcome_tier == OutcomeTier.CRITICAL_FAILURE:
            # "Crits favoring defense cause the attacker's weapon to break,
            # and leave them confused or off-balance."
            # FIXME: Break the attacker's weapon.
            # FIXME: If the attacker is unarmed and attacking with fists or
            # kicking, etc., they pull a muscle and have disadvantage on their
            # next attack. (house rule)
            # FIXME: Give the attacker the condition "confused" or "off-balance".

            miss_color = colors.ORANGE  # A warning color for critical miss
            publish_event(
                MessageEvent(
                    (
                        f"Critical miss! {intent.attacker.name}'s attack on "
                        f"{intent.defender.name} fails."
                    ),
                    miss_color,
                )
            )
        else:
            miss_color = colors.GREY  # Standard miss color
            publish_event(
                MessageEvent(
                    f"{intent.attacker.name} misses {intent.defender.name}.",
                    miss_color,
                )
            )

        # Emit floating text for miss (on the attacker, showing their failure)
        publish_event(
            FloatingTextEvent(
                text="MISS",
                target_actor_id=intent.attacker.actor_id,
                valence=FloatingTextValence.NEUTRAL,
                world_x=intent.attacker.x,
                world_y=intent.attacker.y,
            )
        )

        # Handle 'awkward' weapon property on miss
        if (
            weapon
            and weapon.melee_attack
            and attack is weapon.melee_attack
            and WeaponProperty.AWKWARD in weapon.melee_attack.properties
        ):
            publish_event(
                MessageEvent(
                    f"{intent.attacker.name} is off balance from the awkward swing "
                    f"with {weapon.name}!",
                    colors.LIGHT_BLUE,
                )
            )
            intent.attacker.status_effects.apply_status_effect(
                status_effects.OffBalanceEffect()
            )

        # For ranged misses, trace bullet to find environmental impact
        if attack == weapon.ranged_attack and weapon.ranged_attack is not None:
            self._emit_ranged_miss_impact(intent)

    def _emit_ranged_miss_impact(self, intent: AttackIntent) -> None:
        """Trace missed bullet trajectory and play impact sound at first blocking tile.

        When a ranged attack misses, the bullet continues past the target until
        it hits an environmental surface. This method traces that trajectory
        and plays the appropriate material impact sound at the collision point.
        """
        assert intent.defender is not None  # Tile shots handled separately
        game_map = intent.controller.gw.game_map

        # Calculate direction from attacker through defender
        dx = intent.defender.x - intent.attacker.x
        dy = intent.defender.y - intent.attacker.y

        # Extend the line past the defender to find what the bullet hits
        # Use 10 tiles as a reasonable extension distance
        extend_distance = 10
        divisor = max(abs(dx), abs(dy), 1)
        end_x = intent.defender.x + (dx * extend_distance // divisor)
        end_y = intent.defender.y + (dy * extend_distance // divisor)

        # Clamp to map bounds
        end_x = max(0, min(game_map.width - 1, end_x))
        end_y = max(0, min(game_map.height - 1, end_y))

        # Trace line and find first non-transparent tile
        line = ranges.get_line(intent.attacker.x, intent.attacker.y, end_x, end_y)

        impact_x, impact_y = None, None
        for x, y in line[1:]:  # Skip attacker's tile
            if not (0 <= x < game_map.width and 0 <= y < game_map.height):
                break
            if not game_map.transparent[x, y]:
                impact_x, impact_y = x, y
                break

        # If we found an impact point, play the tile material sound
        if impact_x is not None and impact_y is not None:
            tile_type_id = int(game_map.tiles[impact_x, impact_y])
            material = AudioMaterialResolver.resolve_tile_material(tile_type_id)
            impact_sound_id = get_impact_sound_id(material)

            is_player = intent.attacker == intent.controller.gw.player
            publish_event(
                PresentationEvent(
                    sound_events=[
                        SoundEvent(
                            sound_id=impact_sound_id,
                            x=impact_x,
                            y=impact_y,
                            pitch_jitter=(0.92, 1.08),
                        )
                    ],
                    source_x=impact_x,
                    source_y=impact_y,
                    is_player_action=is_player,
                )
            )

    def _log_hit_message(
        self,
        intent: AttackIntent,
        attack_result: ResolutionResult,
        weapon: Item,
        damage: int,
    ) -> None:
        """Log appropriate hit message based on critical status."""
        assert intent.defender is not None  # Tile shots handled separately

        # Get verb from weapon based on attack mode
        verb = self._get_attack_verb(weapon, intent.attack_mode)
        verb_conjugated = self._conjugate_verb(verb)

        # Include weapon name for non-unarmed attacks
        is_unarmed = weapon.melee_attack and WeaponProperty.UNARMED in (
            weapon.melee_attack._spec.properties or set()
        )
        weapon_part = "" if is_unarmed else f" with {weapon.name}"

        if attack_result.outcome_tier == OutcomeTier.CRITICAL_SUCCESS:
            hit_color = colors.YELLOW
            message = (
                f"Critical hit! {intent.attacker.name} {verb_conjugated} "
                f"{intent.defender.name}{weapon_part} for {damage} damage."
            )
        else:
            hit_color = colors.WHITE  # Default color for a standard hit
            message = (
                f"{intent.attacker.name} {verb_conjugated} {intent.defender.name}"
                f"{weapon_part} for {damage} damage."
            )
        hp_message_part = (
            f" ({intent.defender.name} has {intent.defender.health.hp} HP left.)"
        )
        publish_event(MessageEvent(message + hp_message_part, hit_color))

    def _get_attack_verb(self, weapon: Item, attack_mode: str | None) -> str:
        """Get the verb for an attack based on weapon and attack mode."""
        if attack_mode == "melee" and weapon.melee_attack:
            return weapon.melee_attack._spec.verb
        if attack_mode == "ranged" and weapon.ranged_attack:
            return weapon.ranged_attack._spec.verb
        # Fallback for weapons without explicit verb
        return "hit"

    def _conjugate_verb(self, verb: str) -> str:
        """Conjugate a verb to third person singular present tense.

        Examples: punch -> punches, stab -> stabs, hit -> hits
        """
        if verb.endswith(("s", "x", "z", "ch", "sh")):
            return verb + "es"
        return verb + "s"

    def _handle_post_attack_effects(
        self,
        intent: AttackIntent,
        attack_result: ResolutionResult,
        attack: Attack,
        weapon: Item,
        damage: int,
    ) -> None:
        """Handle post-attack effects like screen shake and AI disposition changes."""
        assert intent.defender is not None  # Tile shots handled separately
        # Screen shake only when PLAYER gets hit
        if (
            attack_result.outcome_tier
            in (
                OutcomeTier.SUCCESS,
                OutcomeTier.CRITICAL_SUCCESS,
                OutcomeTier.PARTIAL_SUCCESS,
            )
            and intent.defender == intent.controller.gw.player
        ):
            self._apply_screen_shake(intent, attack_result, attack, weapon, damage)

        # Update relationship hostility after aggressive attacks.
        escalate_hostility(intent.attacker, intent.defender, intent.controller)

    def _apply_screen_shake(
        self,
        intent: AttackIntent,
        attack_result: ResolutionResult,
        attack: Attack,
        weapon: Item,
        damage: int,
    ) -> None:
        """Apply screen shake when player is hit."""
        # Screen shake only when PLAYER gets hit
        # (player's perspective getting jarred)
        base_damage = damage

        # Different shake for different attack types
        if weapon.melee_attack and attack is weapon.melee_attack:
            # Melee attacks - use probability instead of pixel distance
            shake_intensity = min(
                base_damage * Combat.MELEE_SHAKE_INTENSITY_MULT,
                Combat.MELEE_SHAKE_INTENSITY_CAP,
            )
            shake_duration = (
                Combat.MELEE_SHAKE_DURATION_BASE
                + base_damage * Combat.MELEE_SHAKE_DURATION_MULT
            )
        else:
            # Ranged attacks - lower probability
            shake_intensity = min(
                base_damage * Combat.RANGED_SHAKE_INTENSITY_MULT,
                Combat.RANGED_SHAKE_INTENSITY_CAP,
            )
            shake_duration = (
                Combat.RANGED_SHAKE_DURATION_BASE
                + base_damage * Combat.RANGED_SHAKE_DURATION_MULT
            )

        # Extra shake for critical hits against player
        if attack_result.outcome_tier == OutcomeTier.CRITICAL_SUCCESS:
            shake_intensity *= Combat.CRIT_SHAKE_INTENSITY_MULT
            shake_duration *= Combat.CRIT_SHAKE_DURATION_MULT

        publish_event(ScreenShakeEvent(shake_intensity, DeltaTime(shake_duration)))


class ReloadExecutor(ActionExecutor):
    """Executes reload intents."""

    def execute(self, intent: ReloadIntent) -> GameActionResult | None:  # type: ignore[override]
        ranged_attack = intent.weapon.ranged_attack
        if not ranged_attack:
            return GameActionResult(succeeded=False)

        ammo_item = None
        for item in intent.actor.inventory:
            if (
                isinstance(item, Item)
                and item.ammo
                and item.ammo.ammo_type == ranged_attack.ammo_type
            ):
                ammo_item = item
                break

        if ammo_item:
            # Calculate shells to load BEFORE updating current_ammo
            shells_to_load = ranged_attack.max_ammo - ranged_attack.current_ammo

            intent.actor.inventory.remove_from_inventory(ammo_item)
            ranged_attack.current_ammo = ranged_attack.max_ammo
            intent.actor.inventory._increment_revision()

            # Play reload sound if one exists for this ammo type
            reload_sound_id = get_reload_sound_id(ranged_attack.ammo_type)
            if reload_sound_id:
                publish_event(
                    SoundEvent(
                        sound_id=reload_sound_id,
                        x=intent.actor.x,
                        y=intent.actor.y,
                        params={"shell_count": shells_to_load},
                    )
                )

            publish_event(
                MessageEvent(
                    f"{intent.actor.name} reloaded {intent.weapon.name}.",
                    colors.GREEN,
                )
            )
            return GameActionResult(duration_ms=Combat.RELOAD_DURATION_MS)

        publish_event(
            MessageEvent(
                f"No {ranged_attack.ammo_type} ammo available!",
                colors.RED,
            )
        )
        return GameActionResult(succeeded=False)
