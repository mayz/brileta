from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

from catley import colors

from .core_discovery import ActionCategory

if TYPE_CHECKING:
    from catley.game.items.capabilities import MeleeAttack, RangedAttack
    from catley.game.items.item_core import Item


class ActionFormatter:
    """Utility class for formatting action display text."""

    @staticmethod
    def get_probability_descriptor(probability: float) -> tuple[str, str]:
        """Convert probability to qualitative descriptor and color."""
        from catley import config

        for max_prob, descriptor, color in config.PROBABILITY_DESCRIPTORS:
            if probability <= max_prob:
                return (descriptor, color)

        return ("Unknown", "white")

    # Verbs where the weapon is the direct object (e.g., "Throw Knife at Target")
    # rather than the instrument (e.g., "Shoot Target with Pistol")
    THROWN_VERBS: ClassVar[set[str]] = {"throw", "hurl", "fling", "toss", "lob"}

    @staticmethod
    def get_attack_display_name(
        weapon: Item, attack_mode: str, target_name: str
    ) -> str:
        """Generate attack display name using the weapon's verb.

        Most weapons use instrument phrasing: "Verb Target with Weapon"
        Thrown weapons use direct object phrasing: "Verb Weapon at Target"
        """
        if attack_mode == "melee" and weapon.melee_attack:
            melee = cast("MeleeAttack", weapon.melee_attack)
            verb = melee._spec.verb
            return f"{verb.title()} {target_name} with {weapon.name}"
        if attack_mode == "ranged" and weapon.ranged_attack:
            ranged = cast("RangedAttack", weapon.ranged_attack)
            verb = ranged._spec.verb
            # Thrown weapons use different grammar: "Throw Knife at Target"
            if verb.lower() in ActionFormatter.THROWN_VERBS:
                return f"{verb.title()} {weapon.name} at {target_name}"
            return f"{verb.title()} {target_name} with {weapon.name}"
        return f"Attack {target_name} with {weapon.name}"

    @staticmethod
    def format_menu_text(action_option) -> str:
        """Formatted text for menus."""
        text = action_option.display_text or action_option.name
        if action_option.success_probability is not None:
            descriptor, _color = ActionFormatter.get_probability_descriptor(
                action_option.success_probability
            )
            text += f" ({descriptor})"
        if action_option.cost_description:
            text += f" - {action_option.cost_description}"
        return text

    @staticmethod
    def get_category_color(category: ActionCategory) -> colors.Color:
        """Get display color for action category."""
        color_map = {
            ActionCategory.COMBAT: colors.RED,
            ActionCategory.MOVEMENT: colors.BLUE,
            ActionCategory.ITEMS: colors.GREEN,
            ActionCategory.ENVIRONMENT: colors.ORANGE,
            ActionCategory.SOCIAL: colors.MAGENTA,
        }
        return color_map.get(category, colors.WHITE)
