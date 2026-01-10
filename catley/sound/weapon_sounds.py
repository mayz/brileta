"""Mapping from weapon properties to sound families.

This module maps weapon ammo types to their corresponding gunfire sound IDs.
Each weapon type has its own distinct sound (Fallout 1/2 style) rather than
grouping weapons into generic categories.
"""

from __future__ import annotations

# Maps ammo type to sound definition ID.
# Each weapon category has its own sound for distinct audio feedback.
AMMO_TO_SOUND_FAMILY: dict[str, str] = {
    "pistol": "gun_fire_pistol",
    "rifle": "gun_fire_rifle",
    "shotgun": "gun_fire_shotgun",
}


def get_weapon_sound_id(ammo_type: str) -> str | None:
    """Get the sound ID for a weapon's ammo type.

    Args:
        ammo_type: The weapon's ammo type (e.g., "pistol", "rifle").

    Returns:
        The sound definition ID to use, or None if no sound is defined
        for this ammo type.
    """
    return AMMO_TO_SOUND_FAMILY.get(ammo_type)
