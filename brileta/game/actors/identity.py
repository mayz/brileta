"""NPC identity data for future dialogue and text generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Gender(Enum):
    """Binary NPC gender used for names, pronouns, and dialogue grammar."""

    MALE = "male"
    FEMALE = "female"


@dataclass(frozen=True, slots=True)
class Pronouns:
    """English third-person pronoun forms for dialogue templates."""

    subject: str
    accusative: str
    possessive_adjective: str
    possessive_pronoun: str
    reflexive: str


@dataclass(frozen=True, slots=True)
class NPCIdentity:
    """Identity fields that affect text, not gameplay behavior."""

    gender: Gender
    pronouns: Pronouns


MALE_PRONOUNS = Pronouns(
    subject="he",
    accusative="him",
    possessive_adjective="his",
    possessive_pronoun="his",
    reflexive="himself",
)
FEMALE_PRONOUNS = Pronouns(
    subject="she",
    accusative="her",
    possessive_adjective="her",
    possessive_pronoun="hers",
    reflexive="herself",
)

PRONOUNS_BY_GENDER: dict[Gender, Pronouns] = {
    Gender.MALE: MALE_PRONOUNS,
    Gender.FEMALE: FEMALE_PRONOUNS,
}


def identity_for_gender(gender: Gender) -> NPCIdentity:
    """Build standard identity data for ``gender``."""
    return NPCIdentity(gender=gender, pronouns=PRONOUNS_BY_GENDER[gender])
