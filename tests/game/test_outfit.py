"""Unit tests for the outfit (armor/clothing) system."""

from brileta.game.items.capabilities import OutfitSpec
from brileta.game.outfit import (
    LEATHER_ARMOR_TYPE,
    RAGS_TYPE,
    SHELTER_JUMPSUIT_TYPE,
    DamageResult,
    OutfitCapability,
    apply_hit,
    calculate_damage,
    get_outfit_spec,
    is_outfit_type,
    repair,
    rest_recovery,
)


class TestOutfitSpec:
    """Tests for OutfitSpec dataclass."""

    def test_default_values(self) -> None:
        """Default spec has 0 protection and 0 AP."""
        spec = OutfitSpec()
        assert spec.protection == 0
        assert spec.max_ap == 0

    def test_custom_values(self) -> None:
        """Spec stores custom protection and AP."""
        spec = OutfitSpec(protection=3, max_ap=5)
        assert spec.protection == 3
        assert spec.max_ap == 5


class TestOutfitCapability:
    """Tests for OutfitCapability runtime state."""

    def test_initialization_from_spec(self) -> None:
        """Capability is initialized from spec values."""
        spec = OutfitSpec(protection=2, max_ap=4)
        cap = OutfitCapability(spec)

        assert cap.protection == 2
        assert cap.max_ap == 4
        assert cap.ap == 4  # Starts at max
        assert cap.damaged_since_rest is False

    def test_is_broken_false_at_full_AP(self) -> None:
        """Outfit is not broken at full AP."""
        spec = OutfitSpec(protection=2, max_ap=4)
        cap = OutfitCapability(spec)

        assert cap.is_broken is False

    def test_is_broken_true_at_zero_AP(self) -> None:
        """Outfit is broken when AP reaches 0."""
        spec = OutfitSpec(protection=2, max_ap=4)
        cap = OutfitCapability(spec)
        cap.ap = 0

        assert cap.is_broken is True

    def test_is_broken_false_for_no_AP_outfit(self) -> None:
        """Outfit with 0 max_ap is never 'broken' (no AP to track)."""
        spec = OutfitSpec(protection=0, max_ap=0)
        cap = OutfitCapability(spec)

        assert cap.is_broken is False

    def test_has_protection_true(self) -> None:
        """has_protection is True when protection > 0."""
        spec = OutfitSpec(protection=2, max_ap=4)
        cap = OutfitCapability(spec)

        assert cap.has_protection is True

    def test_has_protection_false(self) -> None:
        """has_protection is False when protection == 0."""
        spec = OutfitSpec(protection=0, max_ap=0)
        cap = OutfitCapability(spec)

        assert cap.has_protection is False

    def test_effective_protection_when_functional(self) -> None:
        """effective_protection returns protection value when not broken."""
        spec = OutfitSpec(protection=3, max_ap=5)
        cap = OutfitCapability(spec)

        assert cap.effective_protection == 3

    def test_effective_protection_when_broken(self) -> None:
        """effective_protection returns 0 when broken."""
        spec = OutfitSpec(protection=3, max_ap=5)
        cap = OutfitCapability(spec)
        cap.ap = 0

        assert cap.effective_protection == 0


class TestCalculateDamage:
    """Tests for the calculate_damage function."""

    def test_no_outfit_full_damage(self) -> None:
        """Without outfit, all damage goes to HP."""
        result = calculate_damage(5, None)

        assert result.damage_to_hp == 5
        assert result.penetrated is True
        assert result.blocked is False

    def test_damage_blocked_by_armor(self) -> None:
        """Damage less than protection is fully blocked."""
        spec = OutfitSpec(protection=3, max_ap=4)
        cap = OutfitCapability(spec)

        result = calculate_damage(2, cap)

        assert result.damage_to_hp == 0
        assert result.penetrated is False
        assert result.blocked is True

    def test_damage_equal_to_protection_blocked(self) -> None:
        """Damage equal to protection is fully blocked."""
        spec = OutfitSpec(protection=3, max_ap=4)
        cap = OutfitCapability(spec)

        result = calculate_damage(3, cap)

        assert result.damage_to_hp == 0
        assert result.penetrated is False
        assert result.blocked is True

    def test_damage_exceeds_protection(self) -> None:
        """Damage exceeding protection penetrates."""
        spec = OutfitSpec(protection=3, max_ap=4)
        cap = OutfitCapability(spec)

        result = calculate_damage(5, cap)

        assert result.damage_to_hp == 2  # 5 - 3 = 2
        assert result.penetrated is True
        assert result.blocked is False

    def test_broken_armor_no_protection(self) -> None:
        """Broken armor provides no protection."""
        spec = OutfitSpec(protection=3, max_ap=4)
        cap = OutfitCapability(spec)
        cap.ap = 0

        result = calculate_damage(5, cap)

        assert result.damage_to_hp == 5  # Full damage
        assert result.penetrated is True


class TestApplyHit:
    """Tests for the apply_hit function."""

    def test_penetrating_hit_reduces_AP(self) -> None:
        """Penetrating hit reduces AP by 1."""
        spec = OutfitSpec(protection=2, max_ap=4)
        cap = OutfitCapability(spec)

        result = apply_hit(cap, penetrated=True)

        assert result is True
        assert cap.ap == 3
        assert cap.damaged_since_rest is True

    def test_blocked_hit_no_AP_loss(self) -> None:
        """Blocked hit does not reduce AP."""
        spec = OutfitSpec(protection=2, max_ap=4)
        cap = OutfitCapability(spec)

        result = apply_hit(cap, penetrated=False)

        assert result is False
        assert cap.ap == 4
        assert cap.damaged_since_rest is False

    def test_no_AP_outfit_not_affected(self) -> None:
        """Outfit with no AP tracking is not affected."""
        spec = OutfitSpec(protection=0, max_ap=0)
        cap = OutfitCapability(spec)

        result = apply_hit(cap, penetrated=True)

        assert result is False
        assert cap.ap == 0

    def test_already_broken_not_reduced_further(self) -> None:
        """Already broken armor is not reduced further."""
        spec = OutfitSpec(protection=2, max_ap=4)
        cap = OutfitCapability(spec)
        cap.ap = 0

        result = apply_hit(cap, penetrated=True)

        assert result is False
        assert cap.ap == 0


class TestRestRecovery:
    """Tests for the rest_recovery function."""

    def test_recovers_one_AP(self) -> None:
        """Rest recovers 1 AP when damaged since last rest."""
        spec = OutfitSpec(protection=2, max_ap=4)
        cap = OutfitCapability(spec)
        cap.ap = 2
        cap.damaged_since_rest = True

        result = rest_recovery(cap)

        assert result is True
        assert cap.ap == 3
        assert cap.damaged_since_rest is False

    def test_no_recovery_if_not_damaged(self) -> None:
        """Rest does not recover if outfit wasn't damaged."""
        spec = OutfitSpec(protection=2, max_ap=4)
        cap = OutfitCapability(spec)
        cap.ap = 3
        cap.damaged_since_rest = False

        result = rest_recovery(cap)

        assert result is False
        assert cap.ap == 3

    def test_no_recovery_if_broken(self) -> None:
        """Broken armor cannot be recovered through rest."""
        spec = OutfitSpec(protection=2, max_ap=4)
        cap = OutfitCapability(spec)
        cap.ap = 0
        cap.damaged_since_rest = True

        result = rest_recovery(cap)

        assert result is False
        assert cap.ap == 0
        # Flag should be cleared even though recovery failed
        assert cap.damaged_since_rest is False

    def test_no_recovery_at_max_ap(self) -> None:
        """Cannot recover beyond max AP."""
        spec = OutfitSpec(protection=2, max_ap=4)
        cap = OutfitCapability(spec)
        cap.ap = 4
        cap.damaged_since_rest = True

        result = rest_recovery(cap)

        assert result is False
        assert cap.ap == 4
        assert cap.damaged_since_rest is False

    def test_no_AP_outfit_not_affected(self) -> None:
        """Outfit with no AP tracking is not affected."""
        spec = OutfitSpec(protection=0, max_ap=0)
        cap = OutfitCapability(spec)

        result = rest_recovery(cap)

        assert result is False


class TestRepair:
    """Tests for the repair function."""

    def test_full_repair(self) -> None:
        """None amount fully repairs the outfit."""
        spec = OutfitSpec(protection=2, max_ap=4)
        cap = OutfitCapability(spec)
        cap.ap = 0
        cap.damaged_since_rest = True

        amount_repaired = repair(cap, None)

        assert amount_repaired == 4
        assert cap.ap == 4
        assert cap.damaged_since_rest is False

    def test_partial_repair(self) -> None:
        """Specific amount repairs that much AP."""
        spec = OutfitSpec(protection=2, max_ap=4)
        cap = OutfitCapability(spec)
        cap.ap = 1

        amount_repaired = repair(cap, 2)

        assert amount_repaired == 2
        assert cap.ap == 3

    def test_repair_capped_at_max(self) -> None:
        """Repair cannot exceed max AP."""
        spec = OutfitSpec(protection=2, max_ap=4)
        cap = OutfitCapability(spec)
        cap.ap = 3

        amount_repaired = repair(cap, 5)

        assert amount_repaired == 1
        assert cap.ap == 4

    def test_no_AP_outfit_not_affected(self) -> None:
        """Outfit with no AP tracking cannot be repaired."""
        spec = OutfitSpec(protection=0, max_ap=0)
        cap = OutfitCapability(spec)

        amount_repaired = repair(cap, 2)

        assert amount_repaired == 0


class TestOutfitItemTypes:
    """Tests for outfit item type definitions."""

    def test_is_outfit_type_true(self) -> None:
        """Known outfit types are recognized."""
        assert is_outfit_type(LEATHER_ARMOR_TYPE) is True
        assert is_outfit_type(RAGS_TYPE) is True
        assert is_outfit_type(SHELTER_JUMPSUIT_TYPE) is True

    def test_is_outfit_type_false(self) -> None:
        """Non-outfit types are not recognized."""
        from brileta.game.items.item_types import PISTOL_TYPE

        assert is_outfit_type(PISTOL_TYPE) is False

    def test_get_outfit_spec(self) -> None:
        """get_outfit_spec returns correct spec for outfit types."""
        spec = get_outfit_spec(LEATHER_ARMOR_TYPE)

        assert spec is not None
        assert spec.protection == 2
        assert spec.max_ap == 4

    def test_get_outfit_spec_none_for_non_outfit(self) -> None:
        """get_outfit_spec returns None for non-outfit types."""
        from brileta.game.items.item_types import PISTOL_TYPE

        assert get_outfit_spec(PISTOL_TYPE) is None


class TestDamageResultDataclass:
    """Tests for DamageResult dataclass."""

    def test_blocked_true_when_no_damage_and_not_penetrated(self) -> None:
        """blocked is True when damage is 0 and not penetrated."""
        result = DamageResult(damage_to_hp=0, penetrated=False)
        assert result.blocked is True

    def test_blocked_false_when_damage_dealt(self) -> None:
        """blocked is False when damage was dealt."""
        result = DamageResult(damage_to_hp=5, penetrated=True)
        assert result.blocked is False

    def test_blocked_false_when_penetrated(self) -> None:
        """blocked is False when penetrated (even with 0 damage)."""
        result = DamageResult(damage_to_hp=0, penetrated=True)
        assert result.blocked is False


class TestOutfitCapabilityPersistence:
    """Tests for outfit capability persistence on items.

    The outfit capability (AP, damage state) should persist on the Item itself,
    not be created fresh on each equip. This ensures damaged armor stays damaged
    after unequip/re-equip cycles.
    """

    def test_item_has_outfit_capability_on_creation(self) -> None:
        """Outfit items have outfit_capability set when created."""
        item = LEATHER_ARMOR_TYPE.create()

        assert item.outfit_capability is not None
        assert item.outfit_capability.ap == 4  # Leather armor max_ap
        assert item.outfit_capability.max_ap == 4
        assert item.outfit_capability.protection == 2

    def test_non_outfit_item_has_no_capability(self) -> None:
        """Non-outfit items have outfit_capability as None."""
        from brileta.game.items.item_types import PISTOL_TYPE

        item = PISTOL_TYPE.create()

        assert item.outfit_capability is None

    def test_damage_state_persists_after_unequip_reequip(self) -> None:
        """Armor keeps damage state after unequip/re-equip cycle."""
        from brileta.game.actors.components import CharacterInventory, StatsComponent

        # Setup inventory
        stats = StatsComponent(strength=5)  # Enough capacity
        inv = CharacterInventory(stats)

        # Create and equip armor
        armor_item = LEATHER_ARMOR_TYPE.create()
        inv.add_voluntary_item(armor_item)
        success, _ = inv.equip_outfit(armor_item)
        assert success

        # Damage the armor (simulate combat)
        assert armor_item.outfit_capability is not None
        armor_item.outfit_capability.ap = 2  # Damaged from 4 to 2
        assert armor_item.outfit_capability.ap == 2

        # Unequip
        success, _ = inv.unequip_outfit()
        assert success

        # Verify damage state persisted on the item
        assert armor_item.outfit_capability.ap == 2

        # Re-equip
        success, _ = inv.equip_outfit(armor_item)
        assert success

        # Verify damage state still persisted
        assert armor_item.outfit_capability.ap == 2

    def test_broken_armor_stays_broken_after_unequip_reequip(self) -> None:
        """Broken armor (0 AP) stays broken after unequip/re-equip cycle."""
        from brileta.game.actors.components import CharacterInventory, StatsComponent

        # Setup inventory
        stats = StatsComponent(strength=5)
        inv = CharacterInventory(stats)

        # Create and equip armor
        armor_item = LEATHER_ARMOR_TYPE.create()
        inv.add_voluntary_item(armor_item)
        success, _ = inv.equip_outfit(armor_item)
        assert success

        # Break the armor completely
        assert armor_item.outfit_capability is not None
        armor_item.outfit_capability.ap = 0
        assert armor_item.outfit_capability.is_broken

        # Unequip
        success, _ = inv.unequip_outfit()
        assert success

        # Verify broken state persisted
        assert armor_item.outfit_capability.is_broken
        assert armor_item.outfit_capability.ap == 0

        # Re-equip
        success, _ = inv.equip_outfit(armor_item)
        assert success

        # Verify still broken after re-equip
        assert armor_item.outfit_capability.is_broken
        assert armor_item.outfit_capability.ap == 0

    def test_equipped_outfit_uses_item_capability(self) -> None:
        """Equipped outfit tuple references the same capability as the item."""
        from brileta.game.actors.components import CharacterInventory, StatsComponent

        stats = StatsComponent(strength=5)
        inv = CharacterInventory(stats)

        armor_item = LEATHER_ARMOR_TYPE.create()
        inv.add_voluntary_item(armor_item)
        inv.equip_outfit(armor_item)

        # The equipped outfit's capability should be the same object as item's
        assert inv.equipped_outfit is not None
        _, equipped_cap = inv.equipped_outfit
        assert equipped_cap is armor_item.outfit_capability

    def test_set_starting_outfit_uses_item_capability(self) -> None:
        """set_starting_outfit uses the item's existing capability."""
        from brileta.game.actors.components import CharacterInventory, StatsComponent

        stats = StatsComponent(strength=5)
        inv = CharacterInventory(stats)

        armor_item = LEATHER_ARMOR_TYPE.create()
        inv.set_starting_outfit(armor_item)

        # Verify capability is shared
        assert inv.equipped_outfit is not None
        _, equipped_cap = inv.equipped_outfit
        assert equipped_cap is armor_item.outfit_capability

        # Modifying one should affect the other
        assert armor_item.outfit_capability is not None
        armor_item.outfit_capability.ap = 1
        assert equipped_cap.ap == 1
