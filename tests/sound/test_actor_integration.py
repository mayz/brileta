"""Tests for Actor-SoundEmitter integration."""

from brileta import colors
from brileta.game.actors.core import Actor
from brileta.game.actors.environmental import ContainedFire
from brileta.sound.emitter import SoundEmitter
from brileta.types import SoundId


class TestActorSoundEmitterIntegration:
    """Test Actor integration with SoundEmitter components."""

    def test_actor_sound_emitters_default_none(self) -> None:
        """Test that actors start with no sound emitters."""
        actor = Actor(x=5, y=5, ch="@", color=colors.WHITE, name="Test Actor")

        assert actor.sound_emitters is None

    def test_add_sound_emitter_to_actor(self) -> None:
        """Test adding a sound emitter to an actor."""
        actor = Actor(x=5, y=5, ch="@", color=colors.WHITE, name="Test Actor")
        emitter = SoundEmitter("fire_ambient")

        actor.add_sound_emitter(emitter)

        assert actor.sound_emitters is not None
        assert len(actor.sound_emitters) == 1
        assert actor.sound_emitters[0] is emitter

    def test_add_multiple_sound_emitters(self) -> None:
        """Test adding multiple sound emitters to an actor."""
        actor = Actor(x=5, y=5, ch="@", color=colors.WHITE, name="Test Actor")

        emitter1 = SoundEmitter("fire_ambient")
        emitter2 = SoundEmitter("waterfall_ambient")
        emitter3 = SoundEmitter("engine_loop")

        actor.add_sound_emitter(emitter1)
        actor.add_sound_emitter(emitter2)
        actor.add_sound_emitter(emitter3)

        assert actor.sound_emitters is not None
        assert len(actor.sound_emitters) == 3
        assert emitter1 in actor.sound_emitters
        assert emitter2 in actor.sound_emitters
        assert emitter3 in actor.sound_emitters

    def test_add_sound_emitter_preserves_existing(self) -> None:
        """Test that adding emitters preserves existing ones."""
        actor = Actor(x=5, y=5, ch="@", color=colors.WHITE, name="Test Actor")

        emitter1 = SoundEmitter("fire_ambient")
        emitter2 = SoundEmitter("waterfall_ambient")

        # Add first emitter
        actor.add_sound_emitter(emitter1)
        original_list = actor.sound_emitters

        # Add second emitter
        actor.add_sound_emitter(emitter2)

        # Should be the same list object
        assert actor.sound_emitters is original_list
        assert actor.sound_emitters is not None
        assert len(actor.sound_emitters) == 2
        assert emitter1 in actor.sound_emitters
        assert emitter2 in actor.sound_emitters

    def test_add_same_emitter_multiple_times(self) -> None:
        """Test adding the same emitter multiple times."""
        actor = Actor(x=5, y=5, ch="@", color=colors.WHITE, name="Test Actor")
        emitter = SoundEmitter("fire_ambient")

        actor.add_sound_emitter(emitter)
        actor.add_sound_emitter(emitter)
        actor.add_sound_emitter(emitter)

        assert actor.sound_emitters is not None
        assert len(actor.sound_emitters) == 3  # Same emitter added 3 times
        assert all(e is emitter for e in actor.sound_emitters)

    def test_sound_emitters_independence(self) -> None:
        """Test that sound emitters on different actors are independent."""
        actor1 = Actor(x=1, y=1, ch="1", color=colors.WHITE, name="Actor 1")
        actor2 = Actor(x=2, y=2, ch="2", color=colors.WHITE, name="Actor 2")

        emitter1 = SoundEmitter("fire_ambient")
        emitter2 = SoundEmitter("waterfall_ambient")

        actor1.add_sound_emitter(emitter1)
        actor2.add_sound_emitter(emitter2)

        # Each actor should have their own list
        assert actor1.sound_emitters is not actor2.sound_emitters
        assert actor1.sound_emitters is not None
        assert actor2.sound_emitters is not None
        assert len(actor1.sound_emitters) == 1
        assert len(actor2.sound_emitters) == 1
        assert actor1.sound_emitters[0] is emitter1
        assert actor2.sound_emitters[0] is emitter2


class TestContainedFireSoundIntegration:
    """Test ContainedFire integration with sound system."""

    def test_contained_fire_has_sound_emitter(self) -> None:
        """Test that ContainedFire automatically gets a sound emitter."""
        fire = ContainedFire(x=10, y=10, ch="*", color=colors.RED, name="Test Fire")

        assert fire.sound_emitters is not None
        assert len(fire.sound_emitters) == 1

        emitter = fire.sound_emitters[0]
        assert emitter.sound_id == "fire_ambient"
        assert emitter.active is True
        assert emitter.volume_multiplier == 1.0

    def test_campfire_sound_emitter(self) -> None:
        """Test that campfire has proper sound emitter."""
        campfire = ContainedFire.create_campfire(5, 5)

        assert campfire.sound_emitters is not None
        assert len(campfire.sound_emitters) == 1

        emitter = campfire.sound_emitters[0]
        assert emitter.sound_id == "fire_ambient"
        assert emitter.active is True

    def test_barrel_fire_sound_emitter(self) -> None:
        """Test that barrel fire has proper sound emitter."""
        barrel_fire = ContainedFire.create_barrel_fire(8, 8)

        assert barrel_fire.sound_emitters is not None
        assert len(barrel_fire.sound_emitters) == 1

        emitter = barrel_fire.sound_emitters[0]
        assert emitter.sound_id == "fire_ambient"
        assert emitter.active is True

    def test_torch_sound_emitter(self) -> None:
        """Test that torch has proper sound emitter."""
        torch = ContainedFire.create_torch(12, 12)

        assert torch.sound_emitters is not None
        assert len(torch.sound_emitters) == 1

        emitter = torch.sound_emitters[0]
        assert emitter.sound_id == "fire_ambient"
        assert emitter.active is True

    def test_contained_fire_with_additional_emitters(self) -> None:
        """Test adding additional sound emitters to ContainedFire."""
        fire = ContainedFire(x=15, y=15, ch="*", color=colors.RED, name="Complex Fire")

        # Should start with fire_ambient
        assert fire.sound_emitters is not None
        assert len(fire.sound_emitters) == 1
        assert fire.sound_emitters[0].sound_id == "fire_ambient"

        # Add additional emitter
        steam_emitter = SoundEmitter("steam_hiss", volume_multiplier=0.3)
        fire.add_sound_emitter(steam_emitter)

        # Should now have both
        assert fire.sound_emitters is not None
        assert len(fire.sound_emitters) == 2
        sound_ids = [e.sound_id for e in fire.sound_emitters]
        assert "fire_ambient" in sound_ids
        assert "steam_hiss" in sound_ids

    def test_contained_fire_emitter_properties(self) -> None:
        """Test that ContainedFire emitter has correct properties."""
        fire = ContainedFire(
            x=20, y=20, ch="*", color=colors.RED, name="Property Test Fire"
        )

        assert fire.sound_emitters is not None
        emitter = fire.sound_emitters[0]

        # Test default properties
        assert emitter.sound_id == "fire_ambient"
        assert emitter.volume_multiplier == 1.0
        assert emitter.active is True
        assert emitter.playing_instances == []

        # Test that we can modify properties
        emitter.set_volume_multiplier(0.5)
        emitter.set_active(False)

        assert emitter.volume_multiplier == 0.5
        assert emitter.active is False


class TestActorSoundEmitterEdgeCases:
    """Test edge cases for Actor-SoundEmitter integration."""

    def test_actor_with_empty_sound_id(self) -> None:
        """Test actor with emitter that has empty sound ID."""
        actor = Actor(x=5, y=5, ch="@", color=colors.WHITE, name="Test Actor")
        emitter = SoundEmitter("")  # Empty sound ID

        actor.add_sound_emitter(emitter)

        assert actor.sound_emitters is not None
        assert len(actor.sound_emitters) == 1
        assert actor.sound_emitters[0].sound_id == ""

    def test_actor_with_various_sound_ids(self) -> None:
        """Test actor with emitters having various sound IDs."""
        actor = Actor(x=5, y=5, ch="@", color=colors.WHITE, name="Test Actor")

        sound_ids = [
            "fire_ambient",
            "waterfall_ambient",
            "engine_loop",
            "steam_hiss",
            "custom_sound_123",
            "sound-with-dashes",
            "UPPERCASE_SOUND",
        ]

        for sound_id in sound_ids:
            emitter = SoundEmitter(sound_id)
            actor.add_sound_emitter(emitter)

        assert actor.sound_emitters is not None
        assert len(actor.sound_emitters) == len(sound_ids)

        actual_sound_ids = [e.sound_id for e in actor.sound_emitters]
        assert actual_sound_ids == sound_ids

    def test_actor_sound_emitter_state_changes(self) -> None:
        """Test that sound emitter state changes work properly on actors."""
        actor = Actor(x=5, y=5, ch="@", color=colors.WHITE, name="Test Actor")

        emitter1 = SoundEmitter("fire_ambient", volume_multiplier=0.8, active=True)
        emitter2 = SoundEmitter(
            "waterfall_ambient", volume_multiplier=0.6, active=False
        )

        actor.add_sound_emitter(emitter1)
        actor.add_sound_emitter(emitter2)

        # Modify emitter states
        emitter1.set_active(False)
        emitter1.set_volume_multiplier(0.2)
        emitter2.set_active(True)
        emitter2.set_volume_multiplier(0.9)

        # Check that changes are reflected
        assert actor.sound_emitters is not None
        assert actor.sound_emitters[0].active is False
        assert actor.sound_emitters[0].volume_multiplier == 0.2
        assert actor.sound_emitters[1].active is True
        assert actor.sound_emitters[1].volume_multiplier == 0.9

    def test_actor_multiple_emitters_same_sound_id(self) -> None:
        """Test actor with multiple emitters for the same sound ID."""
        actor = Actor(x=5, y=5, ch="@", color=colors.WHITE, name="Test Actor")

        # Create multiple emitters with same sound ID but different properties
        emitter1 = SoundEmitter("fire_ambient", volume_multiplier=1.0, active=True)
        emitter2 = SoundEmitter("fire_ambient", volume_multiplier=0.5, active=False)
        emitter3 = SoundEmitter("fire_ambient", volume_multiplier=0.8, active=True)

        actor.add_sound_emitter(emitter1)
        actor.add_sound_emitter(emitter2)
        actor.add_sound_emitter(emitter3)

        assert actor.sound_emitters is not None
        assert len(actor.sound_emitters) == 3

        # All should have same sound_id but different properties
        for emitter in actor.sound_emitters:
            assert emitter.sound_id == "fire_ambient"

        volumes = [e.volume_multiplier for e in actor.sound_emitters]
        active_states = [e.active for e in actor.sound_emitters]

        assert volumes == [1.0, 0.5, 0.8]
        assert active_states == [True, False, True]


class TestActorSoundEmitterTypeConsistency:
    """Test type consistency for Actor-SoundEmitter integration."""

    def test_sound_emitter_with_sound_id_type(self) -> None:
        """Test that SoundEmitter works with SoundId type."""
        actor = Actor(x=5, y=5, ch="@", color=colors.WHITE, name="Test Actor")

        sound_id: SoundId = "fire_ambient"
        emitter = SoundEmitter(sound_id)

        actor.add_sound_emitter(emitter)

        assert actor.sound_emitters is not None
        assert actor.sound_emitters[0].sound_id == sound_id
        assert actor.sound_emitters[0].sound_id == "fire_ambient"

    def test_multiple_sound_id_types(self) -> None:
        """Test multiple emitters with SoundId type."""
        actor = Actor(x=5, y=5, ch="@", color=colors.WHITE, name="Test Actor")

        sound_ids: list[SoundId] = ["fire_ambient", "waterfall_ambient", "engine_loop"]

        for sound_id in sound_ids:
            emitter = SoundEmitter(sound_id)
            actor.add_sound_emitter(emitter)

        assert actor.sound_emitters is not None
        assert len(actor.sound_emitters) == len(sound_ids)

        for i, emitter in enumerate(actor.sound_emitters):
            assert emitter.sound_id == sound_ids[i]
