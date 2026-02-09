"""Tests for CharacterLayer and multi-character composition rendering.

This module tests the CharacterLayer system including:
- CharacterLayer dataclass for visual composition
- Actor integration with character_layers property
- Bookcase container creation with layers
"""

from __future__ import annotations

from brileta.game.actors.container import Container, create_bookcase
from brileta.game.actors.core import Actor, CharacterLayer

# Re-export for easier test access


# =============================================================================
# CHARACTER LAYER DATACLASS TESTS
# =============================================================================


class TestCharacterLayer:
    """Tests for CharacterLayer dataclass."""

    def test_character_layer_creation_defaults(self) -> None:
        """CharacterLayer should have default offset and scale values."""
        layer = CharacterLayer(char="X", color=(255, 0, 0))

        assert layer.char == "X"
        assert layer.color == (255, 0, 0)
        assert layer.offset_x == 0.0
        assert layer.offset_y == 0.0
        assert layer.scale_x == 1.0
        assert layer.scale_y == 1.0

    def test_character_layer_creation_with_offsets(self) -> None:
        """CharacterLayer should accept custom offset values."""
        layer = CharacterLayer(
            char="|", color=(100, 100, 100), offset_x=-0.3, offset_y=0.25
        )

        assert layer.char == "|"
        assert layer.color == (100, 100, 100)
        assert layer.offset_x == -0.3
        assert layer.offset_y == 0.25

    def test_character_layer_negative_offsets(self) -> None:
        """CharacterLayer should support negative offsets for positioning."""
        layer = CharacterLayer(
            char="*", color=(200, 200, 0), offset_x=-0.5, offset_y=-0.5
        )

        assert layer.offset_x == -0.5
        assert layer.offset_y == -0.5

    def test_character_layer_with_various_colors(self) -> None:
        """CharacterLayer should work with various RGB color tuples."""
        # Black
        layer1 = CharacterLayer(char=".", color=(0, 0, 0))
        assert layer1.color == (0, 0, 0)

        # White
        layer2 = CharacterLayer(char=".", color=(255, 255, 255))
        assert layer2.color == (255, 255, 255)

        # Various colors
        layer3 = CharacterLayer(char=".", color=(128, 64, 32))
        assert layer3.color == (128, 64, 32)

    def test_character_layer_custom_scale(self) -> None:
        """CharacterLayer should accept custom scale_x and scale_y values."""
        layer = CharacterLayer(char="I", color=(200, 100, 50), scale_x=0.5, scale_y=0.8)

        assert layer.scale_x == 0.5
        assert layer.scale_y == 0.8

    def test_character_layer_non_uniform_scale(self) -> None:
        """CharacterLayer should support non-uniform scaling (different x and y)."""
        # Wide and short
        layer1 = CharacterLayer(
            char="|", color=(100, 100, 100), scale_x=1.2, scale_y=0.6
        )
        assert layer1.scale_x == 1.2
        assert layer1.scale_y == 0.6

        # Tall and thin
        layer2 = CharacterLayer(
            char="|", color=(100, 100, 100), scale_x=0.5, scale_y=1.0
        )
        assert layer2.scale_x == 0.5
        assert layer2.scale_y == 1.0

    def test_character_layer_scale_with_offsets(self) -> None:
        """CharacterLayer should support scale combined with offsets."""
        layer = CharacterLayer(
            char="*",
            color=(100, 100, 100),
            offset_x=0.2,
            offset_y=-0.1,
            scale_x=0.4,
            scale_y=0.6,
        )

        assert layer.offset_x == 0.2
        assert layer.offset_y == -0.1
        assert layer.scale_x == 0.4
        assert layer.scale_y == 0.6

    def test_character_layer_very_small_scale(self) -> None:
        """CharacterLayer should support very small scale values."""
        layer = CharacterLayer(
            char=".", color=(255, 255, 255), scale_x=0.1, scale_y=0.1
        )

        assert layer.scale_x == 0.1
        assert layer.scale_y == 0.1


# =============================================================================
# ACTOR INTEGRATION TESTS
# =============================================================================


class TestActorCharacterLayers:
    """Tests for Actor integration with character_layers."""

    def test_actor_without_character_layers(self) -> None:
        """Actor should default to None for character_layers."""
        actor = Actor(x=5, y=5, ch="@", color=(255, 255, 255), name="Test Actor")

        assert actor.character_layers is None

    def test_actor_with_character_layers(self) -> None:
        """Actor should accept character_layers parameter."""
        layers = [
            CharacterLayer(char="|", color=(100, 100, 100), offset_x=-0.3),
            CharacterLayer(char="|", color=(100, 100, 100), offset_x=0.3),
        ]
        actor = Actor(
            x=5,
            y=5,
            ch="@",
            color=(255, 255, 255),
            name="Test Actor",
            character_layers=layers,
        )

        assert actor.character_layers is not None
        assert len(actor.character_layers) == 2
        assert actor.character_layers[0].char == "|"
        assert actor.character_layers[0].offset_x == -0.3

    def test_actor_with_empty_layers_list(self) -> None:
        """Actor should handle an empty layers list."""
        actor = Actor(
            x=5,
            y=5,
            ch="@",
            color=(255, 255, 255),
            name="Test Actor",
            character_layers=[],
        )

        assert actor.character_layers == []

    def test_actor_with_single_layer(self) -> None:
        """Actor should work with a single character layer."""
        layer = CharacterLayer(
            char="*", color=(255, 200, 0), offset_x=0.1, offset_y=-0.1
        )
        actor = Actor(
            x=5,
            y=5,
            ch="@",
            color=(255, 255, 255),
            name="Test Actor",
            character_layers=[layer],
        )

        assert actor.character_layers is not None
        assert len(actor.character_layers) == 1
        assert actor.character_layers[0] is layer


# =============================================================================
# CONTAINER WITH LAYERS TESTS
# =============================================================================


class TestContainerCharacterLayers:
    """Tests for Container actors with character_layers."""

    def test_container_without_layers(self) -> None:
        """Container should work without character_layers (default)."""
        container = Container(x=10, y=10, ch="~", color=(100, 100, 100), name="Box")

        assert container.character_layers is None

    def test_container_with_layers(self) -> None:
        """Container should accept and store character_layers."""
        layers = [
            CharacterLayer(char="[", color=(80, 60, 40), offset_x=-0.2),
            CharacterLayer(char="]", color=(80, 60, 40), offset_x=0.2),
        ]
        container = Container(
            x=10,
            y=10,
            ch="~",
            color=(100, 100, 100),
            name="Fancy Box",
            character_layers=layers,
        )

        assert container.character_layers is not None
        assert len(container.character_layers) == 2


# =============================================================================
# BOOKCASE FACTORY TESTS
# =============================================================================


class TestCreateBookcase:
    """Tests for the create_bookcase factory function."""

    def test_bookcase_has_character_layers(self) -> None:
        """Bookcase should be created with character_layers."""
        bookcase = create_bookcase(x=5, y=5)

        assert bookcase.character_layers is not None
        assert len(bookcase.character_layers) > 0

    def test_bookcase_has_expected_name(self) -> None:
        """Bookcase should have the name 'Bookcase'."""
        bookcase = create_bookcase(x=5, y=5)

        assert bookcase.name == "Bookcase"

    def test_bookcase_has_expected_capacity(self) -> None:
        """Bookcase should have default capacity of 12."""
        bookcase = create_bookcase(x=5, y=5)

        assert bookcase.inventory.capacity == 12

    def test_bookcase_custom_capacity(self) -> None:
        """Bookcase should accept custom capacity."""
        bookcase = create_bookcase(x=5, y=5, capacity=20)

        assert bookcase.inventory.capacity == 20

    def test_bookcase_blocks_movement(self) -> None:
        """Bookcase should block movement."""
        bookcase = create_bookcase(x=5, y=5)

        assert bookcase.blocks_movement is True

    def test_bookcase_has_fallback_glyph(self) -> None:
        """Bookcase should have a fallback glyph for non-layer rendering."""
        bookcase = create_bookcase(x=5, y=5)

        # The fallback glyph should be set (used when layers aren't rendered)
        assert bookcase.ch == "["

    def test_bookcase_layers_have_varied_colors(self) -> None:
        """Bookcase layers should include varied book spine colors."""
        bookcase = create_bookcase(x=5, y=5)

        assert bookcase.character_layers is not None
        # Collect unique colors from layers
        colors = {layer.color for layer in bookcase.character_layers}
        # Should have multiple colors (frame + various book colors)
        assert len(colors) > 1

    def test_bookcase_layers_include_frame_and_books(self) -> None:
        """Bookcase layers should include frame characters and book characters."""
        bookcase = create_bookcase(x=5, y=5)

        assert bookcase.character_layers is not None
        chars = {layer.char for layer in bookcase.character_layers}
        # Should include frame characters ([ and ]) and book characters (|)
        assert "[" in chars  # Left frame bracket
        assert "]" in chars  # Right frame bracket
        assert "|" in chars  # Book spines

    def test_bookcase_uses_scaled_layers(self) -> None:
        """Bookcase should use per-layer scaling for visual composition."""
        bookcase = create_bookcase(x=5, y=5)

        assert bookcase.character_layers is not None
        # Collect all scale values (both x and y)
        scales_x = {layer.scale_x for layer in bookcase.character_layers}
        scales_y = {layer.scale_y for layer in bookcase.character_layers}
        # Should have varied scales - frame at 1.0, books with varied sizes
        assert 1.0 in scales_x or 1.0 in scales_y  # Frame at full size
        assert any(s < 1.0 for s in scales_x) or any(s < 1.0 for s in scales_y)

    def test_bookcase_with_initial_items(self) -> None:
        """Bookcase should accept initial items."""
        from brileta.game.enums import ItemSize
        from brileta.game.items.item_core import Item, ItemType

        item = Item(
            ItemType(name="Old Book", description="A dusty tome", size=ItemSize.NORMAL)
        )
        bookcase = create_bookcase(x=5, y=5, items=[item])

        assert item in bookcase.inventory

    def test_bookcase_position(self) -> None:
        """Bookcase should be placed at specified coordinates."""
        bookcase = create_bookcase(x=15, y=20)

        assert bookcase.x == 15
        assert bookcase.y == 20
