"""Generic Wave Function Collapse solver.

This module provides a reusable WFC solver that can be used with any pattern set.
The solver is pattern-agnostic - it takes any dict[PatternType, WFCPattern] and
solves the constraint satisfaction problem.

Usage:
    from catley.environment.generators.wfc_solver import WFCSolver, WFCPattern

    # Define patterns with adjacency rules
    patterns = {
        PatternA: WFCPattern(PatternA, TileTypeID.GRASS, valid_neighbors={...}),
        PatternB: WFCPattern(PatternB, TileTypeID.DIRT, valid_neighbors={...}),
    }

    # Create and run solver
    solver = WFCSolver(width, height, patterns, rng)
    result = solver.solve()  # Returns grid of pattern IDs

Performance Optimizations:
    This implementation uses three key optimizations:

    1. Bitset representation: Each cell's possibilities are stored as a uint8 bitmask
       in a numpy array. For 4 patterns, bit 0 = pattern 0, bit 1 = pattern 1, etc.
       Set operations become fast bitwise operations: & for intersection, popcount
       for cardinality.

    2. Precomputed propagation masks: For each direction, we precompute a uint8 mask
       that maps "current cell possibilities" to "valid neighbor possibilities".
       This enables fast constraint propagation without iterating over patterns.

    3. Native solve loop: The collapse/selection hot loop runs in C for production
       execution, while this module keeps constraint setup and pattern mapping in Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from catley.environment.tile_types import TileTypeID
from catley.util.rng import RNG

try:
    from catley.util._native import (
        WFCContradictionError as _NativeWFCContradictionError,
    )
    from catley.util._native import wfc_solve as _c_wfc_solve
except ImportError as exc:  # pragma: no cover - fails fast by design
    raise ImportError(
        "catley.util._native is required. "
        "Build native extensions with `make` (or `uv pip install -e .`)."
    ) from exc


class WFCContradiction(Exception):
    """Raised when WFC reaches an unsolvable state.

    This occurs when constraint propagation eliminates all possibilities
    for a cell, meaning no valid solution exists with the current constraints.
    """

    pass


@dataclass
class WFCPattern[PatternType]:
    """A single WFC tile pattern with adjacency rules.

    Attributes:
        pattern_id: Unique identifier for this pattern.
        tile_type: The tile type this pattern maps to when rendered.
        weight: Relative probability weight for selection (higher = more common).
        valid_neighbors: Dict mapping direction ("N", "E", "S", "W") to sets of
            pattern IDs that can be adjacent in that direction.
    """

    pattern_id: PatternType
    tile_type: TileTypeID
    weight: float = 1.0
    valid_neighbors: dict[str, set[PatternType]] = field(default_factory=dict)


# Direction utilities
DIRECTIONS = ["N", "E", "S", "W"]
OPPOSITE_DIR = {"N": "S", "E": "W", "S": "N", "W": "E"}
DIR_OFFSETS = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}

# Precomputed popcount lookup table for uint8 values (0-255)
_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount(mask: int) -> int:
    """Count the number of set bits in a uint8 mask."""
    return _POPCOUNT_TABLE[mask]


class WFCSolver[PatternType]:
    """Core Wave Function Collapse solver with optimized bitset representation.

    This solver implements the WFC algorithm for constraint propagation:
    1. Initialize all cells with all possible patterns (as bitmasks)
    2. Find the cell with minimum entropy using a priority queue
    3. Collapse that cell to a single pattern (weighted random choice)
    4. Propagate constraints to neighbors using bitwise operations
    5. Repeat until all cells are collapsed or contradiction occurs

    The solver is generic over the pattern type, allowing it to work with
    any pattern set (terrain, settlements, dungeons, etc.).
    """

    def __init__(
        self,
        width: int,
        height: int,
        patterns: dict[PatternType, WFCPattern[PatternType]],
        rng: RNG,
    ):
        """Initialize the WFC solver.

        Args:
            width: Grid width in cells.
            height: Grid height in cells.
            patterns: Dict mapping pattern IDs to WFCPattern definitions.
            rng: Random number generator for deterministic results.
        """
        self.width = width
        self.height = height
        self.patterns = patterns
        self.rng = rng

        # Build mapping from pattern IDs to bit indices (0, 1, 2, ...)
        self.pattern_ids = list(patterns.keys())
        self.num_patterns = len(self.pattern_ids)

        if self.num_patterns > 8:
            raise ValueError(
                f"WFCSolver supports at most 8 patterns, got {self.num_patterns}. "
                "Extend to uint16/uint64 for more patterns."
            )

        # Map pattern_id -> bit index
        self.pattern_to_bit: dict[PatternType, int] = {
            pid: i for i, pid in enumerate(self.pattern_ids)
        }
        # Map bit index -> pattern_id
        self.bit_to_pattern: dict[int, PatternType] = dict(enumerate(self.pattern_ids))

        # Pattern weights indexed by bit position
        self.pattern_weights = np.array(
            [patterns[pid].weight for pid in self.pattern_ids], dtype=np.float64
        )

        # Initial mask with all patterns possible (e.g., 0b1111 for 4 patterns)
        self.all_patterns_mask = (1 << self.num_patterns) - 1

        # Wave: uint8 bitmask per cell. Shape is (width, height).
        # Each cell stores which patterns are still possible.
        self.wave = np.full((width, height), self.all_patterns_mask, dtype=np.uint8)

        # Track which cells are collapsed (have exactly one pattern)
        self.collapsed = np.zeros((width, height), dtype=bool)

        # Precompute propagation rules as bitmasks
        self._precompute_propagation_masks()

    def _precompute_propagation_masks(self) -> None:
        """Precompute propagation masks for fast constraint propagation.

        For each direction, we build a lookup table: given a cell's current
        possibility mask, what possibilities are valid for the neighbor?

        This turns the inner propagation loop from O(num_patterns) to O(1).
        """
        # propagation_masks[direction][current_mask] = valid_neighbor_mask
        self.propagation_masks: dict[str, np.ndarray] = {}

        for direction in DIRECTIONS:
            # Build lookup table for all possible 256 input masks
            lookup = np.zeros(256, dtype=np.uint8)

            for current_mask in range(256):
                valid_neighbor = 0
                # For each pattern that might be in current cell
                for bit_idx in range(self.num_patterns):
                    if current_mask & (1 << bit_idx):
                        # This pattern is possible - add its valid neighbors
                        pid = self.bit_to_pattern[bit_idx]
                        pattern = self.patterns[pid]
                        for neighbor_pid in pattern.valid_neighbors.get(direction, ()):
                            neighbor_bit = self.pattern_to_bit[neighbor_pid]
                            valid_neighbor |= 1 << neighbor_bit

                lookup[current_mask] = valid_neighbor

            self.propagation_masks[direction] = lookup

    def _propagate(self, start_x: int, start_y: int) -> None:
        """Propagate constraints from a cell to its neighbors.

        Uses precomputed propagation masks for O(1) constraint lookup per cell.
        """
        stack = [(start_x, start_y)]
        in_stack = {(start_x, start_y)}

        iterations = 0
        max_iterations = self.width * self.height * 10

        while stack:
            iterations += 1
            if iterations >= max_iterations:
                raise WFCContradiction("Propagation exceeded maximum iterations")

            x, y = stack.pop()
            in_stack.discard((x, y))

            current_mask = self.wave[x, y]

            for direction in DIRECTIONS:
                dx, dy = DIR_OFFSETS[direction]
                nx, ny = x + dx, y + dy

                # Bounds check
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue

                neighbor_mask = self.wave[nx, ny]

                # Skip if already collapsed to single pattern
                if _popcount(neighbor_mask) <= 1:
                    continue

                # Use precomputed lookup to find valid neighbor patterns
                valid_for_neighbor = self.propagation_masks[direction][current_mask]

                # Apply constraint
                new_mask = neighbor_mask & valid_for_neighbor

                if new_mask != neighbor_mask:
                    if new_mask == 0:
                        raise WFCContradiction(
                            f"No valid patterns at ({nx}, {ny}) after propagation"
                        )

                    self.wave[nx, ny] = new_mask

                    if _popcount(new_mask) == 1:
                        self.collapsed[nx, ny] = True

                    if (nx, ny) not in in_stack:
                        stack.append((nx, ny))
                        in_stack.add((nx, ny))

    def constrain_cell(self, x: int, y: int, allowed: set[PatternType]) -> None:
        """Constrain a cell to only allow specific patterns.

        This is useful for applying boundary conditions or seeding specific
        patterns before solving. The constraint is immediately propagated
        to neighboring cells.

        Args:
            x: Cell x coordinate.
            y: Cell y coordinate.
            allowed: Set of pattern IDs that are allowed for this cell.
        """
        # Convert set to bitmask
        allowed_mask = 0
        for pid in allowed:
            if pid in self.pattern_to_bit:
                allowed_mask |= 1 << self.pattern_to_bit[pid]

        old_mask = self.wave[x, y]
        new_mask = old_mask & allowed_mask

        if new_mask == 0:
            raise WFCContradiction(f"No valid patterns at ({x}, {y}) after constraint")

        self.wave[x, y] = new_mask

        if _popcount(new_mask) == 1:
            self.collapsed[x, y] = True

        # Propagate if possibilities changed
        if new_mask != old_mask:
            self._propagate(x, y)

    def constrain_cells(self, cells: list[tuple[int, int, set[PatternType]]]) -> None:
        """Constrain multiple cells and propagate constraints.

        Args:
            cells: List of (x, y, allowed_patterns) tuples.
        """
        changed_cells: list[tuple[int, int]] = []

        for x, y, allowed in cells:
            # Convert set to bitmask
            allowed_mask = 0
            for pid in allowed:
                if pid in self.pattern_to_bit:
                    allowed_mask |= 1 << self.pattern_to_bit[pid]

            old_mask = self.wave[x, y]
            new_mask = old_mask & allowed_mask

            if new_mask == 0:
                # Fallback to first allowed pattern
                for pid in allowed:
                    if pid in self.pattern_to_bit:
                        new_mask = 1 << self.pattern_to_bit[pid]
                        break

            if new_mask != old_mask:
                self.wave[x, y] = new_mask
                changed_cells.append((x, y))
                if _popcount(new_mask) == 1:
                    self.collapsed[x, y] = True

        self._batch_propagate(changed_cells)

    def _batch_propagate(self, cells: list[tuple[int, int]]) -> None:
        """Propagate constraints from multiple cells efficiently."""
        if not cells:
            return

        stack = list(cells)
        in_stack = set(cells)

        iterations = 0
        max_iterations = self.width * self.height * 10

        while stack and iterations < max_iterations:
            iterations += 1
            x, y = stack.pop()
            in_stack.discard((x, y))

            current_mask = self.wave[x, y]

            for direction in DIRECTIONS:
                dx, dy = DIR_OFFSETS[direction]
                nx, ny = x + dx, y + dy

                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue

                neighbor_mask = self.wave[nx, ny]

                if _popcount(neighbor_mask) <= 1:
                    continue

                valid_for_neighbor = self.propagation_masks[direction][current_mask]
                new_mask = neighbor_mask & valid_for_neighbor

                if new_mask != neighbor_mask:
                    if new_mask == 0:
                        raise WFCContradiction(
                            f"No valid patterns at ({nx}, {ny}) after propagation"
                        )

                    self.wave[nx, ny] = new_mask

                    if _popcount(new_mask) == 1:
                        self.collapsed[nx, ny] = True

                    if (nx, ny) not in in_stack:
                        stack.append((nx, ny))
                        in_stack.add((nx, ny))

        if iterations >= max_iterations:
            raise WFCContradiction("Propagation exceeded maximum iterations")

    def _solve_native(self) -> list[list[PatternType]]:
        """Solve using the native C extension and map bit indices to pattern IDs."""
        propagation_masks = np.zeros((4, 256), dtype=np.uint8)
        for direction_idx, direction in enumerate(DIRECTIONS):
            propagation_masks[direction_idx] = self.propagation_masks[direction]

        native_result = _c_wfc_solve(
            self.width,
            self.height,
            self.num_patterns,
            propagation_masks,
            self.pattern_weights,
            np.ascontiguousarray(self.wave, dtype=np.uint8),
            self.rng.getrandbits(64),
        )

        if len(native_result) != self.width:
            raise WFCContradiction(
                f"Native solver returned invalid width: {len(native_result)}"
            )

        result: list[list[PatternType]] = []
        for x, bit_column in enumerate(native_result):
            if len(bit_column) != self.height:
                raise WFCContradiction(
                    f"Native solver returned invalid height at x={x}: {len(bit_column)}"
                )

            column: list[PatternType] = []
            for y, bit_idx in enumerate(bit_column):
                if not (0 <= bit_idx < self.num_patterns):
                    raise WFCContradiction(
                        f"Native solver returned invalid bit index {bit_idx} at "
                        f"({x}, {y})"
                    )

                self.wave[x, y] = 1 << bit_idx
                self.collapsed[x, y] = True
                column.append(self.bit_to_pattern[bit_idx])

            result.append(column)

        return result

    def solve(self) -> list[list[PatternType]]:
        """Run the WFC algorithm to completion.

        Uses the native solver for production execution.
        """
        try:
            return self._solve_native()
        except _NativeWFCContradictionError as exc:
            raise WFCContradiction(str(exc)) from exc

    # Legacy API compatibility: expose wave as sets for tests that access it directly
    @property
    def wave_as_sets(self) -> list[list[set[PatternType]]]:
        """Convert the internal bitmask wave to sets for debugging/testing."""
        result: list[list[set[PatternType]]] = []
        for x in range(self.width):
            column: list[set[PatternType]] = []
            for y in range(self.height):
                mask = self.wave[x, y]
                patterns: set[PatternType] = set()
                for bit_idx in range(self.num_patterns):
                    if mask & (1 << bit_idx):
                        patterns.add(self.bit_to_pattern[bit_idx])
                column.append(patterns)
            result.append(column)
        return result
