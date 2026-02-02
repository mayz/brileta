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
    This implementation uses three key optimizations to achieve O(n^2 log n) complexity:

    1. Bitset representation: Each cell's possibilities are stored as a uint8 bitmask
       in a numpy array. For 4 patterns, bit 0 = pattern 0, bit 1 = pattern 1, etc.
       Set operations become fast bitwise operations: & for intersection, popcount
       for cardinality.

    2. Priority queue for min-entropy: Instead of scanning all cells O(n^2) per
       collapse, we maintain a heap of (entropy, x, y) tuples. When propagation
       changes a cell, we push a new entry. Stale entries are skipped.

    3. Precomputed propagation masks: For each direction, we precompute a uint8 mask
       that maps "current cell possibilities" to "valid neighbor possibilities".
       This enables fast constraint propagation without iterating over patterns.
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field

import numpy as np

from catley.environment.tile_types import TileTypeID


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
        rng: random.Random,
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

        # Priority queue for min-entropy: (entropy, tiebreaker, x, y)
        # The tiebreaker ensures deterministic ordering for equal entropies
        self._entropy_heap: list[tuple[float, int, int, int]] = []
        self._heap_counter = 0  # Monotonic counter for tiebreaking

        # Initialize entropy heap with all cells
        self._initialize_entropy_heap()

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

    def _initialize_entropy_heap(self) -> None:
        """Initialize the priority queue with all uncollapsed cells."""
        for x in range(self.width):
            for y in range(self.height):
                entropy = self._calculate_entropy(x, y)
                heapq.heappush(
                    self._entropy_heap,
                    (entropy, self._heap_counter, x, y),
                )
                self._heap_counter += 1

    def _push_entropy(self, x: int, y: int) -> None:
        """Push a cell onto the entropy heap with its current entropy."""
        entropy = self._calculate_entropy(x, y)
        heapq.heappush(
            self._entropy_heap,
            (entropy, self._heap_counter, x, y),
        )
        self._heap_counter += 1

    def _calculate_entropy(self, x: int, y: int) -> float:
        """Calculate Shannon entropy for a cell based on remaining possibilities.

        Uses the bitset representation for fast popcount.
        """
        mask = self.wave[x, y]
        count = _popcount(mask)

        if count <= 1:
            return 0.0

        # Extract weights for patterns that are still possible
        total_weight = 0.0
        entropy = 0.0

        for bit_idx in range(self.num_patterns):
            if mask & (1 << bit_idx):
                total_weight += self.pattern_weights[bit_idx]

        if total_weight == 0:
            return 0.0

        for bit_idx in range(self.num_patterns):
            if mask & (1 << bit_idx):
                weight = self.pattern_weights[bit_idx]
                if weight > 0:
                    p_normalized = weight / total_weight
                    entropy -= p_normalized * np.log(p_normalized)

        # Add small random noise to break ties deterministically
        entropy += self.rng.random() * 0.001
        return entropy

    def _find_min_entropy_cell(self) -> tuple[int, int] | None:
        """Find the uncollapsed cell with minimum entropy using heap.

        Returns:
            (x, y) tuple of the cell with lowest entropy, or None if all collapsed.

        Raises:
            WFCContradiction: If any cell has no valid patterns.
        """
        while self._entropy_heap:
            entropy, _counter, x, y = heapq.heappop(self._entropy_heap)

            mask = self.wave[x, y]
            count = _popcount(mask)

            if count == 0:
                raise WFCContradiction(f"No valid patterns at ({x}, {y})")

            if count == 1:
                # Already collapsed, skip this stale entry
                continue

            # Verify this isn't a stale entry by checking current entropy
            # (we use a simpler check: if the cell has more than 1 possibility)
            current_entropy = self._calculate_entropy(x, y)

            # Allow small tolerance for floating point comparison
            if abs(current_entropy - entropy) > 0.01:
                # Stale entry - the cell's entropy changed since we pushed it
                # Re-push with current entropy and continue
                self._push_entropy(x, y)
                continue

            return (x, y)

        return None

    def _weighted_choice(self, mask: int) -> int:
        """Choose a pattern bit from mask using weighted random selection.

        Args:
            mask: Bitmask of possible patterns.

        Returns:
            Bit index of the chosen pattern.
        """
        if mask == 0:
            raise WFCContradiction("No patterns to choose from")

        # Collect possible patterns and their weights
        bits = []
        weights = []
        for bit_idx in range(self.num_patterns):
            if mask & (1 << bit_idx):
                bits.append(bit_idx)
                weights.append(self.pattern_weights[bit_idx])

        total = sum(weights)

        if total == 0:
            return self.rng.choice(bits)

        r = self.rng.random() * total
        cumulative = 0.0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return bits[i]

        return bits[-1]

    def _collapse_cell(self, x: int, y: int) -> None:
        """Collapse a cell to a single pattern using weighted random choice."""
        mask = self.wave[x, y]
        chosen_bit = self._weighted_choice(mask)
        self.wave[x, y] = 1 << chosen_bit
        self.collapsed[x, y] = True

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
                    else:
                        # Push updated entropy to heap
                        self._push_entropy(nx, ny)

                    if (nx, ny) not in in_stack:
                        stack.append((nx, ny))
                        in_stack.add((nx, ny))

    def _is_fully_collapsed(self) -> bool:
        """Check if all cells have been collapsed to a single pattern.

        Uses vectorized popcount for efficiency.
        """
        # Check all cells have exactly one bit set
        popcounts = _POPCOUNT_TABLE[self.wave]
        return bool(np.all(popcounts == 1))

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

    def solve(self) -> list[list[PatternType]]:
        """Run the WFC algorithm to completion.

        Returns:
            2D grid (list of columns) where each cell contains the collapsed pattern ID.

        Raises:
            WFCContradiction: If no valid solution can be found.
        """
        max_iterations = self.width * self.height * 2
        iterations = 0

        while not self._is_fully_collapsed():
            iterations += 1
            if iterations >= max_iterations:
                raise WFCContradiction("Solve exceeded maximum iterations")

            cell = self._find_min_entropy_cell()
            if cell is None:
                break

            x, y = cell
            self._collapse_cell(x, y)
            self._propagate(x, y)

        # Convert bitmask wave back to pattern IDs
        result: list[list[PatternType]] = []
        for x in range(self.width):
            column: list[PatternType] = []
            for y in range(self.height):
                mask = self.wave[x, y]
                if _popcount(mask) != 1:
                    raise WFCContradiction(
                        f"Cell ({x}, {y}) not collapsed: mask={bin(mask)}"
                    )
                # Find the single set bit
                for bit_idx in range(self.num_patterns):
                    if mask & (1 << bit_idx):
                        column.append(self.bit_to_pattern[bit_idx])
                        break
            result.append(column)

        return result

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
