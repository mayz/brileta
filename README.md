# Brileta

Brileta is a Python game engine and sandbox for experimenting with game mechanics, graphics, and procedural generation. It has strong roguelike bones - glyph-based rendering, turn-based play, procedural worlds - backed by deep systems.

The current sandbox generates a settlement and drops you into it. Everything is deterministic from a single seed.

This is a personal hobby project. There's no release timeline, no roadmap promises, and no guarantee that anything works.

## Notable Systems

### Shader-Driven Lighting

- Per-tile illumination computed in WGPU fragment shaders, composited over the glyph layer in a separate pass.
- Multiple concurrent point lights with distance falloff and dynamic flicker driven by shader-side noise.
- Directional sun with ray-marched terrain shadows that shift in angle and length with time of day.
- Actors and terrain cast glyph-shaped shadows that shift with sun angle and nearby point lights.
- Sky exposure for indoor/outdoor transitions and emission for glowing surfaces.

### Composable World Generation

- Pipeline of generation layers that each transform a shared context.
- Terrain uses Wave Function Collapse with weighted adjacency rules.
- Buildings placed from templates with configurable footprints.
- Generated regions track properties like sky exposure that feed into the lighting system.
- Each subsystem gets its own isolated RNG stream derived from a master seed, so changes to one system's random consumption don't cascade to others.

### Utility AI with Goals

- Each tick, every available action is scored against a context built from health, threat proximity, escape routes, and other inputs.
- Multi-turn behaviors like fleeing or patrolling create Goals that compete in the same scoring system as one-shot actions.
- Persistence bonus scales with goal progress - an NPC most of the way through fleeing is harder to distract than one that just started.

### Declarative Actions

- Player and NPC actions go through the same pipeline.
- Plans are declarative sequences of movement and action steps, with optional skip-if predicates.
- Each step produces a pure-data intent, dispatched to a specialized executor.
- Pathfinding supports hierarchical cross-region routing via a region graph built during world generation.

### Effects & Presentation

- Sub-tile particle system layered over the glyph grid.
- Directional muzzle flash cones and blood splatter that persists as floor decals.
- Presentation manager staggers NPC action feedback so simultaneous turns read as a sequence.
- Positional audio with distance-based falloff, listener position smoothing, and variant selection.

### C Extensions for Hot Paths

- A shared native extension for A\* pathfinding, symmetric shadowcasting FOV, and Wave Function Collapse.

## Getting Started

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Run `uv sync`.
3. Run `make`.
4. Run the sandbox with `make run`.

## License

Licensed under [AGPL-3.0](LICENSE). See [NOTICE](NOTICE) for third-party attributions.
