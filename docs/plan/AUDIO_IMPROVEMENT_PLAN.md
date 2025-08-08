# Audio System Improvement Plan

## Overview
This document outlines critical issues and improvements needed for the Catley audio/sound system, prioritized by importance and impact. Each item includes specific implementation details to guide development.

## Priority 2: Performance & Architecture Improvements

### 4. Add Spatial Partitioning for Audio
**Problem:** System iterates through ALL actors regardless of distance, causing unnecessary processing overhead.

**Current Code:** `catley/sound/system.py:119` - iterates `for actor in actors`

**Fix:**
Leverage the existing `SpatialHashGrid` from `catley/util/spatial.py`:

```python
# catley/sound/system.py
class SoundSystem:
    def __init__(self, max_concurrent_sounds: int = 16):
        # ... existing init code ...
        self._max_audio_distance: float | None = None
        
    def _calculate_max_audio_distance(self) -> float:
        """Calculate the maximum audible distance across all sound definitions."""
        if self._max_audio_distance is not None:
            return self._max_audio_distance
            
        max_dist = 0.0
        from .definitions import SOUND_DEFINITIONS
        for sound_def in SOUND_DEFINITIONS.values():
            max_dist = max(max_dist, sound_def.max_distance)
        
        # Add buffer for smooth transitions (currently max is 17.0 for campfire)
        self._max_audio_distance = max_dist + 5.0  # ~22.0 tiles
        return self._max_audio_distance

    def update(
        self,
        listener_x: WorldTileCoord,
        listener_y: WorldTileCoord,
        actor_spatial_index: SpatialIndex[Actor],  # NEW PARAMETER
        delta_time: float,
    ) -> None:
        # ... existing listener interpolation code ...
        
        # Query only actors within maximum audio distance
        audio_cull_distance = self._calculate_max_audio_distance()
        nearby_actors = actor_spatial_index.get_in_radius(
            int(self.audio_listener_x),  # Use interpolated position
            int(self.audio_listener_y),
            radius=int(audio_cull_distance)
        )
        
        # Collect active emitters from nearby actors only
        active_emitters: list[tuple[SoundEmitter, WorldTileCoord, WorldTileCoord]] = []
        for actor in nearby_actors:
            if hasattr(actor, "sound_emitters") and actor.sound_emitters:
                # Additional distance check with actual Euclidean distance
                dx = actor.x - self.audio_listener_x
                dy = actor.y - self.audio_listener_y
                if dx * dx + dy * dy <= audio_cull_distance * audio_cull_distance:
                    active_emitters.extend(
                        (emitter, actor.x, actor.y)
                        for emitter in actor.sound_emitters
                        if emitter.active
                    )
```

**Controller Update:**
```python
# catley/controller.py:243
self.sound_system.update(
    self.gw.player.x,
    self.gw.player.y,
    self.gw.actor_spatial_index,  # Pass spatial index
    self.fixed_timestep,
)
```

**Benefits:**
- O(1) spatial hash lookups instead of O(n) iteration over all actors
- Only processes actors within ~22 tiles (vs potentially 100s-1000s)
- Reuses existing, tested `SpatialHashGrid` infrastructure
- Minimal code changes required
- Note: `get_in_radius` uses Chebyshev distance (square), so we add Euclidean check for accuracy

### 5. Separate Positional Audio Processing
**Problem:** Distance calculations and volume updates are intertwined with emitter management.

**Fix:**
Create a new `PositionalAudioProcessor` class:
```python
# New file: catley/sound/positional_audio.py
class PositionalAudioProcessor:
    def calculate_volume(self, emitter_pos, listener_pos, sound_def):
        """Pure function for audio calculations"""
        pass

    def apply_occlusion(self, volume, emitter_pos, listener_pos, world):
        """Check line-of-sight and apply muffling"""
        pass

    def get_pan(self, emitter_pos, listener_pos):
        """Calculate stereo panning based on position"""
        pass
```

### 6. Implement Audio Channel Prioritization
**Problem:** No voice stealing or priority system when channels are exhausted.

**Current Code:** `catley/backends/tcod/audio.py:196-209`

**Fix:**
```python
class TCODAudioBackend:
    def get_channel(self, priority=5):
        # First try to find idle channel
        for channel in self._channels:
            if not channel.is_playing():
                return channel

        # Voice stealing: find lowest priority channel
        lowest_priority = priority
        steal_candidate = None
        for channel in self._channels:
            if channel.current_priority < lowest_priority:
                lowest_priority = channel.current_priority
                steal_candidate = channel

        if steal_candidate:
            steal_candidate.stop()
            return steal_candidate

        return None
```

### 7. Add Audio Occlusion System
**Problem:** Sounds play through walls without any muffling or obstruction.

**Implementation:**
```python
def check_audio_occlusion(self, emitter_x, emitter_y, listener_x, listener_y):
    """Check if sound path is blocked by walls"""
    # Use bresenham line algorithm to check tiles between emitter and listener
    tiles = get_line(emitter_x, emitter_y, listener_x, listener_y)

    occlusion_factor = 1.0
    for tile_x, tile_y in tiles:
        if self.world.is_wall(tile_x, tile_y):
            occlusion_factor *= 0.3  # Each wall reduces volume by 70%

    return occlusion_factor
```

## Priority 3: Data & Configuration

### 8. Move to Data-Driven Sound Definitions
**Problem:** Sound definitions are hardcoded in Python.

**Current:** `catley/sound/definitions.py:47-110`

**Fix:**
Create `assets/sounds/definitions.json`:
```json
{
  "fire_ambient": {
    "layers": [
      {
        "file": "fire_crackle_loop.ogg",
        "volume": 1.0,
        "loop": true
      },
      {
        "file": "fire_pops.ogg",
        "volume": 0.4,
        "loop": false,
        "interval": [1.5, 6.0],
        "pitch_variation": [0.8, 1.6]
      }
    ],
    "base_volume": 0.7,
    "falloff_start": 2.8,
    "max_distance": 17.0,
    "priority": 6,
    "rolloff_curve": "logarithmic"
  }
}
```

## Priority 4: Additional Optimizations

### 9. Implement Audio LOD System
**Problem:** All sounds are processed with same fidelity regardless of distance.

**Fix:**
```python
def get_audio_lod_level(self, distance, max_distance):
    """Determine level of detail based on distance"""
    distance_ratio = distance / max_distance

    if distance_ratio < 0.3:
        return AudioLOD.FULL  # All effects, variations
    elif distance_ratio < 0.6:
        return AudioLOD.MEDIUM  # No pitch variation
    else:
        return AudioLOD.LOW  # Reduced update rate, no variations
```

### 10. Add Debug Visualization
**Problem:** No way to debug audio issues visually.

**Implementation:**
```python
class AudioDebugRenderer:
    def render_audio_debug(self, console, sound_system, camera):
        """Overlay audio information on game view"""
        for playing in sound_system.playing_sounds:
            # Draw circles showing sound falloff ranges
            # Color intensity based on current volume
            # Show emitter positions and active state
            pass
```

## Testing Strategy

### Unit Tests Needed:
- Test new logarithmic falloff calculations
- Test teleportation detection
- Test occlusion calculations
- Test priority-based voice stealing

### Integration Tests:
- Test smooth camera/listener transitions
- Test multiple overlapping sound sources
- Test sound cutoff at max distance
- Test save/load with active sounds

## Migration Notes

1. The removal of `use_interpolated_listener` is backward compatible - just always use interpolation
2. New rolloff curves should be configurable per sound definition
3. Occlusion should be optional (some sounds like UI might not want it)
4. Debug visualization should be togglable via config/hotkey

## Success Metrics

- Sounds should smoothly fade during camera transitions (no pops/clicks)
- Distant sounds should feel naturally quiet, not artificially cut off
- Moving sounds (NPCs walking) should pan smoothly in stereo
- System should handle 50+ simultaneous emitters without performance issues
- Sounds behind walls should be noticeably muffled