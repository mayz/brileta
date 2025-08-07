# Audio System Improvement Plan

## Overview
This document outlines critical issues and improvements needed for the Catley audio/sound system, prioritized by importance and impact. Each item includes specific implementation details to guide development.

## Priority 1: Critical Fixes

### 3. Improve Listener Interpolation
**Problem:** Fixed-duration interpolation (1 second) creates unnatural movement, especially during teleportation.

**Current Code:** `catley/sound/system.py:83-117`

**Fix:**
```python
class SoundSystem:
    def __init__(self):
        # Add velocity-based smoothing parameters
        self.listener_smoothing_factor = 0.15  # 0-1, higher = less smoothing
        self.teleport_threshold = 10.0  # Distance to consider as teleportation
        self.previous_listener_x = 0.0
        self.previous_listener_y = 0.0

    def update(self, listener_x, listener_y, actors, delta_time):
        # Detect teleportation
        if self._listener_initialized:
            movement_distance = math.sqrt(
                (listener_x - self.previous_listener_x) ** 2 +
                (listener_y - self.previous_listener_y) ** 2
            )

            if movement_distance > self.teleport_threshold:
                # Instant update for teleportation
                self.audio_listener_x = listener_x
                self.audio_listener_y = listener_y
            else:
                # Smooth interpolation using exponential smoothing
                alpha = 1.0 - math.exp(-delta_time * 5.0)  # 5.0 = smoothing speed
                self.audio_listener_x += (listener_x - self.audio_listener_x) * alpha
                self.audio_listener_y += (listener_y - self.audio_listener_y) * alpha

        self.previous_listener_x = listener_x
        self.previous_listener_y = listener_y
```

## Priority 2: Architecture Improvements

### 4. Separate Positional Audio Processing
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

### 5. Implement Audio Channel Prioritization
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

### 6. Add Audio Occlusion System
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

### 7. Move to Data-Driven Sound Definitions
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

## Priority 4: Performance Optimizations

### 8. Add Spatial Partitioning
**Problem:** System checks all actors/emitters regardless of distance.

**Fix:**
```python
class SoundSystem:
    def __init__(self):
        self.audio_cull_distance = 50.0  # Don't process sounds beyond this

    def update(self, listener_x, listener_y, actors, delta_time):
        # Early culling based on distance
        nearby_actors = [
            actor for actor in actors
            if abs(actor.x - listener_x) <= self.audio_cull_distance
            and abs(actor.y - listener_y) <= self.audio_cull_distance
        ]
        # Process only nearby_actors...
```

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