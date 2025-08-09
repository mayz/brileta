# Audio System Improvement Plan

## Overview
This document outlines critical issues and improvements needed for the Catley audio/sound system, prioritized by importance and impact. Each item includes specific implementation details to guide development.

## Priority 1: Critical Gameplay Features

### 1. Implement Audio Channel Prioritization
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

### 2. Environmental Audio System (Reverb/Echo)
**Problem:** No reverb or echo modeling - sounds play identically in small rooms, large halls, and outdoors.

**Solution:** Leverage existing MapRegion system that already tracks indoor/outdoor spaces.

**Implementation:**
```python
class EnvironmentalAudio:
    def get_acoustic_environment(self, x, y, game_map):
        """Determine acoustic environment using existing region data"""
        region = game_map.get_region_at((x, y))
        
        if not region:
            return AudioEnvironment.OUTDOOR
        
        # Use sky_exposure to determine indoor/outdoor
        if region.sky_exposure > 0.5:
            return AudioEnvironment.OUTDOOR
        elif region.sky_exposure > 0.0:
            return AudioEnvironment.SEMI_OUTDOOR  # Ruins, covered markets
        
        # For indoor, use region type and size
        room_size = sum(rect.width * rect.height for rect in region.bounds)
        
        if region.region_type == "hallway":
            return AudioEnvironment.CORRIDOR  # Long, narrow reverb
        elif room_size < 20:
            return AudioEnvironment.SMALL_ROOM  # Tight reverb
        elif room_size < 100:
            return AudioEnvironment.MEDIUM_ROOM  # Medium reverb
        else:
            return AudioEnvironment.LARGE_HALL  # Cathedral reverb
    
    def apply_environmental_audio(self, sound, source_env, listener_env):
        """Apply reverb based on source and listener environments"""
        if source_env == listener_env:
            # Same environment - full reverb if indoor
            return apply_reverb_preset(sound, source_env)
        elif source_env.is_indoor and listener_env.is_outdoor:
            # Indoor sound heard from outside - muffled + distant reverb
            return apply_muffled_reverb(sound, source_env)
        elif source_env.is_outdoor and listener_env.is_indoor:
            # Outdoor sound heard from inside - muffled, no reverb
            return apply_muffling(sound, 0.5)
        else:
            # Outdoor to outdoor - no processing
            return sound
```

**Key Benefits:**
- Uses existing region.sky_exposure (0.0-1.0) for indoor/outdoor detection
- Accurate room size from region.bounds
- Special handling for region_type ("hallway", "cave", etc.)
- No duplicate work - region system already tracks all spaces

### 3. Add Audio Occlusion System
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

## Priority 2: Performance & Architecture

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

### 5. Implement Audio LOD System
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

## Priority 3: Developer Experience

### 6. Move to Data-Driven Sound Definitions
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

### 7. Add Debug Visualization
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