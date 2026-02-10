# Audio System Improvement Plan

## Overview
This document outlines critical issues and improvements needed for the Brileta audio/sound system, prioritized by importance and impact. Each item includes specific implementation details to guide development.

## Priority 1: Critical Gameplay Features

### 2. Environmental Audio System (Reverb/Echo)
**Problem:** No reverb or echo modeling - sounds play identically in small rooms, large halls, and outdoors.

**Solution:** Use delayed sound copies + frequency filtering to simulate environmental acoustics, leveraging existing MapRegion system.

**Implementation:**
```python
class EnvironmentalAudioProcessor:
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
            return AudioEnvironment.SMALL_ROOM  # Short echo
        elif room_size < 100:
            return AudioEnvironment.MEDIUM_ROOM  # Medium echo
        else:
            return AudioEnvironment.LARGE_HALL  # Long echo

    def get_environmental_effects(self, emitter_env, listener_env):
        """Calculate echo and filtering parameters for environmental audio"""
        if emitter_env == listener_env:
            # Same environment - apply room acoustics
            return self._get_room_acoustics(emitter_env)
        else:
            # Cross-environment - apply transition effects
            return self._get_transition_effects(emitter_env, listener_env)
    
    def _get_room_acoustics(self, environment):
        """Get echo delay, volume, and filtering for room types"""
        acoustics = {
            AudioEnvironment.OUTDOOR: {
                "echo_delay_ms": 0,      # No echo
                "echo_volume": 0.0,      # No echo
                "low_pass_cutoff": 1.0   # Bright sound
            },
            AudioEnvironment.SMALL_ROOM: {
                "echo_delay_ms": 80,     # Short echo
                "echo_volume": 0.25,     # Quiet echo
                "low_pass_cutoff": 0.8   # Slightly muffled
            },
            AudioEnvironment.MEDIUM_ROOM: {
                "echo_delay_ms": 150,    # Medium echo
                "echo_volume": 0.35,     # Moderate echo
                "low_pass_cutoff": 0.7   # More muffled
            },
            AudioEnvironment.LARGE_HALL: {
                "echo_delay_ms": 300,    # Long echo
                "echo_volume": 0.45,     # Strong echo
                "low_pass_cutoff": 0.6   # Cathedral-like muffling
            },
            AudioEnvironment.CORRIDOR: {
                "echo_delay_ms": 200,    # Medium-long echo
                "echo_volume": 0.4,      # Emphasis on reflections
                "low_pass_cutoff": 0.75  # Moderate filtering
            }
        }
        return acoustics.get(environment, acoustics[AudioEnvironment.OUTDOOR])

    def _get_transition_effects(self, source_env, listener_env):
        """Handle audio transitions between different environments"""
        if source_env.is_indoor and listener_env.is_outdoor:
            # Indoor sound heard from outside - muffled + distant echo
            return {
                "echo_delay_ms": 120,
                "echo_volume": 0.2,     # Distant echo
                "low_pass_cutoff": 0.4, # Heavy muffling
                "volume_modifier": 0.7  # Reduce overall volume
            }
        elif source_env.is_outdoor and listener_env.is_indoor:
            # Outdoor sound heard from inside - muffled, no echo
            return {
                "echo_delay_ms": 0,     # No echo indoors
                "echo_volume": 0.0,
                "low_pass_cutoff": 0.5, # Muffled by walls
                "volume_modifier": 0.6  # Walls block sound
            }
        else:
            # Default to no processing for other transitions
            return self._get_room_acoustics(AudioEnvironment.OUTDOOR)
```

**Technical Implementation Details:**

**Echo Simulation via Delayed Playback:**
```python
def play_sound_with_echo(self, sound, base_volume, echo_effects):
    """Play original sound plus delayed echo copy"""
    # Play original sound immediately
    channel1 = self.audio_backend.play_sound(sound, volume=base_volume)
    
    # Schedule echo copy if echo is enabled
    if echo_effects["echo_delay_ms"] > 0 and echo_effects["echo_volume"] > 0:
        # Use audio backend's delayed playback or schedule via timer
        echo_volume = base_volume * echo_effects["echo_volume"]
        self._schedule_delayed_sound(
            sound, 
            delay_ms=echo_effects["echo_delay_ms"],
            volume=echo_volume,
            low_pass=echo_effects["low_pass_cutoff"]
        )
    
    return channel1

def _schedule_delayed_sound(self, sound, delay_ms, volume, low_pass):
    """Schedule a sound to play after a delay with filtering"""
    # Option 1: Use audio backend delay (if available)
    if hasattr(self.audio_backend, 'play_delayed'):
        self.audio_backend.play_delayed(sound, delay_ms, volume, low_pass)
    
    # Option 2: Use game timer system
    else:
        future_time = self.current_time + (delay_ms / 1000.0)
        self._delayed_sounds.append({
            "trigger_time": future_time,
            "sound": sound,
            "volume": volume,
            "low_pass": low_pass
        })
```

**Frequency Filtering Implementation:**
```python
def apply_audio_filtering(self, channel, low_pass_cutoff):
    """Apply frequency filtering to simulate environmental effects"""
    # Option 1: Use SDL2's built-in filtering (preferred)
    if hasattr(channel, 'set_low_pass_filter'):
        channel.set_low_pass_filter(cutoff_frequency=low_pass_cutoff * 22050)
    
    # Option 2: Simulate filtering via volume bands (fallback)
    else:
        # Reduce high frequencies by reducing overall brightness
        filtered_volume = channel.volume * low_pass_cutoff
        channel.set_volume(filtered_volume)
        
        # Add slight pitch reduction to simulate muffling
        if low_pass_cutoff < 0.7:
            pitch_factor = 0.95 + (low_pass_cutoff * 0.05)  # Slight pitch down
            # Apply pitch shift if available
```

**Integration with SoundSystem:**
```python
def _calculate_audio_effects(self, emitter_x, emitter_y, listener_x, listener_y, sound_def, volume_multiplier):
    """Extended version of _calculate_volume with environmental effects"""
    # 1. Calculate base distance volume (existing logic)
    base_volume = self._calculate_distance_volume(
        emitter_x, emitter_y, listener_x, listener_y, sound_def, volume_multiplier
    )
    
    # 2. Get environmental and occlusion effects
    combined_effects = self.audio_effects_manager.calculate_final_effects(
        (emitter_x, emitter_y), (listener_x, listener_y), self.game_map
    )
    
    # 3. Apply volume modifiers
    final_volume = base_volume * combined_effects["volume_modifier"]
    
    # 4. Return volume and effect parameters for audio backend
    return {
        "volume": final_volume,
        "echo_delay_ms": combined_effects["echo_delay_ms"],
        "echo_volume": combined_effects["echo_volume"],
        "low_pass_cutoff": combined_effects["low_pass_cutoff"]
    }
```

**Key Benefits:**
- Uses existing region.sky_exposure (0.0-1.0) for indoor/outdoor detection
- Accurate room size from region.bounds
- Special handling for region_type ("hallway", "cave", etc.)
- No duplicate work - region system already tracks all spaces
- Shared infrastructure between environmental and occlusion effects
- Performance-friendly: effects calculated once per frame, applied as channel parameters
- Extensible: easy to add new effect types (doppler, pitch shifting, etc.)

### 3. Add Audio Occlusion System
**Problem:** Sounds play through walls without any muffling or obstruction.

**Solution:** Use line-of-sight checking with shared muffling infrastructure from environmental audio system.

**Implementation:**
```python
class AudioOcclusionProcessor:
    def get_occlusion_effects(self, emitter_x, emitter_y, listener_x, listener_y, game_map):
        """Calculate occlusion effects when walls block sound path"""
        # Use Bresenham line algorithm to check tiles between emitter and listener
        tiles = self._get_line(emitter_x, emitter_y, listener_x, listener_y)
        
        wall_count = 0
        for tile_x, tile_y in tiles:
            if game_map.is_wall_at((tile_x, tile_y)):
                wall_count += 1
        
        # Each wall adds progressive occlusion
        if wall_count == 0:
            return {
                "volume_modifier": 1.0,     # No occlusion
                "low_pass_cutoff": 1.0,     # No filtering
                "add_muffling": False
            }
        elif wall_count == 1:
            return {
                "volume_modifier": 0.6,     # Single wall: 40% volume reduction
                "low_pass_cutoff": 0.6,     # Light muffling
                "add_muffling": True
            }
        elif wall_count == 2:
            return {
                "volume_modifier": 0.3,     # Double wall: 70% volume reduction
                "low_pass_cutoff": 0.4,     # Heavy muffling
                "add_muffling": True
            }
        else:
            return {
                "volume_modifier": 0.1,     # Multiple walls: 90% volume reduction
                "low_pass_cutoff": 0.2,     # Very heavy muffling
                "add_muffling": True
            }

    def _get_line(self, x0, y0, x1, y1):
        """Bresenham line algorithm to get tiles between two points"""
        tiles = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        n = 1 + dx + dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        error = dx - dy
        
        dx *= 2
        dy *= 2
        
        for _ in range(n):
            tiles.append((int(x), int(y)))
            
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        
        return tiles

# Shared infrastructure with EnvironmentalAudioProcessor
class AudioEffectsManager:
    """Combines environmental and occlusion effects"""
    
    def __init__(self):
        self.env_processor = EnvironmentalAudioProcessor()
        self.occlusion_processor = AudioOcclusionProcessor()
    
    def calculate_final_effects(self, emitter_pos, listener_pos, game_map):
        """Combine environmental and occlusion effects"""
        # Get environmental effects
        emitter_env = self.env_processor.get_acoustic_environment(*emitter_pos, game_map)
        listener_env = self.env_processor.get_acoustic_environment(*listener_pos, game_map)
        env_effects = self.env_processor.get_environmental_effects(emitter_env, listener_env)
        
        # Get occlusion effects
        occlusion_effects = self.occlusion_processor.get_occlusion_effects(
            *emitter_pos, *listener_pos, game_map
        )
        
        # Combine effects (occlusion overrides environmental filtering)
        combined_effects = {
            "echo_delay_ms": env_effects["echo_delay_ms"],
            "echo_volume": env_effects["echo_volume"] * occlusion_effects["volume_modifier"],
            "low_pass_cutoff": min(env_effects["low_pass_cutoff"], 
                                 occlusion_effects["low_pass_cutoff"]),
            "volume_modifier": env_effects.get("volume_modifier", 1.0) * 
                             occlusion_effects["volume_modifier"]
        }
        
        return combined_effects
```

## Priority 2: Performance & Architecture

### 4. Separate Positional Audio Processing
**Problem:** Distance calculations and volume updates are intertwined with emitter management.

**Fix:**
Create a new `PositionalAudioProcessor` class:
```python
# New file: brileta/sound/positional_audio.py
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

**Current:** `brileta/sound/definitions.py:47-110`

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