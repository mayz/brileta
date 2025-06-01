import math
import random

import numpy as np

from catley import colors

# Usage example:
# particle_system = SubTileParticleSystem(map_width, map_height)
#
# # In combat:
# particle_system.emit_blood_splatter(enemy.x, enemy.y, damage // 2)
#
# Emitting methods:
#   - emit_blood_splatter()      # Pure particles (no background)
#   - emit_muzzle_flash()        # Pure particles (no background)
#   - emit_explosion()           # Particles + background flash
#   - emit_flamethrower_burn()   # Particles + background glow
#   - emit_smoke_cloud()         # Particles + background tint
#   - emit_poison_gas()          # Particles + background tint
#
# # In render loop:
# particle_system.update(delta_time)
# particle_system.render_to_console(self.game_map_console)


class SubParticle:
    def __init__(
        self,
        sub_x: float,
        sub_y: float,
        vel_x: float,
        vel_y: float,
        lifetime: float,
        char: str = "*",
        color: colors.Color = (255, 0, 0),
        bg_color: colors.Color | None = None,
        bg_blend_mode: str = "tint",  # "tint", "overlay", "replace"
    ):
        self.sub_x = sub_x  # Position in sub-tile coordinates
        self.sub_y = sub_y
        self.vel_x = vel_x  # Velocity in sub-tiles per second
        self.vel_y = vel_y
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.char = char
        self.color = color
        self.bg_color = bg_color
        self.bg_blend_mode = bg_blend_mode
        self.gravity = 0.0  # Optional gravity in sub-tiles/sec^2
        self._flash_intensity: float | None = None  # For explosion flash effects

    def update(self, delta_time: float) -> bool:
        """Update particle, return False when dead"""
        # Update position
        self.sub_x += self.vel_x * delta_time
        self.sub_y += self.vel_y * delta_time

        # Apply gravity
        self.vel_y += self.gravity * delta_time

        # Update lifetime
        self.lifetime -= delta_time
        return self.lifetime > 0

    @property
    def intensity(self) -> float:
        """Get current intensity based on remaining lifetime"""
        return self.lifetime / self.max_lifetime

    @property
    def tile_pos(self) -> tuple[int, int]:
        """Get the tile position this particle occupies"""
        return int(self.sub_x), int(self.sub_y)


class SubTileParticleSystem:
    def __init__(self, map_width: int, map_height: int, subdivision: int = 3):
        self.subdivision = subdivision  # 3x3 sub-tiles per tile
        self.sub_width = map_width * subdivision
        self.sub_height = map_height * subdivision
        self.map_width = map_width
        self.map_height = map_height
        self.particles: list[SubParticle] = []

    def add_particle(
        self,
        tile_x: float,
        tile_y: float,
        vel_x: float,
        vel_y: float,
        lifetime: float = 1.0,
        char: str = "*",
        color: colors.Color = (255, 0, 0),
        bg_color: colors.Color | None = None,
        bg_blend_mode: str = "tint",
    ) -> None:
        """Add a particle at tile coordinates with sub-tile velocity"""
        # Convert tile coordinates to sub-tile coordinates
        sub_x = tile_x * self.subdivision + random.uniform(-0.5, 0.5)
        sub_y = tile_y * self.subdivision + random.uniform(-0.5, 0.5)

        # Convert velocity to sub-tile velocity
        sub_vel_x = vel_x * self.subdivision
        sub_vel_y = vel_y * self.subdivision

        particle = SubParticle(
            sub_x,
            sub_y,
            sub_vel_x,
            sub_vel_y,
            lifetime,
            char,
            color,
            bg_color,
            bg_blend_mode,
        )
        self.particles.append(particle)

    def emit_blood_splatter(self, tile_x: float, tile_y: float, intensity: int = 5):
        """Create blood splatter particles (no background effects)"""
        for _ in range(intensity * 3):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2.0)  # tiles per second

            vel_x = math.cos(angle) * speed
            vel_y = math.sin(angle) * speed

            lifetime = random.uniform(0.3, 0.8)
            color = (120 + random.randint(0, 40), 0, 0)  # Red variations
            char = random.choice(["*", ".", ","])

            # No background color - just flying blood particles
            self.add_particle(tile_x, tile_y, vel_x, vel_y, lifetime, char, color)

    def emit_muzzle_flash(
        self, tile_x: float, tile_y: float, direction_x: float, direction_y: float
    ):
        """Create muzzle flash particles (no background effects)"""
        for _ in range(8):  # More particles for visibility
            # Particles in a cone in the direction of fire
            base_angle = math.atan2(direction_y, direction_x)
            spread = random.uniform(-0.3, 0.3)  # Tighter cone
            angle = base_angle + spread
            speed = random.uniform(3.0, 6.0)  # Faster

            vel_x = math.cos(angle) * speed
            vel_y = math.sin(angle) * speed

            lifetime = random.uniform(0.05, 0.15)
            color: colors.Color = (
                255,
                random.randint(200, 255),
                random.randint(0, 100),
            )
            char = random.choice(["+", "*", "x"])

            # No background color - just bright flying particles
            self.add_particle(tile_x, tile_y, vel_x, vel_y, lifetime, char, color)

    def emit_explosion(self, tile_x: float, tile_y: float, radius: int = 3):
        """Create explosion with background flash + flying particles"""

        # 1. Create bright background flash across explosion area
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                distance = (dx * dx + dy * dy) ** 0.5
                if distance <= radius:
                    # Bright flash intensity falls off with distance
                    flash_intensity = 1.0 - (distance / radius)
                    flash_color = (255, 255, 200)  # Bright yellow-white

                    flash_particle = SubParticle(
                        (tile_x + dx) * self.subdivision,
                        (tile_y + dy) * self.subdivision,
                        0,
                        0,  # No movement, just flash
                        lifetime=random.uniform(0.1, 0.2),  # Brief flash
                        char="",  # No character
                        color=(0, 0, 0),  # No foreground
                        bg_color=flash_color,
                        bg_blend_mode="replace",
                    )
                    # Store intensity for flash effect
                    flash_particle._flash_intensity = flash_intensity
                    self.particles.append(flash_particle)

        # 2. Create flying debris particles
        particle_count = radius * 8
        for _ in range(particle_count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1.0, 3.0)

            vel_x = math.cos(angle) * speed
            vel_y = math.sin(angle) * speed

            lifetime = random.uniform(0.4, 1.0)
            # Mix of fire colors and chars
            colors_and_chars: list[tuple[colors.Color, str]] = [
                ((255, 100, 0), "*"),  # Orange
                ((255, 200, 0), "+"),  # Yellow
                ((200, 50, 0), "."),  # Dark red
                ((100, 100, 100), ","),  # Smoke
            ]
            color, char = random.choice(colors_and_chars)

            particle = SubParticle(
                tile_x * self.subdivision,
                tile_y * self.subdivision,
                vel_x * self.subdivision,
                vel_y * self.subdivision,
                lifetime,
                char,
                color,
            )
            particle.gravity = 2.0 * self.subdivision  # Particles fall
            self.particles.append(particle)

    def emit_flamethrower_burn(
        self, tile_x: float, tile_y: float, direction_x: float, direction_y: float
    ):
        """Create flamethrower effect with background burning glow"""
        for _ in range(15):
            # Flame particles in direction of spray
            base_angle = math.atan2(direction_y, direction_x)
            spread = random.uniform(-0.3, 0.3)
            angle = base_angle + spread
            speed = random.uniform(1.0, 3.0)

            # Fire colors for both foreground and background
            fire_colors = [(255, 100, 0), (255, 150, 0), (200, 50, 0)]
            fg_color = random.choice(fire_colors)
            bg_color = tuple(int(c * 0.6) for c in fg_color)  # Dimmer background

            particle = SubParticle(
                tile_x * self.subdivision,
                tile_y * self.subdivision,
                math.cos(angle) * speed * self.subdivision,
                math.sin(angle) * speed * self.subdivision,
                lifetime=random.uniform(0.3, 0.6),
                char=random.choice(["*", "+", "x"]),
                color=fg_color,
                bg_color=bg_color,
                bg_blend_mode="overlay",  # Fire glow
            )
            self.particles.append(particle)

    def emit_smoke_cloud(self, tile_x: float, tile_y: float, radius: int = 2):
        """Create smoke cloud with gray background tinting"""
        for _ in range(radius * 6):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.2, 1.0)  # Slow drift

            vel_x = math.cos(angle) * speed
            vel_y = math.sin(angle) * speed - 0.3  # Slight upward drift

            smoke_gray = random.randint(60, 120)
            fg_color = (smoke_gray, smoke_gray, smoke_gray)
            bg_color = (smoke_gray // 2, smoke_gray // 2, smoke_gray // 2)

            particle = SubParticle(
                tile_x * self.subdivision,
                tile_y * self.subdivision,
                vel_x * self.subdivision,
                vel_y * self.subdivision,
                lifetime=random.uniform(1.0, 2.0),  # Long-lasting
                char=random.choice([".", ",", " "]),
                color=fg_color,
                bg_color=bg_color,
                bg_blend_mode="tint",  # Fills area with smoke
            )
            self.particles.append(particle)

    def emit_poison_gas(self, tile_x: float, tile_y: float, radius: int = 2):
        """Create poison gas cloud with green background tinting"""
        for _ in range(radius * 4):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.1, 0.8)  # Slow spread

            vel_x = math.cos(angle) * speed
            vel_y = math.sin(angle) * speed

            green_intensity = random.randint(40, 100)
            fg_color = (0, green_intensity, 0)
            bg_color = (0, green_intensity // 3, 0)

            particle = SubParticle(
                tile_x * self.subdivision,
                tile_y * self.subdivision,
                vel_x * self.subdivision,
                vel_y * self.subdivision,
                lifetime=random.uniform(2.0, 4.0),  # Very long-lasting
                char=random.choice([".", ",", "~"]),
                color=fg_color,
                bg_color=bg_color,
                bg_blend_mode="tint",  # Contaminates area
            )
            self.particles.append(particle)

    def update(self, delta_time: float):
        """Update all particles"""
        # Update particles and remove dead ones
        self.particles = [p for p in self.particles if p.update(delta_time)]

        # Remove particles that are out of bounds
        self.particles = [
            p
            for p in self.particles
            if 0 <= p.sub_x < self.sub_width and 0 <= p.sub_y < self.sub_height
        ]

    def render_to_console(self, console):
        """Render particles to the console using sub-tile blending"""
        # Create arrays to track particle effects per tile
        particle_intensity = np.zeros((self.map_width, self.map_height))
        particle_chars = {}  # tile_pos -> list of (char, intensity, color)
        particle_bg_effects = {}  # tile_pos -> list of
        # (bg_color, intensity, blend_mode)

        # Accumulate particle effects
        for particle in self.particles:
            # Convert sub-tile position to tile position
            tile_x = int(particle.sub_x // self.subdivision)
            tile_y = int(particle.sub_y // self.subdivision)

            if 0 <= tile_x < self.map_width and 0 <= tile_y < self.map_height:
                # Use flash intensity if available, otherwise normal intensity
                intensity = (
                    particle._flash_intensity
                    if particle._flash_intensity is not None
                    else particle.intensity
                )
                particle_intensity[tile_x, tile_y] += intensity

                # Track character and color info
                tile_pos = (tile_x, tile_y)
                if tile_pos not in particle_chars:
                    particle_chars[tile_pos] = []

                # Only track particles that have actual characters
                if particle.char:  # Skip empty characters
                    particle_chars[tile_pos].append(
                        (particle.char, intensity, particle.color)
                    )

                # Track background effects
                if particle.bg_color:
                    if tile_pos not in particle_bg_effects:
                        particle_bg_effects[tile_pos] = []
                    particle_bg_effects[tile_pos].append(
                        (particle.bg_color, intensity, particle.bg_blend_mode)
                    )

        # Render to console
        for x in range(self.map_width):
            for y in range(self.map_height):
                # Apply background effects first
                if (x, y) in particle_bg_effects:
                    self._apply_background_effects(
                        console, x, y, particle_bg_effects[(x, y)]
                    )

                # Then apply foreground effects
                total_intensity = particle_intensity[x, y]
                if total_intensity > 0 and (x, y) in particle_chars:
                    # Choose the most intense particle for this tile
                    char_data = particle_chars[(x, y)]

                    # Skip if no particles with actual characters
                    # (e.g., explosion flash only)
                    if not char_data:
                        continue

                    char_data.sort(
                        key=lambda x: x[1], reverse=True
                    )  # Sort by intensity

                    best_char, best_intensity, best_color = char_data[0]

                    # Scale intensity (multiple particles can make it brighter)
                    display_intensity = min(total_intensity, 1.0)

                    # Blend color based on intensity
                    final_color = tuple(int(c * display_intensity) for c in best_color)

                    # Don't overwrite important map elements, blend instead
                    current_char = chr(console.rgb["ch"][x, y])
                    if current_char == " ":  # Empty space, can overwrite
                        console.rgb["ch"][x, y] = ord(best_char)
                        console.rgb["fg"][x, y] = final_color
                    else:  # Blend with existing
                        # Just modify the foreground color for blend effect
                        current_fg = console.rgb["fg"][x, y]
                        blended_fg = tuple(
                            min(255, int(current_fg[i] + final_color[i] * 0.3))
                            for i in range(3)
                        )
                        console.rgb["fg"][x, y] = blended_fg

    def _apply_background_effects(self, console, x, y, bg_effects):
        """Apply background color effects to a tile"""
        current_bg = console.rgb["bg"][x, y]

        for bg_color, intensity, blend_mode in bg_effects:
            if blend_mode == "tint":
                # Subtle blend with existing background
                blended_bg = tuple(
                    min(255, int(current_bg[i] + bg_color[i] * intensity * 0.4))
                    for i in range(3)
                )
                console.rgb["bg"][x, y] = blended_bg

            elif blend_mode == "overlay":
                # Stronger overlay effect
                blend_factor = intensity * 0.6
                blended_bg = tuple(
                    int(current_bg[i] * (1 - blend_factor) + bg_color[i] * blend_factor)
                    for i in range(3)
                )
                console.rgb["bg"][x, y] = blended_bg

            elif blend_mode == "replace":
                # Full replacement (for bright flashes)
                console.rgb["bg"][x, y] = tuple(int(c * intensity) for c in bg_color)

            # Update current_bg for next effect in the list
            current_bg = console.rgb["bg"][x, y]
