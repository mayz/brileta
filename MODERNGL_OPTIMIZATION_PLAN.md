# ModernGL Rendering Performance Optimization Plan

## Current Situation
- **Performance Issue**: Dropped from 700-800 FPS (TCOD+SDL) to 15-19 FPS (Pyglet/ModernGL)
- **Main Bottleneck**: `TextureRenderer._encode_glyph_buffer_to_vertices()` at line 125
- **Root Cause**: Creating 6,400 numpy arrays and 38,400 individual assignments per frame

## Optimization Strategy

### ✅ Step 1: Apply UITextureRenderer Pattern to TextureRenderer (COMPLETED)
**Goal**: Use the existing proven pattern from `UITextureRenderer` 

**What Was Done**:
- ✅ Pre-allocated CPU vertex buffer (`self.cpu_vertex_buffer`) with max capacity (200x100 tiles)
- ✅ Created persistent VBO/VAO during initialization instead of per-frame
- ✅ Modified `_encode_glyph_buffer_to_vertices()` to write directly to pre-allocated buffer
- ✅ Eliminated temporary VBO/VAO creation in `render()` method
- ✅ Updated tests to reflect new persistent resource architecture
- ✅ Maintained backward-compatible API

**Results**: Infrastructure in place for major optimizations. Eliminated VBO/VAO churn.

---

### Step 2: Eliminate Individual Vertex Array Allocations
**Goal**: Remove the remaining 6,400 numpy array allocations per frame

**Current Problem** (still exists):
```python
# These are still happening in the nested loops:
bg_vertices = np.zeros(6, dtype=VERTEX_DTYPE)  # Called 3,200 times  
fg_vertices = np.zeros(6, dtype=VERTEX_DTYPE)  # Called 3,200 times
```

**Solution**: 
- Replace individual `np.zeros(6, dtype=VERTEX_DTYPE)` calls with direct buffer writes
- Use vectorized operations where possible
- Eliminate intermediate vertex arrays entirely

**Expected Gain**: Major performance improvement - this addresses the core bottleneck

---

### Step 3: Implement Glyph Atlas + Shader-Based Coloring
**Goal**: Eliminate per-frame vertex generation entirely

**Current Problem**: 
- Generating geometry for each glyph every frame
- Encoding colors into vertex data

**Solution**:
- Create texture atlas with white/alpha glyphs
- Use fragment shader to multiply glyph texture by color
- Pass colors as vertex attributes instead of encoding in vertices
- Reference existing games/UI frameworks that use this pattern

**Technical Notes**:
- Store glyphs as white textures with alpha channel
- Fragment shader: `gl_FragColor = texture(glyph_atlas, uv) * vertex_color`
- Vertex format becomes simpler: position + UV + color

**Expected Gain**: 2-5x performance improvement, potentially reaching TCOD performance

---

### Step 4: Vectorize Color Normalization
**Goal**: Replace per-cell color processing with batch operations

**Current Problem**:
```python
# Done for each cell individually
fg_color_norm = tuple(c / 255.0 for c in fg_color_rgba)
bg_color_norm = tuple(c / 255.0 for c in bg_color_rgba)
```

**Solution**:
```python
# Process entire arrays at once
fg_colors = glyph_buffer["fg"].astype(np.float32) / 255.0
bg_colors = glyph_buffer["bg"].astype(np.float32) / 255.0
```

**Note**: May become simpler or unnecessary after Step 3

---

### Step 5: Reuse GPU Resources
**Goal**: Prevent VBO/VAO creation/destruction overhead

**Current Problem**:
```python
temp_vbo = self.mgl_context.buffer(vertex_data.tobytes())
temp_vao = self.mgl_context.vertex_array(...)
# ... render ...
temp_vao.release()
temp_vbo.release()
```

**Solution**:
```python
# Initialize once
self.vbo = ctx.buffer(reserve=size, dynamic=True)
self.vao = ctx.vertex_array(...)

# Update per frame
self.vbo.write(vertex_data.tobytes())
```

**Note**: Should be largely handled by Step 1

---

### Step 6: Cache Rendered Textures
**Goal**: Skip re-rendering when glyph buffer unchanged

**Solution**:
```python
buffer_hash = hash(glyph_buffer.tobytes())
if buffer_hash == self.last_buffer_hash:
    return self.cached_texture
```

**Expected Gain**: Massive gains when screen is static, minimal overhead when changing

---

## Implementation Progress

### ✅ Completed
- **Step 1**: Applied UITextureRenderer pattern - persistent GPU resources in place

### 🔄 Next Steps
- **Step 2**: Eliminate remaining 6,400 array allocations (major bottleneck)
- **Step 3**: Implement glyph atlas + shader coloring (architectural improvement)
- **Steps 4-6**: Final optimizations and caching

## Implementation Notes

### Testing Strategy
- ✅ Updated tests to reflect new persistent resource architecture
- ✅ All 440 tests passing after Step 1
- 🔄 Profile after each remaining step to measure gains

### Risk Mitigation
- ✅ Maintained backward-compatible API
- ✅ Incremental changes with version control
- ✅ Visual correctness maintained

### Success Metrics
- Target: 300+ FPS (competitive with TCOD)
- Minimum acceptable: 60+ FPS for smooth gameplay
- Stretch goal: 500+ FPS (better than TCOD)

### Architecture Benefits
After optimization, the rendering pipeline will:
- Pre-allocate all GPU resources
- Use efficient batch operations
- Leverage GPU for color calculations
- Support easy extension (effects, animations)
- Match or exceed TCOD performance while maintaining flexibility