# ModernGL Rendering Performance Optimization Plan

## Current Situation
- **Original Performance**: 700-800 FPS (TCOD+SDL) → 15-19 FPS (Pyglet/ModernGL)
- **After Steps 1-2**: 25-29 FPS (~50% improvement) ✅ **STABLE BASELINE**
- **Current Focus**: Implementing Steps 3-5 for additional performance gains
- **Target**: Still aiming for 300+ FPS to match TCOD performance

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

### ✅ Step 2: Eliminate Individual Vertex Array Allocations (COMPLETED)
**Goal**: Remove the remaining 6,400 numpy array allocations per frame

**What Was Done**:
- ✅ Implemented vectorized color normalization: `fg_colors_norm = fg_colors_raw.astype(np.float32) / 255.0`
- ✅ Pre-calculated background UV coordinates (solid block character)
- ✅ Replaced individual vertex assignments with array slice operations
- ✅ Eliminated 6,400 per-frame color tuple conversions
- ✅ Cached frequently used values to reduce redundant calculations
- ✅ Optimized vertex coordinate calculations

**Performance Result**: 🎉 **~50% improvement: 15-19 FPS → 25-29 FPS**

---

### 🔄 Step 3: Vectorize Color Normalization (NEXT)
**Goal**: Replace per-cell color processing with batch operations

**Current Problem**:
```python
# Done for each cell individually in nested loops
for y_console in range(h):
    for x_console in range(w):
        fg_color_norm = fg_colors_norm[x_console, y_console]
        bg_color_norm = bg_colors_norm[x_console, y_console]
```

**Solution**:
```python
# Process entire arrays at once before the loop
fg_colors_norm = glyph_buffer.data["fg"].astype(np.float32) / 255.0
bg_colors_norm = glyph_buffer.data["bg"].astype(np.float32) / 255.0
# Then use direct array indexing in the loop
```

**Expected Gain**: 10-20% improvement by eliminating redundant per-cell color processing

---

### ✅ Step 4: Enhance GPU Resource Reuse (COMPLETED)
**Goal**: Further optimize VBO/VAO usage patterns

**What Was Done**:
- ✅ Pre-calculated screen coordinates using vectorized `np.arange()` operations
- ✅ Eliminated per-cell multiplication by caching coordinate arrays
- ✅ Optimized memory access patterns with row-wise Y coordinate caching
- ✅ Reduced redundant coordinate calculations in nested loops

**Results**: Micro-optimizations implemented for improved cache efficiency and reduced arithmetic operations

---

### 🔄 Step 5: Cache Rendered Textures (NEXT)
**Goal**: Skip re-rendering when glyph buffer unchanged

**Solution**:
```python
buffer_hash = hash(glyph_buffer.tobytes())
if buffer_hash == self.last_buffer_hash:
    return self.cached_texture
```

**Expected Gain**: 
- **Massive gains** when screen is static (UI panels, paused game)
- **Minimal overhead** when content changes frequently
- Particularly effective for inventory screens, menus, and status panels

---

### ⚠️ Step 6: Glyph Atlas + Shader-Based Coloring (POSTPONED - COMPLEX ISSUES)
**Goal**: Eliminate per-frame vertex generation with color palette system

## DETAILED POST-MORTEM: Step 6 Implementation Attempt

### What We Implemented ✅
1. **Smart Color Quantization System**:
   - Protected 30 base game colors to preserve them exactly
   - 6-bit quantization for lighting gradients (64 levels per channel)
   - Perceptual color distance matching (CIE76-based with 5.0 threshold)
   - LRU eviction system for palette management

2. **Optimized Vertex Format**:
   ```python
   OPTIMIZED_VERTEX_DTYPE = np.dtype([
       ("position", "2f4"),     # (x, y) - 8 bytes
       ("uv", "2f4"),          # (u, v) - 8 bytes  
       ("color_index", "u1"),   # Index - 1 byte
       ("padding", "3u1"),      # Padding - 3 bytes
   ])  # Total: 20 bytes vs original 32 bytes (37.5% reduction)
   ```

3. **Dual-Mode Rendering**:
   - Legacy mode: Direct RGBA colors (working, 25-29 FPS)
   - Optimized mode: Color palette indices (broken)

4. **GPU Shader System**:
   - White/alpha atlas texture conversion (luminance-based)
   - Fragment shader color lookup: `v_color = u_color_palette[in_color_index]`
   - Proper uniform array upload to GPU

### Critical Issues Discovered 🚨

1. **Severe Visual Corruption**:
   - Random flashing colors across the entire screen
   - "Epileptic flashing" - rapidly changing bright colors
   - Complete breakdown of visual coherence
   - Screenshot evidence: `~/Downloads/ss3.jpg`

2. **Root Cause Analysis**:
   - **NOT** a color palette issue (debugging showed correct palette upload)
   - **NOT** color index bounds problems (indices were within 0-255 range)
   - **NOT** color quantization artifacts (base colors preserved correctly)
   
   **Likely Root Cause**: **Vertex Format/VAO Binding Mismatch**
   - VAO format string: `"2f 2f 1u 3x"` (position, uv, color_index, padding)
   - Potential byte alignment issues between CPU and GPU
   - OpenGL expects specific data layouts that we may not be providing
   - The `uint8` color index might not map correctly to `uint` in vertex shader

3. **Technical Debugging Evidence**:
   ```
   Protected 30 base game colors
   Uploading palette: 86 used colors, 256 total
   First 5 used colors: [(1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0), ...]
   ```
   - Color system working correctly (30 base + ~56 lighting variations)
   - Palette upload successful
   - No runtime errors or crashes
   - Problem is in GPU vertex/shader pipeline

### Lessons Learned 📚

**What Worked Well**:
- Color analysis and quantization strategy was sound
- Smart base color preservation approach
- Dual-mode rendering for safe fallback
- Comprehensive debugging and logging

**Critical Mistakes**:
1. **Insufficient Vertex Format Testing**: Should have tested the VAO binding with simple test data first
2. **Complex Multi-System Change**: Combined vertex format + color palette + shader changes in one step
3. **Assumption About GPU Data Layout**: Assumed CPU numpy array layout would map directly to GPU vertex attributes

### Recommended Re-implementation Strategy 💡

**Phase 1: Vertex Format Validation** (Essential before any shader work)
1. Create minimal test with `OPTIMIZED_VERTEX_DTYPE` rendering solid colors
2. Verify VAO binding works correctly: `"2f 2f 1u 3x"`
3. Test with simple color indices (0, 1, 2) to solid colors (red, green, blue)
4. Ensure byte alignment and data interpretation is correct

**Phase 2: Incremental Shader Integration**
1. Start with static 4-color palette instead of 256
2. Test shader uniform array upload with tiny palette
3. Verify color lookup works before expanding to full system

**Phase 3: Alternative Approaches to Consider**
1. **Texture-Based Palette**: Use 256x1 texture instead of uniform arrays (better GPU limits)
2. **Instanced Rendering**: Draw each character as an instance with color attribute
3. **Compute Shader Pre-pass**: Pre-process colors on GPU before vertex stage
4. **Simpler Optimization**: Focus on reducing vertex data size without color palettes

**Phase 4: Risk Mitigation**
- Implement each component in isolation with unit tests
- Maintain visual validation at each step
- Create automated screenshot comparison tests
- Keep fallback modes for every optimization

### Current Decision: Postpone Step 6 ⏸️

**Rationale**:
- Steps 3-5 offer safer, incremental performance gains (targeting 60+ FPS)
- Complex vertex format debugging would require significant time investment
- Current 25-29 FPS baseline is stable and usable
- Better to build momentum with easier wins first

**When to Revisit**:
- After Steps 3-5 implementation and performance measurement
- If we need additional performance beyond what Steps 3-5 provide
- When we have dedicated time for complex graphics debugging
- Consider as a separate "Advanced Optimization" phase

---

## Implementation Progress

### ✅ Completed
- **Step 1**: Applied UITextureRenderer pattern - persistent GPU resources in place
- **Step 2**: Eliminated vertex allocation bottleneck - **50% performance gain (15-19 → 25-29 FPS)**
- **Step 4**: Enhanced GPU resource reuse - coordinate calculation optimizations

### 🔄 Next Steps (Priority Order)
1. **Step 3**: Vectorize color normalization (low risk, moderate gain)
2. **Step 5**: Texture caching system (low risk, high situational gain)
3. **Step 6**: Consider advanced shader optimizations (high risk, high potential gain)

### 📊 Performance Targets
- **Current**: 25-29 FPS (2x improvement)
- **Target after Steps 3-5**: 60+ FPS (4x improvement, acceptable for smooth gameplay)
- **Stretch goal**: 150+ FPS (10x improvement)
- **Ultimate goal**: 300+ FPS (20x improvement, matching TCOD)

## Implementation Notes

### Testing Strategy
- ✅ Updated tests to reflect new persistent resource architecture
- ✅ All 440 tests passing after Step 2
- ✅ Visual validation confirmed for Steps 1-2
- 🔄 Plan automated performance benchmarks for Steps 3-5
- 🔄 Consider screenshot-based visual regression testing

### Risk Mitigation
- ✅ Maintained backward-compatible API throughout
- ✅ Incremental changes with git version control
- ✅ Working fallback modes for all optimizations
- ✅ Comprehensive debugging and logging systems
- 🔄 Focus on low-risk optimizations first

### Current State Summary
- **Stable Baseline**: 25-29 FPS with excellent visual quality
- **Architecture**: Clean, maintainable, extensible rendering pipeline  
- **Next Focus**: Incremental optimizations (Steps 3-5) for additional performance
- **Long-term**: Advanced shader techniques (Step 6) when time permits

### Success Metrics
- **Minimum Success**: 60+ FPS for smooth gameplay (Steps 3-5 target)
- **Good Success**: 150+ FPS competitive performance  
- **Exceptional Success**: 300+ FPS (matching TCOD)
- **Visual Quality**: Must maintain or exceed current visual fidelity
- **Maintainability**: Code must remain clean and debuggable

### Architecture Benefits Achieved
Current rendering pipeline features:
- ✅ Pre-allocated GPU resources (no allocation overhead)
- ✅ Efficient batch operations (vectorized color processing)
- ✅ Persistent VBO/VAO (no creation/destruction overhead)  
- ✅ Backward compatibility (legacy mode always available)
- ✅ Clean separation of concerns (easy to extend)
- ✅ Comprehensive testing (440 tests passing)

The architecture is now well-positioned for both incremental optimizations (Steps 3-5) and future advanced techniques (Step 6 when ready).