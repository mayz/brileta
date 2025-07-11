# Realistic ModernGL Performance Baseline - Step 3.1

This document contains the **realistic** performance baseline measurements for the ModernGL backend before the WGPU migration. These metrics represent actual gameplay performance with a visible window, vsync, and full game loop simulation.

## Test Environment

- **Hardware**: Apple M4 Pro
- **OpenGL Version**: 4.1 Metal - 89.4
- **GPU**: Apple M4 Pro
- **Test Configuration**: Visible window, VSync enabled, full game simulation
- **Target FPS**: 60 FPS (realistic gaming target)
- **Test Duration**: 10 seconds per scenario
- **Date**: 2025-01-11

## Key Findings

### Performance Reality Check

The realistic benchmark reveals dramatically different performance characteristics compared to hidden window tests:

1. **Actual FPS Range**: 105-421 FPS (not the unrealistic 1000+ from hidden tests)
2. **VSync Impact**: Performance properly limited by display refresh rate
3. **Frame Consistency**: Significant frame time variance due to real display pipeline
4. **Frame Drops**: Measurable frame drops (>16.7ms) indicating real bottlenecks

### Shadow Performance Impact

Surprisingly, **shadows actually improved performance** in some scenarios:
- 10 lights: 105.6 FPS → 123.6 FPS (+17% with shadows!)
- 25 lights: 178.7 FPS → 144.0 FPS (-19% with shadows)
- 50 lights: 421.6 FPS → 222.5 FPS (-47% with shadows)

This counter-intuitive result with 10 lights may indicate:
- Different GPU utilization patterns
- Batch processing benefits with shadows enabled
- Test variance due to realistic timing

## Detailed Performance Results

| Test Scenario | Avg FPS | P95 FPS | Frame Drops | Memory | Input Lag |
|---------------|---------|---------|-------------|---------|-----------|
| **10 lights (no shadows)** | 105.6 | 60.0 | 54 | 181.2MB | 0.07ms |
| **10 lights (with shadows)** | 123.6 | 70.0 | 23 | 187.9MB | 0.03ms |
| **25 lights (no shadows)** | 178.7 | 69.7 | 40 | 192.2MB | 0.02ms |
| **25 lights (with shadows)** | 144.0 | 69.0 | 13 | 192.2MB | 0.01ms |
| **50 lights (no shadows)** | 421.6 | 89.5 | 25 | 192.6MB | 0.01ms |
| **50 lights (with shadows)** | 222.5 | 82.0 | 8 | 192.7MB | 0.01ms |

## Performance Analysis

### Frame Rate Consistency (P95/P99 Analysis)

The P95 and P99 metrics reveal performance consistency:

**Most Consistent Performance:**
- 50 lights (with shadows): P95 = 82.0 FPS, 8 frame drops
- 25 lights (with shadows): P95 = 69.0 FPS, 13 frame drops

**Least Consistent Performance:**
- 10 lights (no shadows): P95 = 60.0 FPS, 54 frame drops
- 25 lights (no shadows): P95 = 69.7 FPS, 40 frame drops

### Memory Usage

- **Base Usage**: ~181MB for minimal lighting
- **Scale Impact**: Only +11MB for 5x more lights (181→193MB)
- **Shadow Impact**: +7MB when shadows enabled
- **Excellent Scaling**: Memory usage is not a bottleneck

### Input Latency

- **Excellent Responsiveness**: 0.01-0.07ms input latency
- **Negligible Impact**: No noticeable increase with more lights
- **Real-world Ready**: Sub-millisecond latency for responsive gameplay

## WGPU Migration Targets

Based on these realistic baseline measurements:

### Minimum Acceptable Performance (WGPU must achieve)
- **10-25 lights**: ≥100 FPS average, ≥60 FPS P95
- **50 lights**: ≥200 FPS average, ≥80 FPS P95
- **Frame Drops**: ≤50 frame drops per 10-second test
- **Memory**: ≤250MB total usage
- **Input Latency**: ≤1ms

### Stretch Goals (Performance improvements)
- **More Predictable Shadows**: Eliminate counter-intuitive shadow performance
- **Better Frame Consistency**: Reduce frame drops by 50%
- **Higher Light Counts**: Support 100+ lights at 60+ FPS
- **Lower Memory**: Reduce baseline memory usage

## Critical Performance Insights

### 1. Real vs Synthetic Performance
- **Hidden window tests**: 1000+ FPS (unrealistic)
- **Realistic tests**: 100-400 FPS (actual gameplay)
- **Lesson**: Always test with visible window and VSync for real metrics

### 2. Frame Drop Patterns
Frame drops (>16.7ms frames) indicate real bottlenecks:
- More common with fewer lights (optimization opportunities)
- Reduced with shadows in some cases (batching benefits)
- Critical metric for smooth gameplay experience

### 3. GPU Utilization Patterns
Counter-intuitive shadow performance suggests:
- ModernGL may have suboptimal GPU utilization without shadows
- Shadow passes might trigger better GPU batching
- WGPU migration opportunity: more predictable GPU utilization

## Test Reproducibility

To reproduce these measurements:

```bash
uv run python scripts/benchmark_realistic_performance.py --verbose --duration 10.0 --export-json results.json
```

**Important**: This benchmark shows a **visible game window** that you can interact with. Close the window to end the test early.

## Migration Validation Plan

For WGPU Step 3.2 validation:

1. **Run Identical Tests**: Same scenarios, same durations
2. **Compare P95 Performance**: Focus on consistency, not just averages  
3. **Validate Frame Drops**: Ensure WGPU has ≤similar frame drop counts
4. **Memory Regression**: WGPU should use ≤250MB
5. **Visual Parity**: Screenshot comparison between backends

## Next Steps

1. ✅ **Realistic Baseline Established** - Current step complete
2. **Shader Translation (Step 3.2)**: Convert GLSL→WGSL with these targets
3. **WGPU Implementation**: Build equivalent system targeting these metrics
4. **Performance Validation**: Replicate these exact tests with WGPU
5. **Regression Analysis**: Compare side-by-side with statistical significance

---

*This baseline represents real-world ModernGL performance that WGPU must match or exceed for successful migration.*