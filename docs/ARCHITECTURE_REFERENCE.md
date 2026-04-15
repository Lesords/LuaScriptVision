# Architecture Reference

This document contains detailed architecture, design decisions, and performance data.
Extracted from CLAUDE.md for reference when working on architecture-level changes.

See also: `docs/architecture/` for comprehensive system documentation.

---

## Target Platform: Embedded Linux

This project is designed to run on **low-end embedded Linux systems** with limited resources.

**Performance requirements**:
- CPU: RISC-V(C906)/ARMv8 or similar low-power processors (e.g., Sophgo SG2002, RK3566)
- Memory: < 256MB RAM available
- Storage: Limited flash/eMMC (minimize binary size)
- No GPU acceleration available in baseline configuration

**Design principles for embedded systems**:

1. **Memory efficiency**:
   - Minimize heap allocations (use stack/static when possible)
   - Avoid unnecessary copies (prefer move semantics, zero-copy views)
   - Consider small buffer optimization (SBO) for containers
   - Be aware of memory fragmentation

2. **CPU efficiency**:
   - Cache-friendly data structures (prefer contiguous memory)
   - Avoid virtual function calls in hot paths
   - Minimize atomic operations (shared_ptr pass-by-value)
   - Consider SIMD optimization for batch operations

3. **Binary size**:
   - Avoid template bloat (explicit instantiation when possible)
   - Minimize header-only libraries
   - Use compiler optimization flags carefully

4. **Power consumption**:
   - Avoid busy-waiting loops
   - Batch operations to reduce CPU wake-ups
   - Consider thermal throttling on sustained workloads

**When implementing features**:
- Always benchmark performance impact
- Profile hot paths before optimization
- Test on actual embedded hardware when possible
- Document performance characteristics in comments
- Do NOT sacrifice correctness for premature optimization
- Do NOT add features that significantly increase binary size without clear benefits

---

## Critical Design Decisions

### 1. Lua Compiled as C++

**Location**: `CMakeLists.txt:18-20`
```cmake
target_compile_options(lua PRIVATE -x c++ -O3 -Wall -DLUA_USE_POSIX)
```
**Why**: Ensures exception safety when C++ exceptions cross the Lua boundary. Without this, throwing exceptions from C++ through Lua causes undefined behavior.

### 2. DeviceBuffer Abstraction Layer

**Location**: `src/modules/tensor/`

The tensor system uses a **virtual interface pattern** to support multiple devices:

```
DeviceBuffer (interface) - Device buffer abstraction
    +-- CpuMemory (CPU implementation) - CPU memory management
    +-- [Future: NpuMemory, TpuMemory]

Tensor (user-facing class)
    +-- uses shared_ptr<DeviceBuffer>
```

**Key files**:
- `device_buffer.h`: Abstract buffer interface with virtual methods
- `cpu_memory.h/cpp`: CPU memory management implementation
- `tensor.h`: User-facing tensor operations with stride-based indexing
- `tensor_*.cpp`: Modular implementation (10 files by functionality)

**Naming rationale**:
- **DeviceBuffer**: Emphasizes cross-device data buffer abstraction
- **CpuMemory**: Focuses on CPU-side memory allocation/deallocation
- Combines precision: buffer (interface), memory (implementation), allocation (operations)

**Design tradeoff**: Virtual function overhead (~few nanoseconds per call) vs. device abstraction. Current performance bottlenecks are NOT the virtual calls but rather:
- Memory allocation (`CpuMemory::allocate` with memset)
- Non-contiguous tensor copying (`contiguous_copy` recursive implementation)

### 3. Zero-Copy View Operations

**Location**: `src/modules/tensor/tensor_shape.cpp`

Operations like `slice()`, `transpose()`, `squeeze()` are **zero-copy** - they share the same underlying `DeviceBuffer` but modify metadata:
- `shape_`: Logical dimensions
- `strides_`: Memory layout (enables non-contiguous views)
- `offset_`: Starting position in storage
- `contiguous_`: Flag indicating if data is contiguous in memory

**Critical invariant**: When `contiguous_ == false`, must use stride-based indexing, NOT direct pointer arithmetic.

---

## Tensor API Performance Characteristics

**Location**: See `API_IMPROVEMENTS.md` and README benchmarks

| Operation | Speed | Notes |
|-----------|-------|-------|
| `slice()`, `transpose()` | **Instant** (~us) | Zero-copy view |
| `contiguous()` | **Fast** (0.02-4ms) | Optimized batch memcpy |
| `max_with_argmax()` | **Fast** (~0.6ms) | Fused operation, single pass |
| `where_indices()` | **Fast** (~0.07ms) | C++ vector scan |
| `extract_columns()` | **Fast** (~0.02ms) | Direct Lua table output |
| `to_table()` | **Fast for small data** | Only use after filtering |

**Golden rule**: Filter data in C++ (using `where_indices`, `index_select`), THEN convert small result sets to Lua tables.

### Performance Analysis (YOLO11n, 640x640)

**Time distribution**:
| Stage | Time | Percentage |
|-------|------|------------|
| ONNX inference | ~100ms | **50%** |
| Model loading | ~80ms | One-time |
| Image load + preprocess | ~15ms | 7.5% |
| **Postprocess (Tensor API)** | **~4.5ms** | **2.2%** |

**Postprocess breakdown**:
```
contiguous scores [80,8400]:  3.77 ms  (672K elements copy)
max_with_argmax:              0.57 ms
where_indices:                0.07 ms
extract_columns:              0.02 ms
other:                        0.14 ms
```

**Conclusion**: Tensor API postprocess is highly efficient (~4.5ms). The bottleneck is ONNX inference (~100ms), which is determined by model complexity.

### Optimizations Implemented

1. **OPT-1**: Removed memset in allocate
2. **OPT-2**: Inlined hot-path functions (at, data, raw_data, device)
3. **OPT-3**: extract_columns returns direct Lua table (row format)
4. **OPT-4**: Cached device type to avoid virtual calls
5. **OPT-5**: Added in-place operations (add_, sub_, mul_, div_)
6. **OPT-6**: Optimized contiguous_copy with batch memcpy
7. **OPT-7**: Added `max_with_argmax()` fused operation (architecture-level)

**Result**: 210ms → 200ms (~5% improvement)
