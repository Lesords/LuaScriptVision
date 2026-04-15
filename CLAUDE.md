# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Build project
mkdir build && cd build
cmake ..
make

# Run Lua-based inference (recommended)
./build/lua_runner scripts/yolo11_tensor_detector.lua models/yolo11n.onnx images/zidane.jpg

# Run pure C++ inference 
./build/cpp_infer models/yolov5n.onnx images/zidane.jpg

# Run tensor API tests
./build/lua_runner tests/run_all_tests.lua

# Video inference with options
./build/lua_runner scripts/yolo11_detector.lua models/yolo11n.onnx video.mp4 show save=out.mp4 frames=100

# Rebuild after changes
cd build && make -j8

# SG200X Device: Parallel inference + streaming (with camera)
# Build for SG200X:
cmake -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-sg200x.cmake -DENABLE_CVI_CAMERA=ON -B build
make -j8 parallel_infer_stream
#
# Run on device:
./build/parallel_infer_stream scripts/yolo11_tensor_detector.lua /path/to/model.cvimodel --duration 300
#
# Command format:
#   parallel_infer_stream <script.lua> <model.cvimodel> [--duration <seconds>]
#
# Example on-device deployment:
#   /userdata/parallel_infer_stream /userdata/scripts/yolo11_tensor_detector.lua \
#                                    /userdata/Models/model.cvimodel --duration 300

# JPEG Decode Method Selection
# Default: Software decode (OpenCV, no VB pool dependency)
cmake -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-sg200x.cmake -B build

# Optional: Hardware VDEC decode (requires VB pool, may cause resource conflicts)
cmake -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-sg200x.cmake -DUSE_VDEC_DECODE=ON -B build
```

### JPEG Decode Options

LuaScriptVision supports two JPEG decoding methods:

**Software Decode (Default, Recommended)**:
- Uses OpenCV for JPEG decoding
- No VB pool dependency
- Avoids VDEC resource conflicts
- Slightly slower decode (~5-10ms extra per image)
- **Best for**: Testing, development, and most production scenarios

**Hardware VDEC Decode (Optional)**:
- Uses Sophgo VDEC hardware decoder
- Requires VB pool allocation
- May conflict with other video processing pipelines
- Faster decode (~1-2ms per image)
- **Best for**: High-throughput video processing with dedicated VB pool planning

**How to switch**:
```bash
# Software decode (default)
cmake -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-sg200x.cmake -B build

# Hardware decode
cmake -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-sg200x.cmake -DUSE_VDEC_DECODE=ON -B build
```

**Affected components**:
- Test image loading in performance benchmarks
- ImageSource class (image file input)
- End-to-end pipeline tests

**Note**: All Lua scripts and core inference functionality work identically with both methods. The only difference is JPEG decode performance and VB pool usage.

## Architecture Overview

### Target Platform: Embedded Linux

**CRITICAL**: This project is designed to run on **low-end embedded Linux systems** with limited resources (RISC-V/ARMv8, <256MB RAM, no GPU).

When implementing features: benchmark performance impact, profile hot paths, minimize heap allocations, avoid unnecessary copies. See `docs/ARCHITECTURE_REFERENCE.md` for full design principles.

### Hybrid C++/Lua Design
The project implements a **dual-language architecture** where C++ handles performance-critical operations while Lua provides scripting flexibility:

- **C++ Core** (`src/`): ONNX Runtime inference, OpenCV operations, tensor math
- **Lua Scripts** (`scripts/`): Preprocessing, postprocessing, business logic
- **Binding Layer** (`lua-intf-ex/`): Bridges C++ and Lua using LuaIntf library

### Critical Design Decisions

See `docs/ARCHITECTURE_REFERENCE.md` for detailed explanations:

1. **Lua Compiled as C++** (`CMakeLists.txt:18-20`): Ensures exception safety across Lua boundary
2. **DeviceBuffer Abstraction Layer** (`src/modules/tensor/`): Virtual interface for multi-device support (CPU/NPU/TPU)
3. **Zero-Copy View Operations** (`tensor_shape.cpp`): `slice()`, `transpose()`, `squeeze()` share underlying buffer, modify metadata only
   - **Critical invariant**: When `contiguous_ == false`, must use stride-based indexing, NOT direct pointer arithmetic

### Module Structure

#### `lua_cv` (Computer Vision)
**Location**: `src/modules/lua_cv.h/cpp`

Exposes OpenCV 4.x operations to Lua:
```lua
local img = cv.Image.load("path.jpg")
img:resize(640, 640)
img:pad(top, bottom, left, right, 114)
local tensor = img:to_tensor(scale, mean, std)
```

**Implementation**: All operations use OpenCV Mat internally. The `to_tensor()` method creates a `tensor::Tensor` wrapping the Mat data.

#### `lua_nn` (Neural Network)
**Location**: `src/modules/lua_nn.h/cpp`

Provides:
1. **Session class**: ONNX Runtime wrapper
2. **Tensor alias**: Maps to `tensor::Tensor`

```lua
local session = nn.Session.new("model.onnx")
local outputs = session:run(input_tensor)
```

**Important**: `lua_nn::Tensor` is a typedef to `tensor::Tensor`, not a separate implementation.

#### `lua_utils` (Utilities)
**Location**: `src/modules/lua_utils.h/cpp`

Pure Lua/C++ utility functions:
- NMS (Non-Maximum Suppression)
- Box format conversion (xywh to xyxy)
- Coordinate scaling

### Tensor API Performance

**Golden rule**: Filter data in C++ (using `where_indices`, `index_select`), THEN convert small result sets to Lua tables.

See `docs/ARCHITECTURE_REFERENCE.md` for full performance benchmarks and optimization history.

## Development Guidelines

### ⚠️ Lua Binding Architecture Rule

**CRITICAL**: Code under `src/` directory **MUST NOT** directly use Lua C API functions (e.g., `lua_push*`, `lua_to*`, `luaL_*`, etc.).

**Required approach**:
- ✅ All Lua interface calls and optimizations MUST go through `lua-intf-ex/` library
- ✅ Use LuaIntf wrappers: `LuaRef`, `LuaBinding::Class<T>`, `LuaIntf::LuaRef::fromValue()`
- ❌ DO NOT use raw Lua C API: `lua_pushnumber()`, `lua_tonumber()`, `luaL_checktype()`, etc.

**Rationale**:
1. **Type safety**: LuaIntf provides C++ type checking at compile time
2. **Exception safety**: Proper RAII and exception handling across Lua boundary
3. **Maintainability**: Consistent interface layer, easier to refactor
4. **Future-proofing**: Single point of modification if Lua binding strategy changes

**If optimization needed**: Modify `lua-intf-ex/` library, not `src/` code.

---

### Git Commit Guidelines

**CRITICAL**: When committing code changes, follow these rules:

1. **Single-file commits**: Each commit should contain changes to ONE file only
2. **No Claude signatures**: Do NOT include Claude Code attribution in commits
3. **Commit message format**: Use conventional commits style
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `perf:` - Performance improvement
   - `refactor:` - Code restructuring
   - `docs:` - Documentation
   - `test:` - Test changes
   - `chore:` - Build/tooling changes

**Rationale**:
- Single-file commits make history cleaner and easier to review/revert
- No attribution clutter in project history
- Clear, professional commit messages

---

### Documentation Policy

**CRITICAL**: Do NOT proactively create summary or documentation files.

**Rules**:
- ❌ DO NOT create summary files (e.g., `REFACTORING.md`, `CHANGES.md`, `SUMMARY.md`)
- ❌ DO NOT create documentation for work completed unless explicitly requested
- ✅ DO communicate results directly to user in conversation
- ✅ DO update existing documentation (README, CLAUDE.md) when explicitly needed

**Rationale**:
- Keeps repository clean from redundant documentation
- User will request documentation if needed
- Conversation history already contains complete work record

---

### ⚠️ Critical Deployment Requirements

- **MUST** use base64 pipeline + MD5 verification for binary transfer
- **NEVER** use `/remote-device deploy`, `cat`, or `scp` — they strip null bytes from ELF binaries
- See `docs/DEVICE_OPERATIONS.md` for the `deploy_and_verify()` function and full deployment instructions

### 🔄 Device Testing Workflow

After ANY code change affecting runtime behavior, follow this order:

1. Build → Stop processes → Deploy (base64+MD5) → Verify with `parallel_infer_stream` → Test `node_server` → Clean up
2. Testing order: `parallel_infer_stream` → `node_server` → Node-RED (from simple to complex)

See `docs/DEVICE_OPERATIONS.md` for the full 11-step workflow with commands.

### ⚠️ Node-RED Rules

- **NEVER** `killall -9 node-red` or `systemctl restart node-red-service`
- **DO** use `/flows/state` API: `curl -H 'Content-Type: application/json' http://localhost:1880/flows/state -d '{"state": "stop"}'`
- Working directory: `/home/recamera/.node-red/`
- See `docs/DEVICE_OPERATIONS.md` for full commands and debugging reference

---

### When Adding Tensor Operations

1. **Decide contiguity requirement**:
   - Can you support stride-based access? → Use `data() + offset_`
   - Need contiguous memory? → Call `contiguous()` first (documents the copy)

2. **Return value**:
   - Shape/metadata changes only? → Return `Tensor(storage_, new_shape, new_strides, ...)`
   - Need new data? → Allocate new storage and return new Tensor

3. **Lua binding**:
   ```cpp
   // In tensor bindings
   .addFunction("operation", &Tensor::operation)
   ```

### When Writing Lua Inference Scripts

**Study these references**:
- `scripts/yolo11_tensor_detector.lua`: Vectorized filtering (best performance)
- `scripts/yolov5_tensor_detector.lua`: Row-major tensor handling
- README "Performance Best Practices" section

**Anti-patterns to avoid**:
```lua
-- ❌ DON'T: Convert large tensors to tables
local all_data = tensor:to_table()  -- 230ms for 8400 elements!

-- ✅ DO: Filter in C++, convert small results
local indices = tensor:where_indices(0.25, "ge")  -- Fast C++
local filtered = tensor:index_select(0, indices):to_table()  -- Small conversion
```

### OpenCV Integration Notes

All computer vision operations MUST use OpenCV 4.x (specifically 4.6.0+):
- Image I/O: `cv::imread/imwrite`
- Preprocessing: `cv::resize`, `cv::copyMakeBorder`, `cv::cvtColor`
- Video: `cv::VideoCapture`, `cv::VideoWriter`

The `Image` class in `lua_cv.cpp` wraps `cv::Mat` and exposes methods to Lua.

## Testing

```bash
# Basic functionality test
./build/lua_runner tests/run_all_tests.lua

# Benchmark against C++ baseline
./build/cpp_infer models/yolo11n.onnx images/zidane.jpg  # Should be ~180ms
./build/lua_runner scripts/yolo11_tensor_detector.lua models/yolo11n.onnx images/zidane.jpg  # Target: ~190ms

# Memory leak detection (video mode)
./build/lua_runner scripts/yolo11_detector.lua models/yolo11n.onnx video.mp4 frames=1000
# Check output for "Memory leak detected" warnings
```

## Important Caveats

### Lua 1-Based Indexing
Lua tables use 1-based indexing, but C++ uses 0-based. When converting:
```lua
local class_ids = tensor:argmax(0)  -- Returns Lua table [1,2,3,...]
local actual_class = class_ids[i + 1]  -- Lua index needs +1 adjustment
```

### Non-Contiguous Tensor Gotcha
After `slice()` or `transpose()`, tensors may be non-contiguous:
```cpp
// ❌ WRONG: Assumes contiguous
const float* ptr = data();
return ptr[i * shape_[1] + j];  // May skip actual data!

// ✅ CORRECT: Use strides
return data()[i * strides_[0] + j * strides_[1]];
```

### Video Memory Monitoring
The video inference mode tracks memory usage per frame. If memory grows >10KB/frame consistently, it warns about potential leaks. This is a diagnostic tool, not a guarantee.

## File Organization Logic

```
src/
  ├── main.cpp              # Unified Lua runner (inference + testing)
  ├── cpp_main.cpp          # Pure C++ benchmark entry
  ├── modules/
  │   ├── lua_cv.*          # OpenCV bindings
  │   ├── lua_nn.*          # ONNX Runtime + Tensor typedef
  │   ├── lua_utils.*       # NMS, box utils
  │   └── tensor/           # Tensor implementation (modular)
  │       ├── tensor.h                  # Tensor class interface
  │       ├── device_buffer.h/cpp       # DeviceBuffer abstract interface
  │       ├── cpu_memory.h/cpp          # CpuMemory implementation
  │       ├── device_type.h             # Device enum (CPU/NPU/TPU)
  │       ├── sync_handle.h/cpp         # SyncHandle for async operations
  │       ├── tensor_core.cpp           # Constructors, data access
  │       ├── tensor_device.cpp         # Device ops (to, contiguous, view)
  │       ├── tensor_shape.cpp          # Shape ops (slice, reshape, transpose)
  │       ├── tensor_math.cpp           # Math ops (+,-,*,/)
  │       ├── tensor_activation.cpp     # Activation functions
  │       ├── tensor_compare.cpp        # Comparison operations
  │       ├── tensor_reduction.cpp      # Reduction ops (sum, max, argmax)
  │       ├── tensor_select.cpp         # Selection and indexing
  │       ├── tensor_advanced.cpp       # Gather, concat, split
  │       └── tensor_legacy.cpp         # Legacy YOLO filter methods
  └── bindings/
      └── register_modules.cpp      # Lua module registration

scripts/
  ├── yolo11_*.lua          # YOLO11 variants (detector, pose, seg)
  └── yolov5_*.lua          # YOLOv5 scripts

tests/
  ├── run_all_tests.lua     # Test runner
  ├── test_helpers.lua      # Shared test utilities
  └── test_*.lua            # Modular test suites (14 files)
```

## ONNX Runtime Notes

Models must be in `models/` directory. The system supports:
- Dynamic input shapes (auto-padding handles this)
- Float16/Float32 automatic conversion
- Multi-output models (returns Lua table of tensors)

Session creation is expensive (~100-200ms), so scripts should reuse sessions for video/batch processing.

## Device Testing and Debugging

### Memory Issues

After multiple test runs, the device may hit `ion ioctl fail:: Out of memory`. Fix: `/rd cmd "reboot"`. Before testing: `/rd cmd "killall -9 node_server parallel_stream 2>/dev/null"`.

### Known Issues

All known issues (VPSS MEM input, sensor linking, binary corruption, camera node NULL) are documented in `docs/DEVICE_OPERATIONS.md` with symptoms, root causes, and workarounds.


