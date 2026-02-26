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

### 🎯 Target Platform: Embedded Linux

**CRITICAL**: This project is designed to run on **low-end embedded Linux systems** with limited resources.

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
- ✅ Always benchmark performance impact
- ✅ Profile hot paths before optimization
- ✅ Test on actual embedded hardware when possible
- ✅ Document performance characteristics in comments
- ❌ Do NOT sacrifice correctness for premature optimization
- ❌ Do NOT add features that significantly increase binary size without clear benefits

---

### Hybrid C++/Lua Design
The project implements a **dual-language architecture** where C++ handles performance-critical operations while Lua provides scripting flexibility:

- **C++ Core** (`src/`): ONNX Runtime inference, OpenCV operations, tensor math
- **Lua Scripts** (`scripts/`): Preprocessing, postprocessing, business logic
- **Binding Layer** (`lua-intf-ex/`): Bridges C++ and Lua using LuaIntf library

### Critical Design Decisions

#### 1. Lua Compiled as C++
**Location**: `CMakeLists.txt:18-20`
```cmake
target_compile_options(lua PRIVATE -x c++ -O3 -Wall -DLUA_USE_POSIX)
```
**Why**: Ensures exception safety when C++ exceptions cross the Lua boundary. Without this, throwing exceptions from C++ through Lua causes undefined behavior.

#### 2. DeviceBuffer Abstraction Layer
**Location**: `src/modules/tensor/`

The tensor system uses a **virtual interface pattern** to support multiple devices:

```
DeviceBuffer (interface) - 设备缓冲区抽象
    ├── CpuMemory (CPU implementation) - CPU内存管理
    └── [Future: NpuMemory, TpuMemory]

Tensor (user-facing class)
    └── uses shared_ptr<DeviceBuffer>
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

#### 3. Zero-Copy View Operations
**Location**: `src/modules/tensor/tensor_shape.cpp`

Operations like `slice()`, `transpose()`, `squeeze()` are **zero-copy** - they share the same underlying `DeviceBuffer` but modify metadata:
- `shape_`: Logical dimensions
- `strides_`: Memory layout (enables non-contiguous views)
- `offset_`: Starting position in storage
- `contiguous_`: Flag indicating if data is contiguous in memory

**Critical invariant**: When `contiguous_ == false`, must use stride-based indexing, NOT direct pointer arithmetic.

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
- Box format conversion (xywh ↔ xyxy)
- Coordinate scaling

### Tensor API Performance Characteristics

**Location**: See `API_IMPROVEMENTS.md` and README benchmarks

| Operation | Speed | Notes |
|-----------|-------|-------|
| `slice()`, `transpose()` | **Instant** (~μs) | Zero-copy view |
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

### 🔄 Automated Testing Workflow for Device Deployment

**CRITICAL**: After making code changes to fix issues, follow this automated workflow:

**When to apply**: After ANY code change that affects runtime behavior (bug fixes, new features, refactoring)

**Workflow Steps**:

1. **Build the project**:
   ```bash
   cd build && make -j8
   ```

2. **Deploy to device** using base64 transfer (NOT `/remote-device deploy` — see ⚠️ below):
   ```bash
   base64 build/node_server | ssh -J lese@127.0.0.1:2222 root@192.168.42.1 'base64 -d > /userdata/node_server && chmod +x /userdata/node_server'
   ```
   Or via physical host:
   ```bash
   base64 build/node_server | sshpass -p '12345678' ssh lese@127.0.0.1 -p 2222 \
     "sshpass -p 'root' ssh root@192.168.42.1 'base64 -d > /userdata/node_server && chmod +x /userdata/node_server'"
   ```
   **⚠️ CRITICAL**: The `/remote-device deploy` skill script uses `$(cat binary_file)` which **strips null bytes** from ELF binaries, producing a corrupted binary with wrong `e_type`. Always use the base64 pipeline method above.

3. **Stop old process** (if running):
   ```bash
   /remote-device stop node_server
   ```

4. **⚠️ Stop Node-RED flows before testing** (execute on device):
   ```bash
   /remote-device cmd "curl -H 'Content-Type: application/json' http://localhost:1880/flows/state -d '{\"state\": \"stop\"}'"
   ```
   **Why**: Prevents Node-RED from interfering with node_server during testing

5. **Run and test** on device:
   ```bash
   /remote-device run "cd /userdata && ./node_server --host localhost --port 1883 --client-id recamera"
   ```
   Or for background testing:
   ```bash
   /remote-device cmd "cd /userdata && nohup ./node_server --host localhost --port 1883 --client-id recamera > /tmp/node_server.log 2>&1 &"
   ```

   **Command parameters**:
   - `--host localhost` - MQTT broker address
   - `--port 1883` - MQTT broker port
   - `--client-id recamera` - MQTT client identifier

6. **Start Node-RED flows** (to trigger camera/model initialization):
   ```bash
   /remote-device cmd "curl -H 'Content-Type: application/json' http://localhost:1880/flows/state -d '{\"state\": \"start\"}'"
   ```
   **Why**: Triggers Node-RED to create camera/model nodes and initialize hardware

7. **⚠️ Analyze logs** (after 5-10 seconds for initialization):
   ```bash
   /remote-device cmd "tail -100 /tmp/node_server.log"
   ```
   **Look for**:
   - `[CviCamera] VI device timing enabled: fps=30` - VI timing OK
   - `[INFO] CviCamera: warmup dropped X/3 frames` - Frame capture OK
   - `CviVpssProcessor - CVI_VPSS_GetChnFrame failed: 0xc006800e` - VPSS error
   - `[ERROR]` lines - Any initialization failures

8. **⚠️ Restart Node-RED flows after testing** (when done):
   ```bash
   /remote-device cmd "curl -H 'Content-Type: application/json' http://localhost:1880/flows/state -d '{\"state\": \"stop\"}'"
   ```
   **Why**: Clean shutdown before stopping node_server

9. **Stop node_server** (when done testing):
   ```bash
   /remote-device stop node_server
   ```

10. **Iterate based on results**:
   - If test passes → Continue with next task
   - If test fails → Analyze logs, fix issue, repeat from step 1

**Example session**:
```
User: Fix the crash in camera_node

Claude: [Analyzes issue, makes fix]
       [Builds] cd build && make -j8
       [Stops Node-RED] /remote-device cmd "curl -H 'Content-Type: application/json' \
                                         http://localhost:1880/flows/state -d '{\"state\": \"stop\"}'"
       [Deploys] /remote-device deploy build/node_server
       [Starts node_server] /remote-device cmd "cd /userdata && nohup ./node_server \
                         --host localhost --port 1883 --client-id recamera > /tmp/node_server.log 2>&1 &"
       [Starts Node-RED] /remote-device cmd "curl -H 'Content-Type: application/json' \
                                         http://localhost:1880/flows/state -d '{\"state\": \"start\"}'"
       [Waits 5-10 seconds for initialization]
       [Analyzes logs] /remote-device cmd "tail -100 /tmp/node_server.log"
       [Reports] Found errors: CviVpssProcessor failed → Need to fix VPSS Group 5
```

**Node-RED Flow Control Notes**:
- `state: "stop"` - Pauses all Node-RED flows, stops camera/model/stream nodes
- `state: "start"` - Resumes all Node-RED flows, creates and starts nodes
- **Always stop Node-RED before deploying/testing node_server**
- **Start Node-RED AFTER node_server is running** to trigger camera/model creation
- **Always restart Node-RED after testing** to restore normal operation

**Quick test commands**:
```bash
/remote-device status              # Overall device status
/remote-device logs 100            # Recent kernel logs
/remote-device cmd "free -h"        # Memory usage
/remote-device cmd "ps aux"         # Running processes
```

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

### Memory Issues and Device Restart

**⚠️ IMPORTANT**: After multiple test runs, the device may experience memory fragmentation or depletion, resulting in errors like:
```
ion ioctl fail:: Out of memory
Assertion failed: mem_alloc_raw
```

**When this happens, you MUST manually restart the device:**
```bash
# Via remote-device skill
/rd cmd "reboot"

# Or via SSH directly
ssh root@192.168.42.1 "reboot"
```

**Before testing**, always ensure previous processes are stopped:
```bash
/rd cmd "killall -9 node_server parallel_stream 2>/dev/null"
```

**Common causes of memory issues:**
- Multiple parallel_infer_stream instances running simultaneously
- VB pool exhaustion (check `cat /proc/cvitek/vb`)
- VPSS groups not properly cleaned up after crash

### Testing Workflow After Code Changes

**Always follow this sequence after modifying camera/VPSS-related code:**

1. **Stop all running processes:**
   ```bash
   /rd cmd "killall -9 node_server parallel_stream 2>/dev/null; sleep 2"
   ```

2. **Rebuild affected binaries:**
   ```bash
   cd build && make -j8 node_server parallel_infer_stream
   ```

3. **Deploy to device** (use base64, NOT `/rd deploy` which corrupts binaries):
   ```bash
   base64 build/node_server | sshpass -p '12345678' ssh lese@127.0.0.1 -p 2222 \
     "sshpass -p 'root' ssh root@192.168.42.1 'base64 -d > /userdata/node_server && chmod +x /userdata/node_server'"
   # or for parallel_infer_stream:
   base64 build/parallel_infer_stream | sshpass -p '12345678' ssh lese@127.0.0.1 -p 2222 \
     "sshpass -p 'root' ssh root@192.168.42.1 'base64 -d > /userdata/parallel_infer_stream && chmod +x /userdata/parallel_infer_stream'"
   ```

4. **Restart device if previous run had memory issues:**
   ```bash
   /rd cmd "reboot"
   # Wait ~30 seconds for device to boot
   sleep 30
   ```

5. **Verify basic functionality first:**
   ```bash
   # CRITICAL: Always verify parallel_infer_stream works before testing other components
   /rd cmd "cd /userdata && timeout 15 ./parallel_infer_stream /userdata/scripts/yolo11_tensor_detector.lua /usr/share/supervisor/models/yolo11n_detection_cv181x_int8.cvimodel --duration 10"
   ```

   **Expected results:**
   - `[INFO] CviCamera: warmup dropped 1/3 frames` or similar (NOT `captured 0/3`)
   - `rtsp://192.168.42.1:554/live` RTSP server started
   - `NMS final boxes: 0` (or other number, indicating inference is running)

   **If parallel_infer_stream fails:**
   - Do NOT proceed with node_server testing
   - Revert recent changes and investigate
   - Common issues: VPSS not enabled, sensor not detected, VI/VPSS binding failure

6. **Test other components:**
   ```bash
   # Only after parallel_infer_stream works, test node_server
   # (deploy via base64, not /rd deploy)
   /rd stop node_server
   /rd run "cd /userdata && ./node_server --host localhost --port 1883 --client-id recamera"
   ```

### ⚠️ Development Best Practice

**Before adding new features, always verify baseline functionality:**

1. **First test with `parallel_infer_stream`** - This is the minimal, reference implementation
2. **If baseline fails** - Fix camera/VPSS issues first
3. **Then test with `node_server`** - Adds MQTT/node management on top
4. **Finally test with Node-RED** - Full integration test

**Why this order matters:**
- `parallel_infer_stream` tests pure camera/VPSS/inference pipeline
- `node_server` adds MQTT complexity
- Node-RED adds flow management complexity
- Testing in this order isolates issues faster

### Known Issues

#### VPSS Preprocessing with MEM Input

**Status**: Not Working on this hardware/driver

**Symptoms**:
- `CVI_VPSS_SendFrame` succeeds (returns 0x0)  
- `CVI_VPSS_GetChnFrame` fails with `0xc006800e` (buffer empty)
- Group 5 does not appear in `/proc/cvitek/vpss` despite successful `CVI_VPSS_CreateGrp`

**Root Cause**: The VPSS driver on this hardware does not properly support dynamic MEM input group creation. DRV WORK STATUS DEV0 remains empty even after group creation.

**Workaround**: ModelNode uses CPU/OpenCV preprocessing instead of VPSS for offline frame preprocessing.

#### Sensor Library Linking

**Status**: Fixed ✅

**Problem**: `Sensor object is NULL - library not linked?`

**Solution**: Ensure `libsns_full.a` is wrapped with `--whole-archive` in CMakeLists.txt to preserve weak sensor symbols:
```cmake
target_link_libraries(parallel_infer_stream PRIVATE
    ...
    -Wl,--whole-archive
    ${CVI_MPI_LIB_DIR}/libsns_full.a
    -Wl,--no-whole-archive
    ...
)
```

#### Binary Deployment Corruption via `/remote-device deploy`

**Status**: Known Limitation ⚠️

**Symptoms**:
- Deployed binary crashes immediately or shows `Exec format error`
- `xxd /userdata/node_server | head -1` shows wrong ELF magic (`e_type=0x05f1` instead of `0x0002`)
- `md5sum` of local build vs device binary mismatch

**Root Cause**: The `/remote-device deploy` skill script uses `local file_content=$(cat "$local_file")` in bash, which strips null bytes (`\0`) from binary files. ELF binaries contain null bytes in headers and code, so the resulting file is corrupted.

**Solution**: Always deploy binaries using the base64 pipeline:
```bash
base64 build/node_server | sshpass -p '12345678' ssh lese@127.0.0.1 -p 2222 \
  "sshpass -p 'root' ssh root@192.168.42.1 \
   'base64 -d > /userdata/node_server && chmod +x /userdata/node_server && echo OK'"
```

**Verification**:
```bash
# Compare md5 before and after deploy
md5sum build/node_server
/rd cmd "md5sum /userdata/node_server"
# Also check ELF header
/rd cmd "xxd /userdata/node_server | head -2"
# Correct first line: 7f45 4c46 0201 0100 0000 ... (.ELF)
# Correct second line: 0200 f300 ... (e_type=2 ET_EXEC, e_machine=0xf3 RISC-V)
```

#### node_server Camera Node Shows NULL in Node-RED

**Status**: Fixed ✅

**Symptoms**:
- Node-RED UI shows camera node status as "NULL" or error
- `[NodeFactory] onCreate failed for detector with code 22` in logs
- `[WARN] CviCamera: warmup captured 0/3` on first cold boot

**Root Causes & Fixes** (all applied to `src/node/`):

1. **`model_node.cpp`: Wrong LUA_PATH — script `require()` fails**
   - Lua scripts use `require("scripts.lib.preprocess")` which needs `LUA_PATH` to include the **parent** of the `scripts/` directory
   - Old code set `LUA_PATH = /userdata/scripts/?.lua` → looked for `/userdata/scripts/scripts/lib/preprocess.lua` (wrong)
   - Fix: include parent dir — `LUA_PATH = /userdata/?.lua;/userdata/scripts/?.lua`

2. **`node_server.cpp`: Auto-script detection picked incompatible script**
   - Fallback to `yolo11_detector.lua` (non-tensor) which calls `filter_yolo()` on CVI model output
   - CVI model output shape is incompatible → `Invalid YOLO output shape` error in postprocess
   - Fix: prefer `yolo11_tensor_detector.lua` in fallback search order

3. **`stream_node.cpp`: Camera lookup in `onCreate()` fails if model created before camera**
   - Node-RED may send `create model` before `create camera`; `onCreate()` finds no camera dependency → `MA_EINVAL`
   - Fix: move camera lookup to `onStart()` (lazy resolution)

4. **`node_factory.cpp`: Factory mutex held during camera `open()` (ISP warmup ~300ms)**
   - `std::lock_guard` held for entire `create()` including `start()` → blocks all MQTT for 300ms+
   - Fix: `std::unique_lock` + `lock.unlock()` before calling `start()`

5. **`camera_node.cpp`, `model_node.cpp`: `enabled` action unhandled**
   - Node-RED sends `enabled {value: true}` after `create`; nodes returned `MA_EINVAL` (not implemented)
   - Fix: add `enabled` handlers that set `inference_enabled_` / `infer_enabled_` atomics

**Diagnosis tip**: Run `parallel_infer_stream` first (not node_server) to verify camera hardware works:
```bash
/rd cmd "cd /userdata && timeout 15 ./parallel_infer_stream \
  scripts/yolo11_tensor_detector.lua \
  /usr/share/supervisor/models/yolo11n_detection_cv181x_int8.cvimodel --duration 10"
```
Expected: `[INFO] CviCamera: warmup dropped 1/3 frames` + `NMS final boxes: N`

**Note on cold-boot warmup**: On first boot, `warmup captured 0/3` is normal (ISP needs ~1s to stabilize). On second run it shows `warmup dropped 1/3` (success). The `captureLoop` continues retrying so inference starts normally after warmup.

