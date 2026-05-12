# 9. Current Multi-Model Assessment

Current multi-model deployment status on CV181x/SG2002-class devices, based on code analysis of:

- `src/node/model_node.cpp`
- `src/node/model_lua_runtime.cpp`
- `src/node/model_inference_executor.cpp`
- `src/node/camera_node.cpp`
- `src/inference/cvi_session.cpp` / `session_manager.h`
- `src/modules/cv/mmf_context.cpp`
- `src/modules/cv/cvi_vpss_processor.cpp`
- `src/node/resource_estimator.cpp`

---

## 9.1 What the Current Implementation Already Supports

| Capability | Current Status | Notes |
|------------|----------------|-------|
| Multiple model nodes under one camera | Supported | `CameraNode` fans out infer frames to downstream models |
| Per-model model file | Supported | Each `ModelNode` has independent `model_path` |
| Per-model Lua script | Supported | Each `ModelNode` has independent `script_path` and Lua VM |
| Lua postprocess | Supported | `postprocess(outputs, meta)` is the main required script hook |
| ROI chain between models | Supported | Upstream result + `select_rois` script, uses full-resolution stream frame |
| Parallel sibling models | Partially supported | Works structurally, but resource contention is not solved |

### Responsibility Split

| Layer | Responsibility |
|-------|----------------|
| C++ | model loading, `CviSession`, preprocess execution, `CVI_NN_Forward`, output readback |
| Lua | `preprocess_config`, `postprocess`, optional `select_rois` |

The current system is: **C++-driven inference with Lua-configured preprocessing and Lua postprocessing**.

---

## 9.2 Main Memory Pressure Points

### 1. Per-Model-Path `CviSession` and TPU Buffers

`SessionManager` maps `model_path` to `CviSession` with ref-counting. Same model path shares one session; different model paths each allocate:

- `CVI_NN_RegisterModel` registration memory
- TPU input/output device buffers via `CVI_RT_MemAlloc`
- Per-model Lua VM
- Per-model inference thread
- Optional preview/websocket path
- Optional `CviVpssProcessor`

This is the primary ION pressure source — duplicated model sessions and TPU buffers, not camera VB pools.

### 2. Camera VB Pools Are Shared Globally

`CameraNode` uses the singleton `MmfContext` for MMF/VB setup. Camera-side VB pools initialize once and are shared across all model nodes. The bottleneck is therefore at TPU/session memory, not camera pools.

### 3. Offline MEM VPSS Preprocess Is a Global Conflict Point

- `CviVpssProcessor` is created per `ModelNode` at the C++ object level
- But the underlying MEM VPSS group is shared across instances
- Multiple models can fight over the same offline preprocess group
- VB fallback can consume blocks from pools intended for other purposes

### 4. One Slow Model Can Throttle Sibling Models

`CameraNode` derives infer pacing from downstream processing time. The slowest model under the same camera reduces effective infer rate for all others.

### 5. Per-Node Configuration Is Still Global or Conflict-Prone

- `LUA_PATH` is process-global rather than per-node isolated
- Default websocket port usage can conflict across model nodes

---

## 9.3 Current VB Pool Layout

Source: `src/modules/cv/mmf_context.cpp`.

| Pool | Usage | Format | Resolution | Blocks | Approx. Total |
|------|-------|--------|------------|--------|---------------|
| Pool 0 | `CAMERA_VI` | NV21 | 1920×1080 | 3 | ~8.9 MiB |
| Pool 1 | `CAMERA_STREAM` | NV21 | 1920×1080 | 3 | ~8.9 MiB |
| Pool 2 | `CAMERA_PREVIEW` | NV21 | 1280×720 | 3 | ~4.0 MiB |
| Pool 3 | `VPSS_PREPROCESS` | RGB | 640×640 | 2 | ~2.3 MiB |
| Pool 4 | `CAMERA_INFER` | RGB | 640×640 | 3 | ~3.5 MiB |
| Pool 5 | Reserved | — | — | — | — |

Approximate VB subtotal: **~27.6 MiB**

### Notes

1. This pool plan is global and shared.
2. It is not the main reason multi-model mode hits ION OOM.
3. It becomes a secondary problem when models retain frames longer or when preprocess consumes fallback pools.

---

## 9.4 Why Multi-Model Can Hit `ION out of memory`

### Direct Causes

1. independent `CviSession` instances for different model paths
2. duplicated TPU input/output device buffers per session
3. model registration memory for each loaded model
4. extra preview/stream/websocket overhead per model
5. longer infer-frame retention across multiple downstream consumers

### Hidden Amplifiers

- offline VPSS preprocess group conflicts
- VB fallback taking blocks from unrelated pools
- lack of hard admission control
- enabling preview/stream together with multi-model mode
- running multiple full-frame heavy models at the same rate
