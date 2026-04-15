# Device Operations Reference

This document contains detailed instructions for device deployment, testing, and debugging.
Extracted from CLAUDE.md for reference when working with the physical device.

## Critical Deployment Requirements

**Binary transfer rules**:
- MUST use base64 pipeline + MD5 verification (see `deploy_and_verify()` below)
- NEVER use `/remote-device deploy`, `cat`, or `scp` — they strip null bytes from ELF binaries, producing corrupted binaries

**Why**: ELF binaries contain null bytes (`\0`) that get stripped by bash string expansion. Corrupted binaries crash with "Exec format error". MD5 verification is the ONLY reliable way to confirm deployment success.

```bash
# Reusable deployment function with automatic MD5 verification
deploy_and_verify() {
  local binary="$1"
  local remote_path="$2"

  local local_md5=$(md5sum "$binary" | awk '{print $1}')
  echo "Local MD5:  $local_md5"

  local remote_md5=$(base64 "$binary" | sshpass -p '12345678' ssh lese@127.0.0.1 -p 2222 \
    "sshpass -p 'root' ssh root@192.168.42.1 \
     'base64 -d > $remote_path && chmod +x $remote_path && md5sum $remote_path' | awk '{print \$1}'")

  echo "Remote MD5: $remote_md5"

  if [ "$local_md5" != "$remote_md5" ]; then
    echo "MD5 MISMATCH! Deployment failed."
    return 1
  fi

  echo "Deployment verified."
  return 0
}

# Usage:
deploy_and_verify "build/node_server" "/userdata/node_server" || exit 1
deploy_and_verify "build/parallel_infer_stream" "/userdata/parallel_infer_stream" || exit 1
```

---

## Device Testing Workflow

**CRITICAL**: After ANY code change that affects runtime behavior (bug fixes, new features, refactoring), follow this workflow.

**Workflow Steps**:

1. **Build the project**:
   ```bash
   cd build && make -j8
   # Or rebuild specific targets:
   cd build && make -j8 node_server parallel_infer_stream
   ```

2. **Stop all running processes** (MANDATORY before deployment):
   ```bash
   /remote-device cmd "killall -9 node_server parallel_stream 2>/dev/null; sleep 1"
   /remote-device cmd "pidof node_server parallel_stream 2>/dev/null" || echo "OK: All processes stopped"
   ```
   **Why**: Avoid file write conflicts (binary in use), incomplete deployments, version mismatches, and port binding failures.

3. **Deploy to device** (base64 + MD5 — see `deploy_and_verify()` above):
   ```bash
   deploy_and_verify "build/node_server" "/userdata/node_server"
   deploy_and_verify "build/parallel_infer_stream" "/userdata/parallel_infer_stream"
   ```
   **NEVER use `/remote-device deploy`** — it corrupts binaries (see Known Issues).

4. **Restart device** if previous run had memory issues:
   ```bash
   /rd cmd "reboot"
   # Wait ~30 seconds for device to boot
   ```

5. **Verify baseline with `parallel_infer_stream`** (especially after camera/VPSS changes):
   ```bash
   /rd cmd "cd /userdata && timeout 15 ./parallel_infer_stream /userdata/scripts/yolo11_tensor_detector.lua /usr/share/supervisor/models/yolo11n_detection_cv181x_int8.cvimodel --duration 10"
   ```
   **Expected**: `[INFO] CviCamera: warmup dropped X/3 frames` + `NMS final boxes: N`
   **If this fails**: Do NOT proceed. Fix camera/VPSS issues first.

6. **Stop Node-RED flows** (see Node-RED Rules below):
   ```bash
   /remote-device cmd "curl -H 'Content-Type: application/json' http://localhost:1880/flows/state -d '{\"state\": \"stop\"}'"
   ```

7. **Run node_server**:
   ```bash
   /remote-device cmd "cd /userdata && nohup ./node_server --host localhost --port 1883 --client-id recamera > /tmp/node_server.log 2>&1 &"
   ```
   Parameters: `--host localhost` (MQTT broker), `--port 1883`, `--client-id recamera`

8. **Start Node-RED flows** (triggers camera/model initialization):
   ```bash
   /remote-device cmd "curl -H 'Content-Type: application/json' http://localhost:1880/flows/state -d '{\"state\": \"start\"}'"
   ```

9. **Analyze logs** (after 5-10 seconds):
   ```bash
   /remote-device cmd "tail -100 /tmp/node_server.log"
   ```
   **Look for**:
   - `[CviCamera] VI device timing enabled: fps=30` - VI timing OK
   - `[INFO] CviCamera: warmup dropped X/3 frames` - Frame capture OK
   - `CviVpssProcessor - CVI_VPSS_GetChnFrame failed: 0xc006800e` - VPSS error
   - `[ERROR]` lines - Any initialization failures

10. **Clean up** (when done testing):
    ```bash
    /remote-device cmd "curl -H 'Content-Type: application/json' http://localhost:1880/flows/state -d '{\"state\": \"stop\"}'"
    /remote-device stop node_server
    ```

11. **Iterate**: If test fails → analyze logs, fix issue, repeat from step 1

**Testing order** (from simple to complex):
1. `parallel_infer_stream` → pure camera/VPSS/inference pipeline
2. `node_server` → adds MQTT/node management
3. Node-RED → full integration test

Testing in this order isolates issues faster.

---

## Node-RED Management Rules

**CRITICAL: Prohibit arbitrary restarts or kills of Node-RED service**

**Rules**:
- DO NOT use `killall -9 node-red` or `killall -9 node` — this affects local development environments
- DO NOT use `systemctl restart node-red-service` — unless user explicitly requests
- DO use `/flows/state` API to control flow start/stop

**Correct approach**:
```bash
# Stop flows (without stopping Node-RED service)
curl -H 'Content-Type: application/json' http://localhost:1880/flows/state -d '{"state": "stop"}'

# Start flows (without restarting Node-RED service)
curl -H 'Content-Type: application/json' http://localhost:1880/flows/state -d '{"state": "start"}'
```

**Why**:
- Node-RED service may serve both local dev and remote device
- Arbitrary restarts can cause config loss or workflow interruption
- `/flows/state` API is the recommended flow control method

**Node-RED Flow Control Notes**:
- `state: "stop"` - Pauses all Node-RED flows, stops camera/model/stream nodes
- `state: "start"` - Resumes all Node-RED flows, creates and starts nodes
- Always stop Node-RED before deploying/testing node_server
- Start Node-RED AFTER node_server is running to trigger camera/model creation
- Always restart Node-RED after testing to restore normal operation

**Node-RED Working Directory**:
- **Location**: `/home/recamera/.node-red/`
- **Purpose**: Debug Node-RED workflows and node configurations

```bash
# Access Node-RED working directory
/rd cmd "cd /home/recamera/.node-red && ls -la"

# View current flows configuration
/rd cmd "cat /home/recamera/.node-red/flows.json"

# Check Node-RED logs
/rd cmd "cat /home/recamera/.node-red/node-red.log | tail -50"

# Backup current flows
/rd cmd "cd /home/recamera/.node-red && cp flows.json flows.json.backup"

# View node status and configuration
/rd cmd "curl -s http://localhost:1880/flows | jq '.'"

# Check installed nodes
/rd cmd "ls /home/recamera/.node-red/node_modules/"

# View settings
/rd cmd "cat /home/recamera/.node-red/settings.js | grep -E 'debug|flowFile|userDir'"
```

**Quick device test commands**:
```bash
/remote-device status              # Overall device status
/remote-device logs 100            # Recent kernel logs
/remote-device cmd "free -h"        # Memory usage
/remote-device cmd "ps aux"         # Running processes
```

---

## Memory Issues and Device Restart

**IMPORTANT**: After multiple test runs, the device may experience memory fragmentation or depletion, resulting in errors like:
```
ion ioctl fail:: Out of memory
Assertion failed: mem_alloc_raw
```

**When this happens, you MUST manually restart the device**:
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

**Common causes of memory issues**:
- Multiple parallel_infer_stream instances running simultaneously
- VB pool exhaustion (check `cat /proc/cvitek/vb`)
- VPSS groups not properly cleaned up after crash

---

## Known Issues

### VPSS Preprocessing with MEM Input

**Status**: Not Working on this hardware/driver

**Symptoms**:
- `CVI_VPSS_SendFrame` succeeds (returns 0x0)
- `CVI_VPSS_GetChnFrame` fails with `0xc006800e` (buffer empty)
- Group 5 does not appear in `/proc/cvitek/vpss` despite successful `CVI_VPSS_CreateGrp`

**Root Cause**: The VPSS driver on this hardware does not properly support dynamic MEM input group creation. DRV WORK STATUS DEV0 remains empty even after group creation.

**Workaround**: ModelNode uses CPU/OpenCV preprocessing instead of VPSS for offline frame preprocessing.

### Sensor Library Linking

**Status**: Fixed

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

### Binary Deployment Corruption via `/remote-device deploy`

**Status**: Known Limitation

**Symptoms**:
- Deployed binary crashes immediately or shows `Exec format error`
- `xxd /userdata/node_server | head -1` shows wrong ELF magic (`e_type=0x05f1` instead of `0x0002`)
- `md5sum` of local build vs device binary mismatch

**Root Cause**: The `/remote-device deploy` skill script uses `local file_content=$(cat "$local_file")` in bash, which strips null bytes (`\0`) from binary files. ELF binaries contain null bytes in headers and code, so the resulting file is corrupted.

**Solution**: Use the `deploy_and_verify()` function defined in the [Critical Deployment Requirements](#critical-deployment-requirements) section above.

**Verification checklist** (if deployment issues suspected):
```bash
# 1. Compare md5 before and after deploy
md5sum build/node_server
/rd cmd "md5sum /userdata/node_server"

# 2. Check ELF header is correct
/rd cmd "xxd /userdata/node_server | head -2"
# Expected: 7f45 4c46 0201 0100 ... (ELF magic, e_type=2 ET_EXEC, e_machine=0xf3 RISC-V)

# 3. Verify binary is executable
/rd cmd "file /userdata/node_server"
# Expected: ELF 32-bit LSB executable, RISC-V, version 1 (SYSV)
```

### node_server Camera Node Shows NULL in Node-RED

**Status**: Fixed

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
