#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# run_app_on_device.sh - Deploy and run any application on embedded device
# ==============================================================================
# Usage:
#   ./run_app_on_device.sh <app_binary> [app_args...]
#
# Options (via environment variables):
#   DEVICE_IP       - Device IP address (default: 192.168.42.1)
#   DEVICE_USER     - SSH username (default: recamera)
#   DEVICE_PASS     - SSH password (default: 11)
#   DEVICE_LIBS     - Additional library paths on device (colon-separated)
#   RUN_TIMEOUT     - Application timeout in seconds (default: 60)
#   RUN_STDIN_HOLD  - Keep stdin open for apps that block on getchar (default: 0)
#   CAPTURE_CVI_PROC - Capture /proc/cvitek/{sys,vi,vpss,vb} into app log (default: 0)
#   LOG_DIR         - Local log directory (default: ./debug/device_logs)
#   --no-reboot     - Skip device reboot before deployment
#
# Examples:
#   # Run video_demo
#   ./run_app_on_device.sh ./video_demo
#
#   # Run with custom library paths
#   DEVICE_LIBS="/mnt/system/usr/lib:/mnt/system/usr/lib/3rd" \
#     ./run_app_on_device.sh ./video_demo
#
#   # Run with arguments and no reboot
#   ./run_app_on_device.sh ./test_cv_func /tmp/image.jpg --no-reboot
# ==============================================================================

DEVICE_IP="${DEVICE_IP:-192.168.42.1}"
DEVICE_USER="${DEVICE_USER:-recamera}"
DEVICE_PASS="${DEVICE_PASS:-11}"
DEVICE_LIBS="${DEVICE_LIBS:-/mnt/system/usr/lib:/mnt/system/usr/lib/3rd:/usr/lib}"

RUN_TIMEOUT="${RUN_TIMEOUT:-60}"
RUN_SSH_KILL="${RUN_SSH_KILL:-5}"
RUN_SSH_TIMEOUT="${RUN_SSH_TIMEOUT:-$((RUN_TIMEOUT + 10))}"
RUN_STDIN_HOLD="${RUN_STDIN_HOLD:-0}"
CAPTURE_CVI_PROC="${CAPTURE_CVI_PROC:-0}"

REMOTE_DIR="/tmp"
REMOTE_LOG="${REMOTE_DIR}/app_run.log"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/../debug/device_logs}"
mkdir -p "${LOG_DIR}"

# Parse arguments
NO_REBOOT=0
APP_BINARY=""
APP_ARGS=()

for arg in "$@"; do
    case "${arg}" in
        --no-reboot)
            NO_REBOOT=1
            ;;
        *)
            if [[ -z "${APP_BINARY}" ]]; then
                APP_BINARY="${arg}"
            else
                APP_ARGS+=("${arg}")
            fi
            ;;
    esac
done

if [[ -z "${APP_BINARY}" ]]; then
    echo "Usage: $0 <app_binary> [app_args...] [--no-reboot]"
    echo ""
    echo "Environment variables:"
    echo "  DEVICE_IP       Device IP (default: 192.168.42.1)"
    echo "  DEVICE_USER     SSH username (default: recamera)"
    echo "  DEVICE_PASS     SSH password (default: 11)"
    echo "  DEVICE_LIBS     Library paths (default: /mnt/system/usr/lib:/mnt/system/usr/lib/3rd:/usr/lib)"
    echo "  RUN_TIMEOUT     Timeout in seconds (default: 60)"
    echo "  LOG_DIR         Log directory (default: ./debug/device_logs)"
    exit 1
fi

if [[ ! -f "${APP_BINARY}" ]]; then
    echo "Error: Application binary not found: ${APP_BINARY}"
    exit 1
fi

# Check dependencies
require_cmd() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "Error: Missing dependency: ${cmd}"
        exit 1
    fi
}

require_cmd sshpass
require_cmd ssh
require_cmd scp
require_cmd timeout

APP_NAME="$(basename "${APP_BINARY}")"
REMOTE_BIN="${REMOTE_DIR}/${APP_NAME}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
HOST_LOG="${LOG_DIR}/${APP_NAME}_${TIMESTAMP}.log"
REMOTE_APP_CMD="${REMOTE_BIN} ${APP_ARGS[*]:-} > ${REMOTE_LOG} 2>&1"
BANG='$!'

ssh_opts=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 -o LogLevel=ERROR)

echo "[runner] =========================================="
echo "[runner] Application: ${APP_BINARY}"
echo "[runner] Arguments:   ${APP_ARGS[*]:-<none>}"
echo "[runner] Device:      ${DEVICE_USER}@${DEVICE_IP}"
echo "[runner] Timeout:     ${RUN_TIMEOUT}s"
echo "[runner] Log:         ${HOST_LOG}"
echo "[runner] =========================================="

# Reboot device if requested
if [[ "${NO_REBOOT}" -eq 0 ]]; then
    echo "[runner] Rebooting device..."
    sshpass -p "${DEVICE_PASS}" ssh "${ssh_opts[@]}" \
        "${DEVICE_USER}@${DEVICE_IP}" \
        "echo '${DEVICE_PASS}' | sudo -S -p '' reboot" 2>/dev/null || true

    echo "[runner] Waiting for device to come back up..."
    for i in $(seq 1 60); do
        sleep 2
        if sshpass -p "${DEVICE_PASS}" ssh "${ssh_opts[@]}" \
            "${DEVICE_USER}@${DEVICE_IP}" "echo 1" >/dev/null 2>&1; then
            echo "[runner] Device is up (waited ${i}x2s)"
            break
        fi
    done
    sleep 2
fi

# Deploy binary
echo "[runner] Deploying ${APP_NAME} to device..."
sshpass -p "${DEVICE_PASS}" scp "${ssh_opts[@]}" \
    "${APP_BINARY}" \
    "${DEVICE_USER}@${DEVICE_IP}:${REMOTE_BIN}"

# Run application with library path and capture output
echo "[runner] Running application (timeout: ${RUN_TIMEOUT}s)..."

# Start application on device in background
echo "[runner] Starting application..."

# Create startup script on device
startup_script="/tmp/app_start.sh"
cat > /tmp/local_start.sh << 'EOFSCRIPT'
#!/bin/sh
export LD_LIBRARY_PATH="DEVICE_LIBS_PLACEHOLDER:${LD_LIBRARY_PATH}"
cd /tmp
if [ "RUN_STDIN_HOLD_PLACEHOLDER" -eq 1 ]; then
    FIFO="/tmp/app.stdin"
    rm -f "${FIFO}"
    mkfifo "${FIFO}"
    tail -f /dev/null > "${FIFO}" &
    STDIN_PID=$!
    echo "${STDIN_PID}" > /tmp/app.stdin.pid
    APP_CMD_PLACEHOLDER < "${FIFO}" > APP_LOG_PLACEHOLDER 2>&1 &
else
    APP_CMD_PLACEHOLDER > APP_LOG_PLACEHOLDER 2>&1 &
fi
APP_PID=$!
echo ${APP_PID} > /tmp/app.pid
chmod 666 /tmp/app.pid /tmp/app_run.log 2>/dev/null
echo "Started PID: ${APP_PID}"
EOFSCRIPT

# Replace placeholders
sed -i "s|DEVICE_LIBS_PLACEHOLDER|${DEVICE_LIBS}|g" /tmp/local_start.sh
sed -i "s|APP_CMD_PLACEHOLDER|${REMOTE_BIN} ${APP_ARGS[*]:-}|g" /tmp/local_start.sh
sed -i "s|APP_LOG_PLACEHOLDER|${REMOTE_LOG}|g" /tmp/local_start.sh
sed -i "s|RUN_STDIN_HOLD_PLACEHOLDER|${RUN_STDIN_HOLD}|g" /tmp/local_start.sh

# Deploy startup script
sshpass -p "${DEVICE_PASS}" scp "${ssh_opts[@]}" \
    /tmp/local_start.sh \
    "${DEVICE_USER}@${DEVICE_IP}:${startup_script}" >> "${HOST_LOG}" 2>&1

# Cleanup old files and run startup script
sshpass -p "${DEVICE_PASS}" ssh "${ssh_opts[@]}" \
    "${DEVICE_USER}@${DEVICE_IP}" \
    "echo '${DEVICE_PASS}' | sudo -S sh -c '\
        chmod +x ${REMOTE_BIN} ${startup_script}; \
        rm -f /tmp/app.pid /tmp/app.stdin.pid ${REMOTE_LOG}; \
        sh ${startup_script}'" 2>&1 | tee -a "${HOST_LOG}"

# Wait for PID file
sleep 2
set +e
APP_PID=$(timeout 5 sshpass -p "${DEVICE_PASS}" ssh "${ssh_opts[@]}" \
    "${DEVICE_USER}@${DEVICE_IP}" \
    "cat /tmp/app.pid 2>/dev/null" | tr -d '\r\n ')
set -e

if [[ -z "${APP_PID}" || ! "${APP_PID}" =~ ^[0-9]+$ ]]; then
    echo "[runner] ERROR: Failed to get valid application PID (got: '${APP_PID}')"
    echo "[runner] Checking for error output..."
    sshpass -p "${DEVICE_PASS}" ssh "${ssh_opts[@]}" \
        "${DEVICE_USER}@${DEVICE_IP}" \
        "cat ${REMOTE_LOG} 2>/dev/null || echo 'No log available'" | tee -a "${HOST_LOG}"
    exit 1
fi

echo "[runner] Application started with PID: ${APP_PID}"
{
    echo ""
    echo "=========================================="
    echo "Application Info"
    echo "=========================================="
    echo "Binary: ${APP_BINARY}"
    echo "Args: ${APP_ARGS[*]:-<none>}"
    echo "Device PID: ${APP_PID}"
    echo "Start time: $(date)"
    echo "=========================================="
    echo ""
} >> "${HOST_LOG}"

# Start log streaming in background (with timeout per iteration)
echo "[runner] Streaming application logs..."
TAIL_PID=""
(
    for i in $(seq 1 $((RUN_TIMEOUT + 10))); do
        timeout 3 sshpass -p "${DEVICE_PASS}" ssh "${ssh_opts[@]}" \
            "${DEVICE_USER}@${DEVICE_IP}" \
            "echo '${DEVICE_PASS}' | sudo -S -p '' timeout 2 tail -n 100 -f ${REMOTE_LOG} 2>/dev/null || true" 2>/dev/null || true
        sleep 0.5
    done
) >> "${HOST_LOG}" 2>&1 &
TAIL_PID=$!

# Capture CVI proc info immediately after startup if requested
if [[ "${CAPTURE_CVI_PROC}" -ne 0 ]]; then
    sleep 2  # Give app time to initialize
    echo "[runner] Capturing /proc/cvitek info..."
    {
        echo ""
        echo "=========================================="
        echo "CVI System Info (captured at startup)"
        echo "=========================================="
    } >> "${HOST_LOG}"

    timeout 15 sshpass -p "${DEVICE_PASS}" ssh "${ssh_opts[@]}" \
        "${DEVICE_USER}@${DEVICE_IP}" \
        "echo '${DEVICE_PASS}' | sudo -S -p '' sh -c '\
            echo \"=== /proc/cvitek/sys ===\";\
            cat /proc/cvitek/sys 2>&1 || echo \"N/A\";\
            echo \"\";\
            echo \"=== /proc/cvitek/vi ===\";\
            cat /proc/cvitek/vi 2>&1 || echo \"N/A\";\
            echo \"\";\
            echo \"=== /proc/cvitek/vpss ===\";\
            cat /proc/cvitek/vpss 2>&1 || echo \"N/A\";\
            echo \"\";\
            echo \"=== /proc/cvitek/venc ===\";\
            cat /proc/cvitek/venc 2>&1 || echo \"N/A\";\
            echo \"\";\
            echo \"=== /proc/cvitek/vb ===\";\
            cat /proc/cvitek/vb 2>&1 || echo \"N/A\";\
            echo \"\";\
        '" 2>/dev/null >> "${HOST_LOG}" || true
fi

# Monitor application
echo "[runner] Monitoring application (timeout: ${RUN_TIMEOUT}s)..."
ELAPSED=0
IS_RUNNING="1"

while [[ ${ELAPSED} -lt ${RUN_TIMEOUT} ]]; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))

    # Check if application is still running (with timeout)
    set +e
    IS_RUNNING=$(timeout 3 sshpass -p "${DEVICE_PASS}" ssh "${ssh_opts[@]}" \
        "${DEVICE_USER}@${DEVICE_IP}" \
        "kill -0 ${APP_PID} 2>/dev/null && echo 1 || echo 0" 2>/dev/null | tr -d '\r\n')
    check_rc=$?
    set -e

    # If check times out or fails, assume still running
    if [[ ${check_rc} -ne 0 ]]; then
        echo "[runner] WARNING: Process check timed out or failed at ${ELAPSED}s"
        continue
    fi

    if [[ "${IS_RUNNING}" != "1" ]]; then
        echo "[runner] Application exited at ${ELAPSED}s"
        break
    fi

    # Print progress every 10 seconds
    if [[ $((ELAPSED % 10)) -eq 0 ]]; then
        echo "[runner] Still running... ${ELAPSED}/${RUN_TIMEOUT}s"
    fi
done

# Kill application if still running
if [[ "${IS_RUNNING}" == "1" ]]; then
    echo "[runner] Timeout reached, terminating application..."
    sshpass -p "${DEVICE_PASS}" ssh "${ssh_opts[@]}" \
        "${DEVICE_USER}@${DEVICE_IP}" \
        "echo '${DEVICE_PASS}' | sudo -S -p '' kill -TERM ${APP_PID} 2>/dev/null || true" 2>/dev/null || true
    sleep 2
    sshpass -p "${DEVICE_PASS}" ssh "${ssh_opts[@]}" \
        "${DEVICE_USER}@${DEVICE_IP}" \
        "echo '${DEVICE_PASS}' | sudo -S -p '' kill -KILL ${APP_PID} 2>/dev/null || true" 2>/dev/null || true
fi

if [[ "${RUN_STDIN_HOLD}" -ne 0 ]]; then
    sshpass -p "${DEVICE_PASS}" ssh "${ssh_opts[@]}" \
        "${DEVICE_USER}@${DEVICE_IP}" \
        "echo '${DEVICE_PASS}' | sudo -S -p '' sh -c '\
            if [ -f /tmp/app.stdin.pid ]; then \
                kill -KILL \$(cat /tmp/app.stdin.pid) 2>/dev/null || true; \
                rm -f /tmp/app.stdin.pid /tmp/app.stdin; \
            fi' " 2>/dev/null || true
fi

# Stop log streaming
if [[ -n "${TAIL_PID}" ]]; then
    kill ${TAIL_PID} 2>/dev/null || true
    wait ${TAIL_PID} 2>/dev/null || true
fi

# Give time for final log flush
sleep 1

# Collect system status
echo "[runner] Collecting system status..."
{
    echo ""
    echo "=========================================="
    echo "System Status at $(date)"
    echo "=========================================="
    timeout 15 sshpass -p "${DEVICE_PASS}" ssh "${ssh_opts[@]}" \
        "${DEVICE_USER}@${DEVICE_IP}" \
        "echo '${DEVICE_PASS}' | sudo -S -p '' ps aux | head -20; \
         echo '---'; \
         echo '${DEVICE_PASS}' | sudo -S -p '' free -h; \
         echo '---'; \
         echo '${DEVICE_PASS}' | sudo -S -p '' df -h | grep -E '(Filesystem|mnt|tmp|userdata)'; \
         echo '---'; \
         echo '${DEVICE_PASS}' | sudo -S -p '' dmesg | tail -20" 2>/dev/null
} >> "${HOST_LOG}" || true

echo "[runner] =========================================="
echo "[runner] Run complete"
echo "[runner] Log saved to: ${HOST_LOG}"
echo "[runner] =========================================="

# Display last 50 lines of log
echo ""
echo "Last 50 lines of output:"
tail -50 "${HOST_LOG}"
