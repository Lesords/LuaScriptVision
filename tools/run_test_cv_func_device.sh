#!/usr/bin/env bash
set -euo pipefail

DEVICE_IP="${DEVICE_IP:-192.168.42.1}"
DEVICE_USER="${DEVICE_USER:-recamera}"
DEVICE_PASS="${DEVICE_PASS:-11}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TEST_BIN="${TEST_BIN:-${ROOT_DIR}/build/test_cv_func}"
IMAGE_PATH="${IMAGE_PATH:-${ROOT_DIR}/images/zidane.jpg}"
RUNNER_PATH="${RUNNER_PATH:-${ROOT_DIR}/scripts/run_test_cv_func.py}"
RUN_TIMEOUT="${RUN_TIMEOUT:-60}"
RUN_INTERVAL="${RUN_INTERVAL:-10}"
RUN_SSH_KILL="${RUN_SSH_KILL:-5}"
RUN_SSH_TIMEOUT="${RUN_SSH_TIMEOUT:-$((RUN_TIMEOUT + 10))}"

REMOTE_DIR="/tmp"
REMOTE_BIN="${REMOTE_DIR}/test_cv_func"
REMOTE_IMAGE="${REMOTE_DIR}/$(basename "${IMAGE_PATH}")"
REMOTE_RUNNER="${REMOTE_DIR}/run_test_cv_func.py"
REMOTE_LOG="${REMOTE_DIR}/test_cv_func.log"

LOG_DIR="${LOG_DIR:-${ROOT_DIR}/debug/device_logs}"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
HOST_LOG="${LOG_DIR}/test_cv_func_${TIMESTAMP}.log"

NO_REBOOT=0
for arg in "$@"; do
    case "${arg}" in
        --no-reboot)
            NO_REBOOT=1
            ;;
        *)
            echo "Unknown argument: ${arg}"
            echo "Usage: $0 [--no-reboot]"
            exit 1
            ;;
    esac
done

require_file() {
    local path="$1"
    if [[ ! -f "${path}" ]]; then
        echo "Missing file: ${path}"
        exit 1
    fi
}

require_cmd() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "Missing dependency: ${cmd}"
        exit 1
    fi
}

require_cmd sshpass
require_cmd ssh
require_cmd scp
require_cmd timeout

require_file "${TEST_BIN}"
require_file "${IMAGE_PATH}"
require_file "${RUNNER_PATH}"

ssh_opts=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5)

if [[ "${NO_REBOOT}" -eq 0 ]]; then
    echo "[runner] rebooting device..."
    sshpass -p "${DEVICE_PASS}" ssh "${ssh_opts[@]}" \
        "${DEVICE_USER}@${DEVICE_IP}" \
        "echo '${DEVICE_PASS}' | sudo -S -p '' reboot" || true

    echo "[runner] waiting for device..."
    for _ in $(seq 1 60); do
        sleep 2
        if sshpass -p "${DEVICE_PASS}" ssh "${ssh_opts[@]}" \
            "${DEVICE_USER}@${DEVICE_IP}" "echo up" >/dev/null 2>&1; then
            echo "[runner] device up"
            break
        fi
    done
fi

echo "[runner] deploying files..."
sshpass -p "${DEVICE_PASS}" scp "${ssh_opts[@]}" \
    "${TEST_BIN}" "${IMAGE_PATH}" "${RUNNER_PATH}" \
    "${DEVICE_USER}@${DEVICE_IP}:${REMOTE_DIR}/"

echo "[runner] running test_cv_func with watchdog (timeout ${RUN_TIMEOUT}s)..."
remote_cmd="if command -v timeout >/dev/null 2>&1; then \
echo '${DEVICE_PASS}' | sudo -S -p '' timeout -k 5 ${RUN_TIMEOUT} \
python3 ${REMOTE_RUNNER} --binary ${REMOTE_BIN} --image ${REMOTE_IMAGE} \
--log ${REMOTE_LOG} --timeout ${RUN_TIMEOUT} --interval ${RUN_INTERVAL}; \
else \
echo '${DEVICE_PASS}' | sudo -S -p '' python3 ${REMOTE_RUNNER} --binary ${REMOTE_BIN} \
--image ${REMOTE_IMAGE} --log ${REMOTE_LOG} \
--timeout ${RUN_TIMEOUT} --interval ${RUN_INTERVAL}; \
fi"
set +e
timeout -k "${RUN_SSH_KILL}" "${RUN_SSH_TIMEOUT}" \
    sshpass -p "${DEVICE_PASS}" ssh "${ssh_opts[@]}" \
    "${DEVICE_USER}@${DEVICE_IP}" \
    "${remote_cmd}"
run_rc=$?
set -e
if [[ "${run_rc}" -eq 124 ]]; then
    echo "[runner] run step exceeded ${RUN_SSH_TIMEOUT}s; continuing to collect logs"
fi

echo "[runner] collecting log..."
set +e
timeout -k 5 20 \
    sshpass -p "${DEVICE_PASS}" ssh "${ssh_opts[@]}" \
    "${DEVICE_USER}@${DEVICE_IP}" \
    "echo '${DEVICE_PASS}' | sudo -S -p '' cat ${REMOTE_LOG}" > "${HOST_LOG}"
log_rc=$?
set -e
if [[ "${log_rc}" -ne 0 ]]; then
    echo "[runner] failed to fetch log (rc=${log_rc})"
fi

echo "[runner] log saved to ${HOST_LOG}"
