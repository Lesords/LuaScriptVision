#!/usr/bin/env python3
import argparse
import os
import subprocess
import time


def append_log(log_path, text):
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(text)
        if not text.endswith("\n"):
            log.write("\n")


def read_proc(path, max_bytes):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read(max_bytes)
    except Exception as exc:
        return "[runner] failed to read %s: %s\n" % (path, exc)


def dump_proc(log_path, tag, proc_paths, max_bytes):
    append_log(log_path, tag)
    for path in proc_paths:
        append_log(log_path, "==== %s ====" % path)
        append_log(log_path, read_proc(path, max_bytes))


def parse_args():
    parser = argparse.ArgumentParser(description="Run test_cv_func with watchdog logging.")
    parser.add_argument("--binary", default="/tmp/test_cv_func", help="Path to test binary")
    parser.add_argument("--image", default="/tmp/zidane.jpg", help="Path to JPEG image")
    parser.add_argument("--log", default="/tmp/test_cv_func.log", help="Output log path")
    parser.add_argument("--interval", type=float, default=10.0, help="Proc dump interval (s)")
    parser.add_argument("--timeout", type=float, default=180.0, help="Timeout before kill (s)")
    parser.add_argument("--max-proc-bytes", type=int, default=16384,
                        help="Max bytes to read per /proc file")
    return parser.parse_args()


def main():
    args = parse_args()
    log_path = args.log
    proc_paths = [
        "/proc/cvitek/sys",
        "/proc/cvitek/vpss",
        "/proc/cvitek/vb",
        "/proc/cvitek/vi",
        "/proc/cvitek/vdec",
    ]

    try:
        os.remove(log_path)
    except FileNotFoundError:
        pass

    append_log(log_path, "[runner] start")

    env = os.environ.copy()
    env["LUA_VB_TRACE"] = env.get("LUA_VB_TRACE", "1")

    cmd = [args.binary, "--image", args.image]
    with open(log_path, "a", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)

    start = time.monotonic()
    next_dump = start + args.interval

    while True:
        rc = proc.poll()
        if rc is not None:
            append_log(log_path, "[runner] exit code %s" % rc)
            dump_proc(log_path, "[runner] proc snapshot after exit",
                      proc_paths, args.max_proc_bytes)
            break

        now = time.monotonic()
        if now >= next_dump:
            elapsed = int(now - start)
            dump_proc(log_path, "[runner] still running after %ss" % elapsed,
                      proc_paths, args.max_proc_bytes)
            next_dump = now + args.interval

        if now - start >= args.timeout:
            append_log(log_path, "[runner] timeout, killing test_cv_func")
            try:
                proc.terminate()
                time.sleep(1)
            except Exception:
                pass
            try:
                proc.kill()
            except Exception:
                pass
            dump_proc(log_path, "[runner] proc snapshot after timeout",
                      proc_paths, args.max_proc_bytes)
            break

        time.sleep(1)

    append_log(log_path, "[runner] done")

    with open(log_path, "r", encoding="utf-8", errors="replace") as log:
        tail = log.read()[-4096:]
    print(tail)


if __name__ == "__main__":
    main()
