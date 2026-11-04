# SPDX-License-Identifier: GPL-2.0-only
"""Run the opt-in CUDA comparison with telemetry and bounded execution."""
import argparse
import datetime
import json
import pathlib
import shutil
import subprocess
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    directory = pathlib.Path(__file__).resolve().parent
    parser.add_argument("--exe", type=pathlib.Path, default=directory / "build/matmul-tune.exe")
    parser.add_argument("--size", choices=["4096", "8192"])
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if not args.exe.is_file():
        parser.error("Executable missing; run tools/performance/build.cmd first")
    if shutil.which("nvidia-smi") is None:
        parser.error("nvidia-smi is required on PATH")
    output = args.output or directory / "results" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    command = [str(args.exe.resolve())] + ([args.size] if args.size else [])
    fields = "timestamp,name,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,memory.used"
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stop = threading.Event()
    samples, limits = [], []

    def monitor():
        while not stop.is_set():
            try:
                query = subprocess.run(
                    ["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5, check=False,
                )
                for line in query.stdout.splitlines():
                    if not line.strip():
                        continue
                    samples.append(line)
                    try:
                        if float(line.split(",")[3]) >= 85:
                            limits.append("Temperature reached 85 C")
                            process.terminate()
                    except (IndexError, ValueError):
                        pass
            except (OSError, subprocess.TimeoutExpired) as error:
                limits.append("Telemetry unavailable: " + str(error))
                process.terminate()
            if time.monotonic() - started > 120:
                limits.append("Runtime exceeded 120 seconds")
                process.terminate()
            stop.wait(0.5)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    lines = []
    try:
        for line in process.stdout:
            print(line.rstrip(), flush=True)
            lines.append(line)
        errors = process.stderr.read()
        code = process.wait()
    finally:
        stop.set()
        if process.poll() is None:
            process.terminate()
        thread.join(timeout=6)
    (output / "results.jsonl").write_text("".join(lines), encoding="utf-8")
    (output / "stderr.txt").write_text(errors, encoding="utf-8")
    (output / "telemetry.csv").write_text(fields + "\n" + "\n".join(samples) + "\n", encoding="utf-8")
    metadata = {
        "time": datetime.datetime.now().astimezone().isoformat(),
        "duration_seconds": time.monotonic() - started,
        "exit_code": code, "limits": limits,
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Saved:", output)
    if errors:
        print(errors)
    if code or limits:
        raise SystemExit(code or 1)


if __name__ == "__main__":
    main()
