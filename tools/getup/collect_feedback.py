"""Collect compact Getup training feedback; no GPU CSV or checkpoint upload."""
import argparse
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", default="output/a5000_getup_2048_full_r01")
    args = parser.parse_args()
    folder = Path(args.run_dir)
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    blocks = []
    for name in ("runtime.txt", "run_manifest.json", "engine_config.yaml",
                 "env_config.yaml", "agent_config.yaml"):
        path = folder / name
        blocks.append(f"=== {name} ===\n" + (path.read_text(encoding="utf-8") if path.is_file() else "not yet available\n"))
    log = folder / "log.txt"
    if log.is_file() and log.stat().st_size:
        lines = [line.split() for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
        headers, data = lines[0], lines[1:]
        rows = [dict(zip(headers, values)) for values in data if len(values) == len(headers)]
        nonfinite = []
        for row in rows:
            for key, value in row.items():
                try:
                    if not math.isfinite(float(value)):
                        nonfinite.append((row.get("Iteration"), key, value))
                except ValueError:
                    pass
        blocks.append(f"=== log audit ===\nrows={len(rows)} incomplete_rows={len(data)-len(rows)} "
                      f"nonfinite={nonfinite[:20]}\n")
        indices = sorted(set([0, len(rows) // 2, *range(max(0, len(rows) - 8), len(rows))]))
        fields = [key for key in headers if key in (
            "Iteration", "Samples", "Wall_Time", "Samples_Per_Second", "Train_Return",
            "Test_Return", "Train_Episode_Length", "Test_Episode_Length")
            or key.lower().startswith("getup_") or "smp" in key.lower() or "sds" in key.lower()
            or "task_reward" in key.lower()]
        blocks.append("=== selected training rows ===\n" + "\n".join(
            " | ".join(f"{key}={rows[i][key]}" for key in fields) for i in indices if i < len(rows)) + "\n")
    console = folder / "console.txt"
    if console.is_file():
        blocks.append("=== console last 45 lines ===\n" + "\n".join(
            console.read_text(encoding="utf-8", errors="replace").splitlines()[-45:]) + "\n")
    path = folder / "feedback.txt"
    path.write_text("\n".join(blocks), encoding="utf-8")
    print(path.resolve())


if __name__ == "__main__":
    main()
