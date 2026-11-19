"""Check RSP dependency boundaries and an optional Solver relocation baseline.

This check parses source only: it imports no simulation or rendering modules.
"""
import argparse
import ast
from importlib.util import resolve_name
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
MIMICKIT = ("engines", "envs", "learning", "util", "mimickit")
FORBIDDEN = {
    "sim_ready": (*MIMICKIT, "RSP.app", "RSP.physics", "RSP.simulation", "RSP.integrations", "RSP.runtime", "RSP.launcher"),
    "rendering": (*MIMICKIT, "RSP.app", "RSP.integrations", "RSP.runtime", "RSP.launcher"),
    "simulation": (*MIMICKIT, "RSP.app", "RSP.integrations", "RSP.runtime", "RSP.launcher"),
    "physics": (*MIMICKIT, "RSP.app", "RSP.simulation", "RSP.integrations", "RSP.runtime", "RSP.launcher"),
    "app": MIMICKIT,
}


def check_imports():
    failures = []
    count = 0
    for domain, banned in FORBIDDEN.items():
        for path in sorted((ROOT / "RSP" / domain).rglob("*.py")):
            count += 1
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [item.name for item in node.names]
                elif isinstance(node, ast.ImportFrom):
                    package = ".".join(path.parent.relative_to(ROOT).parts)
                    module = resolve_name("." * node.level + (node.module or ""), package)
                    names = [module, *(f"{module}.{item.name}" for item in node.names)]
                for name in names:
                    if name and any(name == prefix or name.startswith(prefix + ".") for prefix in banned):
                        failures.append(f"{path.relative_to(ROOT)}:{node.lineno}: forbidden import {name}")
    return count, failures


def check_solver(baseline):
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", baseline, "mimickit/solvers/pbf"],
        cwd=ROOT, text=True).splitlines()
    paths += ["RSP/runtime/coupling.py", "RSP/runtime/fluid_clock.py"]
    failures = []
    for source in paths:
        target = (source.replace("mimickit/solvers/pbf", "RSP/physics/pbf")
                  .replace("RSP/runtime/coupling.py", "RSP/physics/clocked_coupling.py")
                  .replace("RSP/runtime/fluid_clock.py", "RSP/physics/fluid_clock.py"))
        expected = subprocess.check_output(["git", "show", f"{baseline}:{source}"], cwd=ROOT).decode("utf-8")
        expected = expected.replace("\r\n", "\n")
        if source.endswith("boundary_model.py"):
            expected = expected.replace("RSP.sim_ready.pipelines.boundary.schema", "RSP.sim_ready.boundary.schema")
            expected = expected.replace("RSP.sim_ready.pipelines.boundary.load", "RSP.sim_ready.boundary.load")
        if source == "RSP/runtime/fluid_clock.py":
            expected = expected.replace("from solvers.pbf.pbf_helper", "from .pbf.pbf_helper")
        if source == "RSP/runtime/coupling.py":
            expected = expected.replace("from solvers.pbf.coupling", "from .pbf.coupling")
        if source.endswith("coupling.py"):
            expected = expected.replace('"solvers.pbf.', '"RSP.physics.pbf.')
        path = ROOT / target
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            failures.append(f"{target}: differs beyond authorized relocation/import changes")
    return len(paths), failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-baseline", help="Git revision before the directory-only Solver migration")
    args = parser.parse_args()
    count, failures = check_imports()
    if args.solver_baseline:
        solver_count, solver_failures = check_solver(args.solver_baseline)
        failures.extend(solver_failures)
        print(f"Checked {solver_count} Solver files against {args.solver_baseline}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"PASS: {count} RSP modules respect dependency boundaries")


if __name__ == "__main__":
    main()
