#!/usr/bin/env python3
"""Command-line entry point for the BUD paper's reproducibility package."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
TARGETS = ("simulation", "gbsg", "supplement")
ENV = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    # Parallelism is at the cohort/draw level; keep numerical libraries single-threaded.
    ENV[variable] = "1"


def call(script, *args):
    subprocess.run([sys.executable, "-u", str(ROOT / "code" / f"{script}.py"), *map(str, args)],
                   cwd=ROOT, env=ENV, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command, description in (
        ("doctor", "Check installed dependencies and the requested backend."),
        ("smoke", "Run reduced-size checks in smoke_results/."),
        ("run", "Run the paper's complete experiment protocols."),
        ("verify", "Compare full outputs with the paper reference values."),
        ("data", "Download and verify the two public application data files."),
    ):
        child = commands.add_parser(command, help=description)
        if command in {"smoke", "run", "verify"}:
            child.add_argument("--target", choices=("all", *TARGETS), default="all")
        if command in {"smoke", "run", "doctor"}:
            child.add_argument("--backend", choices=("c", "python"), default="c")
        if command in {"smoke", "run"}:
            child.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
        if command == "verify":
            child.add_argument("--outputs", type=Path, default=ROOT,
                               help="Path to the checkout containing results/ and figures/.")
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        parser.error("Python 3.12 or later is required. Use the versions in requirements.txt.")

    if args.command == "doctor":
        call("verify_results", "doctor", "--backend", args.backend)
    elif args.command == "verify":
        call("verify_results", "verify", "--target", args.target, "--outputs", args.outputs.resolve())
    elif args.command == "data":
        call("reproduce_gbsg", "--data-only")
    else:
        if args.workers < 1:
            parser.error("--workers must be at least 1")
        targets = TARGETS if args.target == "all" else (args.target,)
        for target in targets:
            options = ["--quick"] if args.command == "smoke" else []
            call(f"reproduce_{target}", "--backend", args.backend, "--workers", args.workers, *options)
        if args.command == "run":
            call("verify_results", "verify", "--target", args.target, "--outputs", ROOT)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(f"Stage failed with exit status {error.returncode}. See the stage output above.", file=sys.stderr)
        raise SystemExit(error.returncode or 1) from None
