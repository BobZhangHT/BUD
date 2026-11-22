"""Verify numerical inputs to the paper tables and figures, excluding runtime metadata."""

import argparse
import csv
from hashlib import sha256
from importlib.metadata import version
import json
from math import isclose
from pathlib import Path
import platform


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference_results.json"
# Tight enough to catch a changed draw or reported rounded value, while tolerating
# harmless floating-point differences between supported compiler/BLAS builds.
RTOL = 2e-9
ATOL = 1e-11


def extract(root, check):
    path = root / check["path"]
    if path.suffix == ".pdf":
        # PDFs carry build-dependent metadata. Their underlying numerical inputs
        # are verified separately, so here we check that the output was rendered.
        with path.open("rb") as file:
            if file.read(5) != b"%PDF-":
                raise ValueError(f"Invalid PDF header in {path}")
        return None
    if path.suffix == ".tex":
        return sha256(path.read_text(encoding="utf8").strip().encode("utf8")).hexdigest()
    if path.suffix == ".csv":
        with path.open(encoding="utf8", newline="") as file:
            rows = csv.DictReader(file)
            return sorted([row[column] for column in check["columns"]] for row in rows
                          if all(row[key] in values for key, values in check.get("where", {}).items()))
    values = json.loads(path.read_bytes())
    selected = []
    for fields in check["fields"]:
        value = values
        for field in fields:
            value = value[field]
        selected.append(value)
    return selected


def compare(actual, expected, where):
    if type(expected) is dict:
        if type(actual) is not dict or actual.keys() != expected.keys():
            raise ValueError(f"{where}: changed object keys")
        for key, reference in expected.items():
            compare(actual[key], reference, f"{where}.{key}")
    elif type(expected) is list:
        if type(actual) is not list or len(actual) != len(expected):
            raise ValueError(f"{where}: expected {len(expected)} items")
        for index, (candidate, reference) in enumerate(zip(actual, expected)):
            compare(candidate, reference, f"{where}[{index}]")
    elif type(expected) is bool or type(actual) is bool:
        if actual is not expected:
            raise ValueError(f"{where}: expected {expected!r}, found {actual!r}")
    elif actual != expected:
        # CSV readers return strings. Compare numbers as numbers, but keep
        # labels, exact booleans, and missing-value markers as literal values.
        try:
            if isclose(float(actual), float(expected), rel_tol=RTOL, abs_tol=ATOL):
                return
        except (TypeError, ValueError):
            pass
        raise ValueError(f"{where}: expected {expected!r}, found {actual!r}")


def verify(target, outputs):
    reference = json.loads(REFERENCE.read_bytes(), parse_constant=lambda value: value)
    targets = reference if target == "all" else {target: reference[target]}
    failures = []
    for name, checks in targets.items():
        for check in checks:
            try:
                compare(extract(outputs, check), check.get("expected"), check["path"])
            except (OSError, KeyError, ValueError) as error:
                failures.append(f"{name}: {error}")
        print(f"{name}: checked {len(checks)} output files")
    if failures:
        raise SystemExit("Verification failed:\n" + "\n".join(failures))
    print("All selected paper outputs matched the reference.")


def doctor(backend):
    from bud_experiment import _new_backend

    print(f"Python {platform.python_version()} | {platform.platform()}", flush=True)
    for package in ("numpy", "scipy", "pandas", "scikit-learn", "matplotlib"):
        print(f"{package} {version(package)}", flush=True)
    configuration = _new_backend(backend).metadata()
    if backend == "c":
        from design_backend import native_metadata
        configuration.update(native_metadata())
    print(json.dumps(configuration, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("doctor").add_argument("--backend", choices=("c", "python"), required=True)
    verify_parser = modes.add_parser("verify")
    verify_parser.add_argument("--target", choices=("all", "simulation", "gbsg", "supplement"), required=True)
    verify_parser.add_argument("--outputs", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "doctor":
        doctor(args.backend)
    else:
        verify(args.target, args.outputs)
