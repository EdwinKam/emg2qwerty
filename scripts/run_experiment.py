"""Run a checked-in emg2qwerty experiment manifest."""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import yaml


REPO = Path(__file__).resolve().parents[1]
SLUG = re.compile(r"[a-z0-9][a-z0-9_-]*")
LAUNCHER_KEYS = {
    "user", "dataset", "train", "checkpoint", "seed", "trainer.accelerator", "trainer.devices",
    "trainer.max_epochs", "evaluate_test", "write_results", "benchmark_inference",
}


@dataclass(frozen=True)
class Case:
    name: str
    frontend_context_ms: int | None
    overrides: tuple[str, ...]


@dataclass(frozen=True)
class Experiment:
    path: Path
    name: str
    title: str
    split: str
    seed: int
    epochs: int
    overrides: tuple[str, ...]
    cases: tuple[Case, ...]
    comparisons: tuple[dict[str, str], ...]


def require_slug(value: str, label: str) -> str:
    if not isinstance(value, str) or not SLUG.fullmatch(value):
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def read_overrides(values: list[str]) -> tuple[str, ...]:
    if not isinstance(values, list):
        raise ValueError("Overrides must be a list of Hydra key=value strings.")
    for override in values:
        if not isinstance(override, str) or "=" not in override:
            raise ValueError(f"Invalid Hydra override: {override!r}")
        key = override.split("=", 1)[0].lstrip("+~")
        if key in LAUNCHER_KEYS or key.startswith(("hydra.", "hydra/", "dataset.")):
            raise ValueError(f"{key} is set by the runner or split. Use the manifest fields.")
    return tuple(values)


def read_experiment(path: Path) -> Experiment:
    path = path.resolve()
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or not isinstance(config.get("cases"), dict):
        raise ValueError("A manifest must contain a mapping of named cases.")
    cases = tuple(
        Case(
            require_slug(name, "case name"),
            case.get("frontend_context_ms"),
            read_overrides(case.get("overrides", [])),
        )
        for name, case in config["cases"].items()
    )
    names = {case.name for case in cases}
    if not names or "selected-test" in names:
        raise ValueError("Declare at least one case. The name selected-test is reserved.")
    comparisons = tuple(config.get("comparisons", []))
    for pair in comparisons:
        if pair["before"] not in names or pair["after"] not in names:
            raise ValueError(f"Comparison refers to an unknown case: {pair}")
    return Experiment(
        path, require_slug(config["id"], "experiment ID"), config["title"],
        require_slug(config["split"], "split"), config["seed"], config["epochs"],
        read_overrides(config.get("overrides", [])), cases, comparisons,
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def write_json(path: Path, data: dict) -> None:
    write_text(path, json.dumps(data, indent=2) + "\n")


def split_sessions(experiment: Experiment) -> dict[str, list[str]]:
    path = REPO / "config" / "user" / f"{experiment.split}.yaml"
    if not path.is_file():
        path = experiment.path.parent / "user" / f"{experiment.split}.yaml"
    split = yaml.safe_load(path.read_text())["dataset"]
    sessions = {
        phase: [row["session"] + ".hdf5" for row in split[phase]]
        for phase in ("train", "val", "test")
    }
    names = [name for files in sessions.values() for name in files]
    if len(set(names)) != len(names) or not all(sessions.values()):
        raise ValueError("The split must have disjoint, nonempty train, val, and test sessions.")
    return sessions


def dataset_manifest(root: Path, sessions: dict[str, list[str]]) -> dict:
    manifest = {}
    for split, names in sessions.items():
        manifest[split] = []
        for name in names:
            path = root / name
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(block)
            manifest[split].append({
                "file": name,
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            })
    return manifest


def case_command(
    experiment: Experiment, case: Case, data: Path, run_dir: Path, args: argparse.Namespace
) -> list[str]:
    command = [
        sys.executable, "-u", "-m", "emg2qwerty.train",
        "--config-dir", str(experiment.path.parent),
        f"user={experiment.split}", f"dataset.root={json.dumps(str(data))}",
        *experiment.overrides, *case.overrides,
        f"seed={args.seed}", f"trainer.accelerator={args.accelerator}", "trainer.devices=1",
        f"trainer.max_epochs={args.epochs}",
        "+trainer.enable_progress_bar=false",
        "evaluate_test=false", "write_results=true", "benchmark_inference=true",
        "hydra.job.chdir=true", f"hydra.run.dir={json.dumps(str(run_dir))}",
    ]
    if args.smoke:
        command += [
            "batch_size=4", "num_workers=0",
            "+trainer.limit_train_batches=2", "+trainer.limit_val_batches=1",
        ]
    return command


def run(command: list[str], directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "command.json", {"argv": command})
    started = monotonic()
    with (directory / "console.log").open("a") as log:
        subprocess.run(
            command,
            cwd=REPO,
            env={**os.environ, "HYDRA_FULL_ERROR": "1", "MPLBACKEND": "Agg"},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    result = read_json(directory / "results.json")
    write_json(directory / "invocation.json", {"wall_seconds": monotonic() - started})
    return result


def validation_cer(result: dict) -> float:
    return result["val_metrics"][0]["val/CER"]


def summarize(
    output: Path,
    experiment: Experiment,
    results: dict[Case, dict],
    failures: dict[str, str],
    winner: dict | None,
    smoke: bool,
) -> None:
    cases = experiment.cases
    rows = [
        f"# {experiment.title}",
        "",
        f"{len(results)} of {len(cases)} configurations have validation results.",
        "",
        "| Configuration | Frontend context | Validation CER | Parameters | Input frames | Median forward pass | Device |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    best = min(results, key=lambda case: validation_cer(results[case]), default=None)
    for case in cases:
        result = results.get(case)
        context = f"{case.frontend_context_ms} ms" if case.frontend_context_ms else "—"
        if result is None:
            state = "Failed" if case.name in failures else "Pending"
            rows.append(f"| {case.name} | {context} | {state} | | | | |")
            continue
        timing = result["forward_pass"]
        rows.append(
            f"| {case.name} | {context} | {validation_cer(result):.2f}% "
            f"| {result['trainable_parameters']:,} | {timing['frames']} | {timing['median_ms']:.2f} ms "
            f"| {timing['device']} |"
        )
    rows += [
        "",
        "Forward-pass time measures one prepared spectrogram and excludes feature "
        "extraction, CTC decoding and live typing delay.",
        "",
    ]
    if experiment.comparisons:
        rows.append("Negative CER differences indicate fewer errors.")
    by_name = {case.name: result for case, result in results.items()}
    for pair in experiment.comparisons:
        before = by_name.get(pair["before"])
        after = by_name.get(pair["after"])
        if before is None or after is None:
            continue
        difference = validation_cer(after) - validation_cer(before)
        comparison = f"- {pair['label']} changed validation CER by {difference:+.2f} pp"
        before_timing = before["forward_pass"]
        after_timing = after["forward_pass"]
        if before_timing["device"] == after_timing["device"]:
            time_change = after_timing["median_ms"] - before_timing["median_ms"]
            comparison += f" and median forward time by {time_change:+.2f} ms"
        rows.append(comparison + ".")
    devices = {result["forward_pass"]["device"] for result in results.values()}
    if len(devices) > 1:
        rows += ["", "The sweep used multiple device models. Compare timing only within the same device."]
    if best is not None:
        rows += ["", f"Lowest observed validation CER belongs to **{best.name}**."]
    if smoke:
        rows.append("Smoke test only. This run skips held-out testing and has a reduced training budget.")
    elif winner is not None:
        test = winner["test_metrics"][0]["test/CER"]
        rows.append(f"The validation-selected checkpoint scored **{test:.2f}% test CER**.")
    elif len(results) == len(cases):
        rows.append("Held-out test evaluation is pending.")
    for name, reason in failures.items():
        rows.append(f"- {name} failed. {reason}")
    rows += ["", f"Split: `{experiment.split}`.", ""]
    write_text(output / "summary.md", "\n".join(rows))


def read_completed(directory: Path) -> dict | None:
    path = directory / "results.json"
    if not path.exists():
        return None
    result = read_json(path)
    if not math.isfinite(validation_cer(result)) or not Path(result["best_checkpoint"]).is_file():
        raise ValueError(f"Invalid validation CER or missing best checkpoint for {directory.name}.")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True, help="Experiment YAML file")
    parser.add_argument("--data", type=Path, required=True, help="Extracted HDF5 session directory")
    output_args = parser.add_mutually_exclusive_group()
    output_args.add_argument("--output", type=Path, help="Exact run directory")
    output_args.add_argument("--output-root", type=Path, help="Directory for experiment runs")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--accelerator", choices=["gpu", "cpu"], default="gpu")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan", action="store_true", help="Print launch commands without running")
    modes.add_argument("--check-data", action="store_true", help="Check which split files are present")
    parser.add_argument("--smoke", action="store_true", help="Train briefly and skip held-out testing")
    args = parser.parse_args(argv)
    try:
        experiment = read_experiment(args.experiment)
        sessions = split_sessions(experiment)
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        parser.error(f"Cannot load experiment: {exc}")
    data = args.data.resolve()
    args.seed = experiment.seed if args.seed is None else args.seed
    if args.epochs is None:
        args.epochs = experiment.epochs
    if args.smoke:
        args.epochs = 1
    if args.output_root is not None:
        run_name = f"{experiment.name}-seed{args.seed}" + ("-smoke" if args.smoke else "")
        args.output = args.output_root / run_name
    if not args.check_data and args.output is None:
        parser.error("--output or --output-root is required")
    if args.plan:
        commands = {
            case.name: case_command(experiment, case, data, args.output.resolve() / case.name, args)
            for case in experiment.cases
        }
        print(json.dumps(commands, indent=2))
        return
    missing = [name for names in sessions.values() for name in names if not (data / name).is_file()]
    if missing:
        raise SystemExit(f"Missing {len(missing)} session files under {data}. First: {missing[0]}")
    if args.check_data:
        print("Session files found. " + ", ".join(f"{phase}={len(names)}" for phase, names in sessions.items()))
        return
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    output = args.output.resolve()
    git = ["git", "-C", str(REPO)]
    try:
        relative_experiment = experiment.path.relative_to(REPO)
    except ValueError:
        parser.error("Put the experiment manifest inside the repository.")
    tracked_manifest = subprocess.run(
        git + ["cat-file", "-e", f"HEAD:{relative_experiment.as_posix()}"],
        capture_output=True,
    )
    if tracked_manifest.returncode:
        parser.error("Commit the experiment manifest before launching a run.")
    tracked_changes = subprocess.check_output(
        git + ["status", "--porcelain", "--untracked-files=no"], text=True
    )
    if tracked_changes.strip():
        raise SystemExit("Commit tracked changes before launching the sweep.")
    print("Checking session files and recording provenance.", flush=True)
    manifest = {
        "commit": subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "epochs": args.epochs,
        "accelerator": args.accelerator,
        "seed": args.seed,
        "experiment": experiment.name,
        "experiment_file": relative_experiment.as_posix(),
        "smoke": args.smoke,
        "dataset": dataset_manifest(data, sessions),
        "definition": yaml.safe_load(experiment.path.read_text()),
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        old_manifest = read_json(manifest_path)
        # Kernel versions can differ on replacement runtimes.
        old_manifest["platform"] = manifest["platform"]
        if old_manifest != manifest:
            raise SystemExit("Existing sweep has different provenance. Choose a new output directory.")
    else:
        write_json(manifest_path, manifest)
    packages = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze", "--exclude-editable"], text=True
    )
    packages_path = output / "packages.txt"
    if packages_path.exists() and packages_path.read_text() != packages:
        raise SystemExit("Package versions differ. Restore packages.txt or choose a new output directory.")
    write_text(packages_path, packages)

    results = {}
    failures = {}
    if not (output / "summary.md").is_file():
        summarize(output, experiment, results, failures, None, args.smoke)
    for case in experiment.cases:
        directory = output / case.name
        try:
            completed = read_completed(directory)
            if completed is None:
                print(f"Training {case.name}. Live log: {directory / 'console.log'}", flush=True)
                command = case_command(experiment, case, data, directory, args)
                run(command, directory)
                completed = read_completed(directory)
                if completed is None:
                    raise ValueError(f"Missing results for {case.name}.")
            results[case] = completed
            print(f"{case.name}: {validation_cer(completed):.2f}% validation CER", flush=True)
        except (subprocess.CalledProcessError, ValueError) as error:
            failures[case.name] = f"{type(error).__name__}. See {case.name}/console.log."
            print(failures[case.name], file=sys.stderr, flush=True)
        summarize(output, experiment, results, failures, None, args.smoke)
    if failures:
        raise SystemExit(f"{len(failures)} incomplete configurations. See {output / 'summary.md'}.")

    if args.smoke:
        print(f"Smoke test passed. Report: {output / 'summary.md'}", flush=True)
        return

    best = min(results, key=lambda case: validation_cer(results[case]))
    directory = output / "selected-test"
    test_path = directory / "results.json"
    if test_path.is_file():
        winner = read_json(test_path)
    else:
        command = case_command(experiment, best, data, directory, args)
        command[command.index("evaluate_test=false")] = "evaluate_test=true"
        command[command.index("benchmark_inference=true")] = "benchmark_inference=false"
        command += ["train=false", f"checkpoint={json.dumps(results[best]['best_checkpoint'])}"]
        print(f"Testing validation winner {best.name}.", flush=True)
        winner = run(command, directory)
    summarize(output, experiment, results, failures, winner, args.smoke)
    print((output / "summary.md").read_text(), flush=True)
    print(f"Report: {output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
