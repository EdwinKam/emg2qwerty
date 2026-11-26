"""Run the four single-user TDS context/depth experiments sequentially."""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import yaml


REPO = Path(__file__).resolve().parents[1]
SEED = 1501


@dataclass(frozen=True)
class Case:
    blocks: int
    kernel: int

    @property
    def name(self) -> str:
        return f"tds-{self.blocks}x{self.kernel}"

    @property
    def context_ms(self) -> int:
        return 32 + 8 * self.blocks * (self.kernel - 1)


CASES = (Case(4, 32), Case(4, 16), Case(2, 63), Case(2, 31))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def write_json(path: Path, data: dict) -> None:
    write_text(path, json.dumps(data, indent=2) + "\n")


def dataset_manifest(root: Path) -> dict:
    split_config = yaml.safe_load((REPO / "config/user/single_user.yaml").read_text())
    manifest = {}
    for split, sessions in split_config["dataset"].items():
        manifest[split] = []
        for session in sessions:
            path = root / (session["session"] + ".hdf5")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(block)
            manifest[split].append({
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            })
    return manifest


def case_command(
    case: Case, data: Path, run_dir: Path, epochs: int, accelerator: str, seed: int
) -> list[str]:
    channels = ",".join(["24"] * case.blocks)
    return [
        sys.executable, "-u", "-m", "emg2qwerty.train",
        "user=single_user", "model=tds_conv_ctc", "decoder=ctc_greedy",
        f"dataset.root={json.dumps(str(data))}",
        f"module.block_channels=[{channels}]", f"module.kernel_width={case.kernel}",
        f"seed={seed}", "batch_size=32", "num_workers=2",
        f"trainer.accelerator={accelerator}", "trainer.devices=1",
        f"trainer.max_epochs={epochs}",
        "+trainer.enable_progress_bar=false",
        "evaluate_test=false", "write_results=true", "benchmark_inference=true",
        "hydra.job.chdir=true", f"hydra.run.dir={json.dumps(str(run_dir))}",
    ]


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
    output: Path, results: dict[Case, dict], failures: dict[str, str], winner: dict | None
) -> None:
    rows = [
        "# TDS context and depth sweep",
        "",
        f"{len(results)} of {len(CASES)} configurations have validation results.",
        "",
        "| Configuration | Raw signal span | Validation CER | Parameters | Median forward pass | Device |",
        "|---|---:|---:|---:|---:|---|",
    ]
    best = min(results, key=lambda case: validation_cer(results[case]), default=None)
    for case in CASES:
        result = results.get(case)
        if result is None:
            state = "Failed" if case.name in failures else "Pending"
            rows.append(f"| {case.name} | {case.context_ms} ms | {state} | | | |")
            continue
        timing = result["forward_pass"]
        rows.append(
            f"| {case.name} | {case.context_ms} ms | {validation_cer(result):.2f}% "
            f"| {result['trainable_parameters']:,} | {timing['median_ms']:.2f} ms "
            f"| {timing['device']} |"
        )
    rows += [
        "",
        "Forward-pass time uses one 497-frame spectrogram and excludes feature "
        "extraction, CTC decoding and live typing delay.",
        "",
        "CER changes below are signed percentage-point differences. Negative values improve CER.",
    ]
    comparisons = (
        ("Longer context at four blocks", Case(4, 16), Case(4, 32)),
        ("Longer context at two blocks", Case(2, 31), Case(2, 63)),
        ("Four versus two blocks at ~0.5 s", Case(2, 31), Case(4, 16)),
        ("Four versus two blocks at ~1.0 s", Case(2, 63), Case(4, 32)),
    )
    for label, before, after in comparisons:
        if before in results and after in results:
            difference = validation_cer(results[after]) - validation_cer(results[before])
            comparison = f"- {label} changed validation CER by {difference:+.2f} pp"
            before_timing = results[before]["forward_pass"]
            after_timing = results[after]["forward_pass"]
            if before_timing["device"] == after_timing["device"]:
                time_change = after_timing["median_ms"] - before_timing["median_ms"]
                comparison += f" and median forward time by {time_change:+.2f} ms"
            rows.append(comparison + ".")
    if len(results) == len(CASES):
        context_gain_at_two = (
            validation_cer(results[Case(2, 63)]) - validation_cer(results[Case(2, 31)])
        )
        context_gain_at_four = (
            validation_cer(results[Case(4, 32)]) - validation_cer(results[Case(4, 16)])
        )
        rows.append(
            f"- Longer context changed CER by {context_gain_at_four - context_gain_at_two:+.2f} pp "
            "more at four blocks than at two."
        )
    devices = {result["forward_pass"]["device"] for result in results.values()}
    if len(devices) > 1:
        rows += ["", "The sweep used multiple device models. Compare timing only within the same device."]
    if best is not None:
        rows += ["", f"Lowest observed validation CER belongs to **{best.name}**."]
    if winner is not None:
        test = winner["test_metrics"][0]["test/CER"]
        rows.append(f"The validation-selected checkpoint scored **{test:.2f}% test CER**.")
    elif len(results) == len(CASES):
        rows.append("Held-out test evaluation is pending.")
    for name, reason in failures.items():
        rows.append(f"- {name} failed. {reason}")
    rows += ["", "This sweep uses one seed and one subject.", ""]
    write_text(output / "summary.md", "\n".join(rows))


def read_completed(directory: Path) -> dict | None:
    path = directory / "results.json"
    if not path.exists():
        return None
    result = read_json(path)
    if not math.isfinite(validation_cer(result)) or not Path(result["best_checkpoint"]).is_file():
        raise ValueError(f"Invalid validation CER or missing best checkpoint for {directory.name}.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Extracted HDF5 session directory")
    parser.add_argument("--output", type=Path, required=True, help="Persistent sweep directory")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--accelerator", choices=["gpu", "cpu"], default="gpu")
    args = parser.parse_args()
    if args.epochs <= 10:
        parser.error("--epochs must exceed the baseline's 10 warmup epochs")
    data, output = args.data.resolve(), args.output.resolve()
    git = ["git", "-C", str(REPO)]
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
        "dataset": dataset_manifest(data),
        "cases": [
            {"name": case.name, "blocks": case.blocks, "kernel": case.kernel,
             "context_ms": case.context_ms}
            for case in CASES
        ],
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
        summarize(output, results, failures, None)
    for case in CASES:
        directory = output / case.name
        try:
            completed = read_completed(directory)
            if completed is None:
                print(f"Training {case.name}. Live log: {directory / 'console.log'}", flush=True)
                command = case_command(case, data, directory, args.epochs, args.accelerator, args.seed)
                run(command, directory)
                completed = read_completed(directory)
                if completed is None:
                    raise ValueError(f"Missing results for {case.name}.")
            results[case] = completed
            print(f"{case.name}: {validation_cer(completed):.2f}% validation CER", flush=True)
        except (subprocess.CalledProcessError, ValueError) as error:
            failures[case.name] = f"{type(error).__name__}. See {case.name}/console.log."
            print(failures[case.name], file=sys.stderr, flush=True)
        summarize(output, results, failures, None)
    if failures:
        raise SystemExit(f"{len(failures)} incomplete configurations. See {output / 'summary.md'}.")

    best = min(results, key=lambda case: validation_cer(results[case]))
    directory = output / "selected-test"
    test_path = directory / "results.json"
    if test_path.is_file():
        winner = read_json(test_path)
    else:
        command = case_command(best, data, directory, args.epochs, args.accelerator, args.seed)
        command[command.index("evaluate_test=false")] = "evaluate_test=true"
        command[command.index("benchmark_inference=true")] = "benchmark_inference=false"
        command += ["train=false", f"checkpoint={json.dumps(results[best]['best_checkpoint'])}"]
        print(f"Testing validation winner {best.name}.", flush=True)
        winner = run(command, directory)
    summarize(output, results, failures, winner)
    print((output / "summary.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
