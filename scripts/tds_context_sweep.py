"""Compatibility entry point for the original TDS sweep and follow-up."""

import argparse
from pathlib import Path

if __package__:
    from .run_experiment import main
else:
    from run_experiment import main


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    manifests = {
        "context-depth": "tds_context_depth/experiment.yaml",
        "longer-context": "tds_longer_context/experiment.yaml",
    }
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--experiment", choices=manifests, default="context-depth")
    args, remaining = parser.parse_known_args()
    main(["--experiment", str(root / "experiments" / manifests[args.experiment]), *remaining])
