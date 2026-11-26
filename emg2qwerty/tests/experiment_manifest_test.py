from argparse import Namespace
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from scripts.run_experiment import REPO, case_command, read_experiment, split_sessions


@pytest.mark.parametrize(
    "manifest,expected",
    [
        ("experiments/tds_longer_context/experiment.yaml", [("tds-4x48", 4, 48, 1024, False)]),
        ("benchmarks/tds_baseline/experiment.yaml", [("tds-4x32", 4, 32, 0, False)]),
        ("benchmarks/tds_bigru_offline/experiment.yaml", [("tds-bigru", 4, 32, 0, True)]),
        (
            "experiments/tds_context_depth/experiment.yaml",
            [
                ("tds-4x32", 4, 32, 0, False),
                ("tds-4x16", 4, 16, 0, False),
                ("tds-2x63", 2, 63, 0, False),
                ("tds-2x31", 2, 31, 0, False),
            ],
        ),
    ],
)
def test_experiment_configs_preserve_the_recorded_protocol(manifest, expected):
    experiment = read_experiment(REPO / manifest)
    sessions = split_sessions(experiment)
    assert {phase: len(files) for phase, files in sessions.items()} == {
        "train": 16, "val": 1, "test": 1,
    }
    assert len(experiment.cases) == len(expected)
    args = Namespace(seed=experiment.seed, epochs=experiment.epochs, accelerator="cpu", smoke=False)
    for case, (name, blocks, width, extra_context, recurrent) in zip(experiment.cases, expected):
        command = case_command(
            experiment, case, Path("/data/raw sessions"), Path("/output/run results"), args
        )
        with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
            config = compose(config_name="base", overrides=command[6:], return_hydra_config=True)
        assert case.name == name
        assert config.dataset.root == "/data/raw sessions"
        assert config.hydra.run.dir == "/output/run results"
        assert config.trainer.max_epochs == 40
        assert config.batch_size == 32
        assert config.seed == 1501
        assert config.module.kernel_width == width
        assert list(config.module.block_channels) == [24] * blocks
        assert config.datamodule.get("extra_left_context", 0) == extra_context
        if recurrent:
            assert config.module.hidden_size == 128
            assert config.module.num_layers == 2
            assert config.module.dropout == 0.1
        else:
            assert config.module._target_.endswith(".TDSConvCTCModule")
