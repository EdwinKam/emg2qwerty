# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import pytest
import torch

import emg2qwerty.data as data
from emg2qwerty.data import LabelData


@pytest.mark.parametrize(
    "text, labels",
    [
        ("aa", [0, 0]),
        ("ab", [0, 1]),
        ("", []),
        ("\n\r\b\x08", [95, 95, 94, 94]),
        ("⏎\na⌫\b", [95, 95, 0, 94, 94]),
        ("’“”—", [68, 63, 63, 74]),
    ],
)
def test_label_data(text: str, labels: list[int]):
    label_data1 = LabelData.from_str(text)
    label_data2 = LabelData.from_labels(labels)

    assert label_data1 == label_data2
    assert label_data1.text == label_data2.text
    assert list(label_data1.labels) == labels
    assert list(label_data2.labels) == labels

    assert LabelData.from_str(text + text) == label_data1 + label_data2
    assert LabelData.from_labels(labels + labels) == label_data1 + label_data2


@pytest.mark.parametrize("window_length,index", [(8000, 0), (8000, 2), (8000, 3), (None, 0)])
def test_extra_left_context_preserves_targets_and_emission_alignment(
    monkeypatch, tmp_path, window_length, index
):
    samples = np.zeros(20_000, dtype=[("time", "i8"), ("emg", "f4")])
    samples["time"] = np.arange(len(samples))
    samples["emg"] = np.arange(1, len(samples) + 1)

    class Session:
        TIMESTAMPS = "time"
        condition = "on_keyboard"

        def __init__(self, path):
            self.target_span = None

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def __len__(self):
            return len(samples)

        def __getitem__(self, key):
            return samples[key]

        def ground_truth(self, start_t, end_t):
            self.target_span = (int(start_t), int(end_t))
            return LabelData.from_str("ab")

    monkeypatch.setattr(data, "EMGSessionData", Session)
    settings = {
        "window_length": window_length,
        "stride": 1000,
        "padding": (1800, 200) if window_length is not None else (0, 0),
        "transform": lambda window: torch.as_tensor(window["emg"].copy()),
    }
    baseline = data.WindowedEMGDataset(tmp_path / "sample.hdf5", **settings)
    extended = data.WindowedEMGDataset(
        tmp_path / "sample.hdf5", extra_left_context=1024, **settings
    )
    base_emg, base_labels = baseline[index]
    extended_emg, extended_labels = extended[index]

    assert baseline.session.target_span == extended.session.target_span
    assert torch.equal(extended_labels, base_labels)
    assert len(extended_emg) == len(base_emg) + 1024
    assert torch.equal(extended_emg[1024:], base_emg)

    expected_prefix = np.zeros(1024, dtype=np.float32)
    base_start = max(index * 1000 - settings["padding"][0], 0)
    observed = min(base_start, 1024)
    if observed:
        expected_prefix[-observed:] = samples["emg"][base_start - observed : base_start]
    assert torch.equal(extended_emg[:1024], torch.as_tensor(expected_prefix))

    base_frames = 1 + (len(base_emg) - 64) // 16
    extended_frames = 1 + (len(extended_emg) - 64) // 16
    assert base_frames - 4 * (32 - 1) == extended_frames - 4 * (48 - 1)
