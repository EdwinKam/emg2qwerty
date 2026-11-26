# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""TDS baseline encoder followed by a bidirectional GRU and CTC."""

from collections.abc import Sequence

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchmetrics import MetricCollection

from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData
from emg2qwerty.lightning import TDSConvCTCModule
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
)


class TDSBiGRUCTCModule(TDSConvCTCModule):
    """Baseline TDS features, followed by a two-directional GRU.

    The TDS convolutions are unpadded and remove ``kernel_width - 1``
    frames per block. The GRU receives packed sequences so its backward
    direction never reads the padded tail of a shorter batch member.
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        # Initialize Lightning without constructing the parent's TDS-only model.
        pl.LightningModule.__init__(self)
        self.save_hyperparameters({
            'in_features': in_features, 'mlp_features': mlp_features,
            'block_channels': block_channels, 'kernel_width': kernel_width,
            'hidden_size': hidden_size, 'num_layers': num_layers,
            'dropout': dropout, 'optimizer': optimizer,
            'lr_scheduler': lr_scheduler, 'decoder': decoder,
        })
        if not block_channels or kernel_width < 1 or num_layers < 1:
            raise ValueError('TDS blocks, kernel width, and GRU layers must be positive')
        self.time_reduction = len(block_channels) * (kernel_width - 1)
        features = self.NUM_BANDS * mlp_features[-1]
        self.normalization = SpectrogramNorm(self.NUM_BANDS * self.ELECTRODE_CHANNELS)
        self.wrist_features = MultiBandRotationInvariantMLP(
            in_features=in_features, mlp_features=mlp_features, num_bands=self.NUM_BANDS
        )
        self.tds = TDSConvEncoder(
            num_features=features, block_channels=block_channels, kernel_width=kernel_width
        )
        self.gru = nn.GRU(
            input_size=features, hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0, bidirectional=True,
        )
        self.classifier = nn.Linear(2 * hidden_size, charset().num_classes)
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict({
            f'{phase}_metrics': metrics.clone(prefix=f'{phase}/')
            for phase in ('train', 'val', 'test')
        })

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Lengths after the baseline's unpadded, stride-one TDS blocks."""
        lengths = input_lengths - self.time_reduction
        if (lengths <= 0).any():
            raise ValueError(
                f'Each input must exceed {self.time_reduction} spectrogram frames '
                'for the TDS encoder.'
            )
        return lengths

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        output_lengths = self.output_lengths(input_lengths)
        x = self.normalization(inputs)
        x = self.wrist_features(x).flatten(start_dim=2)
        x = self.tds(x)
        packed = pack_padded_sequence(x, output_lengths.detach().cpu(), enforce_sorted=False)
        packed_output, _ = self.gru(packed)
        x, _ = pad_packed_sequence(packed_output, total_length=x.size(0))
        return self.classifier(x).log_softmax(dim=-1)

    def _step(self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        inputs, targets = batch['inputs'], batch['targets']
        input_lengths, target_lengths = batch['input_lengths'], batch['target_lengths']
        emission_lengths = self.output_lengths(input_lengths)
        emissions = self.forward(inputs, input_lengths)
        loss = self.ctc_loss(
            log_probs=emissions, targets=targets.transpose(0, 1),
            input_lengths=emission_lengths, target_lengths=target_lengths,
        )
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )
        metrics = self.metrics[f'{phase}_metrics']
        target_arrays = targets.detach().cpu().numpy()
        for i, prediction in enumerate(predictions):
            target = LabelData.from_labels(target_arrays[:int(target_lengths[i]), i])
            metrics.update(prediction=prediction, target=target)
        self.log(f'{phase}/loss', loss, batch_size=len(input_lengths), sync_dist=True)
        return loss
