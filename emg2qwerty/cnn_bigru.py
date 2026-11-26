# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Small convolutional and bidirectional GRU model for CTC decoding."""

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
from emg2qwerty.modules import MultiBandRotationInvariantMLP, SpectrogramNorm


class CNNBiGRUCTCModule(TDSConvCTCModule):
    """Reuse the baseline's training hooks with a compact CNN-BiGRU encoder.

    The input and output time resolutions are identical: the temporal
    convolution uses stride one and same padding, and the GRU does not
    downsample. Packing makes the backward GRU ignore padded time steps.
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        conv_features: int,
        kernel_width: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        # Initialize Lightning directly: the parent initializer constructs a
        # TDS model, which this architecture does not use.
        pl.LightningModule.__init__(self)
        self.save_hyperparameters({
            'in_features': in_features, 'mlp_features': mlp_features,
            'conv_features': conv_features, 'kernel_width': kernel_width,
            'hidden_size': hidden_size, 'num_layers': num_layers,
            'dropout': dropout, 'optimizer': optimizer,
            'lr_scheduler': lr_scheduler, 'decoder': decoder,
        })
        if kernel_width < 1 or kernel_width % 2 != 1:
            raise ValueError('kernel_width must be a positive odd number')
        if num_layers < 1:
            raise ValueError('num_layers must be positive')

        features = self.NUM_BANDS * mlp_features[-1]
        self.normalization = SpectrogramNorm(self.NUM_BANDS * self.ELECTRODE_CHANNELS)
        self.wrist_features = MultiBandRotationInvariantMLP(
            in_features=in_features, mlp_features=mlp_features, num_bands=self.NUM_BANDS
        )
        self.conv = nn.Sequential(
            nn.Conv1d(features, conv_features, kernel_width, padding=kernel_width // 2),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=conv_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.classifier = nn.Linear(2 * hidden_size, charset().num_classes)
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {f'{phase}_metrics': metrics.clone(prefix=f'{phase}/') for phase in ('train', 'val', 'test')}
        )

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        """Return time-first log probabilities with the original time lengths."""
        x = self.normalization(inputs)
        x = self.wrist_features(x).flatten(start_dim=2)  # (time, batch, features)
        # Do not let padded positions enter the temporal convolution.
        mask = torch.arange(x.size(0), device=x.device)[:, None] < input_lengths.to(x.device)[None, :]
        x = x * mask.unsqueeze(-1)
        x = self.conv(x.permute(1, 2, 0)).permute(2, 0, 1)
        packed = pack_padded_sequence(
            x, input_lengths.detach().cpu(), enforce_sorted=False
        )
        packed_output, _ = self.gru(packed)
        x, _ = pad_packed_sequence(packed_output, total_length=inputs.size(0))
        return self.classifier(x).log_softmax(dim=-1)

    def _step(self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        inputs, targets = batch['inputs'], batch['targets']
        input_lengths, target_lengths = batch['input_lengths'], batch['target_lengths']
        emissions = self.forward(inputs, input_lengths)
        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=input_lengths,
            target_lengths=target_lengths,
        )
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=input_lengths.detach().cpu().numpy(),
        )
        metrics = self.metrics[f'{phase}_metrics']
        target_arrays = targets.detach().cpu().numpy()
        for i, prediction in enumerate(predictions):
            target = LabelData.from_labels(target_arrays[:int(target_lengths[i]), i])
            metrics.update(prediction=prediction, target=target)
        self.log(f'{phase}/loss', loss, batch_size=len(input_lengths), sync_dist=True)
        return loss
