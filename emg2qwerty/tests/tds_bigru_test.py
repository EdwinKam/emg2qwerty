# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from omegaconf import OmegaConf

from emg2qwerty.charset import charset
from emg2qwerty.tds_bigru import TDSBiGRUCTCModule


def make_model() -> TDSBiGRUCTCModule:
    return TDSBiGRUCTCModule(
        in_features=528, mlp_features=[4], block_channels=[2, 2], kernel_width=3,
        hidden_size=4, num_layers=2, dropout=0.0,
        optimizer=OmegaConf.create({'_target_': 'torch.optim.Adam', 'lr': 0.001}),
        lr_scheduler=OmegaConf.create({}),
        decoder=OmegaConf.create({'_target_': 'emg2qwerty.decoder.CTCGreedyDecoder'}),
    )


class TDSBiGRUTest(unittest.TestCase):
    def test_lengths_and_padded_values(self):
        torch.manual_seed(1)
        model = make_model().eval()
        lengths = torch.tensor([18, 12], dtype=torch.int32)
        inputs = torch.randn(18, 2, 2, 16, 33)
        altered = inputs.clone()
        altered[12:, 1] = 100.0
        with torch.no_grad():
            original = model(inputs, lengths)
            changed = model(altered, lengths)
        self.assertEqual(original.shape, (14, 2, charset().num_classes))
        self.assertEqual(model.output_lengths(lengths).tolist(), [14, 8])
        torch.testing.assert_close(original[:8, 1], changed[:8, 1])
        torch.testing.assert_close(original.exp().sum(-1), torch.ones(14, 2))

    def test_ctc_training_step_backpropagates(self):
        torch.manual_seed(2)
        model = make_model()
        self.assertEqual(model.hparams.num_layers, 2)
        self.assertEqual(model.hparams.optimizer.lr, 0.001)
        model.log = lambda *args, **kwargs: None
        batch = {
            'inputs': torch.randn(18, 2, 2, 16, 33),
            'input_lengths': torch.tensor([18, 12], dtype=torch.int32),
            'targets': torch.tensor([[1, 2], [2, 3], [3, 0]], dtype=torch.int32),
            'target_lengths': torch.tensor([3, 2], dtype=torch.int32),
        }
        loss = model.training_step(batch, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(model.gru.weight_ih_l0.grad).all())
        self.assertTrue(torch.isfinite(model.tds.tds_conv_blocks[0].conv2d.weight.grad).all())

    def test_rejects_sequences_too_short_for_tds(self):
        model = make_model()
        with self.assertRaisesRegex(ValueError, 'must exceed'):
            model.output_lengths(torch.tensor([18, 4]))


if __name__ == '__main__':
    unittest.main()
