# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from omegaconf import OmegaConf

from emg2qwerty.charset import charset
from emg2qwerty.cnn_bigru import CNNBiGRUCTCModule


def make_model() -> CNNBiGRUCTCModule:
    return CNNBiGRUCTCModule(
        in_features=528, mlp_features=[8], conv_features=8, kernel_width=3,
        hidden_size=4, num_layers=2, dropout=0.0,
        optimizer=OmegaConf.create({'_target_': 'torch.optim.Adam', 'lr': 0.001}),
        lr_scheduler=OmegaConf.create({}),
        decoder=OmegaConf.create({'_target_': 'emg2qwerty.decoder.CTCGreedyDecoder'}),
    )


class CNNBiGRUTest(unittest.TestCase):
    def test_padded_values_do_not_change_valid_bidirectional_outputs(self):
        torch.manual_seed(1)
        model = make_model().eval()
        lengths = torch.tensor([12, 7])
        inputs = torch.randn(12, 2, 2, 16, 33)
        altered = inputs.clone()
        altered[7:, 1] = 100.0
        with torch.no_grad():
            original = model(inputs, lengths)
            changed = model(altered, lengths)
        self.assertEqual(original.shape, (12, 2, charset().num_classes))
        torch.testing.assert_close(original[:7, 1], changed[:7, 1])
        torch.testing.assert_close(original.exp().sum(-1), torch.ones(12, 2))

    def test_ctc_training_step_backpropagates(self):
        torch.manual_seed(2)
        model = make_model()
        self.assertEqual(model.hparams.optimizer.lr, 0.001)
        self.assertEqual(model.hparams.hidden_size, 4)
        model.log = lambda *args, **kwargs: None
        batch = {
            'inputs': torch.randn(12, 2, 2, 16, 33),
            'input_lengths': torch.tensor([12, 9], dtype=torch.int32),
            'targets': torch.tensor([[1, 2], [2, 3], [3, 0]], dtype=torch.int32),
            'target_lengths': torch.tensor([3, 2], dtype=torch.int32),
        }
        loss = model.training_step(batch, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.gru.weight_ih_l0.grad)
        self.assertTrue(torch.isfinite(model.gru.weight_ih_l0.grad).all())


if __name__ == '__main__':
    unittest.main()
