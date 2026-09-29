import unittest
from types import SimpleNamespace

import torch
from datasets import Dataset

from fraise.training.batch import loss_for_batch
from fraise.training.batch_types import PackedTokenBatch
from fraise.training.minibatches import backward_minibatches
from fraise.training.validation import evaluate


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 5)
        self.output = torch.nn.Linear(5, 8)

    def forward(self, ids):
        return self.output(self.embedding(ids))


class TrainingMinibatchTests(unittest.TestCase):
    def test_accumulated_gradients_match_full_batch_with_padding_and_short_tail(self):
        torch.manual_seed(17)
        full, accumulated = TinyModel(), TinyModel()
        accumulated.load_state_dict(full.state_dict())
        batch = PackedTokenBatch(
            torch.tensor([[1, 2, 3, 4], [2, 3, 0, 0], [3, 4, 5, 0]]),
            torch.tensor([[2, 3, 4, 5], [3, 4, 0, 0], [4, 5, 6, 0]]),
            torch.tensor(
                [[True] * 4, [True, True, False, False], [True, False, False, False]]
            ),
        )
        loss = loss_for_batch(full, batch)
        loss.backward()
        actual, count = backward_minibatches(
            accumulated, batch, torch.device("cpu"), minibatch_size=2, use_bf16=False
        )
        self.assertEqual(count, 7)
        torch.testing.assert_close(actual, loss)
        for expected, parameter in zip(full.parameters(), accumulated.parameters()):
            torch.testing.assert_close(
                parameter.grad, expected.grad, rtol=1e-5, atol=1e-6
            )

    def test_validation_respects_minibatch_limit_and_restores_mode(self):
        class LimitedModel(TinyModel):
            def forward(self, ids):
                if len(ids) > 1:
                    raise AssertionError("validation exceeded minibatch_size")
                return super().forward(ids)

        settings = SimpleNamespace(
            sequence_length=2,
            packing_stride=2,
            minibatch_size=1,
            max_validation_tokens=None,
            use_bf16=False,
        )
        model = LimitedModel()
        self.assertIsNotNone(
            evaluate(
                model, Dataset.from_dict({"token_ids": [[1, 2, 3, 4, 5]]}), settings, 0
            )
        )
        self.assertTrue(model.training)
