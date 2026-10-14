import unittest
from types import SimpleNamespace

import torch
from datasets import Dataset

from fraise.training.batch import loss_for_batch
from fraise.training.minibatches import backward_minibatches
from fraise.training.pipeline.batch_types import PackedTokenBatch
from fraise.training.validation.evaluate import evaluate


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 5)
        self.output = torch.nn.Linear(5, 8)

    def forward(self, ids):
        return self.output(self.embedding(ids))


class TrainingMinibatchTests(unittest.TestCase):
    def test_zero_validation_budget_returns_without_reading_a_split_or_running_model(
        self,
    ):
        class UnusedModel(TinyModel):
            def forward(self, ids):
                raise AssertionError("disabled validation ran a forward pass")

        model = UnusedModel()
        self.assertIsNone(
            evaluate(
                model,
                Dataset.from_dict({"unused": ["held-out text"]}),
                SimpleNamespace(max_validation_tokens=0),
                0,
            )
        )
        self.assertTrue(model.training)

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
            deterministic=False,
        )
        model = LimitedModel()
        self.assertIsNotNone(
            evaluate(
                model, Dataset.from_dict({"token_ids": [[1, 2, 3, 4, 5]]}), settings, 0
            )
        )
        self.assertTrue(model.training)

    def test_fixed_shape_padding_preserves_loss_gradients_and_target_count(self):
        class RecordingModel(TinyModel):
            def __init__(self):
                super().__init__()
                self.shapes = []

            def forward(self, ids):
                self.shapes.append(tuple(ids.shape))
                return super().forward(ids)

        batch = PackedTokenBatch(
            torch.tensor([[1, 2, 3, 4], [2, 3, 0, 0], [3, 4, 5, 0]]),
            torch.tensor([[2, 3, 4, 5], [3, 4, 0, 0], [4, 5, 6, 0]]),
            torch.tensor(
                [[True] * 4, [True, True, False, False], [True, False, False, False]]
            ),
            ("first", "second", "first"),
        )
        full, padded = TinyModel(), RecordingModel()
        padded.load_state_dict(full.state_dict())
        expected = loss_for_batch(full, batch)
        expected.backward()
        warmups = []
        actual, count = backward_minibatches(
            padded,
            batch,
            torch.device("cpu"),
            minibatch_size=2,
            use_bf16=False,
            pad_to_minibatch_size=True,
            on_first_backward=warmups.append,
        )
        self.assertEqual(padded.shapes, [(2, 4), (2, 4)])
        self.assertEqual(count, 7)
        self.assertEqual(len(warmups), 1)
        torch.testing.assert_close(actual, expected)
        for reference, parameter in zip(full.parameters(), padded.parameters()):
            torch.testing.assert_close(
                parameter.grad, reference.grad, rtol=1e-5, atol=1e-6
            )


class TokenLossTests(unittest.TestCase):
    def test_ignored_targets_match_gathering_only_valid_positions(self):
        import torch.nn.functional as F

        from fraise.training.batch import token_loss_sum

        torch.manual_seed(3)
        model = TinyModel()
        ids = torch.randint(0, 8, (3, 4))
        targets = torch.randint(0, 8, (3, 4))
        mask = torch.rand(3, 4) > 0.4
        mask[0, 0] = True
        logits = model(ids).flatten(0, 1)
        expected = F.cross_entropy(
            logits[mask.flatten()], targets.flatten()[mask.flatten()], reduction="sum"
        )
        torch.testing.assert_close(token_loss_sum(model, ids, targets, mask), expected)
        batch = PackedTokenBatch(ids, targets, mask)
        torch.testing.assert_close(
            loss_for_batch(model, batch), expected / int(mask.sum())
        )
