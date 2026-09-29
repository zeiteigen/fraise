import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

from datasets import Dataset
import torch


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

from config import (  # noqa: E402
    _OPTIONAL_TRAINING_HYPERPARAMETER_TYPES,
    _TRAINING_HYPERPARAMETER_TYPES,
    _validate_config_types,
)
from train.batch import _loss_for_batch  # noqa: E402
from train.packing import PackedTokenBatch  # noqa: E402
from train.minibatches import _backward_minibatches  # noqa: E402
from train.validation import _evaluate  # noqa: E402


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 5)
        self.output = torch.nn.Linear(5, 8)

    def forward(self, input_ids):
        return self.output(self.embedding(input_ids))


class TrainingMinibatchTests(unittest.TestCase):
    def test_validation_respects_minibatch_memory_limit(self):
        class LimitedModel(_TinyModel):
            def forward(self, input_ids):
                if len(input_ids) > 1:
                    raise AssertionError("validation exceeded minibatch_size")
                return super().forward(input_ids)

        settings = SimpleNamespace(
            sequence_length=2,
            packing_stride=2,
            batch_size=8,
            minibatch_size=1,
            seed=0,
            max_validation_tokens=None,
            use_bf16=False,
        )
        split = Dataset.from_dict({"token_ids": [[1, 2, 3, 4, 5]]})
        self.assertIsNotNone(_evaluate(LimitedModel(), split, settings, 0))

    def test_accumulated_gradient_matches_full_batch_with_padding_and_short_tail(self):
        torch.manual_seed(17)
        full_model = _TinyModel()
        accumulated_model = _TinyModel()
        accumulated_model.load_state_dict(full_model.state_dict())
        batch = PackedTokenBatch(
            input_ids=torch.tensor([[1, 2, 3, 4], [2, 3, 0, 0], [3, 4, 5, 0]]),
            targets=torch.tensor([[2, 3, 4, 5], [3, 4, 0, 0], [4, 5, 6, 0]]),
            target_mask=torch.tensor(
                [
                    [True, True, True, True],
                    [True, True, False, False],
                    [True, False, False, False],
                ]
            ),
        )

        full_loss, full_tokens = _loss_for_batch(full_model, batch)
        full_loss.backward()
        result = _backward_minibatches(
            accumulated_model,
            batch,
            torch.device("cpu"),
            minibatch_size=2,
            use_bf16=False,
        )

        self.assertIsNotNone(result)
        accumulated_loss, accumulated_tokens = result
        self.assertEqual(accumulated_tokens, full_tokens)
        self.assertAlmostEqual(
            float(accumulated_loss.item()), float(full_loss.item()), places=6
        )
        for full_parameter, accumulated_parameter in zip(
            full_model.parameters(), accumulated_model.parameters()
        ):
            torch.testing.assert_close(
                accumulated_parameter.grad, full_parameter.grad, rtol=1e-5, atol=1e-6
            )

    def test_json_schema_accepts_optional_settings_and_rejects_wrong_types(self):
        required = {
            key: 1 for key in _TRAINING_HYPERPARAMETER_TYPES
        }
        required.update(
            betas=[0.9, 0.99],
            dataset_token_counts={"first": 1},
            use_bf16=False,
            shuffle=True,
        )
        self.assertEqual(
            _validate_config_types(
                required,
                _TRAINING_HYPERPARAMETER_TYPES,
                "training hyperparameters",
                _OPTIONAL_TRAINING_HYPERPARAMETER_TYPES,
            ),
            required,
        )
        with_options = {
            **required,
            "minibatch_size": 2,
            "gradient_accumulation_steps": 4,
            "checkpoint_keep_last": 3,
        }
        self.assertEqual(
            _validate_config_types(
                with_options,
                _TRAINING_HYPERPARAMETER_TYPES,
                "training hyperparameters",
                _OPTIONAL_TRAINING_HYPERPARAMETER_TYPES,
            ),
            with_options,
        )
        with self.assertRaisesRegex(TypeError, "minibatch_size"):
            _validate_config_types(
                {**required, "minibatch_size": True},
                _TRAINING_HYPERPARAMETER_TYPES,
                "training hyperparameters",
                _OPTIONAL_TRAINING_HYPERPARAMETER_TYPES,
            )


if __name__ == "__main__":
    unittest.main()
