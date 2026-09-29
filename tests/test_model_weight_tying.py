import io
import unittest

import torch

from fraise.config.model import ModelConfig
from fraise.model.model import FraiseModel
from fraise.model.parameters import estimate_parameter_count


class ModelWeightTyingTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "vocab_size": 8,
            "residual_stream_dim": 8,
            "head_dim": 4,
            "qk_dim": 4,
            "num_heads": 1,
            "swiglu_up_dim": 16,
            "num_layers": 1,
            "max_seq_len": 4,
        }
        self.input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])

    def test_default_and_explicit_false_keep_separate_weights(self):
        for option in ({}, {"tie_word_embeddings": False}):
            with self.subTest(option=option):
                model = FraiseModel(ModelConfig(**(self.config | option)))
                self.assertIsNot(
                    model.lm_head.weight,
                    model.token_embedding.weight,
                )
                self.assertEqual(model(self.input_ids).shape, (2, 3, 8))

    def test_parameter_count_matches_model_with_and_without_tying(self):
        counts = []
        for tied in (False, True):
            with self.subTest(tied=tied):
                config = self.config | {"tie_word_embeddings": tied}
                model = FraiseModel(ModelConfig(**config))
                count = sum(parameter.numel() for parameter in model.parameters())
                self.assertEqual(estimate_parameter_count(ModelConfig(**config)), count)
                self.assertIsNotNone(model.lm_head.bias)
                counts.append(count)
        self.assertEqual(
            counts[0] - counts[1],
            self.config["vocab_size"] * self.config["residual_stream_dim"],
        )

    def test_shared_weights_accumulate_both_gradients_and_update_once(self):
        tied = FraiseModel(
            ModelConfig(**(self.config | {"tie_word_embeddings": True}))
        ).double()
        untied = FraiseModel(ModelConfig(**self.config)).double()
        untied.load_state_dict(tied.state_dict())
        shared_weight = tied.token_embedding.weight
        self.assertIs(tied.lm_head.weight, shared_weight)

        tied_logits = tied(self.input_ids)
        untied_logits = untied(self.input_ids)
        torch.testing.assert_close(tied_logits, untied_logits)
        tied_logits.square().mean().backward()
        untied_logits.square().mean().backward()
        expected_gradient = (
            untied.token_embedding.weight.grad + untied.lm_head.weight.grad
        )
        torch.testing.assert_close(shared_weight.grad, expected_gradient)

        optimizer = torch.optim.SGD(tied.parameters(), lr=0.1)
        self.assertEqual(
            sum(
                parameter is shared_weight
                for parameter in optimizer.param_groups[0]["params"]
            ),
            1,
        )
        before = shared_weight.detach().clone()
        optimizer.step()
        torch.testing.assert_close(shared_weight, before - 0.1 * expected_gradient)
        self.assertIs(tied.lm_head.weight, shared_weight)

    def test_checkpoint_round_trip_preserves_tying_and_predictions(self):
        config = self.config | {"tie_word_embeddings": True}
        original = FraiseModel(ModelConfig(**config))
        checkpoint = io.BytesIO()
        torch.save(original.state_dict(), checkpoint)
        checkpoint.seek(0)

        restored = FraiseModel(ModelConfig(**config))
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        self.assertIs(
            restored.lm_head.weight,
            restored.token_embedding.weight,
        )
        torch.testing.assert_close(restored(self.input_ids), original(self.input_ids))

    def test_fullgraph_compilation_preserves_tied_model_predictions(self):
        model = FraiseModel(
            ModelConfig(**(self.config | {"tie_word_embeddings": True}))
        )
        compiled = torch.compile(model, fullgraph=True, backend="eager")
        torch.testing.assert_close(compiled(self.input_ids), model(self.input_ids))


if __name__ == "__main__":
    unittest.main()
