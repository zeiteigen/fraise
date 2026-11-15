import unittest

import torch

from fraise.architectures import build_model, get_architecture
from fraise.config.model import ModelConfig
from fraise.config.training import TrainingSettings


def _config(**overrides):
    values = dict(
        vocab_size=64,
        residual_stream_dim=16,
        head_dim=4,
        qk_dim=4,
        num_heads=2,
        swiglu_up_dim=32,
        num_layers=1,
        max_seq_len=8,
    )
    values.update(overrides)
    return ModelConfig(**values)


def _settings(**overrides):
    values = dict(
        batch_size=1,
        learning_rate=0.01,
        dataset_token_counts={"a": 8},
        max_train_tokens=8,
    )
    values.update(overrides)
    return TrainingSettings(**values)


class FinalNormTest(unittest.TestCase):
    def test_disabled_keeps_state_dict_keys(self):
        model = build_model(_config())
        self.assertFalse(any("final_norm" in key for key in model.state_dict()))

    def test_enabled_normalizes_head_input_and_counts_parameters(self):
        config = _config(final_norm=True)
        model = build_model(config)
        self.assertIn("final_norm.weight", model.state_dict())
        counted = get_architecture("transformer").count_parameters(config)
        self.assertEqual(counted, sum(p.numel() for p in model.parameters()))
        seen = {}
        model.lm_head.register_forward_hook(
            lambda module, inputs, output: seen.setdefault("x", inputs[0])
        )
        model(torch.randint(0, 64, (2, 5)))
        torch.testing.assert_close(
            seen["x"].mean(-1), torch.zeros(2, 5), atol=1e-5, rtol=0
        )

    def test_rejects_non_bool(self):
        with self.assertRaises(TypeError):
            _config(final_norm=1)


class EmbeddingInitTest(unittest.TestCase):
    def _initialized(self, config, settings):
        torch.manual_seed(0)
        model = build_model(config)
        get_architecture("transformer").initialize(model, config, settings)
        return model.token_embedding.weight

    def test_default_keeps_unit_normal(self):
        weight = self._initialized(_config(vocab_size=4096), _settings())
        self.assertAlmostEqual(weight.std().item(), 1.0, delta=0.05)

    def test_untied_std_is_applied(self):
        weight = self._initialized(
            _config(vocab_size=4096), _settings(embedding_init_std=0.02)
        )
        self.assertAlmostEqual(weight.std().item(), 0.02, delta=0.002)

    def test_tied_uses_tied_std(self):
        weight = self._initialized(
            _config(vocab_size=4096, tie_word_embeddings=True),
            _settings(embedding_init_std=0.5, tied_weight_init_std=0.02),
        )
        self.assertAlmostEqual(weight.std().item(), 0.02, delta=0.002)

    def test_rejects_non_positive(self):
        with self.assertRaises(ValueError):
            _settings(embedding_init_std=0.0)


if __name__ == "__main__":
    unittest.main()
