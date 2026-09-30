import math
import unittest
from unittest.mock import patch

import torch
from fixtures import tiny_config, tiny_settings

from fraise.model.model import FraiseModel
from fraise.training.runtime import build_runtime


class TrainingInitializationTests(unittest.TestCase):
    def setUp(self):
        compile_patch = patch.object(FraiseModel, "compile", lambda self: None)
        compile_patch.start()
        self.addCleanup(compile_patch.stop)
        rng_context = torch.random.fork_rng(devices=[])
        rng_context.__enter__()
        self.addCleanup(rng_context.__exit__, None, None, None)
        self.config = tiny_config(
            vocab_size=512, residual_stream_dim=128, tie_word_embeddings=True
        )

    def test_fresh_shared_weights_use_default_and_custom_standard_deviations(self):
        for changes, expected in (({}, 0.02), ({"tied_weight_init_std": 0.1}, 0.1)):
            with self.subTest(changes=changes):
                runtime = build_runtime(
                    self.config, tiny_settings(**changes), 1, device="cpu"
                )
                model = runtime.model
                weights = model.token_embedding.weight.detach()
                self.assertIs(model.lm_head.weight, model.token_embedding.weight)
                self.assertAlmostEqual(
                    weights.std().item(), expected, delta=expected * 0.02
                )
                self.assertAlmostEqual(weights.mean().item(), 0, delta=expected * 0.02)

    def test_default_training_initialization_avoids_saturated_next_token_loss(self):
        runtime = build_runtime(self.config, tiny_settings(), 1, device="cpu")
        inputs = torch.tensor([[0, 1, 2], [3, 4, 5]])
        loss = torch.nn.functional.cross_entropy(
            runtime.model(inputs).flatten(0, 1), (inputs + 1).flatten()
        )
        self.assertLess(loss.item(), math.log(512) + 1)

    def test_untied_and_resumed_models_skip_training_initialization(self):
        for tied, initialize in ((False, True), (True, False)):
            with self.subTest(tied=tied, initialize=initialize):
                config = tiny_config(tie_word_embeddings=tied)
                settings = tiny_settings(tied_weight_init_std=0.1)
                torch.manual_seed(settings.seed)
                expected = FraiseModel(config)
                runtime = build_runtime(
                    config, settings, 1, device="cpu", initialize_weights=initialize
                )
                for name, weights in expected.state_dict().items():
                    torch.testing.assert_close(
                        runtime.model.state_dict()[name], weights, rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
