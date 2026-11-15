import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from fixtures import eager_loss, tiny_config, tiny_data, tiny_settings, tiny_tokenizer

from fraise.inference.artifacts import load_run
from fraise.inference.generation import generate
from fraise.inference.scoring import score_text
from fraise.io import read_json, write_json
from fraise.model.model import FraiseModel
from fraise.training.run.checkpoints import save_model_state_dict
from fraise.training.training import pretrain_model


class InferenceTests(unittest.TestCase):
    def test_saved_tied_run_loads_with_sharing_and_sampling_defaults(self):
        with (
            TemporaryDirectory() as directory,
            eager_loss(),
        ):
            config = tiny_config(tie_word_embeddings=True)
            original = pretrain_model(
                tiny_tokenizer(),
                tiny_data(),
                tiny_settings(tied_weight_init_std=0.1),
                config,
                directory,
                "tiny",
                device="cpu",
            )
            loaded = load_run(Path(directory) / "tiny", device="cpu")
            archived = read_json(Path(directory) / "tiny" / "model_config.json")
            self.assertNotIn("tied_weight_init_std", archived)
            hyperparameters = read_json(
                Path(directory) / "tiny" / "training_hyperparameters.json"
            )
            self.assertEqual(hyperparameters["tied_weight_init_std"], 0.1)
            self.assertIs(
                loaded.model.token_embedding.weight, loaded.model.lm_head.weight
            )
            inputs = torch.tensor([[1, 3, 4]])
            torch.testing.assert_close(loaded.model(inputs), original(inputs))
            _text, ids = generate(
                loaded.model, loaded.tokenizer, "a", temperature=1, max_new_tokens=2
            )
            self.assertLessEqual(len(ids), 4)
            self.assertTrue(
                torch.isfinite(
                    torch.tensor(
                        score_text(loaded.model, loaded.tokenizer, "a b")["loss"]
                    )
                )
            )

    def test_incomplete_run_selects_latest_checkpoint_numerically(self):
        with (
            TemporaryDirectory() as directory,
            eager_loss(),
        ):
            pretrain_model(
                tiny_tokenizer(),
                tiny_data(),
                tiny_settings(),
                tiny_config(),
                directory,
                "tiny",
                device="cpu",
            )
            run = Path(directory) / "tiny"
            metadata = read_json(run / "model_meta.json")
            metadata["training_finished"] = False
            write_json(run / "model_meta.json", metadata)
            self.assertEqual(
                load_run(run, device="cpu").path.name, "checkpoint_step_00000002.pt"
            )


class IncrementalDecodingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = (
            FraiseModel(tiny_config(vocab_size=50, num_layers=2, max_seq_len=12))
            .double()
            .eval()
        )

    def test_cached_logits_match_recomputation_before_and_after_sliding(self):
        from fraise.inference.generation import IncrementalDecoder

        decoder = IncrementalDecoder(self.model)
        ids = [1, 5, 7]
        for _ in range(20):  # runs past max_seq_len into the sliding window
            actual = decoder.logits(ids)
            with torch.no_grad():
                expected = self.model(torch.tensor([ids[-12:]]))[0, -1]
            torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
            ids.append(int(actual.argmax()))

    def test_multi_token_continuation_uses_bottom_right_causal_masking(self):
        cache = self.model.new_cache()
        with torch.no_grad():
            self.model(torch.tensor([[1, 2, 3]]), cache=cache)
            continued = self.model(torch.tensor([[4, 5, 6]]), cache=cache)
            full = self.model(torch.tensor([[1, 2, 3, 4, 5, 6]]))
        torch.testing.assert_close(continued, full[:, 3:], rtol=1e-9, atol=1e-10)
        self.assertEqual(len(cache), 6)
        with self.assertRaisesRegex(ValueError, "max_seq_len"):
            self.model(torch.zeros(1, 7, dtype=torch.long), cache=cache)

    def test_last_only_projects_just_the_final_position(self):
        inputs = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            last = self.model(inputs, last_only=True)
            full = self.model(inputs)
        self.assertEqual(last.shape, (1, 1, 50))
        torch.testing.assert_close(last[:, 0], full[:, -1])
