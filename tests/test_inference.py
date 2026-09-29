import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from fixtures import tiny_config, tiny_data, tiny_settings, tiny_tokenizer

from fraise.inference.artifacts import load_run
from fraise.inference.generation import generate
from fraise.inference.scoring import score_text
from fraise.io import read_json, write_json
from fraise.model.model import FraiseModel
from fraise.training.checkpoints import save_model_state_dict
from fraise.training.migration import migrate_run
from fraise.training.training import pretrain_transformer


class InferenceTests(unittest.TestCase):
    def test_saved_tied_run_loads_with_sharing_and_sampling_defaults(self):
        with (
            TemporaryDirectory() as directory,
            patch.object(FraiseModel, "compile", lambda self: None),
        ):
            config = tiny_config(tie_word_embeddings=True)
            original = pretrain_transformer(
                tiny_tokenizer(),
                tiny_data(),
                tiny_settings(),
                config,
                directory,
                "tiny",
                device="cpu",
            )
            loaded = load_run(Path(directory) / "tiny", device="cpu")
            self.assertIs(
                loaded.model.token_embedding.weight, loaded.model.lm_head.weight
            )
            inputs = torch.tensor([[1, 3, 4]])
            torch.testing.assert_close(loaded.model(inputs), original(inputs))
            text, ids = generate(
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
            patch.object(FraiseModel, "compile", lambda self: None),
        ):
            pretrain_transformer(
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

    def test_legacy_weights_and_manifest_have_a_migration_path(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            config, settings, tokenizer, data = (
                tiny_config(tie_word_embeddings=True),
                tiny_settings(),
                tiny_tokenizer(),
                tiny_data(),
            )
            model = FraiseModel(config)
            legacy_weights = {
                name.replace("token_embedding.", "token_embedding.embedding.")
                .replace("lm_head.", "lm_head.projection.")
                .replace(".swiglu.", ".swiGLU."): value
                for name, value in model.state_dict().items()
            }
            save_model_state_dict(legacy_weights, directory / "legacy.pt")
            tokenizer.save(str(directory / "tokenizer.json"))
            from fraise.tokenizer.loading import tokenizer_fingerprint

            metadata = {
                "model_name": "legacy",
                "training_finished": True,
                "tokenizer_file": "tokenizer.json",
                "all_messages": ["old log"],
                "manifest": {
                    "manifest_version": 1,
                    "model_config": asdict(config),
                    "training": asdict(settings),
                    "tokenizer": {"fingerprint": tokenizer_fingerprint(tokenizer)},
                },
            }
            write_json(directory / "model_meta.json", metadata)
            loaded = load_run(directory, device="cpu")
            self.assertIs(
                loaded.model.token_embedding.weight, loaded.model.lm_head.weight
            )
            migrate_run(directory, data)
            self.assertTrue((directory / "model_meta.v2.json").is_file())
            self.assertEqual(
                read_json(directory / "model_meta.json")["manifest"][
                    "manifest_version"
                ],
                2,
            )
            migrated = load_run(directory, device="cpu")
            torch.testing.assert_close(
                migrated.model(torch.tensor([[1, 3]])), model(torch.tensor([[1, 3]]))
            )
