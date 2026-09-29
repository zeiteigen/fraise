import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from fixtures import tiny_config, tiny_data, tiny_settings, tiny_tokenizer

from fraise.training.artifacts import RunArtifacts, read_run_model_name
from fraise.training.checkpoints import CheckpointStore
from fraise.training.manifest import run_manifest


class CheckpointRetentionTests(unittest.TestCase):
    def test_retention_keeps_recent_and_pending_validation_files(self):
        with TemporaryDirectory() as directory:
            paths = {
                step: Path(directory) / f"checkpoint_step_{step:08d}.pt"
                for step in (1, 2, 3, 100_000_000)
            }
            for path in paths.values():
                path.write_bytes(b"checkpoint")
            store = CheckpointStore(Path(directory), paths[100_000_000])
            store.prune(2, {paths[1]})
            self.assertEqual(
                {step for step, path in paths.items() if path.exists()},
                {1, 3, 100_000_000},
            )
            store.prune(2)
            self.assertEqual(
                {step for step, path in paths.items() if path.exists()},
                {3, 100_000_000},
            )

    def test_resume_checks_identity_but_records_environment_changes(self):
        tokenizer, data, config, settings = (
            tiny_tokenizer(),
            tiny_data(),
            tiny_config(),
            tiny_settings(),
        )
        manifest = run_manifest(data, tokenizer, config, settings, torch.device("cpu"))
        with TemporaryDirectory() as directory:
            original = RunArtifacts(
                Path(directory), "run", manifest, tokenizer, config, settings
            )
            checkpoint = original.directory / "checkpoint_step_00000001.pt"
            checkpoint.write_bytes(b"checkpoint")
            self.assertEqual(read_run_model_name(original.directory), "run")
            manifest["provenance"]["python_version"] = "different"
            resumed = RunArtifacts(
                Path(directory),
                "run",
                manifest,
                tokenizer,
                config,
                settings,
                resume_from=checkpoint,
            )
            self.assertEqual(
                resumed.metadata["resume_provenance"][-1]["python_version"], "different"
            )
            with self.assertRaisesRegex(ValueError, "model_name"):
                RunArtifacts(
                    Path(directory),
                    "other",
                    manifest,
                    tokenizer,
                    config,
                    settings,
                    resume_from=checkpoint,
                )
            manifest["identity"]["training"]["learning_rate"] = 0.2
            with self.assertRaisesRegex(ValueError, "resume training"):
                RunArtifacts(
                    Path(directory),
                    "run",
                    manifest,
                    tokenizer,
                    config,
                    settings,
                    resume_from=checkpoint,
                )
