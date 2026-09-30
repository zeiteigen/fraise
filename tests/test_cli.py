import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fixtures import tiny_config, tiny_data, tiny_settings, tiny_tokenizer

from fraise.cli import train, visualize_training_data
from fraise.io import write_json
from fraise.model.model import FraiseModel


class CommandTests(unittest.TestCase):
    def test_training_cli_skips_held_out_loading_only_for_zero_budget(self):
        with (
            TemporaryDirectory() as directory,
            patch.object(train, "SOURCE_NAMES", ("fixture",)),
            patch.object(train, "load_training_data", return_value=tiny_data()) as load,
            patch.object(train, "pretrain_transformer"),
        ):
            root = Path(directory)
            arguments = self.write_inputs(root)
            for budget, enabled in ((0, False), (123, True), (None, True)):
                with self.subTest(budget=budget):
                    write_json(
                        root / "settings.json",
                        asdict(tiny_settings(max_validation_tokens=budget)),
                    )
                    train.main(arguments + ["--model-name", "tiny"])
                    self.assertIs(load.call_args.kwargs["validation"], enabled)

    def write_inputs(self, root):
        model, settings, tokenizer = (
            root / "model.json",
            root / "settings.json",
            root / "tokenizer.json",
        )
        write_json(model, asdict(tiny_config()))
        write_json(settings, asdict(tiny_settings()))
        tiny_tokenizer().save(str(tokenizer))
        return [
            "--model-config",
            str(model),
            "--training-hyperparameters",
            str(settings),
            "--tokenizer-file",
            str(tokenizer),
            "--data-dir",
            str(root),
        ]

    def test_train_and_resume_use_canonical_archives_with_the_same_input_basename(self):
        with (
            TemporaryDirectory() as directory,
            patch.object(train, "SOURCE_NAMES", ("fixture",)),
            patch.object(train, "load_training_data", return_value=tiny_data()),
            patch.object(FraiseModel, "compile", lambda self: None),
        ):
            root = Path(directory)
            arguments = self.write_inputs(root)
            model_dir, settings_dir = root / "model", root / "settings"
            model_dir.mkdir()
            settings_dir.mkdir()
            (root / "model.json").rename(model_dir / "config.json")
            (root / "settings.json").rename(settings_dir / "config.json")
            arguments[1] = str(model_dir / "config.json")
            arguments[3] = str(settings_dir / "config.json")
            train.main(arguments + ["--model-name", "tiny"])
            run = root / "models" / "tiny"
            self.assertEqual(
                (run / "model_config.json").read_bytes(),
                (model_dir / "config.json").read_bytes(),
            )
            self.assertEqual(
                (run / "training_hyperparameters.json").read_bytes(),
                (settings_dir / "config.json").read_bytes(),
            )
            train.main(arguments + ["--resume", str(run)])

    def test_first_source_preview_skips_validation_and_writes_html(self):
        with (
            TemporaryDirectory() as directory,
            patch.object(visualize_training_data, "SOURCE_NAMES", ("fixture",)),
            patch.object(
                visualize_training_data, "load_training_data", return_value=tiny_data()
            ) as load,
        ):
            root = Path(directory)
            visualize_training_data.main(
                self.write_inputs(root) + ["--count", "2", "--first-source-only"]
            )
            self.assertFalse(load.call_args.kwargs["validation"])
            self.assertEqual(load.call_args.kwargs["sources"], {"fixture"})
            self.assertIn(
                "Fraise training examples",
                (root / "training_examples.html").read_text(),
            )
