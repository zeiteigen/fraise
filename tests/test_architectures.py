"""A small independent architecture exercises the shared training boundaries."""

import json
import math
import unittest
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from datasets import Dataset
from fixtures import tiny_config, tiny_data, tiny_settings, tiny_tokenizer
from torch import nn

from fraise.architectures import (
    ArchitectureSpec,
    OptimizerLayout,
    build_model,
    get_architecture,
    list_architectures,
    model_config_from_dict,
    register_architecture,
)
from fraise.config import load_model_config
from fraise.config.model import ModelConfig
from fraise.inference.artifacts import load_run
from fraise.inference.generation import IncrementalDecoder
from fraise.model.model import FraiseModel
from fraise.model.parameters import estimate_parameter_count
from fraise.training.optimizer import GROUP_TAGS, MuonWithAdamW
from fraise.training.pipeline.batch_types import PackedTokenBatch
from fraise.training.run.checkpoints import load_model_weights, save_model_state_dict
from fraise.training.run.manifest import check_resume_identity, run_manifest
from fraise.training.runtime import build_runtime
from fraise.training.training import pretrain_model
from fraise.training.validation.evaluate import evaluate
from fraise.training.validation.process import ValidationProcess


@dataclass(frozen=True)
class TinyConfig:
    vocab_size: int = 8
    max_seq_len: int = 4
    width: int = 4
    architecture: str = "test-no-cache"


class TinyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.hidden = nn.Linear(config.width, config.width)
        self.output = nn.Linear(config.width, config.vocab_size)

    def new_cache(self):
        return None

    def forward(self, input_ids, *, cache=None, last_only=False):
        if cache is not None:
            raise ValueError("TinyModel has no cache")
        if input_ids.ndim != 2 or input_ids.size(1) > self.max_seq_len:
            raise ValueError("invalid input shape")
        hidden = self.hidden(self.embedding(input_ids)).tanh()
        if last_only:
            hidden = hidden[:, -1:]
        return self.output(hidden)


def tiny_parameter_count(config):
    return (
        2 * config.vocab_size * config.width
        + config.vocab_size
        + config.width * config.width
        + config.width
    )


def initialize_tiny(model, config, settings):
    nn.init.constant_(model.embedding.weight, settings.tied_weight_init_std)


def tiny_optimizer_layout(model):
    return OptimizerLayout(frozenset({model.hidden.weight}))


TINY = ArchitectureSpec(
    id="test-no-cache",
    config_type=TinyConfig,
    build=TinyModel,
    count_parameters=tiny_parameter_count,
    initialize=initialize_tiny,
    optimizer_layout=tiny_optimizer_layout,
    label="Test architecture without a cache",
    example_config=asdict(TinyConfig()),
)


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Deliberately register only in the parent, rather than on module import.
        # The spawned validation process must receive the resolved hooks.
        registry = patch.dict("fraise.architectures.registry._architectures", {})
        registry.start()
        cls.addClassCleanup(registry.stop)
        register_architecture(TINY)

    def test_baseline_legacy_config_and_registered_config_parse(self):
        legacy = asdict(tiny_config())
        del legacy["architecture"]
        self.assertEqual(model_config_from_dict(legacy), tiny_config())
        self.assertEqual(model_config_from_dict(TINY.example_config), TinyConfig())
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(TINY.example_config))
            self.assertEqual(load_model_config(path), TinyConfig())
        self.assertIn(TINY, list_architectures())
        self.assertEqual(get_architecture("transformer").config_type, ModelConfig)

    def test_registry_rejects_unknown_ids_and_duplicate_registration(self):
        with self.assertRaisesRegex(ValueError, "unknown architecture"):
            model_config_from_dict({"architecture": "missing"})
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_architecture(TINY)
        with self.assertRaisesRegex(ValueError, "only supports architecture"):
            replace(tiny_config(), architecture=TINY.id)
        with self.assertRaisesRegex(TypeError, "requires TinyConfig"):
            build_model(_WrongConfig())

    def test_build_and_count_preserve_baseline_weights_and_model_identity(self):
        for config in (
            tiny_config(),
            tiny_config(tie_word_embeddings=True),
            TinyConfig(),
        ):
            with self.subTest(architecture=config.architecture):
                torch.manual_seed(13)
                model = build_model(config)
                self.assertEqual(
                    estimate_parameter_count(config),
                    sum(parameter.numel() for parameter in model.parameters()),
                )
                with patch(
                    "torch.nn.Embedding", side_effect=AssertionError("allocated")
                ):
                    estimate_parameter_count(config)
                if config.architecture == "transformer":
                    self.assertIs(type(model), FraiseModel)
                    torch.manual_seed(13)
                    expected = FraiseModel(config)
                    self.assertEqual(
                        model.state_dict().keys(), expected.state_dict().keys()
                    )
                    for key, value in expected.state_dict().items():
                        torch.testing.assert_close(
                            model.state_dict()[key], value, rtol=0, atol=0
                        )
        self.assertIsNone(build_model(TinyConfig()).new_cache())

    def test_optimizer_layout_partitions_every_parameter_once(self):
        model = build_model(TinyConfig())
        optimizer = MuonWithAdamW(
            model, tiny_settings(), layout=TINY.optimizer_layout(model)
        )
        self.assertEqual(
            tuple(group["optimizer"] for group in optimizer.param_groups), GROUP_TAGS
        )
        self.assertEqual(
            set(optimizer.param_groups[0]["params"]), {model.hidden.weight}
        )
        groups = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        self.assertEqual(len(groups), len(set(groups)))
        self.assertEqual(set(groups), set(model.parameters()))
        adam_only = MuonWithAdamW(
            model, tiny_settings(), layout=OptimizerLayout(frozenset())
        )
        self.assertEqual(adam_only.param_groups[0]["params"], [])
        with self.assertRaisesRegex(TypeError, "require an optimizer layout"):
            MuonWithAdamW(model, tiny_settings())
        with self.assertRaisesRegex(ValueError, "outside the model"):
            MuonWithAdamW(
                model,
                tiny_settings(),
                layout=OptimizerLayout(frozenset({nn.Parameter(torch.ones(2, 2))})),
            )

    def test_no_cache_inference_recomputes_the_sliding_context(self):
        model = build_model(TinyConfig()).eval()
        decoder = IncrementalDecoder(model)
        ids = [1, 3]
        for _ in range(8):
            actual = decoder.logits(ids)
            with torch.inference_mode():
                expected = model(torch.tensor([ids[-model.max_seq_len :]]))[0, -1]
            torch.testing.assert_close(actual, expected)
            ids.append(int(actual.argmax()))
        self.assertIsNone(decoder.cache)

    def test_runtime_uses_architecture_initialization_and_trains_on_cpu(self):
        settings = tiny_settings(use_compile=False, tied_weight_init_std=0.125)
        runtime = build_runtime(TinyConfig(), settings, 2, device="cpu")
        torch.testing.assert_close(
            runtime.model.embedding.weight, torch.full((8, 4), 0.125)
        )
        before = runtime.model.output.weight.detach().clone()
        batch = PackedTokenBatch(
            torch.tensor([[1, 3]]),
            torch.tensor([[3, 2]]),
            torch.ones((1, 2), dtype=torch.bool),
        )
        loss, tokens, norm = runtime.step(batch)
        self.assertTrue(math.isfinite(loss) and math.isfinite(norm))
        self.assertEqual(tokens, 2)
        self.assertFalse(torch.equal(runtime.model.output.weight, before))
        torch.manual_seed(settings.seed)
        expected = TinyModel(TinyConfig())
        resumed = build_runtime(
            TinyConfig(), settings, 2, device="cpu", initialize_weights=False
        )
        for key, value in expected.state_dict().items():
            torch.testing.assert_close(
                resumed.model.state_dict()[key], value, rtol=0, atol=0
            )

    def test_resume_normalizes_legacy_baseline_architecture(self):
        current = run_manifest(
            tiny_data(),
            tiny_tokenizer(),
            tiny_config(),
            tiny_settings(),
            torch.device("cpu"),
        )
        saved = deepcopy(current)
        del saved["identity"]["model_config"]["architecture"]
        check_resume_identity(saved, current)
        changed = deepcopy(current)
        changed["identity"]["model_config"] = asdict(TinyConfig())
        with self.assertRaisesRegex(ValueError, "resume model_config"):
            check_resume_identity(current, changed)

    def test_independent_architecture_trains_and_loads_final_and_checkpoint_weights(
        self,
    ):
        with TemporaryDirectory() as directory:
            settings = tiny_settings(use_compile=False)
            model = pretrain_model(
                tiny_tokenizer(),
                tiny_data(),
                settings,
                TinyConfig(),
                directory,
                "tiny",
                device="cpu",
            )
            run = Path(directory) / "tiny"
            for checkpoint in (None, "checkpoint_step_00000002.pt"):
                with self.subTest(checkpoint=checkpoint):
                    loaded = load_run(run, checkpoint=checkpoint, device="cpu")
                    self.assertIsInstance(loaded.model, TinyModel)
                    self.assertEqual(loaded.config, TinyConfig())
                    ids = torch.tensor([[1, 3, 2]])
                    torch.testing.assert_close(loaded.model(ids), model(ids))
            resumed = pretrain_model(
                tiny_tokenizer(),
                tiny_data(),
                settings,
                TinyConfig(),
                directory,
                "tiny",
                device="cpu",
                resume_from=run / "checkpoint_step_00000001.pt",
            )
            for key, value in model.state_dict().items():
                torch.testing.assert_close(
                    resumed.state_dict()[key], value, rtol=0, atol=0
                )

    def test_non_baseline_raw_weights_preserve_legacy_looking_names(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            weights = {"token_embedding.embedding.weight": torch.ones(2, 2)}
            save_model_state_dict(weights, path)
            self.assertEqual(
                load_model_weights(path, migrate_legacy=False).keys(), weights.keys()
            )
            self.assertEqual(set(load_model_weights(path)), {"token_embedding.weight"})

    def test_spawned_validator_uses_parent_registered_architecture(self):
        config, settings = (
            TinyConfig(),
            tiny_settings(
                use_compile=False,
                max_validation_tokens=3,
                validation_timeout_seconds=30,
            ),
        )
        split = Dataset.from_dict({"token_ids": [[1, 3, 4, 2]]})
        with TemporaryDirectory() as directory:
            model = pretrain_model(
                tiny_tokenizer(),
                tiny_data(),
                settings,
                config,
                directory,
                "tiny",
                device="cpu",
            )
            expected = evaluate(model, split, settings, 0)
            messages = []
            process = ValidationProcess(config, split, settings, 0, messages.append)
            try:
                actual = process.submit(
                    2, Path(directory) / "tiny" / "checkpoint_step_00000002.pt"
                ).result(timeout=40)
                self.assertAlmostEqual(actual["loss"], expected)
                self.assertTrue(
                    any("Validation worker ready" in message for message in messages)
                )
            finally:
                process.stop()


@dataclass(frozen=True)
class _WrongConfig:
    architecture: str = TINY.id
    vocab_size: int = 8
    max_seq_len: int = 4


if __name__ == "__main__":
    unittest.main()
