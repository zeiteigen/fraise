import unittest
from dataclasses import asdict

from fixtures import tiny_settings

from fraise.config.training import TrainingSettings


class TrainingConfigTests(unittest.TestCase):
    def test_shuffle_flags_are_independent(self):
        values = asdict(tiny_settings())
        del values["shuffle_per_epoch"]
        del values["shuffle_data"]
        settings = TrainingSettings.from_dict(values)
        self.assertTrue(settings.shuffle_per_epoch)
        self.assertFalse(settings.shuffle_data)
        for per_epoch in (False, True):
            for mixed in (False, True):
                with self.subTest(per_epoch=per_epoch, mixed=mixed):
                    settings = TrainingSettings.from_dict(
                        values
                        | {
                            "shuffle_per_epoch": per_epoch,
                            "shuffle_data": mixed,
                        }
                    )
                    self.assertIs(settings.shuffle_per_epoch, per_epoch)
                    self.assertIs(settings.shuffle_data, mixed)
        with self.assertRaises(TypeError):
            TrainingSettings.from_dict(values | {"shuffle": False})

    def test_accumulation_is_derived_from_minibatch_size(self):
        settings = tiny_settings(batch_size=8, minibatch_size=2)
        self.assertEqual(settings.minibatch_size, 2)
        self.assertEqual(settings.gradient_accumulation_steps, 4)
        for extra in (
            {"minibatch_size": 0},
            {"minibatch_size": 3},
            {"gradient_accumulation_steps": 4},
        ):
            with self.subTest(extra=extra), self.assertRaises((ValueError, TypeError)):
                tiny_settings(batch_size=8, **extra)

    def test_counts_validate_names_types_and_sum(self):
        settings = tiny_settings()
        with self.assertRaisesRegex(ValueError, "unknown dataset"):
            settings.validate_sources({"other"})
        with self.assertRaisesRegex(ValueError, "missing dataset"):
            settings.validate_sources({"fixture", "other"})
        for count in (True, 1.5, -1):
            with self.subTest(count=count), self.assertRaises(ValueError):
                tiny_settings(dataset_token_counts={"fixture": count})
        with self.assertRaisesRegex(ValueError, "sum to max_train_tokens"):
            tiny_settings(dataset_token_counts={"fixture": 3})

    def test_nonfinite_and_wrong_primitive_values_fail_at_ingestion(self):
        for changes in (
            {"learning_rate": float("nan")},
            {"adamw_learning_rate": 0},
            {"adamw_learning_rate": float("inf")},
            {"adamw_learning_rate": True},
            {"muon_momentum": -0.1},
            {"muon_momentum": 1},
            {"muon_momentum": float("nan")},
            {"muon_momentum": True},
            {"muon_nesterov": 1},
            {"muon_ns_steps": 0},
            {"muon_ns_steps": 1.5},
            {"muon_ns_steps": True},
            {"weight_decay": -1},
            {"use_bf16": 1},
            {"use_compile": 1},
            {"deterministic": "true"},
            {"compile_threads": 0},
            {"compile_threads": True},
            {"compile_threads": 1.5},
            {"log_interval_steps": 0},
            {"log_interval_steps": True},
            {"shuffle_per_epoch": "true"},
            {"shuffle_data": 1},
            {"betas": [0.9, 1]},
            {"seed": True},
            {"packing_stride": 3},
            {"sequence_length": 1},
            {"max_validation_tokens": -1},
            {"max_validation_tokens": False},
            {"max_validation_tokens": 1.5},
            {"validation_timeout_seconds": 0},
            {"validation_timeout_seconds": float("inf")},
            {"validation_cpu_threads": 0},
            {"validation_device": "auto"},
        ):
            with (
                self.subTest(changes=changes),
                self.assertRaises((ValueError, TypeError)),
            ):
                tiny_settings(**changes)

    def test_execution_and_reporting_settings_have_safe_defaults(self):
        settings = tiny_settings()
        self.assertTrue(settings.use_compile)
        self.assertEqual(settings.compile_threads, 4)
        self.assertFalse(settings.deterministic)
        self.assertEqual(settings.log_interval_steps, 100)
        settings = tiny_settings(
            use_compile=False,
            compile_threads=2,
            deterministic=True,
            log_interval_steps=10,
        )
        self.assertFalse(settings.use_compile)
        self.assertEqual(settings.compile_threads, 2)
        self.assertTrue(settings.deterministic)
        self.assertEqual(settings.log_interval_steps, 10)

    def test_muon_and_auxiliary_adamw_defaults_and_overrides(self):
        settings = tiny_settings()
        self.assertEqual(settings.adamw_learning_rate, 0.0003)
        self.assertEqual(settings.muon_momentum, 0.95)
        self.assertTrue(settings.muon_nesterov)
        self.assertEqual(settings.muon_ns_steps, 5)
        self.assertEqual(settings.betas, (0.9, 0.95))
        settings = tiny_settings(
            adamw_learning_rate=0.001,
            muon_momentum=0,
            muon_nesterov=False,
            muon_ns_steps=3,
        )
        self.assertEqual(settings.adamw_learning_rate, 0.001)
        self.assertEqual(settings.muon_momentum, 0)
        self.assertFalse(settings.muon_nesterov)
        self.assertEqual(settings.muon_ns_steps, 3)

    def test_validation_budget_is_loaded_from_hyperparameters_json(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from fraise.config.loading import load_training_settings

        with TemporaryDirectory() as directory:
            path = Path(directory) / "hyperparameters.json"
            path.write_text(
                json.dumps(
                    {
                        "batch_size": 1,
                        "learning_rate": 0.001,
                        "num_epochs": 1,
                        "max_train_tokens": 4,
                        "dataset_token_counts": {"fixture": 4},
                        "max_validation_tokens": 123,
                        "validation_timeout_seconds": 12,
                        "validation_cpu_threads": 2,
                    }
                )
            )
            settings = load_training_settings(path)
            self.assertEqual(settings.max_validation_tokens, 123)
            self.assertEqual(settings.validation_timeout_seconds, 12)
            self.assertEqual(settings.validation_cpu_threads, 2)

    def test_validation_budget_accepts_disabled_limited_and_full_split_modes(self):
        for budget in (0, 123, None):
            with self.subTest(budget=budget):
                self.assertEqual(
                    tiny_settings(max_validation_tokens=budget).max_validation_tokens,
                    budget,
                )

    def test_tied_weight_initialization_defaults_and_accepts_overrides(self):
        for option, expected in (({}, 0.02), ({"tied_weight_init_std": 0.1}, 0.1)):
            with self.subTest(option=option):
                self.assertEqual(tiny_settings(**option).tied_weight_init_std, expected)

    def test_tied_weight_initialization_rejects_invalid_standard_deviations(self):
        for value in (0, -0.02, float("nan"), float("inf"), True, "0.02", None):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "tied_weight_init_std"),
            ):
                tiny_settings(tied_weight_init_std=value)
