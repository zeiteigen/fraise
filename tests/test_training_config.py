import unittest

from fixtures import tiny_settings


class TrainingConfigTests(unittest.TestCase):
    def test_accumulation_is_derived_from_minibatch_size(self):
        for extra in (
            {"minibatch_size": 2},
            {"gradient_accumulation_steps": 4},
            {"minibatch_size": 2, "gradient_accumulation_steps": 4},
        ):
            settings = tiny_settings(batch_size=8, **extra)
            self.assertEqual(settings.minibatch_size, 2)
            self.assertEqual(settings.gradient_accumulation_steps, 4)
        for extra in (
            {"minibatch_size": 0},
            {"gradient_accumulation_steps": True},
            {"minibatch_size": 3},
            {"gradient_accumulation_steps": 3},
            {"minibatch_size": 2, "gradient_accumulation_steps": 3},
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
            {"weight_decay": -1},
            {"use_bf16": 1},
            {"shuffle": "true"},
            {"betas": [0.9, 1]},
            {"seed": True},
            {"packing_stride": 3},
            {"sequence_length": 1},
        ):
            with (
                self.subTest(changes=changes),
                self.assertRaises((ValueError, TypeError)),
            ):
                tiny_settings(**changes)
