import unittest

from datasets import Dataset
from fixtures import tiny_settings, tiny_tokenizer

from fraise.data.types import TrainingData
from fraise.preview.examples import collect_examples
from fraise.preview.html import render_html
from fraise.training.batches import iter_dataset_packed_batches
from fraise.training.preparation import prepare_data


class VisualizeTrainingDataTests(unittest.TestCase):
    def test_mixed_preview_matches_training_and_only_reshuffles_when_requested(self):
        data = TrainingData(
            {
                "first": Dataset.from_dict({"text": ["a b", "c d", "e a"]}),
                "second": Dataset.from_dict({"text": ["e d", "c b", "a e"]}),
            }
        )
        for per_epoch in (False, True):
            with self.subTest(per_epoch=per_epoch):
                settings = tiny_settings(
                    dataset_token_counts={"first": 8, "second": 8},
                    max_train_tokens=16,
                    num_epochs=2,
                    batch_size=2,
                    shuffle_per_epoch=per_epoch,
                    shuffle_data=True,
                )
                tokenizer = tiny_tokenizer()
                examples = collect_examples(data, tokenizer, settings, 16)
                prepared = prepare_data(data, tokenizer, settings)
                actual = []
                for epoch, budgets in enumerate(prepared.budgets):
                    for _, batch in iter_dataset_packed_batches(
                        prepared.train,
                        budgets,
                        2,
                        2,
                        0,
                        batch_size=settings.batch_size,
                        seed=settings.seed + epoch if per_epoch else settings.seed,
                        shuffle_per_epoch=per_epoch,
                        shuffle_data=True,
                    ):
                        actual.extend(
                            zip(
                                batch.sources,
                                batch.input_ids.tolist(),
                                batch.target_mask.sum(1).tolist(),
                            )
                        )
                self.assertEqual(
                    actual,
                    [
                        (row["source"], row["input_ids"], row["valid_targets"])
                        for row in examples
                    ],
                )
                first, second = actual[:8], actual[8:]
                if per_epoch:
                    self.assertNotEqual(first, second)
                else:
                    self.assertEqual(first, second)
                html = render_html(examples, settings)
                self.assertIn("mix datasets True", html)

    def test_preview_reuses_the_same_source_budget_in_each_epoch(self):
        examples = collect_examples(
            TrainingData({"fixture": Dataset.from_dict({"text": ["a b c d e"]})}),
            tiny_tokenizer(),
            tiny_settings(num_epochs=3),
            6,
        )
        self.assertEqual([item["epoch"] for item in examples], [1, 1, 2, 2, 3, 3])
        self.assertEqual(sum(item["valid_targets"] for item in examples), 12)
        self.assertEqual(examples[0]["input_ids"], examples[2]["input_ids"])
        self.assertEqual(examples[0]["input_ids"], examples[4]["input_ids"])

    def test_source_labels_survive_mixed_batches_and_partial_tails(self):
        data = TrainingData(
            {
                "first": Dataset.from_dict({"text": ["a b c d e"]}),
                "second": Dataset.from_dict({"text": ["e d c b a"]}),
            }
        )
        settings = tiny_settings(
            batch_size=3,
            dataset_token_counts={"first": 5, "second": 5},
            max_train_tokens=10,
            sequence_length=4,
            packing_stride=2,
            shuffle_per_epoch=True,
        )
        examples = collect_examples(data, tiny_tokenizer(), settings, 4)
        self.assertEqual(
            [item["source"] for item in examples],
            ["first", "first", "second", "second"],
        )
        self.assertEqual([item["valid_targets"] for item in examples], [4, 1, 4, 1])
        self.assertEqual([item["batch"] for item in examples], [0, 0, 0, 1])
        self.assertIn("first", render_html(examples, settings))
        self.assertEqual(examples[1]["target_mask"], [True, False, False, False])

    def test_preview_matches_first_source_without_loading_later_sources(self):
        source = Dataset.from_dict({"text": ["a b c d e"]})
        data = TrainingData({"fixture": source})
        examples = collect_examples(data, tiny_tokenizer(), tiny_settings(), 2)
        self.assertEqual([item["source"] for item in examples], ["fixture", "fixture"])
        with self.assertRaisesRegex(ValueError, "only 2"):
            collect_examples(data, tiny_tokenizer(), tiny_settings(), 3)
