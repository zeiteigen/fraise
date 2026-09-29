import unittest

from datasets import Dataset
from fixtures import tiny_settings, tiny_tokenizer

from fraise.data.types import TrainingData
from fraise.preview.examples import collect_examples
from fraise.preview.html import render_html


class VisualizeTrainingDataTests(unittest.TestCase):
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
            shuffle=True,
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
