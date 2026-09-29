import unittest

from datasets import Dataset

from fraise.data.provenance import dataset_fingerprint, describe_dataset
from fraise.data.types import TrainingData


class DatasetProvenanceTests(unittest.TestCase):
    def test_source_size_excludes_arrow_storage(self):
        source = Dataset.from_dict({"text": ["a"]})
        source.info.download_size = 123
        self.assertEqual(describe_dataset(source)["size_in_bytes"], 123)

    def test_named_sources_keep_explicit_boundaries(self):
        first = Dataset.from_dict({"text": ["first"]})
        second = Dataset.from_dict({"text": ["second"]})
        data = TrainingData({"first": first, "second": second})
        self.assertIs(data.train["first"], first)
        self.assertIs(data.train["second"], second)
        self.assertNotEqual(
            dataset_fingerprint(data.train),
            dataset_fingerprint({"first": second, "second": first}),
        )
