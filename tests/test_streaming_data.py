import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fixtures import tiny_tokenizer

from fraise.data import stream
from fraise.data.sources import Source


class StreamTests(unittest.TestCase):
    source = Source("fixture", "fixture/text")

    def test_downloads_stop_at_token_budget_and_reuse_arrow(self):
        rows = [{"text": "hold"}, {"text": "a b"}, {"text": "c"}, {"text": "unused"}]
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            with patch.object(stream, "stream_source", return_value=iter(rows)):
                result = stream.load_stream(
                    self.source,
                    directory,
                    validation_size_gb=4e-9,
                    max_train_tokens=6,
                    tokenizer=tiny_tokenizer(),
                )
            self.assertEqual(result.validation["text"], ["hold"])
            self.assertEqual(result.train["fixture"]["text"], ["a b", "c"])
            with (
                patch.object(
                    stream, "stream_source", side_effect=AssertionError("redownloaded")
                ),
                patch.object(
                    stream, "load_dataset", side_effect=AssertionError("reparsed Arrow")
                ),
            ):
                cached = stream.load_stream(
                    self.source,
                    directory,
                    validation_size_gb=4e-9,
                    max_train_tokens=6,
                    tokenizer=tiny_tokenizer(),
                )
            self.assertEqual(cached.train["fixture"]["text"], ["a b", "c"])

    def test_request_switching_and_validation_only_do_not_overwrite_training(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)

            def remote(source):
                return iter(
                    [{"text": "abc"}, {"text": "def"}, {"text": "ghi"}, {"text": "jkl"}]
                )

            with patch.object(stream, "stream_source", side_effect=remote) as fetch:
                first = stream.load_stream(
                    self.source, directory, size_gb=9e-9, validation_size_gb=3e-9
                )
                second = stream.load_stream(
                    self.source, directory, size_gb=6e-9, validation_size_gb=3e-9
                )
                held_out = stream.load_stream(
                    self.source, directory, validation_size_gb=3e-9, include_train=False
                )
                again = stream.load_stream(
                    self.source, directory, size_gb=9e-9, validation_size_gb=3e-9
                )
                self.assertEqual(fetch.call_count, 2)
            self.assertEqual(first.train["fixture"]["text"], ["def", "ghi"])
            self.assertEqual(second.train["fixture"]["text"], ["def"])
            self.assertEqual(held_out.validation["text"], ["abc"])
            self.assertEqual(again.train["fixture"]["text"], ["def", "ghi"])
            self.assertEqual(
                first.validation._fingerprint, second.validation._fingerprint
            )

    def test_batched_token_counting_selects_the_same_rows_as_per_row_counting(self):
        import random

        tokenizer = tiny_tokenizer()
        generator = random.Random(4)
        texts = [
            " ".join(generator.choices("abcde", k=generator.randrange(0, 5)))
            for _ in range(60)
        ]
        for target in (1, 7, 40, 10_000):
            expected, tokens = [], 0
            for text in texts:
                expected.append(text)
                tokens += len(tokenizer.encode(text).ids) if text else 0
                if tokens >= target:
                    break
            for batch_size in (1, 3, 256):
                with self.subTest(target=target, batch_size=batch_size):
                    counter = stream._TrainTokenCounter(tokenizer, target, batch_size)
                    selected = []
                    for text in texts:
                        if counter.add(text):
                            selected += counter.flush()
                            if counter.reached:
                                break
                    selected += counter.flush()
                    self.assertEqual(selected, expected)

    def test_training_only_load_skips_the_held_out_prefix(self):
        with TemporaryDirectory() as directory:
            rows = [{"text": "abc"}, {"text": "def"}, {"text": "ghi"}]
            with patch.object(stream, "stream_source", return_value=iter(rows)):
                result = stream.load_stream(
                    self.source,
                    Path(directory),
                    size_gb=9e-9,
                    validation_size_gb=3e-9,
                    include_validation=False,
                )
        self.assertIsNone(result.validation)
        self.assertEqual(result.train["fixture"]["text"], ["def", "ghi"])

    def test_changed_raw_file_invalidates_arrow_without_remote_download(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            with patch.object(
                stream, "stream_source", return_value=iter([{"text": "old"}])
            ):
                result = stream.load_stream(self.source, directory, size_gb=1e-8)
            raw = next((directory / "raw").glob("*/text.jsonl"))
            raw.write_text('{"text":"new"}\n')
            with patch.object(
                stream, "stream_source", side_effect=AssertionError("redownloaded")
            ):
                changed = stream.load_stream(self.source, directory, size_gb=1e-8)
            self.assertEqual(changed.train["fixture"]["text"], ["new"])
            self.assertNotEqual(
                result.train["fixture"]._fingerprint,
                changed.train["fixture"]._fingerprint,
            )

    def test_validation_only_stops_before_training_rows(self):
        seen = []

        def rows():
            for text in ("abc", "def", "unused"):
                seen.append(text)
                yield {"text": text}

        with (
            TemporaryDirectory() as directory,
            patch.object(stream, "stream_source", return_value=rows()),
        ):
            result = stream.load_stream(
                self.source,
                Path(directory),
                validation_size_gb=3e-9,
                include_train=False,
            )
        self.assertEqual(result.validation["text"], ["abc"])
        self.assertNotIn("unused", seen)
        self.assertFalse(result.train)

    def test_invalid_limits_fail_before_download(self):
        with (
            TemporaryDirectory() as directory,
            patch.object(stream, "stream_source") as fetch,
        ):
            for options in (
                {"max_train_tokens": 1},
                {"include_train": False},
                {"size_gb": -1},
                {"size_gb": 1e-9, "validation_size_gb": 2e-9},
            ):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    stream.load_stream(self.source, Path(directory), **options)
            fetch.assert_not_called()

    def test_training_preview_reserves_held_out_rows_without_materializing_validation(
        self,
    ):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            with patch.object(
                stream,
                "stream_source",
                return_value=iter([{"text": "abc"}, {"text": "def"}, {"text": "ghi"}]),
            ):
                preview = stream.load_stream(
                    self.source,
                    directory,
                    size_gb=9e-9,
                    validation_size_gb=3e-9,
                    include_validation=False,
                )
            self.assertEqual(preview.train["fixture"]["text"], ["def", "ghi"])
            self.assertIsNone(preview.validation)
            self.assertEqual(len(list((directory / "raw").glob("*/text.jsonl"))), 1)
