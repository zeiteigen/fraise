import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from datasets import Dataset
from fixtures import tiny_tokenizer
from macro_fixtures import (
    MACRO_TEXTS,
    byte_level_tokenizers,
    tiny_macro_tokenizer,
)

from fraise.tokenizer.artifacts import create_macro_tokenizer
from fraise.tokenizer.loading import load_tokenizer, tokenizer_fingerprint
from fraise.tokenizer.macro import MacroTokenizer, counting_tokenizer, unit_lengths
from fraise.tokenizer.macro_training import train_macro_tokenizer
from fraise.training.validation.measures import validation_measures


class MacroTokenizerTests(unittest.TestCase):
    def test_macros_decompose_exactly_into_the_base_encoding(self):
        macro = tiny_macro_tokenizer()
        base = macro.base
        self.assertEqual(macro.get_vocab_size(), 14)
        self.assertEqual(macro.base_vocab_size, 8)
        for text in [*MACRO_TEXTS[:3], "e e a d", "b"]:
            with self.subTest(text=text):
                encoding = macro.encode(text)
                expected = base.encode(text).ids
                self.assertEqual(macro.expand(encoding.ids), expected)
                self.assertLess(len(encoding.ids), len(expected) + 1)
                self.assertEqual(macro.decode(encoding.ids), base.decode(expected))
                self.assertEqual(
                    [e.ids for e in macro.encode_batch([text])], [encoding.ids]
                )
        # Base IDs are their own macros and special tokens never merge.
        self.assertEqual(macro.decomposition(5), (5,))
        ids = macro.encode("a b c").ids
        self.assertEqual((ids[0], ids[-1]), (1, 2))
        self.assertTrue(all(length >= 1 for length in unit_lengths(macro)))
        self.assertIs(counting_tokenizer(macro), base)

    def test_max_macro_length_is_inclusive_and_enforced(self):
        for limit in (2, 3, 4):
            with self.subTest(limit=limit):
                macro = tiny_macro_tokenizer(vocab_size=40, max_macro_length=limit)
                self.assertEqual(macro.max_macro_length, limit)
                self.assertEqual(macro.decompositions.shape[1], limit)

    def test_line_breaks_end_macros(self):
        base, macro = byte_level_tokenizers()
        boundaries = set(macro.boundary_ids)
        self.assertTrue(boundaries)
        for macro_id in range(macro.base_vocab_size, macro.get_vocab_size()):
            parts = macro.decomposition(macro_id)
            self.assertFalse(boundaries & set(parts[:-1]), parts)
        text = "the strawberry field\nstraw bales"
        self.assertEqual(macro.expand(macro.encode(text).ids), base.encode(text).ids)
        self.assertEqual(macro.decode(macro.encode(text).ids), text)
        encoding = macro.encode(text, add_special_tokens=False)
        self.assertEqual("".join(encoding.tokens), text)
        self.assertEqual(
            "".join(text[start:end] for start, end in encoding.offsets), text
        )

    def test_serialization_round_trips_and_loads_by_format(self):
        macro = tiny_macro_tokenizer()
        contents = macro.to_str()
        restored = MacroTokenizer.from_str(contents)
        self.assertEqual(restored.to_str(), contents)
        self.assertEqual(restored.merges, macro.merges)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "macro.json"
            path.write_text(contents)
            loaded = load_tokenizer(path)
            self.assertIsInstance(loaded, MacroTokenizer)
            self.assertEqual(
                tokenizer_fingerprint(loaded), tokenizer_fingerprint(macro)
            )
            base_path = Path(directory) / "base.json"
            base_path.write_text(tiny_tokenizer().to_str())
            self.assertNotIsInstance(load_tokenizer(base_path), MacroTokenizer)
        tampered = contents.replace('"max_macro_length":3', '"max_macro_length":1')
        with self.assertRaisesRegex(ValueError, "max_macro_length"):
            MacroTokenizer.from_str(tampered)

    def test_artifacts_reuse_matching_requests(self):
        sources = {"fixture": Dataset.from_dict({"text": MACRO_TEXTS})}
        with TemporaryDirectory() as directory:
            first = create_macro_tokenizer(
                tiny_tokenizer(), sources, 12, Path(directory), max_macro_length=3
            )
            self.assertTrue(first.name.startswith("fraise-macro-12-"))
            with patch(
                "fraise.tokenizer.macro_training.train_macro_tokenizer",
                side_effect=AssertionError("retrained"),
            ):
                again = create_macro_tokenizer(
                    tiny_tokenizer(), sources, 12, Path(directory), max_macro_length=3
                )
            self.assertEqual(again, first)
            self.assertIsInstance(load_tokenizer(first), MacroTokenizer)

    def test_training_rejects_invalid_requests(self):
        sources = {"fixture": Dataset.from_dict({"text": MACRO_TEXTS})}
        with self.assertRaisesRegex(ValueError, "vocab_size"):
            train_macro_tokenizer(tiny_tokenizer(), sources, 8)
        with self.assertRaisesRegex(ValueError, "max_macro_length"):
            train_macro_tokenizer(tiny_tokenizer(), sources, 12, max_macro_length=1)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            train_macro_tokenizer(
                tiny_tokenizer(), {"x": Dataset.from_dict({"text": []})}, 12
            )

    def test_byte_lengths_cover_macro_decompositions(self):
        base, macro = byte_level_tokenizers()
        base_measures = validation_measures(base)
        macro_measures = validation_measures(macro)
        self.assertIsNone(base_measures.unit_lengths)
        text = BYTE = "The strawberry field had straw."
        for tokenizer, measures in ((base, base_measures), (macro, macro_measures)):
            ids = tokenizer.encode(text).ids
            self.assertEqual(
                int(measures.byte_lengths[ids].sum()), len(BYTE.encode("utf-8"))
            )
        # Word-level vocabularies do not map tokens to exact bytes.
        self.assertIsNone(validation_measures(tiny_tokenizer()).byte_lengths)


if __name__ == "__main__":
    unittest.main()
