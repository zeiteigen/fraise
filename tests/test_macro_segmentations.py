"""Word-bounded, curriculum and greedy (prefix-closed) macro segmentations."""

import unittest

from datasets import Dataset
from macro_fixtures import BYTE_TEXTS

from fraise.tokenizer.macro import MacroTokenizer
from fraise.tokenizer.macro_training import train_macro_tokenizer
from fraise.tokenizer.training import train_tokenizer

SOURCE = {"fixture": Dataset.from_dict({"text": BYTE_TEXTS})}
PROBES = [
    *BYTE_TEXTS[:3],
    "Straw strawberries, straw hats!\n\nThe field.",
    "x",
    "",
]


def tokenizers():
    base = train_tokenizer(SOURCE, 280)
    variants = {
        "cross": train_macro_tokenizer(base, SOURCE, 340, max_macro_length=4),
        "words": train_macro_tokenizer(
            base, SOURCE, 340, max_macro_length=4, word_bounded_vocab_size=340
        ),
        "curriculum": train_macro_tokenizer(
            base, SOURCE, 340, max_macro_length=4, word_bounded_vocab_size=310
        ),
        "greedy": train_macro_tokenizer(
            base, SOURCE, 340, max_macro_length=4, segmentation="greedy",
            merges_per_round=8,
        ),
        "greedy-words": train_macro_tokenizer(
            base, SOURCE, 340, max_macro_length=4, segmentation="greedy",
            word_bounded=True, merges_per_round=8,
        ),
    }
    return base, variants


class MacroSegmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base, cls.variants = tokenizers()

    def test_every_segmentation_expands_to_the_base_encoding(self):
        for name, macro in self.variants.items():
            for text in PROBES:
                with self.subTest(name=name, text=text):
                    encoding = macro.encode(text)
                    base_ids = self.base.encode(text).ids
                    self.assertEqual(macro.expand(encoding.ids), base_ids)
                    self.assertEqual(macro.decode(encoding.ids), text)
                    self.assertEqual(
                        [e.ids for e in macro.encode_batch([text, text])],
                        [encoding.ids, encoding.ids],
                    )
                    self.assertEqual(macro.encode_base_ids(base_ids), encoding.ids)

    def test_vocabularies_fill_and_compress(self):
        text = " ".join(BYTE_TEXTS[:3])
        base_count = len(self.base.encode(text).ids)
        for name, macro in self.variants.items():
            with self.subTest(name=name):
                self.assertEqual(macro.get_vocab_size(), 340)
                self.assertLess(len(macro.encode(text).ids), base_count)
        self.assertEqual(self.variants["curriculum"].word_bounded_merges, 30)

    def test_word_bounded_macros_hold_one_word(self):
        for name in ("words", "greedy-words"):
            macro = self.variants[name]
            for text in PROBES:
                base = self.base.encode(text, add_special_tokens=False)
                starts = macro.word_starts(base.ids)
                cursor = 0
                for macro_id in macro.encode(text, add_special_tokens=False).ids:
                    width = int(macro.unit_lengths[macro_id])
                    with self.subTest(name=name, text=text, cursor=cursor):
                        self.assertFalse(starts[cursor + 1 : cursor + width].any())
                        self.assertEqual(
                            len(set(base.word_ids[cursor : cursor + width])), 1
                        )
                    cursor += width
        # Cross-word merges exist where they are allowed.
        crosses = [
            macro_id
            for macro_id in range(280, 340)
            if " " in self.variants["cross"].decode([macro_id]).strip()
        ]
        self.assertTrue(crosses)

    def test_greedy_vocabularies_are_prefix_closed(self):
        for name in ("greedy", "greedy-words"):
            macro = self.variants[name]
            entries = {macro.decomposition(i) for i in range(macro.get_vocab_size())}
            for parts in entries:
                with self.subTest(name=name, parts=parts):
                    self.assertIn(parts[:-1] or parts, entries)

    def test_greedy_boundaries_need_no_lookahead(self):
        # Encoding any prefix reproduces every macro of the full encoding that
        # ends before the prefix does: no boundary depends on later text.
        for name in ("greedy", "greedy-words"):
            macro = self.variants[name]
            for text in PROBES[:4]:
                ids = self.base.encode(text, add_special_tokens=False).ids
                starts = macro.word_starts(ids)
                full = macro.encode_base_ids(ids, starts)
                ends = [0]
                for macro_id in full:
                    ends.append(ends[-1] + int(macro.unit_lengths[macro_id]))
                for cut in range(1, len(ids) + 1):
                    prefix = macro.encode_base_ids(ids[:cut], starts[:cut])
                    complete = sum(end <= cut for end in ends[1:])
                    with self.subTest(name=name, cut=cut):
                        self.assertEqual(prefix[: complete], full[: complete])

    def test_segmentations_round_trip(self):
        for name, macro in self.variants.items():
            with self.subTest(name=name):
                restored = MacroTokenizer.from_str(macro.to_str())
                self.assertEqual(restored.to_str(), macro.to_str())
                self.assertEqual(restored.segmentation, macro.segmentation)
                self.assertEqual(restored.word_bounded, macro.word_bounded)
                text = PROBES[3]
                self.assertEqual(restored.encode(text).ids, macro.encode(text).ids)
        self.assertIn('"format_version":1', self.variants["cross"].to_str())

    def test_invalid_greedy_vocabularies_are_refused(self):
        with self.assertRaisesRegex(ValueError, "append one base token"):
            MacroTokenizer(
                self.base,
                [(1, 2), (3, 280)],
                max_macro_length=4,
                boundary_ids=[],
                segmentation="greedy",
            )


if __name__ == "__main__":
    unittest.main()
