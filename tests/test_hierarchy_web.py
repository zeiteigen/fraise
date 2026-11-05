"""The tokenization visualizer API compares hierarchy configurations."""

import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from macro_fixtures import tiny_macro_tokenizer

from fraise.hierarchy import hierarchy_from_macro_tokenizer

HAS_WEB = importlib.util.find_spec("fastapi") is not None
if HAS_WEB:
    from fastapi.testclient import TestClient

    from fraise.web.app import create_app


@unittest.skipUnless(HAS_WEB, "install the web extra for local web API tests")
class TokenizationApiTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        tokenizer = tiny_macro_tokenizer(vocab_size=20, max_macro_length=4)
        (root / "tokenizers").mkdir()
        (root / "tokenizers" / "macro.json").write_text(tokenizer.to_str())
        tokenizer.base.save(str(root / "tokenizers" / "base.json"))
        definition = hierarchy_from_macro_tokenizer(tokenizer, (8, 12, 17), [2, 3])
        definition.save(root / "hierarchies" / "three.npz")
        self.client = self.enterContext(
            TestClient(create_app(data_dir=root, device="cpu"))
        )

    def test_compares_hierarchies_and_reports_errors_per_configuration(self):
        self.assertEqual(self.client.get("/static/tokens.js").status_code, 200)
        files = self.client.get("/api/hierarchies").json()["hierarchies"]
        self.assertEqual(files, [{"id": "three.npz", "label": "three"}])
        macro = "tokenizer:macro.json"
        response = self.client.post(
            "/api/tokenize",
            json={
                "text": "a b c d e a b",
                "add_special_tokens": True,
                "configurations": [
                    {"tokenizer_id": macro},
                    {
                        "tokenizer_id": macro,
                        "hierarchy": {"kind": "merges", "level_sizes": [8, 12, 17]},
                    },
                    {
                        "tokenizer_id": macro,
                        "hierarchy": {"kind": "file", "file_id": "three.npz"},
                    },
                    {"tokenizer_id": "tokenizer:base.json"},
                    {
                        "tokenizer_id": macro,
                        "hierarchy": {"kind": "merges", "level_sizes": [8, 9]},
                    },
                ],
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        two, merges, file, base, wrong = response.json()["results"]
        self.assertEqual(len(two["levels"]), 2)
        self.assertEqual(len(merges["levels"]), 3)
        self.assertEqual(merges["rows"], file["rows"])
        self.assertEqual(two["input_tokens"], merges["input_tokens"])
        self.assertEqual(base["levels"], [])
        self.assertFalse(base["macro"])
        self.assertTrue(two["macro"])
        kinds = {
            t["id"]: (t["macro"], t["base_vocab_size"])
            for t in self.client.get("/api/catalog").json()["tokenizers"]
        }
        self.assertEqual(kinds["tokenizer:macro.json"], (True, 8))
        self.assertEqual(kinds["tokenizer:base.json"], (False, 8))
        self.assertGreater(len(base["input_tokens"]), len(two["input_tokens"]))
        self.assertIn("level sizes", wrong["error"])
        rejected = self.client.post(
            "/api/tokenize",
            json={
                "text": "a",
                "configurations": [
                    {"tokenizer_id": macro, "hierarchy": {"kind": "file", "file_id": "../x.npz"}}
                ],
            },
        ).json()["results"][0]
        self.assertIn("error", rejected)

    def test_invalid_merge_configurations_return_individual_errors(self):
        response = self.client.post(
            "/api/tokenize",
            json={
                "text": "a b c",
                "configurations": [
                    {
                        "tokenizer_id": "tokenizer:base.json",
                        "hierarchy": {"kind": "merges", "level_sizes": [8, 12, 17]},
                    },
                    {
                        "tokenizer_id": "tokenizer:macro.json",
                        "hierarchy": {"kind": "merges", "level_sizes": [8, 20, 17]},
                    },
                    {"tokenizer_id": "tokenizer:macro.json"},
                ],
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        plain, levels, valid = response.json()["results"]
        self.assertIn("needs a macro tokenizer", plain["error"])
        self.assertIn("strictly increase", levels["error"])
        self.assertNotIn("error", valid)


if __name__ == "__main__":
    unittest.main()
