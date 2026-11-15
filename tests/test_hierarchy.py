"""Multi-level hierarchical embeddings: definitions, composition, and accounting."""

import json
import io
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from fixtures import tiny_settings, tiny_tokenizer
from macro_fixtures import (
    tiny_fraise_config,
    tiny_hierarchy_config,
    tiny_macro_tokenizer,
)

from fraise.architectures import build_model, get_architecture, model_config_from_dict
from fraise.architectures.hierarchical import hierarchy_report
from fraise.config.hierarchy import HierarchyConfig
from fraise.cli.hierarchy import main as hierarchy_main
from fraise.hierarchy import (
    EmbeddingHierarchy,
    hierarchy_from_macro_tokenizer,
    two_level_hierarchy,
)
from fraise.hierarchy.binding import check_tokenizer_hierarchy
from fraise.hierarchy.runs import definition_from_weights
from fraise.model.hierarchical import MacroEmbedding
from fraise.preview.hierarchy import analyze_text, render_text_tree
from fraise.training.optimizer import MuonWithAdamW
from fraise.training.optimizer import OPTIMIZER_IDENTITY
from fraise.training.run.manifest import check_resume_identity

FRAISE = get_architecture("fraise")
LEVELS = (8, 12, 17)


def macro_tokenizer():
    return tiny_macro_tokenizer(vocab_size=20, max_macro_length=4)


def perturbed(model):
    """Zero-initialized projections and length embeddings hide bugs; randomize."""
    torch.manual_seed(3)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.5)
    return model


def bound(config, tokenizer, definition=None):
    torch.manual_seed(0)
    model = build_model(config)
    if definition is None:
        FRAISE.bind_tokenizer(model, tokenizer)
    else:
        model.bind_decompositions(tokenizer.decompositions, tokenizer.unit_lengths)
        model.embedding.bind_hierarchy(definition)
    return perturbed(model).eval()


def reference_embedding(embedding, definition, node, level):
    """Recursive composition without deduplication or batching."""
    if level == 0:
        return embedding.atoms.weight[node]
    hierarchy_level = embedding.levels[level - 1]
    composer = embedding.composer(hierarchy_level)
    children = definition.children(node, level)
    parts = torch.stack(
        [reference_embedding(embedding, definition, c, level - 1) for c in children]
    )
    count = len(children)
    x = (parts * composer.position_gates[:count]).sum(0) / count**0.5
    x = x + composer.length_embedding.weight[count]
    width = int(definition.residual_widths[node])
    if definition.owner_level(node) == level and width:
        start, _ = definition.owned_range(level)
        offset = int(definition.residual_widths[start:node].sum())
        values = hierarchy_level.values[offset : offset + width]
        x = x + composer.residual_projection.weight[:, :width] @ values
    return x


def custom_definition(tokenizer):
    """Non-uniform widths (including zero) on the merge-prefix structure."""
    merged = hierarchy_from_macro_tokenizer(tokenizer, LEVELS, [2, 3])
    widths = np.array([0, 1, 2, 2, 0, 1, 2, 0, 1, 2, 0, 2, 3, 0, 1, 3, 2])
    return EmbeddingHierarchy(
        merged.level_sizes, merged.compositions, widths, level_names=("a", "b", "c")
    )


def file_config(tokenizer, definition, path, sha256, **changes):
    levels = definition.config_levels()
    hierarchy = {"levels": levels, "source": "file", "path": str(path), "sha256": sha256}
    return tiny_fraise_config(
        tokenizer,
        macro_embedding="hierarchical",
        macro_residual_dim=0,
        embedding_hierarchy=hierarchy | changes,
    )


class HierarchyDefinitionTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = macro_tokenizer()

    def test_merge_prefix_hierarchy_spells_tokenizer_decompositions(self):
        definition = hierarchy_from_macro_tokenizer(self.tokenizer, LEVELS, [2, 3])
        check_tokenizer_hierarchy(definition, self.tokenizer)
        table, counts = definition.atom_expansions()
        np.testing.assert_array_equal(counts, self.tokenizer.unit_lengths)
        for node in range(LEVELS[1], LEVELS[2]):
            children = definition.children(node, 2)
            self.assertTrue(all(child < LEVELS[1] for child in children))
            spelled = [atom for c in children for atom in table[c, : counts[c]]]
            self.assertEqual(spelled, list(self.tokenizer.decomposition(node)))
        # Carried nodes recompose from themselves alone.
        self.assertEqual(definition.children(3, 2), [3])
        summary = definition.level_summary()
        self.assertEqual(summary[1]["owned_nodes"], 12)  # atoms belong to level 1
        self.assertEqual(summary[2]["residual_parameters"], 5 * 3)
        self.assertEqual(definition.owner_level(2), 1)
        self.assertEqual(definition.owner_level(15), 2)

    def test_two_level_hierarchy_matches_merge_prefixes(self):
        tokenizer = self.tokenizer
        legacy = two_level_hierarchy(
            tokenizer.decompositions, tokenizer.unit_lengths, 8, 4
        )
        merged = hierarchy_from_macro_tokenizer(tokenizer, (8, 17), 4)
        np.testing.assert_array_equal(legacy.compositions[0], merged.compositions[0])
        np.testing.assert_array_equal(legacy.residual_widths, merged.residual_widths)

    def test_cli_default_width_applies_to_all_levels(self):
        with TemporaryDirectory() as directory:
            tokenizer = Path(directory) / "tokenizer.json"
            tokenizer.write_text(self.tokenizer.to_str())
            output = Path(directory) / "hierarchy.npz"
            with redirect_stdout(io.StringIO()) as stdout:
                hierarchy_main([
                    "build", "--tokenizer-file", str(tokenizer),
                    "--level-sizes", "8 12 17", "--output", str(output),
                ])
            snippet = json.loads(stdout.getvalue())
            definition = EmbeddingHierarchy.load(output)
            np.testing.assert_array_equal(definition.residual_widths, np.full(17, 32))
            self.assertEqual(snippet["embedding_hierarchy"]["levels"], definition.config_levels())

    def test_merge_prefix_rejects_invalid_levels_before_expansion(self):
        for sizes in ((8, 20, 17), (8, 8, 17), (8, 4, 17)):
            with self.subTest(sizes=sizes), self.assertRaisesRegex(ValueError, "strictly increase"):
                hierarchy_from_macro_tokenizer(self.tokenizer, sizes)

    def test_save_and_load_round_trip_both_formats(self):
        definition = custom_definition(self.tokenizer)
        definition.input_nodes = None
        with TemporaryDirectory() as directory:
            for suffix in (".npz", ".json"):
                path = Path(directory) / f"hierarchy{suffix}"
                digest = definition.save(path)
                self.assertEqual(len(digest), 64)
                loaded = EmbeddingHierarchy.load(path)
                self.assertEqual(loaded.level_sizes, definition.level_sizes)
                self.assertEqual(loaded.level_names, ("a", "b", "c"))
                for a, b in zip(loaded.compositions, definition.compositions):
                    np.testing.assert_array_equal(a, b)
                np.testing.assert_array_equal(
                    loaded.residual_widths, definition.residual_widths
                )
            with self.assertRaisesRegex(ValueError, ".npz or .json"):
                definition.save(Path(directory) / "hierarchy.txt")

    def test_structural_validation(self):
        good = hierarchy_from_macro_tokenizer(self.tokenizer, LEVELS, 2)
        rows = [r.copy() for r in good.compositions]
        widths = good.residual_widths
        cases = {
            "strictly increase": dict(level_sizes=(8, 8, 17)),
            "one composition per new node": dict(compositions=(rows[0][:-1], rows[1])),
            "children must be nodes of level 1": dict(
                compositions=(rows[0], np.where(rows[1] >= 0, 12, -1))
            ),
            "precede -1 padding": dict(
                compositions=(np.array([[3, -1, 4]] * 4), rows[1])
            ),
            "cannot be negative": dict(residual_widths=-widths),
            "share one node": dict(input_nodes=np.zeros(17)),
            "undefined node": dict(input_nodes=np.arange(1, 18)),
        }
        base = dict(
            level_sizes=good.level_sizes,
            compositions=tuple(rows),
            residual_widths=widths,
        )
        for message, change in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                EmbeddingHierarchy(**(base | change))

    def test_tokenizer_consistency_and_input_mapping(self):
        definition = hierarchy_from_macro_tokenizer(self.tokenizer, LEVELS, 2)
        swapped = definition.compositions[1].copy()
        swapped[0, :2] = swapped[0, 1::-1]
        broken = EmbeddingHierarchy(
            LEVELS, (definition.compositions[0], swapped), definition.residual_widths
        )
        with self.assertRaisesRegex(ValueError, "input token 12"):
            check_tokenizer_hierarchy(broken, self.tokenizer)
        # An extra intermediate node that no input token uses: inputs map around it.
        extra = EmbeddingHierarchy(
            (8, 13, 18),
            (np.vstack([definition.compositions[0], [[3, 3, -1]]]),
             definition.compositions[1] + (definition.compositions[1] >= 12)),
            np.full(18, 2),
            input_nodes=np.r_[np.arange(12), np.arange(13, 18)],
        )
        check_tokenizer_hierarchy(extra, self.tokenizer)
        with self.assertRaisesRegex(ValueError, "input vocabulary"):
            extra.input_nodes = None
            extra.input_node_ids(17)


class HierarchyConfigTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = macro_tokenizer()

    def test_round_trip_and_legacy_configs(self):
        config = tiny_hierarchy_config(self.tokenizer, LEVELS, [2, 3])
        self.assertIsInstance(config.embedding_hierarchy, HierarchyConfig)
        values = json.loads(json.dumps(asdict(config)))
        self.assertEqual(model_config_from_dict(values), config)
        hash(config)
        legacy = asdict(tiny_fraise_config(self.tokenizer))
        legacy.pop("embedding_hierarchy")
        parsed = model_config_from_dict(legacy)
        self.assertIsNone(parsed.embedding_hierarchy)
        # Manifests written before this field existed still resume.
        manifest = {
            "manifest_version": 3,
            "identity": {
                "optimizer": OPTIMIZER_IDENTITY,
                "model_config": legacy,
                "training": asdict(tiny_settings()),
                "dataset": {},
                "tokenizer": {},
            },
        }
        current = json.loads(json.dumps(manifest))
        current["identity"]["model_config"] = asdict(parsed)
        check_resume_identity(manifest, current)
        changed = json.loads(json.dumps(current))
        changed["identity"]["model_config"] = values
        with self.assertRaisesRegex(ValueError, "model_config"):
            check_resume_identity(manifest, changed)

    def test_validation(self):
        config = tiny_hierarchy_config(self.tokenizer, LEVELS, [2, 3])
        hierarchy = asdict(config.embedding_hierarchy)
        cases = {
            "macro_residual_dim to 0": dict(macro_residual_dim=2),
            "requires embedding_hierarchy": dict(embedding_hierarchy=None),
            "level 0 must match": dict(base_vocab_size=7),
            "at least two levels": dict(
                embedding_hierarchy=hierarchy | {"levels": hierarchy["levels"][:1]}
            ),
            "strictly increase": dict(
                embedding_hierarchy=hierarchy
                | {"levels": [hierarchy["levels"][0]] * 2}
            ),
            "needs a path": dict(embedding_hierarchy=hierarchy | {"source": "file"}),
            "sha256": dict(
                embedding_hierarchy=hierarchy | {"source": "file", "path": "x.npz"}
            ),
            "omit residual_parameters": dict(
                embedding_hierarchy=hierarchy
                | {"levels": [*hierarchy["levels"][:2], hierarchy["levels"][2] | {"residual_parameters": 3}]}
            ),
            "unknown level settings": dict(
                embedding_hierarchy=hierarchy
                | {"levels": [*hierarchy["levels"][:2], {"vocab_size": 17, "width": 3}]}
            ),
            "composition must be": dict(
                embedding_hierarchy=hierarchy | {"composition": "mixed"}
            ),
        }
        for message, change in cases.items():
            with self.subTest(message), self.assertRaisesRegex(
                (ValueError, TypeError), message
            ):
                replace(config, **change)

    def test_tokenizer_compatibility_errors(self):
        check = FRAISE.check_tokenizer
        config = tiny_hierarchy_config(self.tokenizer, LEVELS, [2, 3])
        check(config, self.tokenizer)
        tight = tiny_hierarchy_config(
            self.tokenizer, LEVELS, [2, 3], max_composition_length=2
        )
        with self.assertRaisesRegex(ValueError, "raise its max_composition_length"):
            check(tight, self.tokenizer)
        with self.assertRaisesRegex(ValueError, "needs a macro tokenizer"):
            check(config, tiny_tokenizer())


class HierarchicalEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = macro_tokenizer()

    def ids(self):
        encoded = self.tokenizer.encode("a b c d e a b c d e c d e a b").ids
        # Real text plus every top-level node, including atoms carried through.
        return torch.tensor(encoded + list(range(17))).view(1, -1)

    def test_composition_matches_recursive_reference(self):
        for composition in ("per_level", "shared"):
            with self.subTest(composition=composition):
                config = tiny_hierarchy_config(
                    self.tokenizer,
                    LEVELS,
                    [2, 3],
                    hierarchy={"composition": composition},
                )
                model = bound(config, self.tokenizer)
                definition = hierarchy_from_macro_tokenizer(
                    self.tokenizer, LEVELS, [2, 3]
                )
                ids = self.ids()
                with torch.no_grad():
                    actual = model.embed(ids)[0]
                    expected = torch.stack(
                        [
                            reference_embedding(model.embedding, definition, i, 2)
                            for i in ids[0].tolist()
                        ]
                    )
                torch.testing.assert_close(actual, expected)
                self.assertEqual(
                    len(model.embedding.composers),
                    1 if composition == "shared" else 2,
                )

    def test_variable_widths_and_input_mapping_match_reference(self):
        definition = custom_definition(self.tokenizer)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "custom.json"
            config = file_config(
                self.tokenizer, definition, path, definition.save(path)
            )
            model = bound(config, self.tokenizer)
        ids = self.ids()
        with torch.no_grad():
            actual = model.embed(ids)[0]
            expected = torch.stack(
                [
                    reference_embedding(model.embedding, definition, i, 2)
                    for i in ids[0].tolist()
                ]
            )
        torch.testing.assert_close(actual, expected)

    def test_two_levels_reproduce_the_original_macro_embedding(self):
        tokenizer = self.tokenizer
        legacy_config = tiny_fraise_config(tokenizer, macro_residual_dim=3)
        legacy = perturbed(build_model(legacy_config))
        FRAISE.bind_tokenizer(legacy, tokenizer)
        config = tiny_hierarchy_config(tokenizer, (8, 17), [3])
        model = build_model(config)
        FRAISE.bind_tokenizer(model, tokenizer)
        old: MacroEmbedding = legacy.embedding
        new = model.embedding
        composer = new.composers[0]
        with torch.no_grad():
            new.atoms.weight.copy_(old.base_embedding.weight)
            composer.position_gates.copy_(old.position_gates)
            composer.length_embedding.weight.copy_(old.length_embedding.weight)
            composer.residual_projection.weight.copy_(old.residual_projection.weight)
            new.levels[0].values.copy_(old.residual.weight.flatten())
        ids = self.ids()
        with torch.no_grad():
            torch.testing.assert_close(model.embed(ids), legacy.embed(ids))
        self.assertEqual(
            FRAISE.count_parameters(config), FRAISE.count_parameters(legacy_config)
        )

    def test_each_node_has_one_residual_even_when_carried(self):
        config = tiny_hierarchy_config(self.tokenizer, LEVELS, [2, 3])
        model = bound(config, self.tokenizer)
        embedding = model.embedding
        with torch.no_grad():
            for composer in embedding.composers:
                composer.position_gates.fill_(1)
                composer.length_embedding.weight.zero_()
            atom, intermediate = 3, 10  # 10 is introduced at level 1
            level_one = embedding.levels[0]
            projection = embedding.composers[0].residual_projection.weight
            # An atom used at the top level: carried through levels 1 and 2,
            # composed from itself each time, but its residual is added once.
            expected_atom = embedding.atoms.weight[atom] + projection @ (
                level_one.values[2 * atom : 2 * atom + 2]
            )
            torch.testing.assert_close(
                model.embed(torch.tensor([[atom]]))[0, 0], expected_atom
            )
            # A level-1 node carried to level 2 keeps only its level-1 residual.
            definition = hierarchy_from_macro_tokenizer(self.tokenizer, LEVELS, [2, 3])
            torch.testing.assert_close(
                model.embed(torch.tensor([[intermediate]]))[0, 0],
                reference_embedding(embedding, definition, intermediate, 1),
            )
        sizes = [level.values.numel() for level in embedding.levels]
        self.assertEqual(sizes, [12 * 2, 5 * 3])

    def test_deduplicated_batches_match_independent_positions(self):
        model = bound(tiny_hierarchy_config(self.tokenizer, LEVELS, [2, 3]), self.tokenizer)
        ids = self.ids()
        with torch.no_grad():
            batched = model.embed(ids)
            single = torch.cat([model.embed(ids[:, i : i + 1]) for i in range(ids.size(1))], 1)
        torch.testing.assert_close(batched, single)

    def test_gradients_reach_every_level(self):
        config = tiny_hierarchy_config(self.tokenizer, LEVELS, [2, 3])
        model = bound(config, self.tokenizer).double()
        embedding = model.embedding
        definition = hierarchy_from_macro_tokenizer(self.tokenizer, LEVELS, [2, 3])
        # Node 15 is a level-2 macro built from a level-1 node and atoms.
        token = 15
        children = definition.children(token, 2)
        self.assertTrue(any(child >= 8 for child in children))
        embedding(torch.tensor([[token]])).square().sum().backward()
        level_one, level_two = embedding.levels
        used_one = [c for c in children if c >= 0]
        for child in used_one:
            self.assertTrue(level_one.values.grad[2 * child : 2 * child + 2].abs().sum() > 0)
        unused = set(range(12)) - set(used_one) - {
            atom for c in used_one if c >= 8 for atom in definition.children(c, 1)
        }
        for node in unused:
            self.assertEqual(level_one.values.grad[2 * node : 2 * node + 2].abs().sum(), 0)
        offset = 3 * (token - 12)
        self.assertTrue(level_two.values.grad[offset : offset + 3].abs().sum() > 0)
        self.assertEqual(level_two.values.grad.abs().sum(), level_two.values.grad[offset : offset + 3].abs().sum())
        atoms = set(self.tokenizer.decomposition(token))
        for atom in range(8):
            touched = embedding.atoms.weight.grad[atom].abs().sum() > 0
            self.assertEqual(bool(touched), atom in atoms, atom)
        for composer in embedding.composers:
            self.assertTrue(composer.position_gates.grad.abs().sum() > 0)
            self.assertTrue(composer.residual_projection.weight.grad.abs().sum() > 0)

    def test_gradcheck(self):
        config = tiny_hierarchy_config(self.tokenizer, LEVELS, [2, 3])
        embedding = bound(config, self.tokenizer).embedding.double()
        names, tensors = zip(*embedding.named_parameters())
        ids = self.ids()[:, :10]

        def function(*values):
            return torch.func.functional_call(embedding, dict(zip(names, values)), (ids,))

        inputs = tuple(t.detach().clone().requires_grad_() for t in tensors)
        self.assertTrue(torch.autograd.gradcheck(function, inputs))


class HierarchyAccountingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = macro_tokenizer()

    def configs(self, directory):
        definition = custom_definition(self.tokenizer)
        path = Path(directory) / "custom.npz"
        sha256 = definition.save(path)
        return {
            "two-level": tiny_hierarchy_config(self.tokenizer, (8, 17), [3]),
            "three-level": tiny_hierarchy_config(self.tokenizer, LEVELS, [2, 3]),
            "shared": tiny_hierarchy_config(
                self.tokenizer, LEVELS, [2, 3], hierarchy={"composition": "shared"}
            ),
            "zero residuals": tiny_hierarchy_config(self.tokenizer, LEVELS, [0, 0]),
            "variable widths": file_config(self.tokenizer, definition, path, sha256),
        }

    def test_parameter_counts_match_modules(self):
        with TemporaryDirectory() as directory:
            for name, config in self.configs(directory).items():
                with self.subTest(name):
                    model = build_model(config)
                    FRAISE.bind_tokenizer(model, self.tokenizer)
                    actual = sum(p.numel() for p in model.parameters())
                    self.assertEqual(FRAISE.count_parameters(config), actual)
                    breakdown = FRAISE.parameter_breakdown(config)
                    self.assertEqual(sum(breakdown.values()), actual)
                    report = hierarchy_report(config)
                    residuals = sum(
                        level.values.numel()
                        for level in model.embedding.levels
                        if level.values is not None
                    )
                    self.assertEqual(
                        sum(r["residual_parameters"] for r in report), residuals
                    )
                    self.assertEqual(
                        sum(v for k, v in breakdown.items() if k.endswith("_residuals")),
                        residuals,
                    )
                    self.assertEqual(report[0]["residual_parameters"], 0)

    def test_variable_width_totals_must_match_the_config(self):
        definition = custom_definition(self.tokenizer)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "custom.npz"
            sha256 = definition.save(path)
            config = file_config(self.tokenizer, definition, path, sha256)
            levels = [dict(level) for level in definition.config_levels()]
            levels[2]["residual_parameters"] += 1
            wrong = file_config(
                self.tokenizer, definition, path, sha256, levels=levels
            )
            with self.assertRaisesRegex(ValueError, "configuration allocates"):
                FRAISE.check_tokenizer(wrong, self.tokenizer)
            FRAISE.check_tokenizer(config, self.tokenizer)
            stale = file_config(self.tokenizer, definition, path, "0" * 64)
            with self.assertRaisesRegex(ValueError, "expects"):
                FRAISE.check_tokenizer(stale, self.tokenizer)

    def test_optimizer_decays_packed_residuals_once(self):
        config = tiny_hierarchy_config(self.tokenizer, LEVELS, [2, 3])
        model = build_model(config)
        optimizer = MuonWithAdamW(
            model, tiny_settings(), layout=FRAISE.optimizer_layout(model)
        )
        grouped = [p for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual(set(grouped), set(model.parameters()))
        decayed = {
            p
            for group in optimizer.param_groups
            if group["optimizer"] == "adamw"
            for p in group["params"]
        }
        for level in model.embedding.levels:
            self.assertIn(level.values, decayed)

    def test_state_dict_is_self_contained(self):
        definition = custom_definition(self.tokenizer)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "custom.npz"
            config = file_config(self.tokenizer, definition, path, definition.save(path))
            model = bound(config, self.tokenizer)
            weights = Path(directory) / "weights.pt"
            torch.save(model.state_dict(), weights)
            path.unlink()  # The weights alone must carry the structure.
            restored = build_model(config)
            restored.load_state_dict(torch.load(weights, weights_only=True))
        ids = torch.arange(17).view(1, -1)
        with torch.no_grad():
            torch.testing.assert_close(restored.embed(ids), model.embed(ids))
        recovered = definition_from_weights(config, restored.state_dict())
        for a, b in zip(recovered.compositions, definition.compositions):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(
            recovered.residual_widths, definition.residual_widths
        )
        self.assertIsNone(recovered.input_nodes)


class HierarchyVisualizationTests(unittest.TestCase):
    def test_analysis_separates_segmentation_from_composition(self):
        tokenizer = macro_tokenizer()
        definition = hierarchy_from_macro_tokenizer(tokenizer, LEVELS, [2, 3])
        text = "a b c d e a b c"
        analysis = analyze_text(text, tokenizer, definition, name="three")
        input_ids = tokenizer.encode(text, add_special_tokens=False).ids
        self.assertEqual([t["id"] for t in analysis["input_tokens"]], input_ids)
        base = tokenizer.base.encode(text, add_special_tokens=False).ids
        self.assertEqual([t["id"] for t in analysis["base_tokens"]], base)
        rows = analysis["rows"]
        self.assertEqual(len(rows), 3)
        self.assertEqual([cell["node"] for cell in rows[0]], input_ids)
        self.assertEqual([cell["node"] for cell in rows[-1]], base)
        for row in rows:  # every level tiles the same base tokens
            self.assertEqual(
                [(c["base_start"], c["base_end"]) for c in row][0][0], 0
            )
            self.assertEqual(row[-1]["base_end"], len(base))
        self.assertEqual(len(analysis["trees"]), len(input_ids))
        self.assertIn("node", render_text_tree(analysis))
        flat = analyze_text(text, tokenizer, None)
        self.assertEqual(flat["rows"], [])


if __name__ == "__main__":
    unittest.main()
