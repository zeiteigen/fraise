import unittest
from dataclasses import asdict, replace

import torch
import torch.nn.functional as F
from fixtures import tiny_settings, tiny_tokenizer
from macro_fixtures import tiny_fraise_config, tiny_macro_tokenizer

from fraise.architectures import build_model, get_architecture, model_config_from_dict
from fraise.config.compatibility import check_compatibility
from fraise.config.hierarchical import FraiseConfig
from fraise.inference.hierarchical import iter_generate_base_tokens, macro_trie
from fraise.inference.scoring import score_text, top_next_tokens
from fraise.inference.streaming import iter_generate
from fraise.training.batch import token_loss_sum
from fraise.training.optimizer import MuonWithAdamW

FRAISE = get_architecture("fraise")


def bound_model(tokenizer, **changes):
    torch.manual_seed(5)
    model = build_model(tiny_fraise_config(tokenizer, **changes))
    FRAISE.bind_tokenizer(model, tokenizer)
    # Residual projections start at zero; perturb them so tests exercise them.
    for parameter in model.parameters():
        if not parameter.abs().sum():
            torch.nn.init.normal_(parameter, std=0.1)
    return model.eval()


def sequential_macro_nll(model, hidden, macro_id):
    """Decode one macro slot by slot exactly as generation does."""
    parts = model.decompositions[macro_id, : model.macro_lengths[macro_id]].tolist()
    state = model.local.context(hidden)
    total = 0.0
    for slot, target in enumerate([*parts, model.end_id]):
        logits = model.local.logits(state[None], torch.tensor([slot]))[0]
        total -= F.log_softmax(logits, -1)[target].item()
        if target != model.end_id:
            state = state + model.local.prefix_term(torch.tensor(target), slot)
    return total


class FraiseModelTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = tiny_macro_tokenizer()

    def test_parameter_count_and_breakdown_match_the_module(self):
        for changes in (
            {},
            {"macro_residual_dim": 0},
            {"macro_embedding": "independent", "macro_residual_dim": 0},
            {"local_dim": 6, "local_num_layers": 2},
        ):
            with self.subTest(**changes):
                config = tiny_fraise_config(self.tokenizer, **changes)
                model = build_model(config)
                actual = sum(parameter.numel() for parameter in model.parameters())
                self.assertEqual(FRAISE.count_parameters(config), actual)
                self.assertEqual(
                    sum(FRAISE.parameter_breakdown(config).values()), actual
                )

    def test_config_validation_and_registry(self):
        config = tiny_fraise_config(self.tokenizer)
        self.assertEqual(model_config_from_dict(asdict(config)), config)
        example = model_config_from_dict(dict(FRAISE.example_config))
        self.assertIsInstance(example, FraiseConfig)
        self.assertEqual((example.vocab_size, example.base_vocab_size), (65536, 8192))
        with self.assertRaisesRegex(ValueError, "macro_residual_dim to 0"):
            replace(config, macro_embedding="independent")
        with self.assertRaisesRegex(ValueError, "macro_embedding"):
            replace(config, macro_embedding="mean")
        with self.assertRaisesRegex(ValueError, "base_vocab_size"):
            replace(config, base_vocab_size=config.vocab_size + 1)

    def test_tokenizer_compatibility(self):
        config = tiny_fraise_config(self.tokenizer)
        settings = tiny_settings()
        check_compatibility(self.tokenizer, config, settings)
        with self.assertRaisesRegex(ValueError, "needs a macro tokenizer"):
            FRAISE.check_tokenizer(config, tiny_tokenizer())
        # A base tokenizer works when every macro is a base token.
        FRAISE.check_tokenizer(replace(config, vocab_size=8), tiny_tokenizer())
        with self.assertRaisesRegex(ValueError, "raise max_macro_length"):
            FRAISE.check_tokenizer(replace(config, max_macro_length=1), self.tokenizer)
        with self.assertRaisesRegex(ValueError, "base vocabulary"):
            FRAISE.check_tokenizer(replace(config, base_vocab_size=7), self.tokenizer)

    def test_compositional_inputs_start_from_constituents(self):
        torch.manual_seed(1)
        model = build_model(tiny_fraise_config(self.tokenizer))
        FRAISE.bind_tokenizer(model, self.tokenizer)
        embedding = model.embedding
        macro_id = self.tokenizer.get_vocab_size() - 1
        parts = list(self.tokenizer.decomposition(macro_id))
        expected = (
            sum(
                embedding.base_embedding.weight[part] * embedding.position_gates[slot]
                for slot, part in enumerate(parts)
            )
            / len(parts) ** 0.5
        )
        actual = embedding(
            torch.tensor([macro_id]), model.decompositions, model.macro_lengths
        )[0]
        torch.testing.assert_close(actual, expected)

    def test_training_loss_equals_sequential_local_decoding(self):
        model = bound_model(self.tokenizer)
        encoded = [
            self.tokenizer.encode(text).ids
            for text in ("a b c d e a b c", "c d e c d e a b")
        ]
        width = min(map(len, encoded))
        ids = torch.tensor([sequence[:width] for sequence in encoded])
        inputs, targets = ids[:, :-1], ids[:, 1:]
        mask = torch.ones_like(targets, dtype=torch.bool)
        mask[1, 3:] = False
        with torch.no_grad():
            hidden = model(inputs)
            expected = sum(
                sequential_macro_nll(model, hidden[row, column], targets[row, column])
                for row, column in mask.nonzero().tolist()
            )
            actual = token_loss_sum(model, inputs, targets, mask)
        self.assertAlmostEqual(actual.item(), expected, places=4)
        self.assertTrue(any(self.tokenizer.unit_lengths[targets[mask]] > 1))

    def test_cached_hidden_states_match_full_recomputation(self):
        model = bound_model(self.tokenizer)
        ids = torch.tensor([self.tokenizer.encode("a b c d e a b").ids[:7]])
        with torch.no_grad():
            full = model(ids)
            cache = model.new_cache()
            first = model(ids[:, :4], cache=cache)
            rest = model(ids[:, 4:], cache=cache, last_only=True)
        torch.testing.assert_close(first, full[:, :4])
        torch.testing.assert_close(rest, full[:, -1:])

    def test_optimizer_layout_covers_each_parameter_once(self):
        model = build_model(tiny_fraise_config(self.tokenizer))
        layout = FRAISE.optimizer_layout(model)
        optimizer = MuonWithAdamW(model, tiny_settings(), layout=layout)
        grouped = [p for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual(set(grouped), set(model.parameters()))
        self.assertIn(model.local.context_projection.weight, layout.muon_parameters)
        self.assertNotIn(model.local.head.weight, layout.muon_parameters)
        self.assertNotIn(model.embedding.residual.weight, layout.muon_parameters)

    def test_generation_walks_the_trie_and_feeds_complete_macros(self):
        model = bound_model(self.tokenizer)
        tokenizer = self.tokenizer
        for temperature in (0.0, 1.0):
            with self.subTest(temperature=temperature):
                generator = torch.Generator().manual_seed(2)
                steps = list(
                    iter_generate_base_tokens(
                        model,
                        tokenizer,
                        "a b",
                        max_new_tokens=12,
                        temperature=temperature,
                        generator=generator,
                    )
                )
                final = steps[-1]
                self.assertIn(final.finished, ("length", "eos"))
                self.assertEqual(
                    [step.generated_tokens for step in steps],
                    list(range(1, len(steps) + 1)),
                )
                # Every completed macro is a real macro and they spell the output.
                completed = tokenizer.expand(final.macro_ids[1:])
                self.assertEqual(final.base_ids[1 : 1 + len(completed)], completed)
                pending = final.base_ids[1 + len(completed) :]
                trie, node = macro_trie(model, (1, 0)), 0
                for base_id in pending:
                    node = trie.children[node][base_id]
                self.assertNotIn(1, final.base_ids[1:])
                self.assertNotIn(0, final.base_ids)

    def test_prompt_healing_reopens_the_final_prompt_macro(self):
        model = bound_model(self.tokenizer)
        prompt_macros = self.tokenizer.encode("a b c", add_special_tokens=False).ids
        healed = list(self.tokenizer.decomposition(prompt_macros[-1]))
        first = next(
            iter_generate_base_tokens(model, self.tokenizer, "a b c", max_new_tokens=8)
        )
        prompt_base = self.tokenizer.base.encode("a b c", add_special_tokens=False).ids
        self.assertEqual(first.base_ids[1 : 1 + len(prompt_base)], prompt_base)
        self.assertEqual(first.macro_ids[1:], prompt_macros[:-1])
        final = list(
            iter_generate_base_tokens(model, self.tokenizer, "a b c", max_new_tokens=8)
        )[-1]
        if len(final.macro_ids) > len(prompt_macros):
            reopened = self.tokenizer.decomposition(final.macro_ids[len(prompt_macros)])
            self.assertEqual(list(reopened[: len(healed)]), healed)

    def test_streaming_scoring_and_next_tokens_use_the_hierarchical_path(self):
        model = bound_model(self.tokenizer)
        steps = list(iter_generate(model, self.tokenizer, "a b", max_new_tokens=5))
        self.assertEqual(steps[-1].generated_tokens, len(steps))
        self.assertEqual(
            steps[-1].text,
            self.tokenizer.decode(steps[-1].ids[1:], skip_special_tokens=True),
        )
        score = score_text(model, self.tokenizer, "a b c d")
        self.assertGreater(score["loss"], 0)
        self.assertGreater(score["bits_per_byte"], 0)
        tokens = top_next_tokens(model, self.tokenizer, "a b", count=3)
        self.assertEqual(len(tokens), 3)
        self.assertTrue(all(token_id < 8 for token_id, _, _ in tokens))
        self.assertLessEqual(sum(probability for _, probability, _ in tokens), 1.0001)


if __name__ == "__main__":
    unittest.main()
