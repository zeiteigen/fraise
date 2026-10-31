"""Streaming snapshots retain generation behavior and decode cumulative bytes."""

import unittest

import torch
from fixtures import tiny_config, tiny_tokenizer
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

from fraise.inference.generation import (
    generate,
    logits_for_context,
    prompt_ids,
    sample_token,
)
from fraise.inference.streaming import iter_generate
from fraise.model.model import FraiseModel
from fraise.tokenizer.loading import special_token_id


class ScriptedModel(torch.nn.Module):
    """An uncached language model with a prescribed continuation."""

    def __init__(self, tokens, vocab_size=8, max_seq_len=4):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.tokens = tokens
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.contexts = []

    def new_cache(self):
        return None

    def forward(self, input_ids, *, cache=None, last_only=False):
        if cache is not None or input_ids.size(1) > self.max_seq_len:
            raise AssertionError("invalid uncached context")
        token = self.tokens[len(self.contexts)]
        self.contexts.append(input_ids.tolist()[0])
        positions = 1 if last_only else input_ids.size(1)
        logits = torch.full((1, positions, self.vocab_size), -100.0)
        logits[:, :, token] = 100.0
        return logits


def reference_generate(
    model, tokenizer, prompt, *, max_new_tokens, generator=None, **sampling
):
    """Independent non-streaming loop recomputes the full visible context."""
    ids = prompt_ids(tokenizer, prompt)
    eos = special_token_id(tokenizer, "<|eos|>")
    for _ in range(max_new_tokens):
        token = sample_token(
            logits_for_context(model, ids), generator=generator, **sampling
        )
        ids.append(token)
        if token == eos:
            break
    return tokenizer.decode(ids[1:], skip_special_tokens=True), ids


def byte_tokenizer():
    tokens = ["<|pad|>", "<|bos|>", "<|eos|>", *sorted(ByteLevel.alphabet())]
    tokenizer = Tokenizer(
        BPE(vocab={token: index for index, token in enumerate(tokens)}, merges=[])
    )
    tokenizer.add_special_tokens(tokens[:3])
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    return tokenizer


class StreamingGenerationTests(unittest.TestCase):
    def test_greedy_and_seeded_sampling_match_nonstreaming_full_context(self):
        tokenizer = tiny_tokenizer()
        torch.manual_seed(123)
        model = FraiseModel(tiny_config()).double().eval()
        with torch.no_grad():
            model.lm_head.bias[2] = -100  # Exercise sliding context before EOS.
        modes = (
            {"temperature": 0.0, "top_k": None, "top_p": 0.95},
            {"temperature": 0.8, "top_k": 4, "top_p": 0.85},
            {"temperature": 1.0, "top_k": None, "top_p": None},
        )
        for sampling in modes:
            with self.subTest(sampling=sampling):
                expected = reference_generate(
                    model,
                    tokenizer,
                    "a b",
                    max_new_tokens=8,
                    generator=torch.Generator().manual_seed(29),
                    **sampling,
                )
                steps = list(
                    iter_generate(
                        model,
                        tokenizer,
                        "a b",
                        max_new_tokens=8,
                        generator=torch.Generator().manual_seed(29),
                        **sampling,
                    )
                )
                actual = generate(
                    model,
                    tokenizer,
                    "a b",
                    max_new_tokens=8,
                    generator=torch.Generator().manual_seed(29),
                    **sampling,
                )
                self.assertEqual(actual, expected)
                self.assertEqual((steps[-1].text, steps[-1].ids), expected)
                self.assertEqual(len(steps), 8)
                self.assertEqual(steps[-1].finish_reason, "length")
                self.assertGreater(len(steps[-1].ids), model.max_seq_len)
                for index, step in enumerate(steps, start=1):
                    self.assertEqual(step.generated_tokens, index)
                    self.assertEqual(len(step.ids), 3 + index)
                    self.assertEqual(
                        step.text,
                        tokenizer.decode(step.ids[1:], skip_special_tokens=True),
                    )
                    self.assertEqual(
                        step.completion,
                        tokenizer.decode(step.ids[3:], skip_special_tokens=True),
                    )
                self.assertIsNot(steps[0].ids, steps[1].ids)

    def test_zero_tokens_returns_the_prompt_without_a_forward_pass(self):
        model = ScriptedModel([])
        tokenizer = tiny_tokenizer()
        steps = list(iter_generate(model, tokenizer, "a", max_new_tokens=0))
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].ids, [1, 3])
        self.assertEqual(steps[0].text, "a")
        self.assertEqual(steps[0].completion, "")
        self.assertEqual(steps[0].generated_tokens, 0)
        self.assertEqual(steps[0].finish_reason, "length")
        self.assertEqual(model.contexts, [])
        self.assertEqual(
            generate(model, tokenizer, "a", max_new_tokens=0), ("a", [1, 3])
        )

    def test_eos_finishes_without_sampling_another_token(self):
        model = ScriptedModel([3, 2, 4])
        tokenizer = tiny_tokenizer()
        tokenizer.add_special_tokens(["<|pad|>", "<|bos|>", "<|eos|>"])
        steps = list(iter_generate(model, tokenizer, "b", max_new_tokens=8))
        self.assertEqual(len(steps), 2)
        self.assertIsNone(steps[0].finish_reason)
        self.assertEqual(steps[-1].finish_reason, "eos")
        self.assertEqual(steps[-1].ids, [1, 4, 3, 2])
        self.assertEqual(steps[-1].completion, "a")
        self.assertEqual(len(model.contexts), 2)

    def test_uncached_model_receives_only_the_visible_sliding_window(self):
        model = ScriptedModel([5, 6, 7, 3], max_seq_len=2)
        steps = list(iter_generate(model, tiny_tokenizer(), "a b", max_new_tokens=4))
        self.assertEqual(model.contexts, [[3, 4], [4, 5], [5, 6], [6, 7]])
        self.assertEqual(steps[-1].ids, [1, 3, 4, 5, 6, 7, 3])
        self.assertEqual(steps[-1].finish_reason, "length")

    def test_cancellation_before_and_between_tokens_skips_further_model_work(self):
        for generated_before_stop in (0, 2):
            with self.subTest(generated_before_stop=generated_before_stop):
                model = ScriptedModel([3, 4, 5])
                cancelled = [generated_before_stop == 0]
                stream = iter_generate(
                    model,
                    tiny_tokenizer(),
                    "a",
                    max_new_tokens=5,
                    cancelled=lambda cancelled=cancelled: cancelled[0],
                )
                previous = [next(stream) for _ in range(generated_before_stop)]
                cancelled[0] = True
                final = next(stream)
                self.assertEqual(final.finish_reason, "cancelled")
                self.assertEqual(final.generated_tokens, generated_before_stop)
                self.assertEqual(len(model.contexts), generated_before_stop)
                if previous:
                    self.assertEqual(final.ids, previous[-1].ids)
                    self.assertEqual(len(previous[0].ids), 3)
                with self.assertRaises(StopIteration):
                    next(stream)

    def test_cumulative_byte_decoding_repairs_incomplete_multibyte_sequences(self):
        tokenizer = byte_tokenizer()
        continuation = "é🙂"
        tokens = tokenizer.encode(continuation, add_special_tokens=False).ids
        self.assertEqual(len(tokens), len(continuation.encode("utf-8")))
        model = ScriptedModel([*tokens, 2], vocab_size=tokenizer.get_vocab_size())
        steps = list(
            iter_generate(model, tokenizer, "a", max_new_tokens=len(tokens) + 1)
        )
        self.assertTrue(any("�" in step.completion for step in steps[:-1]))
        self.assertEqual(steps[-1].completion, continuation)
        self.assertEqual(steps[-1].text, "a" + continuation)
        self.assertEqual(steps[-1].finish_reason, "eos")
        self.assertNotIn("�", steps[-1].text)
        self.assertEqual(
            steps[-1].text,
            tokenizer.decode(steps[-1].ids[1:], skip_special_tokens=True),
        )
