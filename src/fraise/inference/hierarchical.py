"""Generate base tokens from a hierarchical model by walking a macro trie.

The global transformer reads macro IDs. For each next macro, the local decoder
emits base tokens; every choice is restricted to children of the current trie
node, plus END where the path already spells a complete macro. On END (or at a
leaf) the completed macro is appended to the global context.

The final prompt macro is re-decided ("healed"): it is removed from the global
context and its base tokens are forced as the first trie path, so a prompt that
ends mid-word can still continue into a longer macro.
"""

from dataclasses import dataclass

import numpy as np
import torch

from fraise.tokenizer.loading import special_token_id

from .generation import IncrementalDecoder, sample_token, validate_sampling


class MacroTrie:
    """Prefix tree over macro decompositions; node 0 is the root."""

    def __init__(self, decompositions: np.ndarray, lengths: np.ndarray, *, excluded=()):
        self.children: list[dict[int, int]] = [{}]
        self.macro: list[int | None] = [None]
        for macro_id, (parts, length) in enumerate(zip(decompositions, lengths)):
            node = 0
            for base_id in parts[:length].tolist():
                child = self.children[node].get(base_id)
                if child is None:
                    child = len(self.children)
                    self.children[node][base_id] = child
                    self.children.append({})
                    self.macro.append(None)
                node = child
            self.macro[node] = macro_id
        for base_id in excluded:
            self.children[0].pop(base_id, None)
        self._options: dict[tuple[int, torch.device], torch.Tensor] = {}

    def options(self, node: int, end_id: int, device) -> torch.Tensor:
        """Allowed local outputs at ``node``: child base IDs, then END if complete."""
        key = (node, torch.device(device))
        if key not in self._options:
            ids = sorted(self.children[node])
            if self.macro[node] is not None:
                ids.append(end_id)
            self._options[key] = torch.tensor(ids, dtype=torch.long, device=device)
        return self._options[key]


def macro_trie(model, excluded=()) -> MacroTrie:
    """The model's trie, rebuilt only when its decomposition buffers change."""
    key = (
        tuple(excluded),
        model.decompositions.data_ptr(),
        model.decompositions._version,
        model.macro_lengths._version,
    )
    cached = getattr(model, "_fraise_trie", None)
    if cached is None or cached[0] != key:
        trie = MacroTrie(
            model.decompositions.cpu().numpy(),
            model.macro_lengths.cpu().numpy(),
            excluded=excluded,
        )
        model._fraise_trie = cached = (key, trie)
    return cached[1]


@dataclass
class HierarchicalStep:
    """``macro_ids`` are the completed macros fed to the global transformer."""

    base_ids: list[int]
    generated_tokens: int
    macro_ids: list[int]
    global_steps: int
    finished: str | None


class HierarchicalSampler:
    """Global macro context plus trie-constrained local decoding."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.device = next(model.parameters()).device
        bos = special_token_id(tokenizer, "<|bos|>")
        pad = special_token_id(tokenizer, "<|pad|>")
        self.trie = macro_trie(model, (bos, pad))
        self.global_decoder = IncrementalDecoder(model)
        self.global_steps = 0
        self._parts = model.decompositions.cpu().numpy()
        self._lengths = model.macro_lengths.cpu().numpy()

    def decomposition(self, macro_id) -> list[int]:
        return self._parts[macro_id, : self._lengths[macro_id]].tolist()

    def expand(self, macro_ids) -> list[int]:
        return [
            base_id
            for macro_id in macro_ids
            for base_id in self.decomposition(macro_id)
        ]

    def context(self, macro_ids) -> torch.Tensor:
        self.global_steps += 1
        hidden = self.global_decoder.logits(macro_ids)
        return self.model.local.context(hidden)

    def local_logits(self, state, slot) -> torch.Tensor:
        slots = torch.tensor([slot], device=self.device)
        return self.model.local.logits(state[None], slots)[0]

    def prefix_term(self, base_id, slot) -> torch.Tensor:
        return self.model.local.prefix_term(
            torch.tensor(base_id, device=self.device), slot
        )


def iter_generate_base_tokens(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens=48,
    temperature=0.0,
    top_k=None,
    top_p=0.95,
    generator=None,
    cancelled=lambda: False,
):
    """Yield the base-token sequence after each emitted base token.

    ``base_ids`` starts with BOS and the prompt's base tokens; the first yielded
    step also reports how many of them form the prompt via ``generated_tokens``.
    """
    validate_sampling(max_new_tokens, temperature, top_k, top_p)
    sampler = HierarchicalSampler(model, tokenizer)
    trie, end_id = sampler.trie, model.end_id
    bos = special_token_id(tokenizer, "<|bos|>")
    eos = special_token_id(tokenizer, "<|eos|>")
    macros = [bos, *tokenizer.encode(prompt, add_special_tokens=False).ids]
    forced = sampler.decomposition(macros.pop()) if len(macros) > 1 else []
    base_ids = [bos, *sampler.expand(macros[1:]), *forced]
    generated = 0

    def step(finished=None):
        return HierarchicalStep(
            base_ids.copy(), generated, macros.copy(), sampler.global_steps, finished
        )

    if max_new_tokens == 0:
        yield step("length")
        return
    with torch.inference_mode():
        while True:
            state = sampler.context(macros)
            node, slot = 0, 0
            for base_id in forced:
                state = state + sampler.prefix_term(base_id, slot)
                node, slot = trie.children[node][base_id], slot + 1
            forced = []
            while True:
                if cancelled():
                    yield step("cancelled")
                    return
                options = trie.options(node, end_id, sampler.device)
                if len(options) == 1:
                    choice = int(options[0])
                else:
                    logits = sampler.local_logits(state, slot)[options].float()
                    index = sample_token(
                        logits, temperature, top_k, top_p, generator=generator
                    )
                    choice = int(options[index])
                if choice == end_id:
                    break
                base_ids.append(choice)
                generated += 1
                node = trie.children[node][choice]
                if choice == eos:
                    yield step("eos")
                    return
                if generated == max_new_tokens:
                    yield step("length")
                    return
                yield step()
                state = state + sampler.prefix_term(choice, slot)
                slot += 1
            macros.append(trie.macro[node])


def next_base_distribution(model, tokenizer, text: str):
    """Probabilities of the next macro's first base token (trie root)."""
    sampler = HierarchicalSampler(model, tokenizer)
    bos = special_token_id(tokenizer, "<|bos|>")
    macros = [bos, *tokenizer.encode(text, add_special_tokens=False).ids]
    with torch.inference_mode():
        options = sampler.trie.options(0, model.end_id, sampler.device)
        logits = sampler.local_logits(sampler.context(macros), 0)[options].float()
    return options, logits.softmax(-1)
