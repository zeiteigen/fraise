import math
import sys
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch

import torch


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

MHA_PATH = Path(__file__).resolve().parents[1] / "src" / "components" / "mha.py"
MHA_SPEC = spec_from_file_location("fraise_mha_under_test", MHA_PATH)
MHA_MODULE = module_from_spec(MHA_SPEC)
MHA_SPEC.loader.exec_module(MHA_MODULE)
MultiHeadAttention = MHA_MODULE.MultiHeadAttention
RotaryEmbedding = MHA_MODULE.RotaryEmbedding
from components.transformer import Transformer


def dense_reference(module, x):
    batch_size, seq_len, _ = x.shape
    q, k, v = module.qkv_projection(x).split(
        (module.q_size, module.k_size, module.v_size),
        dim=-1,
    )
    q = q.reshape(
        batch_size,
        seq_len,
        module.num_heads,
        module.qk_dim,
    ).transpose(1, 2)
    k = k.reshape(
        batch_size,
        seq_len,
        module.num_heads,
        module.qk_dim,
    ).transpose(1, 2)
    v = v.reshape(
        batch_size,
        seq_len,
        module.num_heads,
        module.head_dim,
    ).transpose(1, 2)
    cos, sin = module.rope(
        seq_len,
        device=x.device,
        dtype=x.dtype,
    )
    q, k = module.rope.apply_rotary_qk(q, k, cos, sin)

    scores = q @ k.transpose(-2, -1) / math.sqrt(module.qk_dim)
    positions = torch.arange(seq_len, device=x.device)
    causal = positions[None, :] <= positions[:, None]
    scores = scores.masked_fill(~causal[None, None, :, :], torch.finfo(scores.dtype).min)
    attention = scores.softmax(dim=-1) @ v
    attention = attention.transpose(1, 2).reshape(
        batch_size,
        seq_len,
        module.v_size,
    )
    return module.output_projection(attention)


class MultiHeadAttentionTests(unittest.TestCase):
    def make_module(self, **kwargs):
        options = {
            "io_dim": 8,
            "head_dim": 4,
            "qk_dim": 4,
            "num_heads": 2,
        }
        options.update(kwargs)
        return MultiHeadAttention(**options).double()

    def test_causal_attention_matches_dense_reference(self):
        module = self.make_module(bias=True)
        x = torch.randn(2, 5, 8, dtype=torch.float64)

        torch.testing.assert_close(module(x), dense_reference(module, x))

    def test_right_padding_cannot_change_valid_prefix_outputs(self):
        module = self.make_module(bias=True)
        valid = torch.randn(2, 3, 8, dtype=torch.float64)
        padded = torch.cat((valid, torch.randn(2, 4, 8, dtype=torch.float64)), dim=1)

        torch.testing.assert_close(module(padded)[:, :3], module(valid))

    def test_asymmetric_qk_and_value_dims_match_dense_reference(self):
        x = torch.randn(2, 5, 8, dtype=torch.float64)
        for qk_dim, head_dim in ((4, 5), (6, 3)):
            with self.subTest(qk_dim=qk_dim, head_dim=head_dim):
                module = self.make_module(qk_dim=qk_dim, head_dim=head_dim)
                actual = module(x)
                self.assertEqual(actual.shape, x.shape)
                torch.testing.assert_close(actual, dense_reference(module, x))

    def test_rope_requires_even_dimension(self):
        with self.assertRaisesRegex(ValueError, "even"):
            RotaryEmbedding(3)
        with self.assertRaisesRegex(ValueError, "even"):
            self.make_module(qk_dim=3)

    def test_attention_does_not_enforce_a_context_length(self):
        module = self.make_module()
        x = torch.randn(1, 12, 8, dtype=torch.float64)

        self.assertEqual(module(x).shape, x.shape)

    def test_explicit_rope_table_matches_standalone_path(self):
        module = self.make_module()
        x = torch.randn(2, 5, 8, dtype=torch.float64)
        cos, sin = module.rope(
            x.size(1),
            device=x.device,
            dtype=x.dtype,
        )

        torch.testing.assert_close(module(x), module(x, cos, sin))

    def test_transformer_shares_one_rope_table_across_layers(self):
        transformer = Transformer(8, 4, 4, 2, 16, 3).double()
        x = torch.randn(2, 5, 8, dtype=torch.float64)

        for layer in transformer.layers:
            self.assertIs(layer.mha.rope, transformer.rope)

        with patch.object(
            transformer.rope,
            "forward",
            wraps=transformer.rope.forward,
        ) as rope_forward:
            transformer(x)

        self.assertEqual(rope_forward.call_count, 1)

    def test_eval_disables_attention_dropout(self):
        dropped = self.make_module(attn_dropout=0.7)
        reference = self.make_module(attn_dropout=0.0)
        reference.load_state_dict(dropped.state_dict())
        dropped.eval()
        reference.eval()
        x = torch.randn(2, 5, 8, dtype=torch.float64)

        torch.testing.assert_close(dropped(x), reference(x))

    def test_attention_dropout_is_only_enabled_during_training(self):
        module = self.make_module(attn_dropout=0.25)
        x = torch.randn(2, 5, 8, dtype=torch.float64)
        original_sdpa = MHA_MODULE.F.scaled_dot_product_attention
        with patch.object(
            MHA_MODULE.F,
            "scaled_dot_product_attention",
            wraps=original_sdpa,
        ) as sdpa:
            module.train()
            module(x)
            self.assertEqual(sdpa.call_args.kwargs["dropout_p"], 0.25)
            self.assertTrue(sdpa.call_args.kwargs["is_causal"])

            module.eval()
            module(x)
            self.assertEqual(sdpa.call_args.kwargs["dropout_p"], 0.0)
            self.assertTrue(sdpa.call_args.kwargs["is_causal"])

    def test_fullgraph_compile(self):
        if not hasattr(torch, "compile"):
            self.skipTest("torch.compile is unavailable")

        module = self.make_module()
        x = torch.randn(2, 5, 8, dtype=torch.float64)
        compiled = torch.compile(module, fullgraph=True, backend="eager")

        torch.testing.assert_close(compiled(x), module(x))


if __name__ == "__main__":
    unittest.main()
