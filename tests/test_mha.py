import math
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch

import torch

MHA_PATH = Path(__file__).resolve().parents[1] / "src" / "components" / "mha.py"
MHA_SPEC = spec_from_file_location("fraise_mha_under_test", MHA_PATH)
MHA_MODULE = module_from_spec(MHA_SPEC)
MHA_SPEC.loader.exec_module(MHA_MODULE)
MultiHeadAttention = MHA_MODULE.MultiHeadAttention
validate_right_padding_mask = MHA_MODULE.validate_right_padding_mask


def dense_reference(module, x, padding_mask=None):
    batch_size, seq_len, _ = x.shape
    q, k, v = module._project_qkv(x)
    q = q.reshape(
        batch_size, seq_len, module.num_heads, module.qk_dim
    ).transpose(1, 2)
    k = k.reshape(
        batch_size, seq_len, module.num_heads, module.qk_dim
    ).transpose(1, 2)
    v = v.reshape(
        batch_size, seq_len, module.num_heads, module.head_dim
    ).transpose(1, 2)

    rotary = module.rotary_embedding
    if rotary.rotary_dim:
        compute_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        positions = torch.arange(seq_len, device=x.device, dtype=compute_dtype)
        frequency_indices = torch.arange(
            0,
            rotary.rotary_dim,
            2,
            device=x.device,
            dtype=compute_dtype,
        )
        inverse_frequencies = torch.exp(
            -frequency_indices * (math.log(rotary.base) / rotary.rotary_dim)
        )
        angles = positions[:, None] * inverse_frequencies[None, :]
        cos = angles.cos()[None, None, :, :]
        sin = angles.sin()[None, None, :, :]

        def rotate(tensor):
            values = tensor[..., : rotary.rotary_dim].to(compute_dtype)
            pairs = values.reshape(
                batch_size,
                module.num_heads,
                seq_len,
                rotary.rotary_dim // 2,
                2,
            )
            even, odd = pairs[..., 0], pairs[..., 1]
            rotated = torch.stack(
                (even * cos - odd * sin, even * sin + odd * cos),
                dim=-1,
            ).flatten(-2).to(tensor.dtype)
            if rotary.rotary_dim == rotary.dim:
                return rotated
            return torch.cat(
                (rotated, tensor[..., rotary.rotary_dim :]),
                dim=-1,
            )

        q, k = rotate(q), rotate(k)

    scores = q @ k.transpose(-2, -1) / math.sqrt(module.qk_dim)
    positions = torch.arange(seq_len, device=x.device)
    allowed = positions[None, :] <= positions[:, None]
    allowed = allowed[None, None, :, :]
    if padding_mask is not None:
        allowed = allowed & ~padding_mask[:, None, None, :]

    scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    attention = scores.softmax(dim=-1) @ v
    output = attention.transpose(1, 2).reshape(
        batch_size,
        seq_len,
        module.num_heads * module.head_dim,
    )
    output = module.output_projection(output)
    if padding_mask is not None:
        output = output.masked_fill(padding_mask.unsqueeze(-1), 0.0)
    return output


class MultiHeadAttentionTests(unittest.TestCase):
    def make_module(self, **kwargs):
        options = {
            "io_dim": 8,
            "head_dim": 4,
            "qk_dim": 4,
            "num_heads": 2,
            "max_seq_len": 8,
        }
        options.update(kwargs)
        return MultiHeadAttention(**options).double()

    def test_right_padding_matches_dense_reference(self):
        module = self.make_module(bias=True)
        x = torch.randn(2, 5, 8, dtype=torch.float64)
        padding = validate_right_padding_mask(
            torch.tensor(
                [[False, False, False, True, True], [False, False, True, True, True]]
            )
        )

        actual = module(x, padding, padding_layout="right")
        expected = dense_reference(module, x, padding)

        torch.testing.assert_close(actual, expected)

    def test_padded_nan_and_inf_values_are_inert(self):
        module = self.make_module(bias=True)
        padding = validate_right_padding_mask(
            torch.tensor([[False, False, False, True, True]])
        )
        clean_x = torch.randn(1, 5, 8, dtype=torch.float64)
        padded_x = clean_x.clone()
        padded_x[0, 3] = float("nan")
        padded_x[0, 4] = float("inf")
        padded_x.requires_grad_()

        actual = module(padded_x, padding, padding_layout="right")
        expected = module(clean_x, padding, padding_layout="right")
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.isfinite(actual).all())

        gradients = torch.autograd.grad(
            actual.square().sum(),
            (padded_x, *module.parameters()),
        )
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
        torch.testing.assert_close(
            gradients[0][padding],
            torch.zeros_like(gradients[0][padding]),
        )

    def test_right_padding_validator_rejects_malformed_layout(self):
        valid_mask = torch.tensor([[False, False, True, True]])
        self.assertIs(validate_right_padding_mask(valid_mask), valid_mask)

        malformed_mask = torch.tensor([[False, True, False, True]])
        with self.assertRaisesRegex(ValueError, "cannot contain padding"):
            validate_right_padding_mask(malformed_mask)

    def test_arbitrary_padding_output_and_gradients_match_dense_reference(self):
        module = self.make_module(bias=True)
        x_actual = torch.randn(3, 5, 8, dtype=torch.float64, requires_grad=True)
        x_expected = x_actual.detach().clone().requires_grad_(True)
        padding = torch.tensor(
            [
                [True, False, False, True, False],
                [False, True, False, False, True],
                [True, True, True, True, True],
            ]
        )

        actual = module(x_actual, padding, padding_layout="arbitrary")
        expected = dense_reference(module, x_expected, padding)
        torch.testing.assert_close(actual, expected)

        parameters = tuple(module.parameters())
        actual_grads = torch.autograd.grad(
            actual.square().sum(), (x_actual, *parameters)
        )
        expected_grads = torch.autograd.grad(
            expected.square().sum(), (x_expected, *parameters)
        )
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_unequal_qk_and_value_dims_match_dense_reference(self):
        x = torch.randn(2, 5, 8, dtype=torch.float64)
        for qk_dim, head_dim in ((3, 5), (5, 3)):
            with self.subTest(qk_dim=qk_dim, head_dim=head_dim):
                module = self.make_module(qk_dim=qk_dim, head_dim=head_dim)
                actual = module(x)
                expected = dense_reference(module, x)
                self.assertEqual(actual.shape, x.shape)
                torch.testing.assert_close(actual, expected)

    def test_unequal_dims_run_on_available_backends(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append(torch.device("mps"))

        for device in devices:
            for qk_dim, head_dim in ((3, 5), (5, 3)):
                with self.subTest(
                    device=device, qk_dim=qk_dim, head_dim=head_dim
                ):
                    module = self.make_module(
                        qk_dim=qk_dim,
                        head_dim=head_dim,
                    ).to(device=device, dtype=torch.float32)
                    x = torch.randn(2, 5, 8, device=device)
                    actual = module(x)
                    expected = dense_reference(module, x)
                    torch.testing.assert_close(
                        actual,
                        expected,
                        rtol=1e-4,
                        atol=1e-5,
                    )

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

            module.eval()
            module(x)
            self.assertEqual(sdpa.call_args.kwargs["dropout_p"], 0.0)

    def test_right_padding_path_supports_fullgraph_compile(self):
        if not hasattr(torch, "compile"):
            self.skipTest("torch.compile is unavailable")

        module = self.make_module()
        x = torch.randn(2, 5, 8, dtype=torch.float64)
        padding = validate_right_padding_mask(
            torch.tensor(
                [[False, False, False, True, True], [False, False, True, True, True]]
            )
        )
        compiled = torch.compile(module, fullgraph=True, backend="eager")

        torch.testing.assert_close(
            compiled(x, padding, padding_layout="right"),
            module(x, padding, padding_layout="right"),
        )

    def test_input_and_padding_validation(self):
        with self.assertRaises(ValueError):
            self.make_module(io_dim=0)

        module = self.make_module()
        with self.assertRaises(TypeError):
            module(torch.ones(1, 2, 8, dtype=torch.int64))
        with self.assertRaises(ValueError):
            module(torch.randn(1, 9, 8, dtype=torch.float64))

        x = torch.randn(1, 3, 8, dtype=torch.float64)
        mask = torch.zeros(1, 3, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "padding_layout"):
            module(x, mask)
        with self.assertRaises(ValueError):
            module(x, torch.zeros(1, 2, dtype=torch.bool), padding_layout="right")
        with self.assertRaises(TypeError):
            module(x, torch.zeros(1, 3), padding_layout="right")
        with self.assertRaisesRegex(ValueError, "same device"):
            module(
                x,
                torch.zeros(1, 3, dtype=torch.bool, device="meta"),
                padding_layout="right",
            )


if __name__ == "__main__":
    unittest.main()
