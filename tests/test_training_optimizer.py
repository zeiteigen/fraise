import unittest
from copy import deepcopy

import torch
from fixtures import tiny_config, tiny_settings
from transformers.optimization import get_cosine_schedule_with_warmup

from fraise.model.model import FraiseModel
from fraise.training.optimizer import MuonWithAdamW


class TrainingOptimizerTests(unittest.TestCase):
    def test_partition_covers_parameters_once_and_excludes_embeddings_and_heads(self):
        for tied in (False, True):
            with self.subTest(tied=tied):
                model = FraiseModel(tiny_config(tie_word_embeddings=tied))
                optimizer = MuonWithAdamW(model, tiny_settings(weight_decay=0.1))
                muon, decay, no_decay = (
                    set(group["params"]) for group in optimizer.param_groups
                )
                self.assertEqual(
                    [group["optimizer"] for group in optimizer.param_groups],
                    ["muon", "adamw", "adamw_no_decay"],
                )
                self.assertEqual(
                    muon,
                    {p for p in model.transformer.parameters() if p.ndim == 2},
                )
                self.assertFalse(muon & decay or muon & no_decay or decay & no_decay)
                self.assertEqual(muon | decay | no_decay, set(model.parameters()))
                self.assertIn(model.token_embedding.weight, decay)
                self.assertIn(model.lm_head.weight, decay)
                self.assertIn(model.lm_head.bias, no_decay)
                self.assertTrue(all(p.ndim == 1 for p in no_decay))
                self.assertEqual(optimizer.param_groups[1]["weight_decay"], 0.1)
                self.assertEqual(optimizer.param_groups[2]["weight_decay"], 0.0)
                self.assertEqual(
                    sum(len(group["params"]) for group in optimizer.param_groups),
                    len(list(model.parameters())),
                )

    def test_updates_match_native_optimizers_with_separate_qkv_blocks(self):
        for tied, nesterov in ((False, False), (True, True)):
            with self.subTest(tied=tied, nesterov=nesterov):
                model = FraiseModel(tiny_config(tie_word_embeddings=tied, qk_dim=6))
                reference = deepcopy(model)
                settings = tiny_settings(
                    learning_rate=0.02,
                    adamw_learning_rate=0.0003,
                    muon_momentum=0.9,
                    muon_nesterov=nesterov,
                    muon_ns_steps=3,
                    weight_decay=0.01,
                )
                optimizer = MuonWithAdamW(model, settings)
                matrices = [
                    p for p in reference.transformer.parameters() if p.ndim == 2
                ]
                fused = {
                    layer.mha.qkv_projection.weight: (
                        layer.mha.q_size,
                        layer.mha.k_size,
                        layer.mha.v_size,
                    )
                    for layer in reference.transformer.layers
                }
                # Reference: every Q, K, and V block is its own native Muon parameter.
                blocks = {
                    matrix: [
                        torch.nn.Parameter(block.clone())
                        for block in matrix.detach().split(
                            fused.get(matrix, (len(matrix),))
                        )
                    ]
                    for matrix in matrices
                }
                muon = torch.optim.Muon(
                    [block for parts in blocks.values() for block in parts],
                    lr=settings.learning_rate,
                    momentum=settings.muon_momentum,
                    nesterov=settings.muon_nesterov,
                    ns_steps=settings.muon_ns_steps,
                    weight_decay=settings.weight_decay,
                    adjust_lr_fn="original",
                )
                others = [p for p in reference.parameters() if p not in blocks]
                adamw = torch.optim.AdamW(
                    [
                        {"params": [p for p in others if p.ndim >= 2]},
                        {
                            "params": [p for p in others if p.ndim < 2],
                            "weight_decay": 0.0,
                        },
                    ],
                    lr=settings.adamw_learning_rate,
                    betas=settings.betas,
                    weight_decay=settings.weight_decay,
                )
                for _ in range(3):
                    for actual, expected in zip(
                        model.parameters(), reference.parameters()
                    ):
                        actual.grad = torch.randn_like(actual)
                        expected.grad = actual.grad.clone()
                    for matrix, parts in blocks.items():
                        for part, grad in zip(
                            parts, matrix.grad.split([len(part) for part in parts])
                        ):
                            part.grad = grad.clone()
                    optimizer.step()
                    muon.step()
                    adamw.step()
                    with torch.no_grad():
                        for matrix, parts in blocks.items():
                            matrix.copy_(torch.cat([part for part in parts]))
                for actual, expected in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                matrix = model.transformer.layers[0].mha.qkv_projection.weight
                self.assertIn("momentum_buffer", optimizer.state[matrix])
                self.assertIn(
                    "exp_avg_sq", optimizer.state[model.token_embedding.weight]
                )
                optimizer.zero_grad(set_to_none=True)
                self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_fused_qkv_blocks_are_not_orthogonalized_together(self):
        model = FraiseModel(tiny_config())
        weight = model.transformer.layers[0].mha.qkv_projection.weight
        whole = torch.optim.Muon([torch.nn.Parameter(weight.detach().clone())], lr=0.1)
        optimizer = MuonWithAdamW(model, tiny_settings(learning_rate=0.1))
        for parameter in model.parameters():
            parameter.grad = torch.randn_like(parameter)
        whole.param_groups[0]["params"][0].grad = weight.grad.clone()
        optimizer.step()
        whole.step()
        self.assertFalse(torch.allclose(weight, whole.param_groups[0]["params"][0]))

    def test_checkpoint_resume_preserves_both_states_and_scheduled_learning_rates(self):
        model = FraiseModel(tiny_config(tie_word_embeddings=True))
        settings = tiny_settings(learning_rate=0.02, warmup_steps=2)
        optimizer = MuonWithAdamW(model, settings)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 2, 8)

        def step(model, optimizer, scheduler):
            for parameter in model.parameters():
                parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            scheduler.step()

        for _ in range(3):
            step(model, optimizer, scheduler)
        restored = deepcopy(model)
        resumed = MuonWithAdamW(restored, settings)
        resumed_scheduler = get_cosine_schedule_with_warmup(resumed, 2, 8)
        resumed.load_state_dict(deepcopy(optimizer.state_dict()))
        resumed_scheduler.load_state_dict(deepcopy(scheduler.state_dict()))
        for _ in range(3):
            step(model, optimizer, scheduler)
            step(restored, resumed, resumed_scheduler)
            self.assertEqual(scheduler.get_last_lr(), resumed_scheduler.get_last_lr())
            self.assertIs(resumed.param_groups[0], resumed.muon.param_groups[0])
            self.assertEqual(resumed.param_groups[1:], resumed.adamw.param_groups)
            for group, delegate in zip(
                resumed.param_groups[1:], resumed.adamw.param_groups
            ):
                self.assertIs(group, delegate)
            for actual, expected in zip(restored.parameters(), model.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            muon_lr, adamw_lr, no_decay_lr = scheduler.get_last_lr()
            self.assertAlmostEqual(muon_lr / adamw_lr, 0.02 / 0.0003)
            self.assertEqual(adamw_lr, no_decay_lr)

    def test_legacy_optimizer_states_are_rejected(self):
        model = FraiseModel(tiny_config())
        optimizer = MuonWithAdamW(model, tiny_settings())
        legacy_adamw = torch.optim.AdamW(model.parameters()).state_dict()
        legacy_muon_adamw = optimizer.state_dict()
        legacy_muon_adamw["param_groups"] = legacy_muon_adamw["param_groups"][:2]
        for legacy in (legacy_adamw, legacy_muon_adamw):
            with self.assertRaisesRegex(ValueError, "start a fresh run"):
                optimizer.load_state_dict(legacy)


if __name__ == "__main__":
    unittest.main()
