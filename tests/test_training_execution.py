import unittest
from unittest.mock import patch

import torch
from fixtures import tiny_config, tiny_settings

from fraise.training.batch import token_loss_sum
from fraise.training.execution import deterministic_execution
from fraise.training.pipeline.batch_types import PackedTokenBatch
from fraise.training.runtime import build_runtime


class TrainingExecutionTests(unittest.TestCase):
    def test_eager_runtime_never_compiles(self):
        with patch(
            "fraise.training.runtime.compile_loss_sum",
            side_effect=AssertionError("compiled"),
        ):
            runtime = build_runtime(
                tiny_config(), tiny_settings(use_compile=False), 2, device="cpu"
            )
        self.assertFalse(runtime.use_compile)
        self.assertIs(runtime.loss_sum, token_loss_sum)
        self.assertIsNone(runtime.model._compiled_call_impl)

    def test_compiler_worker_count_is_applied_before_compilation(self):
        from torch._inductor import config

        previous = config.compile_threads
        self.addCleanup(setattr, config, "compile_threads", previous)
        observed = []

        def compile_loss():
            observed.append(config.compile_threads)
            return token_loss_sum

        with patch("fraise.training.runtime.compile_loss_sum", compile_loss):
            runtime = build_runtime(
                tiny_config(), tiny_settings(compile_threads=2), 2, device="cpu"
            )
        self.assertEqual(observed, [2])
        # The module itself stays uncompiled, so checkpoint keys never change.
        self.assertIsNone(runtime.model._compiled_call_impl)

    def test_compiled_runtime_uses_one_graph_for_full_and_short_minibatches(self):
        graphs = []

        def backend(graph, inputs):
            graphs.append(graph)
            return graph.forward

        def compile_loss():
            return torch.compile(
                token_loss_sum, backend=backend, fullgraph=True, dynamic=False
            )

        settings = tiny_settings(batch_size=4, minibatch_size=2)
        with patch("fraise.training.runtime.compile_loss_sum", compile_loss):
            runtime = build_runtime(tiny_config(), settings, 2, device="cpu")
        batch = PackedTokenBatch(
            torch.tensor([[1, 2], [2, 3], [3, 0]]),
            torch.tensor([[2, 3], [3, 4], [4, 0]]),
            torch.tensor([[True, True], [True, True], [True, False]]),
        )
        messages = []
        runtime.log = messages.append
        _, targets, _ = runtime.step(batch)
        runtime.step(batch)
        self.assertEqual(targets, 5)
        self.assertEqual(len(graphs), 1)
        self.assertIn("cross_entropy", str(graphs[0].code))
        self.assertTrue(runtime.compiled_warmed_up)
        self.assertGreater(runtime.compilation_seconds, 0)
        self.assertEqual(
            sum("forward/backward warmup:" in text for text in messages), 1
        )

    def test_strict_execution_pins_math_attention_and_restores_flags_on_failure(self):
        before = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.get_float32_matmul_precision(),
            torch.backends.cudnn.benchmark,
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
        )
        with (
            self.assertRaisesRegex(RuntimeError, "test failure"),
            deterministic_execution(True),
        ):
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
            self.assertEqual(torch.get_float32_matmul_precision(), "highest")
            self.assertFalse(torch.backends.cudnn.benchmark)
            self.assertFalse(torch.backends.cuda.flash_sdp_enabled())
            self.assertTrue(torch.backends.cuda.math_sdp_enabled())
            raise RuntimeError("test failure")
        after = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.get_float32_matmul_precision(),
            torch.backends.cudnn.benchmark,
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
        )
        self.assertEqual(before, after)
