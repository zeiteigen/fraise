import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from fraise.training.diagnostics import TrainingDiagnostics


class TrainingDiagnosticsTests(unittest.TestCase):
    def test_interval_and_final_reports_exclude_compiled_warmup_and_track_wait(self):
        messages = []
        runtime = SimpleNamespace(
            compilation_seconds=0.0,
            device=torch.device("cpu"),
            scheduler=SimpleNamespace(get_last_lr=lambda: [0.02, 0.0003]),
        )
        state = SimpleNamespace(global_step=0, tokens_seen=0)
        validation = SimpleNamespace(queue_wait_seconds=0.0)
        with patch("fraise.training.diagnostics.monotonic", return_value=10):
            diagnostics = TrainingDiagnostics(runtime, state, 2, messages.append)
        state.global_step, state.tokens_seen = 1, 128
        diagnostics.record(state, 3.0, 1.0, validation)
        self.assertEqual(messages, [])
        state.global_step, state.tokens_seen = 2, 256
        runtime.compilation_seconds = 3.0
        validation.queue_wait_seconds = 1.25
        with patch("fraise.training.diagnostics.monotonic", return_value=17):
            diagnostics.record(state, 2.5, 0.9, validation)
        self.assertIn("throughput 64 tokens/s", messages[0])
        self.assertIn("Muon learning rate 0.02", messages[0])
        self.assertIn("AdamW learning rate 0.0003", messages[0])
        self.assertIn("validation queue wait 1.250s", messages[0])
        state.global_step, state.tokens_seen = 3, 272
        diagnostics.record(state, 2.0, 0.8, validation)
        with patch("fraise.training.diagnostics.monotonic", return_value=18):
            diagnostics.report(state, validation)
            diagnostics.report(state, validation)
        self.assertEqual(len(messages), 2)
        self.assertIn("Step 3", messages[1])
        self.assertIn("throughput 16 tokens/s", messages[1])
