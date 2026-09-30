import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from fixtures import tiny_config, tiny_data, tiny_settings, tiny_tokenizer

from fraise.io import read_json
from fraise.model.model import FraiseModel
from fraise.training.validation_npu import OpenVinoNpuModel


class NpuBackendTests(unittest.TestCase):
    def backend(self, devices):
        core = Mock(available_devices=devices)
        compiled = core.compile_model.return_value
        compiled.get_property.return_value = "NPU"
        compiled.output.return_value = "logits"
        ov = SimpleNamespace(Core=lambda: core, convert_model=Mock())
        return core, compiled, ov

    def test_missing_npu_fails_without_compiling_for_cpu(self):
        core, compiled, ov = self.backend(["CPU"])
        with (
            patch.dict("sys.modules", {"openvino": ov}),
            patch("fraise.training.validation_npu.load_local_npu_runtime"),
        ):
            with self.assertRaisesRegex(RuntimeError, "CPU fallback is disabled"):
                OpenVinoNpuModel(
                    FraiseModel(tiny_config()).eval(), 2, progress=lambda message: None
                )
        core.compile_model.assert_not_called()

    def test_npu_is_explicit_and_outputs_do_not_alias_runtime_buffer(self):
        core, compiled, ov = self.backend(["CPU", "NPU"])
        buffer = np.ones((1, 2, 8), dtype=np.float16)
        compiled.return_value = {"logits": buffer}
        messages = []
        with (
            patch.dict("sys.modules", {"openvino": ov}),
            patch("fraise.training.validation_npu.load_local_npu_runtime"),
            patch("torch.export.export"),
        ):
            model = OpenVinoNpuModel(
                FraiseModel(tiny_config()).eval(), 2, progress=messages.append
            )
            output = model(torch.zeros((1, 2), dtype=torch.long))
        self.assertEqual(core.compile_model.call_args.args[1], "NPU")
        ov.convert_model.return_value.reshape.assert_called_once_with([1, 2])
        self.assertEqual(output.dtype, torch.float32)
        buffer.fill(0)
        self.assertTrue(torch.equal(output, torch.ones_like(output)))
        self.assertTrue(
            any("execution devices: ['NPU']" in message for message in messages)
        )
        with self.assertRaisesRegex(ValueError, "fixed-context"):
            model(torch.zeros((2, 2), dtype=torch.long))

    def test_unexpected_execution_device_is_rejected(self):
        core, compiled, ov = self.backend(["NPU"])
        compiled.get_property.return_value = ["CPU"]
        with (
            patch.dict("sys.modules", {"openvino": ov}),
            patch("fraise.training.validation_npu.load_local_npu_runtime"),
            patch("torch.export.export"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Expected NPU execution"):
                OpenVinoNpuModel(
                    FraiseModel(tiny_config()).eval(), 2, progress=lambda message: None
                )

    def test_missing_openvino_has_install_instruction(self):
        with patch.dict("sys.modules", {"openvino": None}):
            with self.assertRaisesRegex(RuntimeError, "uv sync --extra npu"):
                OpenVinoNpuModel(
                    FraiseModel(tiny_config()).eval(), 2, progress=lambda message: None
                )


@unittest.skipUnless(
    os.environ.get("FRAISE_TEST_NPU") == "1",
    "set FRAISE_TEST_NPU=1 for real Intel NPU tests",
)
class NpuHardwareTests(unittest.TestCase):
    def test_actual_npu_logits_match_cpu_and_refresh_checkpoint_weights(self):
        torch.set_num_threads(2)
        torch.manual_seed(1337)
        for config in (tiny_config(qk_dim=2), tiny_config(tie_word_embeddings=True)):
            with self.subTest(config=config), torch.inference_mode():
                reference = FraiseModel(config).eval()
                npu = OpenVinoNpuModel(reference, 4, progress=lambda message: None)
                ids = torch.tensor([[3, 4, 5, 6]])
                expected = reference(ids)
                old = npu(ids)
                torch.testing.assert_close(old, expected, atol=0.01, rtol=0.01)
                reference.lm_head.bias.add_(0.25)
                updated = OpenVinoNpuModel(reference, 4, progress=lambda message: None)
                torch.testing.assert_close(
                    updated(ids), reference(ids), atol=0.01, rtol=0.01
                )
                torch.testing.assert_close(npu(ids), old, atol=0.001, rtol=0.001)

    def test_checkpoint_worker_reports_npu_losses_for_every_checkpoint(self):
        from fraise.training.checkpoints import load_checkpoint_file
        from fraise.training.preparation import prepare_data
        from fraise.training.training import pretrain_transformer
        from fraise.training.validation import evaluate

        with (
            TemporaryDirectory() as directory,
            patch.object(FraiseModel, "compile", lambda self: None),
        ):
            settings = tiny_settings(
                validation_device="npu", max_validation_tokens=3, checkpoint_keep_last=1
            )
            tokenizer = tiny_tokenizer()
            data = tiny_data(validation=True)
            config = tiny_config()
            pretrain_transformer(
                tokenizer, data, settings, config, directory, "tiny", device="cpu"
            )
            run = Path(directory) / "tiny"
            metadata = read_json(run / "model_meta.json")
            self.assertEqual(
                [result["step"] for result in metadata["validation_results"]], [1, 2]
            )
            self.assertEqual(metadata["validation_failures"], [])
            log = (run / "training.log").read_text()
            self.assertIn("OpenVINO execution devices: ['NPU']", log)
            self.assertIn("Progress 3/3", log)
            checkpoint = load_checkpoint_file(run / "checkpoint_step_00000002.pt")
            reference = FraiseModel(config).eval()
            reference.load_state_dict(checkpoint["model_state_dict"])
            split = prepare_data(data, tokenizer, settings, config).validation
            expected = evaluate(reference, split, settings, 0)
            self.assertAlmostEqual(metadata["validation"][1], expected, delta=0.01)
            self.assertEqual(len(list(run.glob("checkpoint_step_*.pt"))), 1)
