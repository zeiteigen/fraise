"""Check strict BF16 checkpoint replay in a fresh CUDA process."""

import os
import subprocess
import sys
import unittest
from pathlib import Path

import torch


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    "requires a CUDA device with BF16 support",
)
class TrainingCudaTests(unittest.TestCase):
    def test_strict_bf16_resume_matches_weights_optimizer_and_scheduler_exactly(self):
        # Isolation configures cuBLAS before CUDA initialization and keeps Torch
        # execution flags and compiler pools independent of other tests.
        script = """
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from fixtures import tiny_config, tiny_settings, tiny_tokenizer, tiny_data
from fraise.training.training import pretrain_transformer
from fraise.training.run.checkpoints import load_checkpoint_file

settings = tiny_settings(
    batch_size=4, minibatch_size=2, num_epochs=3, max_train_tokens=5,
    dataset_token_counts={'fixture': 5}, checkpoint_keep_last=3,
    use_compile=False, use_bf16=True, deterministic=True,
)
with TemporaryDirectory() as directory:
    args = (tiny_tokenizer(), tiny_data(), settings, tiny_config(), directory, 'tiny')
    pretrain_transformer(*args, device='cuda')
    run = Path(directory) / 'tiny'
    expected = load_checkpoint_file(run / 'checkpoint_step_00000003.pt')
    pretrain_transformer(*args, resume_from=run / 'checkpoint_step_00000001.pt', device='cuda')
    actual = load_checkpoint_file(run / 'checkpoint_step_00000003.pt')
    for name, tensor in expected['model_state_dict'].items():
        assert torch.equal(tensor, actual['model_state_dict'][name]), name
    for index, state in expected['optimizer_state_dict']['state'].items():
        for name, value in state.items():
            other = actual['optimizer_state_dict']['state'][index][name]
            assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
    assert expected['scheduler_state_dict'] == actual['scheduler_state_dict']
    assert expected['training_state'] == actual['training_state']
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(
                None,
                (str(Path(__file__).resolve().parent), environment.get("PYTHONPATH")),
            )
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
