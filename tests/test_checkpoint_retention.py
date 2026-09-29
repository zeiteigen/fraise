import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train.save_mgr import SaveManager, read_run_model_name  # noqa: E402


class CheckpointRetentionTests(unittest.TestCase):
    def test_retention_keeps_recent_and_pending_validation_files(self):
        with TemporaryDirectory() as directory:
            manager = SaveManager(directory, "run")
            checkpoints = {
                step: manager.save_dir / f"checkpoint_step_{step:08d}.pt"
                for step in (1, 2, 3, 100_000_000)
            }
            for path in checkpoints.values():
                path.write_bytes(b"checkpoint")
            manager.latest_checkpoint = checkpoints[100_000_000]

            manager.prune_checkpoints(2, protected={checkpoints[1]})
            self.assertEqual(
                {step for step, path in checkpoints.items() if path.exists()},
                {1, 3, 100_000_000},
            )

            manager.prune_checkpoints(2)
            self.assertEqual(
                {step for step, path in checkpoints.items() if path.exists()},
                {3, 100_000_000},
            )

    def test_resume_rejects_a_different_model_name(self):
        with TemporaryDirectory() as directory:
            manager = SaveManager(directory, "original")
            checkpoint = manager.save_dir / "checkpoint_step_00000001.pt"
            checkpoint.write_bytes(b"checkpoint")

            self.assertEqual(read_run_model_name(manager.save_dir), "original")
            with self.assertRaisesRegex(ValueError, "resume model_name"):
                SaveManager(directory, "different", resume_from=checkpoint)


if __name__ == "__main__":
    unittest.main()
