import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class PackagingTests(unittest.TestCase):
    def test_wheel_imports_and_entry_points_work_without_the_checkout(self):
        project = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            result = subprocess.run(
                ["uv", "build", "--offline", "--wheel", "--out-dir", str(directory)],
                cwd=project,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            wheel = next(directory.glob("*.whl"))
            target = directory / "installed"
            result = subprocess.run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--offline",
                    "--no-deps",
                    "--target",
                    str(target),
                    str(wheel),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            script = """
import contextlib, importlib.metadata, io, pathlib, sys
project, target = map(pathlib.Path, sys.argv[1:])
sys.path = [str(target)] + [path for path in sys.path if not pathlib.Path(path).resolve().is_relative_to(project / "src")]
import fraise
assert pathlib.Path(fraise.__file__).is_relative_to(target), fraise.__file__
from fraise.model.model import FraiseModel
from fraise.training.training import pretrain_transformer
from fraise.tokenizer.loading import load_tokenizer
from fraise.data.loading import load_training_data
entries = importlib.metadata.distribution("fraise").entry_points
assert {entry.name for entry in entries} == {"train", "create-tokenizer", "visualize-training-data", "params", "checkpoint-to-model", "migrate-run"}
for entry in entries:
    sys.argv = [entry.name, "--help"]
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            entry.load()()
        except SystemExit as error:
            assert error.code == 0, (entry.name, error.code)
        else:
            raise AssertionError(entry.name)
"""
            result = subprocess.run(
                [sys.executable, "-I", "-c", script, str(project), str(target)],
                cwd=directory,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
