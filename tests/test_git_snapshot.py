import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fraise.helpers.get_git_commit import get_git_snapshot


class GitSnapshotTests(unittest.TestCase):
    def test_untracked_file_content_changes_snapshot(self):
        with TemporaryDirectory(
            prefix=".snapshot-test-", dir=Path(__file__).resolve().parents[1]
        ) as directory:
            path = Path(directory) / "source.py"
            path.write_text("first", encoding="utf-8")
            first = get_git_snapshot()
            path.write_text("second", encoding="utf-8")
            second = get_git_snapshot()

        self.assertTrue(first["dirty"])
        self.assertNotEqual(first["working_tree_sha256"], second["working_tree_sha256"])


if __name__ == "__main__":
    unittest.main()
