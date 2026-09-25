

from pathlib import Path
import hashlib
import subprocess


def get_git_commit() -> str | None:
    """Return the current repository commit, when the code is in a git checkout."""
    repository_root = Path(__file__).resolve().parents[2]

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    commit = result.stdout.strip()
    return commit or None


def get_git_snapshot() -> dict[str, object]:
    """Return commit, dirty state, and a digest of tracked working-tree edits."""
    repository_root = Path(__file__).resolve().parents[2]
    snapshot: dict[str, object] = {
        "commit": get_git_commit(),
        "dirty": False,
        "working_tree_sha256": None,
    }

    try:
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repository_root,
            check=True,
            capture_output=True,
        )
        diff = subprocess.run(
            ["git", "diff", "HEAD", "--binary"],
            cwd=repository_root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return snapshot

    status_bytes = status.stdout
    diff_bytes = diff.stdout
    snapshot["dirty"] = bool(status_bytes)
    snapshot["working_tree_sha256"] = hashlib.sha256(
        status_bytes + b"\0" + diff_bytes
    ).hexdigest()
    return snapshot
