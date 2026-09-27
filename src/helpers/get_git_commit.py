

from pathlib import Path
import hashlib
import os
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
    """Return commit, dirty state, and a digest of working-tree edits."""
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
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=repository_root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return snapshot

    status_bytes = status.stdout
    diff_bytes = diff.stdout
    snapshot["dirty"] = bool(status_bytes)
    digest = hashlib.sha256(status_bytes + b"\0" + diff_bytes + b"\0")
    for raw_path in sorted(filter(None, untracked.stdout.split(b"\0"))):
        path = repository_root / os.fsdecode(raw_path)
        digest.update(raw_path + b"\0")
        try:
            if path.is_symlink():
                digest.update(os.fsencode(os.readlink(path)))
            else:
                with path.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
        except OSError:
            # A file may disappear while git and the filesystem are scanned.
            digest.update(b"<unavailable>")
        digest.update(b"\0")
    snapshot["working_tree_sha256"] = digest.hexdigest()
    return snapshot
