

from pathlib import Path
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