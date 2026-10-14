"""Create run directories and archive configuration and tokenizer artifacts."""

import time
from dataclasses import asdict, dataclass
from pathlib import Path

from fraise.io import atomic_write, read_json, write_json
from fraise.tokenizer.loading import tokenizer_fingerprint

from .checkpoints import save_model_state_dict, step_checkpoints
from .manifest import check_resume_identity

META_FILE_NAME = "model_meta.json"


def read_run_model_name(path: str | Path) -> str:
    path = Path(path).expanduser()
    metadata = read_json((path if path.is_dir() else path.parent) / META_FILE_NAME)
    return metadata["model_name"]


def resolve_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    paths = step_checkpoints(path)
    if not paths:
        raise FileNotFoundError(f"no step checkpoints in {path}")
    return paths[-1]


@dataclass
class RunDestination:
    """Where a run writes, checked before any expensive preparation starts."""

    directory: Path
    latest: Path | None
    metadata: dict | None


def inspect_run_destination(
    root: Path, name: str, manifest: dict, resume_from: Path | None = None
) -> RunDestination:
    """Validate a new or resumed run directory without writing to it."""
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("model name must be a non-empty directory name")
    if resume_from is None:
        directory = Path(root) / name
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(
                f"run directory is not empty: {directory}; choose a new name or use --resume"
            )
        return RunDestination(directory, None, None)
    latest = resolve_checkpoint(resume_from)
    metadata = read_json(latest.parent / META_FILE_NAME)
    if name != metadata["model_name"]:
        raise ValueError("resume model_name does not match run metadata")
    check_resume_identity(metadata["manifest"], manifest)
    return RunDestination(latest.parent, latest, metadata)


class RunArtifacts:
    def __init__(
        self,
        root: Path,
        name: str,
        manifest: dict,
        tokenizer,
        model_config,
        settings,
        *,
        resume_from: Path | None = None,
        config_paths: dict[str, Path] | None = None,
    ):
        destination = inspect_run_destination(root, name, manifest, resume_from)
        self.latest = destination.latest
        self.directory = destination.directory
        self.path = self.directory / META_FILE_NAME
        if destination.metadata is not None:
            self.metadata = destination.metadata
            self.metadata["training_finished"] = False
            self.metadata.setdefault("resume_provenance", []).append(
                manifest["provenance"]
            )
        else:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.metadata = {
                "schema_version": 3,
                "model_name": name,
                "manifest": manifest,
                "start_time": time.time(),
                "tokenizer_file": "tokenizer.json",
                "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
            }
            atomic_write(
                self.directory / "tokenizer.json",
                lambda handle: handle.write(tokenizer.to_str()),
            )
            for role, config in (
                ("model_config", model_config),
                ("training_hyperparameters", settings),
            ):
                target = self.directory / f"{role}.json"
                if config_paths and role in config_paths:
                    contents = Path(config_paths[role]).read_text()
                    atomic_write(
                        target,
                        lambda handle, contents=contents: handle.write(contents),
                    )
                else:
                    write_json(target, asdict(config))
        self.update()

    def update(self, **values):
        self.metadata.update(values, last_updated=time.time())
        write_json(self.path, self.metadata)

    def finish(self, model, state, *, validation=None, validation_failures=()):
        name = f"{self.metadata['model_name']}.pt"
        save_model_state_dict(model.state_dict(), self.directory / name)
        self.update(
            training_finished=True,
            final_model_path=name,
            end_time=time.time(),
            training_state=state.as_checkpoint(),
            validation=validation,
            validation_failures=list(validation_failures),
        )
