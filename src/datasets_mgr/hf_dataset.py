"""Download and load Hugging Face datasets used by Fraise.

The dataset modules in this package all need the same small set of behaviors:

* download rows without first materializing a very large remote dataset;
* optionally stop after a requested number of GB;
* persist the downloaded portion so subsequent runs are local; and
* attach consistent size metadata to the returned Hugging Face dataset.

``HFDataset`` provides those behaviors for both normal Hub datasets and Hub
repositories containing text files. A size of ``None`` means the complete
source is downloaded. Sizes are decimal gigabytes, matching the convention
used by the existing dataset scripts (1 GB = 1,000,000,000 bytes).
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Literal, TypeAlias
from urllib.parse import quote

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import snapshot_download


BYTES_PER_GB = 1_000_000_000
SourceKind: TypeAlias = Literal["dataset", "text_files"]
DatasetResult: TypeAlias = Dataset | DatasetDict
DownloadedFiles: TypeAlias = dict[str, tuple[Path, ...]]


def _validate_size_gb(value: float | None, name: str) -> float | None:
    """Validate and normalize a user-facing GB limit."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive number of GB or None")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{name} must be a positive number of GB or None")
    return normalized


def _validate_validation_size_gb(value: float) -> float:
    """Validate the optional validation portion size."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("validation_size_gb must be a non-negative number of GB")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError("validation_size_gb must be a non-negative number of GB")
    return normalized


def _safe_component(value: str) -> str:
    """Return a path-safe component for generated cache files."""
    component = "".join(
        character if character.isalnum() or character in "-_." else "-"
        for character in value
    )
    return component.strip(".-") or "dataset"


def _relative_paths(paths: Iterable[Path], root: Path) -> list[str]:
    """Serialize paths relative to the dataset cache directory."""
    return [path.relative_to(root).as_posix() for path in paths]


@dataclass(frozen=True, slots=True)
class HFDataset:
    """Description of one Hugging Face dataset source.

    ``source_kind="dataset"`` loads a normal Hub dataset in streaming mode.
    ``source_kind="text_files"`` treats ``file_patterns`` as text files in a
    dataset repository. Complete text-file sources use
    :func:`huggingface_hub.snapshot_download`, while bounded text-file
    sources are streamed through the same row materializer as normal datasets.
    """

    repo_id: str
    data_dir: str | Path
    dataset_name: str | None = None
    config: str | None = None
    revision: str | None = None
    split: str = "train"
    source_kind: SourceKind = "dataset"
    file_patterns: tuple[str, ...] = ()
    cache_prefix: str | None = None
    text_column: str = "text"
    token: str | bool | None = None

    def __post_init__(self) -> None:
        if not self.repo_id:
            raise ValueError("repo_id cannot be empty")
        if not self.split:
            raise ValueError("split cannot be empty")
        if self.source_kind not in ("dataset", "text_files"):
            raise ValueError(
                "source_kind must be either 'dataset' or 'text_files'"
            )
        if self.source_kind == "text_files" and not self.file_patterns:
            raise ValueError(
                "file_patterns are required when source_kind='text_files'"
            )
        if not self.text_column:
            raise ValueError("text_column cannot be empty")

        object.__setattr__(self, "data_dir", Path(self.data_dir))
        if self.dataset_name is None:
            object.__setattr__(self, "dataset_name", self.repo_id)
        if self.cache_prefix is None:
            object.__setattr__(
                self,
                "cache_prefix",
                _safe_component(self.dataset_name or self.repo_id),
            )

    @property
    def _manifest_path(self) -> Path:
        return self.data_dir / f".{self.cache_prefix}.manifest.json"

    def _request(
        self,
        size_gb: float | None,
        validation_size_gb: float,
    ) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "dataset_name": self.dataset_name,
            "config": self.config,
            "revision": self.revision,
            "split": self.split,
            "source_kind": self.source_kind,
            "file_patterns": list(self.file_patterns),
            "text_column": self.text_column,
            "size_gb": size_gb,
            "validation_size_gb": validation_size_gb,
        }

    def _output_path(self, split: str) -> Path:
        return self.data_dir / f"{self.cache_prefix}_{split}.jsonl"

    def _remote_file_patterns(self) -> list[str]:
        """Return revision-pinned ``hf://`` URLs for text-file streaming."""
        revision = f"@{quote(self.revision, safe='')}" if self.revision else ""
        base = f"hf://datasets/{self.repo_id}{revision}"
        return [f"{base}/{pattern.lstrip('/')}" for pattern in self.file_patterns]

    def _stream_source(self):
        """Create the remote streaming dataset for this source."""
        kwargs: dict[str, Any] = {
            "split": self.split,
            "streaming": True,
        }
        if self.revision is not None:
            kwargs["revision"] = self.revision
        if self.token is not None:
            kwargs["token"] = self.token

        if self.source_kind == "text_files":
            return load_dataset(
                "text",
                data_files=self._remote_file_patterns(),
                **kwargs,
            )

        if self.config is not None:
            kwargs["name"] = self.config
        return load_dataset(self.repo_id, **kwargs)

    def _cache_manifest(self) -> dict[str, Any] | None:
        if not self._manifest_path.is_file():
            return None
        try:
            with self._manifest_path.open(encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
        except (OSError, json.JSONDecodeError):
            return None
        return manifest if isinstance(manifest, dict) else None

    def _cached_files(
        self,
        request: dict[str, Any],
    ) -> DownloadedFiles | None:
        """Return cached files when their request and files are complete."""
        manifest = self._cache_manifest()
        if manifest is None or manifest.get("request") != request:
            return None

        manifest_files = manifest.get("files")
        if not isinstance(manifest_files, dict):
            return None

        cached: DownloadedFiles = {}
        for split, raw_paths in manifest_files.items():
            if not isinstance(split, str) or not isinstance(raw_paths, list):
                return None
            paths = tuple(self.data_dir / path for path in raw_paths)
            if not all(path.is_file() for path in paths):
                return None
            cached[split] = paths
        return cached or None

    def _write_manifest(
        self,
        request: dict[str, Any],
        files: DownloadedFiles,
        *,
        mode: Literal["stream", "snapshot"],
    ) -> None:
        """Atomically write the cache manifest after a successful download."""
        payload = {
            "request": request,
            "mode": mode,
            "files": {
                split: _relative_paths(paths, self.data_dir)
                for split, paths in files.items()
            },
        }
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.cache_prefix}.manifest.",
            suffix=".tmp",
            dir=self.data_dir,
            text=True,
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as temporary_file:
                json.dump(payload, temporary_file, indent=2)
                temporary_file.write("\n")
            os.replace(temporary_name, self._manifest_path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    def _row_record(self, row: Mapping[str, Any]) -> tuple[str, int]:
        """Serialize one source row and return it with its counted size."""
        if self.text_column not in row:
            raise ValueError(
                f"Hugging Face dataset does not contain the text column "
                f"{self.text_column!r}; available columns: {list(row)}"
            )

        record = json.dumps(
            {self.text_column: row[self.text_column]},
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ) + "\n"
        text = row[self.text_column]
        counted_size = (
            len(text.encode("utf-8"))
            if isinstance(text, str)
            else len(record.encode())
        )
        return record, counted_size

    def _download_stream(
        self,
        size_gb: float | None,
        validation_size_gb: float,
        request: dict[str, Any],
    ) -> DownloadedFiles:
        """Stream rows into bounded JSONL files and return their paths."""
        total_limit = None if size_gb is None else int(size_gb * BYTES_PER_GB)
        validation_limit = int(validation_size_gb * BYTES_PER_GB)
        output_paths = {"train": self._output_path("train")}
        if validation_limit > 0:
            output_paths["validation"] = self._output_path("validation")

        temporary_paths: dict[str, Path] = {}
        handles: dict[str, Any] = {}
        try:
            for split, output_path in output_paths.items():
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{output_path.name}.",
                    suffix=".tmp",
                    dir=self.data_dir,
                    text=True,
                )
                os.close(descriptor)
                temporary_paths[split] = Path(temporary_name)
                handles[split] = temporary_paths[split].open(
                    "w", encoding="utf-8"
                )

            stream = self._stream_source()
            written_bytes = 0
            for row in stream:
                if not isinstance(row, Mapping):
                    raise TypeError(
                        "streaming Hugging Face datasets must yield mappings; "
                        f"got {type(row).__name__}"
                    )
                if total_limit is not None and written_bytes >= total_limit:
                    break

                record, row_size = self._row_record(row)
                target_split = (
                    "validation"
                    if validation_limit > 0 and written_bytes < validation_limit
                    else "train"
                )
                handles[target_split].write(record)
                written_bytes += row_size

                # Rows are kept intact, so the final file may be slightly
                # larger than the requested limit by one document.
                if total_limit is not None and written_bytes >= total_limit:
                    break

            for handle in handles.values():
                handle.close()
            handles.clear()

            downloaded: DownloadedFiles = {}
            for split, temporary_path in temporary_paths.items():
                output_path = output_paths[split]
                os.replace(temporary_path, output_path)
                downloaded[split] = (output_path,)

            self._write_manifest(request, downloaded, mode="stream")
            return downloaded
        except BaseException:
            for handle in handles.values():
                handle.close()
            for temporary_path in temporary_paths.values():
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass
            raise

    def _matching_local_files(self) -> tuple[Path, ...]:
        """Find downloaded files matching the configured repository patterns."""
        matches: set[Path] = set()
        for pattern in self.file_patterns:
            if "**" in pattern:
                matches.update(self.data_dir.rglob(pattern.replace("**/", "")))
            else:
                matches.update(self.data_dir.glob(pattern))
        return tuple(sorted(path for path in matches if path.is_file()))

    def _download_snapshot(
        self,
        request: dict[str, Any],
        *,
        force_download: bool,
    ) -> DownloadedFiles:
        """Download all configured text files from a Hub repository."""
        cached = None if force_download else self._cached_files(request)
        if cached is not None:
            return cached

        kwargs: dict[str, Any] = {
            "repo_id": self.repo_id,
            "repo_type": "dataset",
            "local_dir": self.data_dir,
            "allow_patterns": list(self.file_patterns),
            "force_download": force_download,
        }
        if self.revision is not None:
            kwargs["revision"] = self.revision
        if self.token is not None:
            kwargs["token"] = self.token

        snapshot_download(**kwargs)
        source_files = self._matching_local_files()
        if not source_files:
            patterns = ", ".join(self.file_patterns)
            raise FileNotFoundError(
                f"no files matching {patterns!r} were downloaded from "
                f"{self.repo_id}"
            )

        downloaded = {"train": source_files}
        self._write_manifest(request, downloaded, mode="snapshot")
        return downloaded

    def download(
        self,
        size_gb: float | None = None,
        *,
        validation_size_gb: float = 0.0,
        force_download: bool = False,
    ) -> DownloadedFiles:
        """Download this source and return files grouped by output split.

        ``size_gb`` is a total source-size limit. Each source row is kept
        whole, so a final split can exceed the limit by the size of one row.
        A ``None`` limit downloads the complete source.
        """
        normalized_size = _validate_size_gb(size_gb, "size_gb")
        normalized_validation = _validate_validation_size_gb(validation_size_gb)
        if (
            normalized_size is not None
            and normalized_validation > normalized_size
        ):
            raise ValueError("validation_size_gb cannot be greater than size_gb")

        self.data_dir.mkdir(parents=True, exist_ok=True)
        request = self._request(normalized_size, normalized_validation)

        if not force_download:
            cached = self._cached_files(request)
            if cached is not None:
                return cached

        # A complete text-file source can remain in its native format and use
        # snapshot_download, matching the efficient path used by BabyLM.
        # Bounded sources go through the row materializer so the requested
        # size is respected instead of downloading every repository file.
        if (
            self.source_kind == "text_files"
            and normalized_size is None
            and normalized_validation == 0
        ):
            return self._download_snapshot(
                request,
                force_download=force_download,
            )

        return self._download_stream(
            normalized_size,
            normalized_validation,
            request,
        )

    def _annotate(self, dataset: Dataset, files: Sequence[Path]) -> Dataset:
        """Attach consistent source and materialized-size metadata."""
        download_size = sum(path.stat().st_size for path in files)
        dataset.info.dataset_name = self.dataset_name
        dataset.info.download_size = download_size
        dataset.info.dataset_size = int(dataset.data.nbytes)
        dataset.info.size_in_bytes = (
            dataset.info.download_size + dataset.info.dataset_size
        )
        return dataset

    def load(
        self,
        size_gb: float | None = None,
        *,
        validation_size_gb: float = 0.0,
        force_download: bool = False,
    ) -> DatasetResult:
        """Download this source if needed and return a map-style HF dataset."""
        normalized_size = _validate_size_gb(size_gb, "size_gb")
        normalized_validation = _validate_validation_size_gb(validation_size_gb)
        files = self.download(
            normalized_size,
            validation_size_gb=normalized_validation,
            force_download=force_download,
        )

        is_native_snapshot = (
            self.source_kind == "text_files"
            and normalized_size is None
            and normalized_validation == 0
        )
        if is_native_snapshot:
            dataset = load_dataset(
                "text",
                data_files=[str(path) for path in files["train"]],
                split=self.split,
            )
            return self._annotate(dataset, files["train"])

        data_files: dict[str, str] = {}
        for split, split_files in files.items():
            if split_files and any(path.stat().st_size > 0 for path in split_files):
                data_files[split] = str(split_files[0])
        if not data_files:
            raise ValueError(f"no rows were downloaded from {self.repo_id}")

        loaded = load_dataset("json", data_files=data_files)
        if not isinstance(loaded, DatasetDict):
            # A mapping of split names normally returns DatasetDict, but keep
            # this defensive branch for alternate datasets library versions.
            loaded = DatasetDict({"train": loaded})

        annotated: dict[str, Dataset] = {}
        for split, split_dataset in loaded.items():
            annotated[split] = self._annotate(split_dataset, files[split])

        if set(annotated) == {"train"}:
            return annotated["train"]
        return DatasetDict(annotated)


def load_hf_dataset(
    repo_id: str,
    *,
    data_dir: str | Path,
    dataset_name: str | None = None,
    config: str | None = None,
    revision: str | None = None,
    split: str = "train",
    size_gb: float | None = None,
    validation_size_gb: float = 0.0,
    source_kind: SourceKind = "dataset",
    file_patterns: Sequence[str] = (),
    cache_prefix: str | None = None,
    text_column: str = "text",
    token: str | bool | None = None,
    force_download: bool = False,
) -> DatasetResult:
    """Construct an :class:`HFDataset`, download it, and load its rows."""
    source = HFDataset(
        repo_id=repo_id,
        data_dir=data_dir,
        dataset_name=dataset_name,
        config=config,
        revision=revision,
        split=split,
        source_kind=source_kind,
        file_patterns=tuple(file_patterns),
        cache_prefix=cache_prefix,
        text_column=text_column,
        token=token,
    )
    return source.load(
        size_gb=size_gb,
        validation_size_gb=validation_size_gb,
        force_download=force_download,
    )


# Keep both spellings convenient for callers importing this module directly.
HuggingFaceDataset = HFDataset
download_hf_dataset = load_hf_dataset


__all__ = [
    "BYTES_PER_GB",
    "HFDataset",
    "HuggingFaceDataset",
    "download_hf_dataset",
    "load_hf_dataset",
]
