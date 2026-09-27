"""Pin local Arrow shards so source loading survives HF cache changes."""

from collections.abc import Sequence
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from datasets import Dataset, DatasetInfo, concatenate_datasets, load_from_disk
from filelock import FileLock


_CACHE_VERSION = 1


class ArrowCache:
    def __init__(self, root: Path, request: dict[str, object]):
        self.root = root
        self.request = request

    def _path(self, split: str, files: Sequence[Path]) -> Path:
        signature = {
            "version": _CACHE_VERSION,
            "request": self.request,
            "split": split,
            "files": [
                {
                    "path": str(path.resolve()),
                    "size": (stat := path.stat()).st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "ctime_ns": stat.st_ctime_ns,
                    "inode": stat.st_ino,
                }
                for path in files
            ],
        }
        digest = hashlib.sha256(
            json.dumps(signature, sort_keys=True).encode("utf-8")
        ).hexdigest()[:24]
        return self.root / f"{split}-{digest}"

    def load(self, split: str, files: Sequence[Path]) -> Dataset | None:
        cache_path = self._path(split, files)
        if not cache_path.is_dir():
            return None
        try:
            with (cache_path / "cache.json").open(encoding="utf-8") as handle:
                metadata = json.load(handle)
            if metadata["version"] != _CACHE_VERSION:
                return None
            if metadata["storage"] == "saved":
                dataset = load_from_disk(str(cache_path / "dataset"))
            else:
                info = DatasetInfo.from_directory(str(cache_path))
                shards = [
                    Dataset.from_file(str(cache_path / name), info=info, split=split)
                    for name in metadata["shards"]
                ]
                dataset = (
                    concatenate_datasets(shards, info=info)
                    if len(shards) > 1
                    else shards[0]
                )
            dataset._fingerprint = metadata["fingerprint"]
            return dataset
        except (OSError, KeyError, ValueError, IndexError, TypeError):
            return None

    def store(
        self,
        split: str,
        files: Sequence[Path],
        dataset: Dataset,
        *,
        replace: bool = False,
    ) -> Dataset:
        self.root.mkdir(parents=True, exist_ok=True)
        cache_path = self._path(split, files)
        with FileLock(str(self.root / f".{split}.lock")):
            if not replace and (cached := self.load(split, files)) is not None:
                return cached

            temporary_path = Path(tempfile.mkdtemp(prefix=f".{split}-", dir=self.root))
            try:
                cache_files = dataset.cache_files if dataset._indices is None else []
                if cache_files:
                    shard_names = []
                    for index, entry in enumerate(cache_files):
                        name = f"{index:05d}.arrow"
                        source = Path(entry["filename"])
                        destination = temporary_path / name
                        try:
                            os.link(source, destination)
                        except OSError as error:
                            if error.errno != errno.EXDEV:
                                raise
                            shutil.copy2(source, destination)
                        shard_names.append(name)
                    dataset.info.write_to_directory(str(temporary_path))
                    storage = "shards"
                else:
                    dataset.save_to_disk(str(temporary_path / "dataset"))
                    shard_names = []
                    storage = "saved"

                with (temporary_path / "cache.json").open(
                    "w", encoding="utf-8"
                ) as handle:
                    json.dump(
                        {
                            "version": _CACHE_VERSION,
                            "storage": storage,
                            "shards": shard_names,
                            "fingerprint": dataset._fingerprint,
                        },
                        handle,
                    )
                if cache_path.exists():
                    shutil.rmtree(cache_path)
                os.replace(temporary_path, cache_path)
            finally:
                if temporary_path.exists():
                    shutil.rmtree(temporary_path)
        return dataset
