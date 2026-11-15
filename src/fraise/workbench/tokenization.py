"""Tokenize text through configurable hierarchies for the tokenization workbench."""

from collections import OrderedDict
from pathlib import Path

from fraise.hierarchy import EmbeddingHierarchy, hierarchy_from_macro_tokenizer
from fraise.hierarchy.binding import check_tokenizer_hierarchy
from fraise.hierarchy.construction import two_level_hierarchy
from fraise.hierarchy.runs import run_hierarchy
from fraise.preview.hierarchy import analyze_text
from fraise.tokenizer.loading import load_tokenizer

from .catalog import artifact_name, contained

HIERARCHY_SUFFIXES = (".npz", ".json")


class _Cache(OrderedDict):
    """A small LRU keyed by file identity, so edits on disk are picked up."""

    def __init__(self, size):
        super().__init__()
        self.size = size

    def get_or_create(self, key, create):
        if key in self:
            self.move_to_end(key)
            return self[key]
        value = self[key] = create()
        while len(self) > self.size:
            self.popitem(last=False)
        return value


def _file_key(path: Path):
    stat = path.stat()
    return (str(path), stat.st_mtime_ns, stat.st_size)


class TokenizationWorkbench:
    def __init__(self, catalog):
        self.catalog = catalog
        self.tokenizers = _Cache(4)
        self.hierarchies = _Cache(8)

    @property
    def hierarchy_root(self) -> Path:
        return self.catalog.data_dir / "hierarchies"

    def hierarchy_files(self) -> list[dict]:
        root = self.hierarchy_root
        if not root.is_dir():
            return []
        records = []
        for path in sorted(root.iterdir()):
            if path.suffix in HIERARCHY_SUFFIXES and path.is_file():
                records.append({"id": path.name, "label": path.stem})
        return records

    def tokenizer(self, path: Path):
        return self.tokenizers.get_or_create(
            ("tokenizer", *_file_key(path)), lambda: load_tokenizer(path)
        )

    def analyze(self, request) -> dict:
        results = []
        for index, configuration in enumerate(request.configurations):
            try:
                tokenizer, definition, label = self._resolve(configuration)
                analysis = analyze_text(
                    request.text,
                    tokenizer,
                    definition,
                    add_special_tokens=request.add_special_tokens,
                    name=configuration.label or label,
                )
            except (OSError, ValueError, KeyError, TypeError) as error:
                analysis = {"name": configuration.label or f"Configuration {index + 1}"}
                analysis["error"] = str(error)
            results.append(analysis)
        return {"results": results}

    def _resolve(self, configuration):
        spec = configuration.hierarchy
        if spec.kind == "run":
            if spec.run is None:
                raise ValueError("select a saved run")
            directory, checkpoint = self.catalog.resolve(spec.run)
            key = ("run", *_file_key(checkpoint))
            tokenizer, definition, _ = self.hierarchies.get_or_create(
                key, lambda: run_hierarchy(directory, checkpoint)
            )
            return tokenizer, definition, f"{directory.name} · {checkpoint.name}"
        if not configuration.tokenizer_id:
            raise ValueError("select a tokenizer")
        path = self.catalog.tokenizer_path(configuration.tokenizer_id)
        tokenizer = self.tokenizer(path)
        name = configuration.tokenizer_id.partition(":")[2]
        residual = spec.residual_dims or [32]
        macro = getattr(tokenizer, "decompositions", None) is not None
        if spec.kind == "none" or (spec.kind == "auto" and not macro):
            return tokenizer, None, f"{name} · segmentation only"
        if spec.kind == "auto":
            key = ("auto", *_file_key(path), residual[0])
            definition = self.hierarchies.get_or_create(
                key,
                lambda: two_level_hierarchy(
                    tokenizer.decompositions,
                    tokenizer.unit_lengths,
                    tokenizer.base_vocab_size,
                    residual[0],
                ),
            )
            return tokenizer, definition, f"{name} · two-level"
        if spec.kind == "merges":
            if not spec.level_sizes:
                raise ValueError("enter level sizes, base vocabulary first")
            dims = residual if len(residual) > 1 else residual[0]
            key = ("merges", *_file_key(path), tuple(spec.level_sizes), str(dims))
            definition = self.hierarchies.get_or_create(
                key,
                lambda: hierarchy_from_macro_tokenizer(
                    tokenizer, spec.level_sizes, dims
                ),
            )
            sizes = " → ".join(map(str, spec.level_sizes))
            return tokenizer, definition, f"{name} · merges {sizes}"
        if not spec.file_id:
            raise ValueError("select a hierarchy file")
        root = self.hierarchy_root
        file = contained(root / artifact_name(spec.file_id), root)
        definition = self.hierarchies.get_or_create(
            ("file", *_file_key(file)), lambda: EmbeddingHierarchy.load(file)
        )
        check_tokenizer_hierarchy(definition, tokenizer)
        return tokenizer, definition, f"{name} · {file.name}"
