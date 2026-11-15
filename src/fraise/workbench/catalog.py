"""Discover local runs, checkpoints, and tokenizers without loading weights."""

from dataclasses import asdict
from pathlib import Path

from fraise.architectures import model_config_from_dict
from fraise.config.training import TrainingSettings
from fraise.io import read_json
from fraise.model.parameters import estimate_parameter_count
from fraise.tokenizer.loading import load_tokenizer, tokenizer_fingerprint
from fraise.training.run.checkpoints import checkpoint_step, step_checkpoints
from fraise.training.optimizer import OPTIMIZER_IDENTITY


def contained(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("artifact path is outside the configured directory")
    return path


def artifact_name(value):
    if not value or Path(value).name != value or value in (".", ".."):
        raise ValueError("invalid artifact identifier")
    return value


class Catalog:
    def __init__(self, data_dir, models_dir=None):
        self.data_dir = Path(data_dir).resolve()
        self.models_dir = Path(models_dir or self.data_dir / "models").resolve()
        self._tokenizer_cache = {}

    def run_path(self, identifier):
        path = contained(self.models_dir / artifact_name(identifier), self.models_dir)
        if not contained(path / "model_meta.json", path).is_file():
            raise FileNotFoundError("saved run was not found")
        return path

    def _tokenizer_record(self, path, identifier, label):
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        if key not in self._tokenizer_cache:
            try:
                tokenizer = load_tokenizer(path)
            except Exception as error:
                # The tokenizer library reports malformed JSON as a plain Exception.
                raise ValueError(
                    f"Cannot load tokenizer {path.name}: {error}"
                ) from error
            self._tokenizer_cache = {
                k: v for k, v in self._tokenizer_cache.items() if k[0] != str(path)
            }
            self._tokenizer_cache[key] = {
                "vocab_size": tokenizer.get_vocab_size(),
                "fingerprint": tokenizer_fingerprint(tokenizer),
                # Macro tokenizers compose over a smaller base (output) vocabulary.
                "macro": hasattr(tokenizer, "base_vocab_size"),
                "base_vocab_size": getattr(
                    tokenizer, "base_vocab_size", tokenizer.get_vocab_size()
                ),
            }
        return {"id": identifier, "label": label, **self._tokenizer_cache[key]}

    def tokenizers(self):
        records = []
        root = self.data_dir / "tokenizers"
        for path in sorted(root.glob("*.json")):
            if path.name.startswith("metadata"):
                continue
            try:
                path = contained(path, root)
                records.append(
                    self._tokenizer_record(path, f"tokenizer:{path.name}", path.stem)
                )
            except OSError, ValueError, KeyError, TypeError:
                continue
        for run in self.runs():
            if run.get("tokenizer"):
                records.append(
                    {**run["tokenizer"], "label": f"{run['name']} · saved tokenizer"}
                )
        return records

    def tokenizer_path(self, identifier):
        prefix, separator, name = identifier.partition(":")
        if not separator:
            raise ValueError("invalid tokenizer identifier")
        if prefix == "tokenizer":
            root = self.data_dir / "tokenizers"
            path = contained(root / artifact_name(name), root)
        elif prefix == "run":
            directory = self.run_path(name)
            metadata = read_json(directory / "model_meta.json")
            path = contained(directory / metadata["tokenizer_file"], directory)
        else:
            raise ValueError("unknown tokenizer identifier")
        if not path.is_file():
            raise FileNotFoundError("tokenizer was not found")
        return path

    def describe(self, directory):
        metadata = read_json(contained(directory / "model_meta.json", directory))
        manifest = metadata["manifest"]
        identity = manifest["identity"]
        parsed_config = model_config_from_dict(identity["model_config"])
        config = asdict(parsed_config)
        checkpoints = [
            {
                "id": p.name,
                "label": f"Step {checkpoint_step(p):,}",
                "step": checkpoint_step(p),
                "kind": "checkpoint",
            }
            for p in step_checkpoints(directory)
            if contained(p, directory).is_file()
        ]
        final_name = metadata.get("final_model_path", f"{metadata['model_name']}.pt")
        final = contained(directory / final_name, directory)
        finished = bool(metadata.get("training_finished"))
        if finished and final.is_file():
            checkpoints.append(
                {
                    "id": final.name,
                    "label": "Final model",
                    "step": metadata.get("global_step"),
                    "kind": "final",
                }
            )
        tokenizer_path = contained(directory / metadata["tokenizer_file"], directory)
        tokenizer = self._tokenizer_record(
            tokenizer_path, f"run:{directory.name}", directory.name
        )
        if tokenizer["vocab_size"] != config["vocab_size"]:
            raise ValueError("tokenizer/model vocabulary mismatch")
        if tokenizer["fingerprint"] != identity["tokenizer"]["fingerprint"]:
            raise ValueError("tokenizer does not match the saved run")
        can_resume = (
            manifest.get("manifest_version") == 3
            and identity.get("optimizer") == OPTIMIZER_IDENTITY
            and any(p["kind"] == "checkpoint" for p in checkpoints)
        )
        training = identity["training"]
        if can_resume:
            training = asdict(TrainingSettings.from_dict(training))
        return {
            "id": directory.name,
            "name": metadata["model_name"],
            "architecture": config["architecture"],
            "parameter_count": estimate_parameter_count(parsed_config),
            "config": config,
            "training": training,
            "tokenizer": tokenizer,
            "checkpoints": checkpoints,
            "default_checkpoint": checkpoints[-1]["id"] if checkpoints else None,
            "finished": finished,
            "global_step": metadata.get("global_step", 0),
            "tokens_seen": metadata.get("tokens_seen", 0),
            "validation": metadata.get("validation"),
            "validation_results": metadata.get("validation_results", []),
            "can_resume": can_resume,
        }

    def runs(self):
        records = []
        for path in sorted(self.models_dir.glob("*/model_meta.json")):
            try:
                directory = contained(path.parent, self.models_dir)
                records.append(self.describe(directory))
            except (OSError, ValueError, KeyError, TypeError) as error:
                records.append(
                    {
                        "id": path.parent.name,
                        "name": path.parent.name,
                        "error": str(error),
                        "checkpoints": [],
                        "can_resume": False,
                    }
                )
        return records

    def resolve(self, reference, *, resume=False):
        directory = self.run_path(reference.run_id)
        run = self.describe(directory)
        identifier = reference.checkpoint_id or run["default_checkpoint"]
        if resume and not run["can_resume"]:
            raise ValueError("this run cannot be resumed; start a fresh training run")
        choices = [
            item
            for item in run["checkpoints"]
            if not resume or item["kind"] == "checkpoint"
        ]
        if reference.checkpoint_id is None and resume and choices:
            identifier = choices[-1]["id"]
        if identifier not in {item["id"] for item in choices}:
            raise ValueError(
                "select an available training checkpoint"
                if resume
                else "select an available model checkpoint"
            )
        return directory, contained(directory / identifier, directory)
