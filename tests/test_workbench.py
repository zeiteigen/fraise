"""The workbench behind fraise's AblationKit panels, against real tiny saved models."""

import base64
import gzip
import json
import os
import signal
import subprocess
import sys
import time
import types
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fixtures import tiny_config, tiny_data, tiny_settings, tiny_tokenizer
from macro_fixtures import tiny_macro_tokenizer

from fraise.cli.launch import train_arguments
from fraise.data.sources import SOURCE_NAMES
from fraise.data.types import TrainingData
from fraise.hierarchy import hierarchy_from_macro_tokenizer
from fraise.io import read_json, write_json
from fraise.training.events import AblationKitMetrics, fan_out
from fraise.training.training import pretrain_model
from fraise.workbench.catalog import Catalog
from fraise.workbench.generation import InferenceSession, generate_sequentially
from fraise.workbench.payload import ENCODING, encode
from fraise.workbench.requests import GenerationRequest, TokenizeRequest, TrainingRequest
from fraise.workbench.saved_inputs import LIMIT, SavedInputs
from fraise.workbench.tokenization import TokenizationWorkbench
from fraise.workbench.training import training_catalog, validate_training, write_plan

ROOT = Path(__file__).resolve().parents[1]

# The real train CLI, with dataset retrieval replaced by an offline text fixture.
OFFLINE_TRAIN = """
import sys
from datasets import Dataset
from fraise.data.types import TrainingData
from fraise.cli import train
train.load_training_data = lambda *a, **k: TrainingData(
    {{'fineweb-edu': Dataset.from_dict({{'text': ['a b c d e'] * {rows}}})}}, None)
{main}
"""


def offline(main, rows=1):
    return OFFLINE_TRAIN.format(rows=rows, main=main)


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.tokenizer = tiny_tokenizer()
        (self.root / "tokenizers").mkdir()
        self.tokenizer.save(str(self.root / "tokenizers" / "tiny.json"))
        self.config = tiny_config()
        counts = {name: 4 if name == "fineweb-edu" else 0 for name in SOURCE_NAMES}
        self.settings = tiny_settings(
            dataset_token_counts=counts,
            use_compile=False,
            max_validation_tokens=0,
            log_interval_steps=1,
        )
        data = TrainingData({"fineweb-edu": tiny_data().train["fixture"]}, None)
        for name in ("first", "second"):
            pretrain_model(
                self.tokenizer, data, self.settings, self.config, self.root / "models", name, device="cpu"
            )
        self.catalog = Catalog(self.root)

    def training_request(self, **changes):
        return TrainingRequest.parse(
            {
                "model_config": asdict(self.config),
                "training_config": asdict(self.settings),
                "tokenizer_id": "tokenizer:tiny.json",
                "run_name": "third",
                **changes,
            }
        )

    def run_cli(self, script, *args, timeout=60):
        environment = {**os.environ, "HF_HUB_OFFLINE": "1"}
        return subprocess.run(
            [sys.executable, "-u", "-c", script, *map(str, args)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
            cwd=ROOT,
        )

    def test_catalog_lists_runs_and_keeps_paths_inside_its_directories(self):
        (self.root / "tokenizers" / "broken.json").write_text("not a tokenizer")
        runs = self.catalog.runs()
        self.assertEqual({run["id"] for run in runs}, {"first", "second"})
        self.assertTrue(all(run["can_resume"] and run["parameter_count"] > 0 for run in runs))
        self.assertIn("tokenizer:tiny.json", {t["id"] for t in self.catalog.tokenizers()})
        offered = training_catalog(self.catalog, {"resolved": "cpu"})
        self.assertIn("transformer", {item["id"] for item in offered["architectures"]})
        with self.assertRaises(ValueError):
            self.catalog.resolve(GenerationRequest.parse({"models": [{"run_id": "../outside"}], "prompt": "a"}).models[0])
        outside = self.root / "outside.json"
        self.tokenizer.save(str(outside))
        (self.root / "tokenizers" / "linked.json").symlink_to(outside)
        with self.assertRaises(ValueError):
            validate_training(self.training_request(tokenizer_id="tokenizer:linked.json"), self.catalog, "cpu")
        with self.assertRaises(ValueError):
            validate_training(self.training_request(run_name="../escape"), self.catalog, "cpu")

    def test_generation_runs_checkpoints_in_turn_with_the_same_seed_and_raw_prompt(self):
        from fraise.inference.streaming import iter_generate

        request = GenerationRequest.parse(
            {
                "models": [{"run_id": "first"}, {"run_id": "second"}],
                "prompt": "  a\nUser: b\nAssistant: c  ",
                "max_new_tokens": 5,
                "temperature": 1,
                "seed": 42,
            }
        )
        references = [(ref, *self.catalog.resolve(ref)) for ref in request.models]
        events = []
        with patch("fraise.workbench.generation.iter_generate", wraps=iter_generate) as generate:
            results = generate_sequentially(
                request, references, InferenceSession("cpu"), lambda kind, data: events.append((kind, data))
            )
        self.assertEqual(generate.call_args.args[2], request.prompt)
        self.assertEqual([r["index"] for r in results], [0, 1])
        self.assertEqual(results[0]["text"], results[1]["text"])
        self.assertGreater(results[0]["tokens_per_second"], 0)
        kinds = [kind for kind, _ in events]
        self.assertEqual(kinds[0], "loading")
        self.assertIn("output", kinds)
        self.assertNotIn("error", kinds)
        cancelled = generate_sequentially(request, references, InferenceSession("cpu"), lambda *a: None, lambda: True)
        self.assertEqual(cancelled, [])

    def test_generation_requests_keep_fraise_web_bounds(self):
        base = {"models": [{"run_id": "first"}], "prompt": "a"}
        for change in (
            {"mode": "conversation"},
            {"messages": [{"role": "user", "content": "a"}]},
            {"max_new_tokens": True},
            {"max_new_tokens": 4097},
            {"top_p": 0},
            {"temperature": float("nan")},
            {"models": []},
            {"models": [{"run_id": "first"}] * 9},
            {"prompt": ""},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                GenerationRequest.parse(base | change)
        parsed = GenerationRequest.parse(base | {"top_p": None, "top_k": 5})
        self.assertEqual((parsed.top_p, parsed.top_k, parsed.seed), (None, 5, 1337))

    def test_resume_keeps_identity_and_allows_operational_changes(self):
        with self.assertRaises(ValueError):
            validate_training(
                self.training_request(
                    run_name="first", resume={"run_id": "first"}, model_config=asdict(tiny_config(num_layers=2))
                ),
                self.catalog,
                "cpu",
            )
        allowed = self.training_request(
            run_name="first",
            resume={"run_id": "first"},
            training_config=asdict(self.settings) | {"log_interval_steps": 2},
        )
        normalized, checkpoint = validate_training(allowed, self.catalog, "cpu")
        self.assertGreater(normalized["parameter_count"], 0)
        self.assertEqual(checkpoint.parent.name, "first")
        with self.assertRaises(ValueError):
            validate_training(self.training_request(run_name="first"), self.catalog, "cpu")

    def test_plans_train_and_resume_through_the_launch_entrypoint(self):
        plans = self.root / "launch"
        fresh = write_plan(self.training_request(), self.catalog, "cpu", plans)
        directory = Path(fresh["directory"])
        self.assertEqual(directory.parent, plans / "third")
        self.assertEqual(read_json(directory / "model.json"), asdict(self.config))
        self.assertEqual(fresh["mode"], "fresh")
        arguments = train_arguments(directory)
        self.assertIn("--ablationkit-metrics", arguments)
        self.assertNotIn("--resume", arguments)
        script = offline("from fraise.cli import launch; launch.main(sys.argv[1:])")
        finished = self.run_cli(script, directory)
        self.assertEqual(finished.returncode, 0, finished.stderr)
        self.assertTrue((self.root / "models" / "third" / "third.pt").is_file())
        resumed = write_plan(
            self.training_request(resume={"run_id": "third"}), self.catalog, "cpu", plans
        )
        self.assertEqual(resumed["mode"], "resume")
        self.assertNotEqual(resumed["directory"], fresh["directory"])
        self.assertIn("--resume", train_arguments(resumed["directory"]))
        finished = self.run_cli(script, resumed["directory"])
        self.assertEqual(finished.returncode, 0, finished.stderr)

    def test_sigterm_saves_a_resumable_checkpoint_after_the_current_step(self):
        counts = {name: 6000 if name == "fineweb-edu" else 0 for name in SOURCE_NAMES}
        settings = asdict(self.settings) | {
            "dataset_token_counts": counts,
            "max_train_tokens": 6000,
            "checkpoint_interval_steps": 1000000,
            "log_interval_steps": 1,
        }
        write_json(self.root / "model.json", asdict(self.config))
        write_json(self.root / "training.json", settings)
        events = self.root / "events.jsonl"
        log = (self.root / "output.log").open("wb")
        self.addCleanup(log.close)
        script = offline("train.main()", rows=3000)
        process = subprocess.Popen(
            [
                sys.executable, "-u", "-c", script,
                "--model-config", str(self.root / "model.json"),
                "--training-hyperparameters", str(self.root / "training.json"),
                "--tokenizer-file", str(self.root / "tokenizers" / "tiny.json"),
                "--model-name", "stopped",
                "--data-dir", str(self.root),
                "--device", "cpu",
                "--events-file", str(events),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "HF_HUB_OFFLINE": "1"},
            cwd=ROOT,
        )
        self.addCleanup(lambda: process.poll() is None and process.kill())
        output = lambda: (self.root / "output.log").read_text(errors="replace")[-4000:]
        deadline = time.monotonic() + 60
        while not (events.exists() and '"progress"' in events.read_text()):
            if process.poll() is not None:
                self.fail(output())
            self.assertLess(time.monotonic(), deadline, output())
            time.sleep(0.05)
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=30)
        self.assertEqual(process.returncode, 0, output())
        stopped = [json.loads(line) for line in events.read_text().splitlines() if '"stopped"' in line]
        self.assertEqual(len(stopped), 1, output())
        self.assertTrue((self.root / "models" / "stopped" / stopped[0]["path"]).is_file())
        self.assertTrue(next(run for run in Catalog(self.root).runs() if run["id"] == "stopped")["can_resume"])

    def test_ablationkit_metrics_match_metrics_file_names(self):
        logged = []
        sdk = types.SimpleNamespace(log_metrics=lambda metrics, step: logged.append((step, metrics)))
        with patch.dict(sys.modules, {"ablationkit_sdk": sdk}):
            sink = AblationKitMetrics()
        callback = fan_out(None, sink)
        callback({"type": "stage", "phase": "training"})
        callback({"type": "progress", "global_step": 3, "loss": 2.5, "tokens_seen": 12})
        callback({"type": "validation_score", "global_step": 3, "validation": {"loss": 2.0, "bits_per_byte": 1.5}})
        callback({"type": "checkpoint", "global_step": 3, "path": "checkpoint_step_00000003.pt"})
        self.assertEqual(
            logged,
            [
                (3, {"loss": 2.5, "tokens_seen": 12}),
                (3, {"validation/loss": 2.0, "validation/bits_per_byte": 1.5}),
            ],
        )
        self.assertIsNone(fan_out(None, None))
        with patch.dict(sys.modules, {"ablationkit_sdk": None}):
            AblationKitMetrics()({"type": "progress", "global_step": 1, "loss": 1.0})

    def test_saved_inputs_keep_the_newest_and_clear(self):
        store = SavedInputs(self.root / "workspace" / "saved-inputs.json")
        self.assertEqual(store.items(), [])
        first = store.remember({"text": "a", "ref": {"run_id": "first"}})
        for index in range(LIMIT + 2):
            store.remember({"text": f"t{index}", "ref": {"run_id": "first"}})
        updated = store.remember({**first, "text": "edited"})
        items = store.items()
        self.assertEqual(len(items), LIMIT)
        self.assertEqual((items[0]["id"], items[0]["text"]), (updated["id"], "edited"))
        with self.assertRaises(ValueError):
            store.remember({"text": "no checkpoint", "ref": {}})
        store.clear()
        self.assertEqual(store.items(), [])

    def test_payloads_round_trip_and_refuse_oversized_results(self):
        value = {"rows": [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}], "n": None}
        data = encode(value)
        self.assertEqual(data["encoding"], ENCODING)
        packed = json.loads(gzip.decompress(base64.b64decode(data["payload"])))
        self.assertEqual(packed["rows"], {"__cols__": ["a", "b"], "__rows__": [[1, "x"], [2, "y"]]})
        noise = base64.b64encode(os.urandom(300_000)).decode()
        with self.assertRaisesRegex(ValueError, "shorten"):
            encode({"text": noise}, too_large="shorten it")


class TokenizationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        tokenizer = tiny_macro_tokenizer(vocab_size=20, max_macro_length=4)
        (root / "tokenizers").mkdir()
        (root / "tokenizers" / "macro.json").write_text(tokenizer.to_str())
        tokenizer.base.save(str(root / "tokenizers" / "base.json"))
        definition = hierarchy_from_macro_tokenizer(tokenizer, (8, 12, 17), [2, 3])
        definition.save(root / "hierarchies" / "three.npz")
        self.catalog = Catalog(root)
        self.workbench = TokenizationWorkbench(self.catalog)

    def analyze(self, text, configurations, **extra):
        request = TokenizeRequest.parse({"text": text, "configurations": configurations, **extra})
        return self.workbench.analyze(request)["results"]

    def test_compares_hierarchies_and_reports_errors_per_configuration(self):
        self.assertEqual(self.workbench.hierarchy_files(), [{"id": "three.npz", "label": "three"}])
        macro = "tokenizer:macro.json"
        two, merges, file, base, wrong = self.analyze(
            "a b c d e a b",
            [
                {"tokenizer_id": macro},
                {"tokenizer_id": macro, "hierarchy": {"kind": "merges", "level_sizes": [8, 12, 17]}},
                {"tokenizer_id": macro, "hierarchy": {"kind": "file", "file_id": "three.npz"}},
                {"tokenizer_id": "tokenizer:base.json"},
                {"tokenizer_id": macro, "hierarchy": {"kind": "merges", "level_sizes": [8, 9]}},
            ],
            add_special_tokens=True,
        )
        self.assertEqual(len(two["levels"]), 2)
        self.assertEqual(len(merges["levels"]), 3)
        self.assertEqual(merges["rows"], file["rows"])
        self.assertEqual(two["input_tokens"], merges["input_tokens"])
        self.assertEqual(base["levels"], [])
        self.assertFalse(base["macro"])
        self.assertTrue(two["macro"])
        kinds = {t["id"]: (t["macro"], t["base_vocab_size"]) for t in self.catalog.tokenizers()}
        self.assertEqual(kinds[macro], (True, 8))
        self.assertEqual(kinds["tokenizer:base.json"], (False, 8))
        self.assertGreater(len(base["input_tokens"]), len(two["input_tokens"]))
        self.assertIn("level sizes", wrong["error"])
        (rejected,) = self.analyze(
            "a", [{"tokenizer_id": macro, "hierarchy": {"kind": "file", "file_id": "../x.npz"}}]
        )
        self.assertIn("error", rejected)

    def test_invalid_merge_configurations_return_individual_errors(self):
        plain, levels, valid = self.analyze(
            "a b c",
            [
                {"tokenizer_id": "tokenizer:base.json", "hierarchy": {"kind": "merges", "level_sizes": [8, 12, 17]}},
                {"tokenizer_id": "tokenizer:macro.json", "hierarchy": {"kind": "merges", "level_sizes": [8, 20, 17]}},
                {"tokenizer_id": "tokenizer:macro.json"},
            ],
        )
        self.assertIn("needs a macro tokenizer", plain["error"])
        self.assertIn("strictly increase", levels["error"])
        self.assertNotIn("error", valid)

    def test_requests_refuse_unknown_fields_and_bad_shapes(self):
        for request in (
            {"text": "a", "configurations": [{"tokenizer_id": "x", "colour": "red"}]},
            {"text": "a", "configurations": [{"hierarchy": {"kind": "magic"}}]},
            {"text": "a", "configurations": [{"hierarchy": {"level_sizes": [True]}}]},
            {"text": "a", "configurations": []},
            {"text": "", "configurations": [{}]},
        ):
            with self.subTest(request=request), self.assertRaises(ValueError):
                TokenizeRequest.parse(request)


if __name__ == "__main__":
    unittest.main()
