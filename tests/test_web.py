"""Exercise local API behavior against real tiny saved models."""

import importlib.util
import json
import sys
import time
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fixtures import tiny_config, tiny_data, tiny_settings, tiny_tokenizer

from fraise.data.sources import SOURCE_NAMES
from fraise.data.types import TrainingData
from fraise.io import read_json, write_json
from fraise.training.training import pretrain_model

HAS_WEB = importlib.util.find_spec("fastapi") is not None
if HAS_WEB:
    from fastapi.testclient import TestClient

    from fraise.web.app import create_app
    from fraise.web.generation import GenerationStream
    from fraise.web.schemas import GenerationRequest


@unittest.skipUnless(HAS_WEB, "install the web extra for local web API tests")
class WebTests(unittest.TestCase):
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
        self.data = TrainingData({"fineweb-edu": tiny_data().train["fixture"]}, None)
        for name in ("first", "second"):
            pretrain_model(
                self.tokenizer,
                self.data,
                self.settings,
                self.config,
                self.root / "models",
                name,
                device="cpu",
            )
        self.app = create_app(data_dir=self.root, device="cpu")
        self.client = self.enterContext(TestClient(self.app))

    def training_request(self, **changes):
        return {
            "model_config": asdict(self.config),
            "training_config": asdict(self.settings),
            "tokenizer_id": "tokenizer:tiny.json",
            "run_name": "third",
            **changes,
        }

    def test_catalog_assets_validation_and_path_boundaries(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        for name in (
            "app.js",
            "api.js",
            "autocomplete.js",
            "saved-inputs.js",
            "run-information.js",
            "compare.js",
            "train.js",
            "styles.css",
        ):
            self.assertEqual(self.client.get(f"/static/{name}").status_code, 200)
        catalog = self.client.get("/api/catalog").json()
        self.assertIn("transformer", {item["id"] for item in catalog["architectures"]})
        (self.root / "tokenizers" / "broken.json").write_text("not a tokenizer")
        self.assertEqual(self.client.get("/api/catalog").status_code, 200)
        runs = self.client.get("/api/runs").json()["runs"]
        self.assertEqual({run["id"] for run in runs}, {"first", "second"})
        self.assertTrue(all(run["can_resume"] for run in runs))
        self.assertTrue(all(run["parameter_count"] > 0 for run in runs))
        response = self.client.post(
            "/api/config/validate", json=self.training_request()
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertGreater(response.json()["parameter_count"], 0)
        bad = self.training_request(run_name="../escape")
        self.assertEqual(self.client.post("/api/jobs", json=bad).status_code, 400)
        outside = self.root / "outside.json"
        self.tokenizer.save(str(outside))
        (self.root / "tokenizers" / "linked.json").symlink_to(outside)
        bad = self.training_request(tokenizer_id="tokenizer:linked.json")
        self.assertEqual(
            self.client.post("/api/config/validate", json=bad).status_code, 400
        )
        self.assertEqual(
            self.client.post(
                "/api/jobs",
                json=self.training_request(),
                headers={"Origin": "https://other.example"},
            ).status_code,
            403,
        )

    def test_streamed_comparison_runs_same_seed_sequentially(self):
        response = self.client.post(
            "/api/generate",
            json={
                "models": [{"run_id": "first"}, {"run_id": "second"}],
                "prompt": "a",
                "max_new_tokens": 5,
                "temperature": 1,
                "seed": 42,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        results = []
        for block in response.text.strip().split("\n\n"):
            kind, data = block.split("\n", 1)
            if kind == "event: result":
                results.append(json.loads(data.removeprefix("data: ")))
        self.assertEqual([result["index"] for result in results], [0, 1])
        self.assertEqual(results[0]["text"], results[1]["text"])
        self.assertGreater(results[0]["tokens_per_second"], 0)
        self.assertIn("event: done", response.text)
        self.assertIsNone(self.app.state.gate.busy)

    def test_generation_uses_raw_input_and_rejects_conversation_fields(self):
        from fraise.inference.streaming import iter_generate

        text = "  a\nUser: b\nAssistant: c  "
        with patch(
            "fraise.web.generation.iter_generate", wraps=iter_generate
        ) as generate:
            response = self.client.post(
                "/api/generate",
                json={
                    "models": [{"run_id": "first"}],
                    "prompt": text,
                    "max_new_tokens": 1,
                },
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(generate.call_args.args[2], text)
        for extra in (
            {"mode": "conversation"},
            {"messages": [{"role": "user", "content": "a"}]},
        ):
            response = self.client.post(
                "/api/generate",
                json={
                    "models": [{"run_id": "first"}],
                    "prompt": "a",
                    **extra,
                },
            )
            self.assertEqual(response.status_code, 422, response.text)

    def test_compute_conflicts_resume_identity_and_invalid_sampling(self):
        self.app.state.gate.acquire("training", "owned")
        try:
            self.assertEqual(
                self.client.post(
                    "/api/generate",
                    json={
                        "models": [{"run_id": "first"}],
                        "prompt": "a",
                    },
                ).status_code,
                409,
            )
            self.assertEqual(
                self.client.post("/api/jobs", json=self.training_request()).status_code,
                409,
            )
        finally:
            self.app.state.gate.release("owned")
        bad = self.training_request(
            run_name="first",
            resume={"run_id": "first"},
            model_config=asdict(tiny_config(num_layers=2)),
        )
        self.assertEqual(
            self.client.post("/api/config/validate", json=bad).status_code, 400
        )
        allowed = self.training_request(
            run_name="first",
            resume={"run_id": "first"},
            training_config=asdict(self.settings) | {"log_interval_steps": 2},
        )
        self.assertEqual(
            self.client.post("/api/config/validate", json=allowed).status_code, 200
        )
        self.assertEqual(
            self.client.post(
                "/api/generate",
                json={
                    "models": [{"run_id": "first"}],
                    "prompt": "a",
                    "max_new_tokens": True,
                },
            ).status_code,
            422,
        )
        self.assertEqual(
            self.client.post(
                "/api/generate",
                json={
                    "models": [{"run_id": "../outside"}],
                    "prompt": "a",
                },
            ).status_code,
            400,
        )

    def test_disconnect_cancels_worker_and_releases_compute_gate(self):
        request = GenerationRequest(
            models=[{"run_id": "first"}], prompt="a", max_new_tokens=4096
        )
        references = [
            (request.models[0], *self.app.state.catalog.resolve(request.models[0]))
        ]
        stream = GenerationStream(
            request, references, self.app.state.inference, self.app.state.gate
        )
        stream.cancelled.set()
        stream.thread.join(timeout=10)
        self.assertFalse(stream.thread.is_alive())
        self.assertIsNone(self.app.state.gate.busy)

    def test_job_snapshots_progress_stop_and_shutdown(self):
        import subprocess

        real_popen = subprocess.Popen
        script = """
import json, pathlib, signal, sys, time
event_path = pathlib.Path(sys.argv[1])
signal.signal(signal.SIGINT, lambda *_: None)
with event_path.open('a') as out:
 out.write(json.dumps({'type':'stage','phase':'training'})+'\\n'); out.flush()
 while not (event_path.parent / 'stop').exists(): time.sleep(.01)
 out.write(json.dumps({'type':'stopped','path':'checkpoint_step_00000001.pt'})+'\\n')
"""

        def launch(command, **kwargs):
            directory = Path(command[command.index("--events-file") + 1]).parent
            return real_popen(
                [sys.executable, "-u", "-c", script, str(directory / "events.jsonl")],
                **kwargs,
            )

        with patch("fraise.web.jobs.subprocess.Popen", side_effect=launch):
            response = self.client.post("/api/jobs", json=self.training_request())
        self.assertEqual(response.status_code, 201, response.text)
        identifier = response.json()["id"]
        deadline = time.monotonic() + 5
        while self.client.get(f"/api/jobs/{identifier}").json()["phase"] != "training":
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.02)
        directory = self.app.state.jobs.storage.directory(identifier)
        self.assertEqual(read_json(directory / "model.json"), asdict(self.config))
        self.assertEqual(
            self.client.post(f"/api/jobs/{identifier}/stop", json={}).status_code, 200
        )
        while (job := self.client.get(f"/api/jobs/{identifier}").json())[
            "status"
        ] == "stopping":
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.02)
        self.assertEqual(job["status"], "stopped")
        self.assertEqual(job["latest_checkpoint"], "checkpoint_step_00000001.pt")
        self.assertIsNone(self.app.state.gate.busy)

    def test_launch_failure_records_error_and_releases_device(self):
        with patch(
            "fraise.web.jobs.subprocess.Popen", side_effect=OSError("launch failed")
        ):
            response = self.client.post("/api/jobs", json=self.training_request())
        self.assertEqual(response.status_code, 400)
        self.assertIsNone(self.app.state.gate.busy)
        self.assertIsNone(self.app.state.jobs.active)
        jobs = self.client.get("/api/jobs").json()["jobs"]
        self.assertEqual(jobs[0]["status"], "failed")

    def test_unfinished_job_is_marked_interrupted_on_restart(self):
        directory = self.root / "web" / "jobs" / "abandoned"
        write_json(directory / "job.json", {"id": "abandoned", "status": "running"})
        restarted = create_app(data_dir=self.root, device="cpu")
        self.assertEqual(restarted.state.jobs.get("abandoned")["status"], "interrupted")

    def test_real_training_subprocess_can_finish_and_resume_through_api(self):
        import subprocess

        real_popen = subprocess.Popen
        # Keep the actual CLI, signal controller, training, and persistence path;
        # replace only network dataset retrieval with an offline text fixture.
        script = """
from datasets import Dataset
from fraise.data.types import TrainingData
from fraise.cli import train
train.load_training_data = lambda *a, **k: TrainingData(
    {'fineweb-edu': Dataset.from_dict({'text': ['a b c d e']})}, None)
train.main()
"""

        def launch(command, **kwargs):
            return real_popen(
                [sys.executable, "-u", "-c", script, *command[4:]], **kwargs
            )

        def wait_for_job(identifier):
            deadline = time.monotonic() + 20
            while True:
                job = self.client.get(f"/api/jobs/{identifier}").json()
                if job["status"] not in {"starting", "running", "stopping"}:
                    return job
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.05)

        with patch("fraise.web.jobs.subprocess.Popen", side_effect=launch):
            response = self.client.post("/api/jobs", json=self.training_request())
            self.assertEqual(response.status_code, 201, response.text)
            job = wait_for_job(response.json()["id"])
            self.assertEqual(job["status"], "completed", job.get("error"))
            self.assertTrue((self.root / "models" / "third" / "third.pt").is_file())
            resumed = self.training_request(
                run_name="third", resume={"run_id": "third"}
            )
            response = self.client.post("/api/jobs", json=resumed)
            self.assertEqual(response.status_code, 201, response.text)
            job = wait_for_job(response.json()["id"])
            self.assertEqual(job["status"], "completed", job.get("error"))
        self.assertIsNone(self.app.state.gate.busy)


if __name__ == "__main__":
    unittest.main()
