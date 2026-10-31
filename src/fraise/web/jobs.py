"""Own a single subprocess training job and its lifecycle."""

import os
import signal
import subprocess
import sys
from threading import RLock, Thread
from time import time
from uuid import uuid4

from fraise.io import write_json

from .configuration import validate_training
from .job_storage import JobStorage, TERMINAL


class TrainingJobs:
    def __init__(self, catalog, gate, inference, device):
        self.catalog, self.gate, self.inference, self.device = (
            catalog,
            gate,
            inference,
            device,
        )
        self.storage = JobStorage(catalog.data_dir / "web" / "jobs")
        self.lock = RLock()
        self.active = None
        self.process = None
        self.monitor = None
        for job in self.storage.jobs():
            if job["status"] not in TERMINAL:
                job.update(
                    status="interrupted",
                    updated_at=time(),
                    error="Server exited before this job finished; resume a saved checkpoint.",
                )
                self.storage.save(job)

    def start(self, request):
        normalized, resume = validate_training(request, self.catalog, self.device)
        identifier = uuid4().hex
        self.gate.acquire("training", identifier)
        process, job = None, None
        try:
            with self.lock:
                self.inference.unload()
                directory = self.storage.directory(identifier)
                directory.mkdir(parents=True)
                write_json(directory / "model.json", normalized["model_config"])
                write_json(directory / "training.json", normalized["training_config"])
                job = {
                    "id": identifier,
                    "status": "starting",
                    "phase": "starting",
                    "run_id": request.resume.run_id
                    if request.resume
                    else request.run_name,
                    "created_at": time(),
                    "updated_at": time(),
                }
                self.storage.save(job)
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "fraise.cli.train",
                    "--model-config",
                    str(directory / "model.json"),
                    "--training-hyperparameters",
                    str(directory / "training.json"),
                    "--tokenizer-file",
                    str(self.catalog.tokenizer_path(request.tokenizer_id)),
                    "--model-name",
                    request.run_name,
                    "--data-dir",
                    str(self.catalog.data_dir),
                    "--save-path",
                    str(self.catalog.models_dir),
                    "--device",
                    self.device,
                    "--stop-file",
                    str(directory / "stop"),
                    "--events-file",
                    str(directory / "events.jsonl"),
                    "--parent-pid",
                    str(os.getpid()),
                ]
                if resume is not None:
                    command.extend(["--resume", str(resume)])
                with (directory / "output.log").open("wb") as output:
                    process = subprocess.Popen(
                        command,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                self.active, self.process = identifier, process
                job.update(status="running", updated_at=time())
                self.storage.save(job)
                self.monitor = Thread(
                    target=self._watch, args=(identifier, process), daemon=True
                )
                self.monitor.start()
                return job
        except BaseException:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            self.active, self.process = None, None
            self.gate.release(identifier)
            if job is not None:
                job.update(
                    status="failed",
                    updated_at=time(),
                    error="Could not launch training",
                )
                self.storage.save(job)
            raise

    def _refresh(self, identifier):
        job = self.storage.get(identifier)
        events = self.storage.events(identifier)
        changed = False
        for event in reversed(events):
            if event["type"] == "stage":
                if job["phase"] != event["phase"]:
                    job["phase"] = event["phase"]
                    changed = True
                break
        for event in reversed(events):
            if event["type"] in ("checkpoint", "stopped") and event.get("path"):
                if job.get("latest_checkpoint") != event["path"]:
                    job["latest_checkpoint"] = event["path"]
                    changed = True
                break
        if changed:
            job["updated_at"] = time()
            self.storage.save(job)
        return job, events

    def _watch(self, identifier, process):
        try:
            while True:
                try:
                    code = process.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    with self.lock:
                        self._refresh(identifier)
                    continue
                with self.lock:
                    job, events = self._refresh(identifier)
                    kinds = {event["type"] for event in events}
                    if code == 0 and "completed" in kinds:
                        status = "completed"
                    elif "stopped" in kinds or (
                        job["status"] == "stopping" and code in (0, -signal.SIGINT, 130)
                    ):
                        status = "stopped"
                    else:
                        status = "failed"
                    job.update(
                        status=status, phase=status, updated_at=time(), exit_code=code
                    )
                    if status == "failed":
                        job["error"] = (
                            self.storage.snapshot(identifier)["logs"][-4000:]
                            or f"Training exited with code {code}"
                        )
                    self.storage.save(job)
                    self.active, self.process = None, None
                return
        except Exception as error:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            with self.lock:
                self.active, self.process = None, None
                job = self.storage.get(identifier)
                job.update(status="failed", error=str(error), updated_at=time())
                self.storage.save(job)
        finally:
            self.gate.release(identifier)

    def get(self, identifier, cursor=0):
        with self.lock:
            if identifier == self.active:
                self._refresh(identifier)
            return self.storage.snapshot(identifier, cursor)

    def list(self):
        with self.lock:
            return self.storage.jobs()

    def stop(self, identifier):
        with self.lock:
            job = self.storage.get(identifier)
            if job["status"] in TERMINAL:
                return job
            if identifier != self.active or self.process is None:
                raise ValueError("training job is not owned by this server")
            (self.storage.directory(identifier) / "stop").touch()
            job.update(status="stopping", updated_at=time())
            self.storage.save(job)
            try:
                self.process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
            return job

    def close(self):
        with self.lock:
            identifier, process = self.active, self.process
        if process is None:
            return
        self.stop(identifier)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2)
        if self.monitor is not None:
            self.monitor.join(timeout=3)
