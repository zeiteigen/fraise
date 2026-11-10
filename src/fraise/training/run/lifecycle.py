"""Coordinate checkpoints, validation protection, and run metadata."""

from .artifacts import RunArtifacts
from .checkpoints import CheckpointStore
from .logging import RunLog
from ..events import emit


class RunLifecycle:
    def __init__(
        self,
        root,
        name,
        manifest,
        tokenizer,
        config,
        settings,
        *,
        resume_from=None,
        config_paths=None,
        on_event=None,
    ):
        self.artifacts = RunArtifacts(
            root,
            name,
            manifest,
            tokenizer,
            config,
            settings,
            resume_from=resume_from,
            config_paths=config_paths,
        )
        self.checkpoints = CheckpointStore(
            self.artifacts.directory, self.artifacts.latest
        )
        self.log = RunLog(self.artifacts.directory)
        self.keep_last = settings.checkpoint_keep_last
        self.last_validation_step = None
        self.on_event = on_event
        self.previous_validation_results = list(
            self.artifacts.metadata.get("validation_results", [])
        )
        self.previous_validation_failures = list(
            self.artifacts.metadata.get("validation_failures", [])
        )
        self.scored_validation_steps = set()

    def restore(self, runtime):
        state = self.checkpoints.restore(
            runtime.model, runtime.optimizer, runtime.scheduler
        )
        if self.checkpoints.latest is not None:
            superseded = self.checkpoints.supersede_after(state.global_step)
            if superseded is not None:
                target, paths = superseded
                self.log(
                    f"Resuming from step {state.global_step}: moved {len(paths)} later "
                    f"checkpoint(s) from the abandoned continuation to {target.name}/."
                )
        return state

    def record_validation(self, *, validation, validation_failures, validation_results):
        self.artifacts.update(
            validation=validation or self.artifacts.metadata.get("validation"),
            validation_failures=self.previous_validation_failures + validation_failures,
            validation_results=self.previous_validation_results + validation_results,
        )
        emit(
            self.on_event,
            "validation",
            validation=validation,
            validation_results=validation_results,
        )
        # One flat event per completed evaluation, so JSON-lines readers get scalars.
        for result in validation_results:
            if (
                result["status"] == "completed"
                and result["step"] not in self.scored_validation_steps
            ):
                self.scored_validation_steps.add(result["step"])
                scores = {
                    key: result[key]
                    for key in ("loss", "bits_per_byte")
                    if key in result
                }
                emit(
                    self.on_event,
                    "validation_score",
                    global_step=result["step"],
                    validation=scores,
                )

    def checkpoint(self, runtime, state, validation):
        path = self.checkpoints.save(
            runtime.model, runtime.optimizer, runtime.scheduler, state
        )
        if self.last_validation_step != state.global_step:
            validation.submit(state.global_step, path)
            self.last_validation_step = state.global_step
        self.artifacts.update(
            latest_checkpoint=path.name,
            global_step=state.global_step,
            tokens_seen=state.tokens_seen,
            training_state=state.as_checkpoint(),
        )
        emit(self.on_event, "checkpoint", global_step=state.global_step, path=path.name)
        validation.report()
        self.checkpoints.prune(self.keep_last, validation.protected_paths)

    def stop(self, runtime, state, validation):
        """Persist a completed optimizer step without queuing more validation."""
        validation.cancel()
        path = self.checkpoints.save(
            runtime.model, runtime.optimizer, runtime.scheduler, state
        )
        self.artifacts.update(
            latest_checkpoint=path.name,
            global_step=state.global_step,
            tokens_seen=state.tokens_seen,
            training_state=state.as_checkpoint(),
            training_finished=False,
            training_status="stopped",
        )
        self.checkpoints.prune(self.keep_last)
        self.log(f"Training stopped at step {state.global_step}; saved {path.name}.")
        emit(self.on_event, "stopped", global_step=state.global_step, path=path.name)

    def finish(self, runtime, state, validation, *, stop_requested=None):
        validation.report(wait=True)
        if stop_requested is not None and stop_requested():
            self.stop(runtime, state, validation)
            return
        self.checkpoints.prune(self.keep_last, validation.protected_paths)
        self.artifacts.finish(
            runtime.model,
            state,
            validation=validation.latest or self.artifacts.metadata.get("validation"),
            validation_failures=self.artifacts.metadata.get("validation_failures", []),
        )
        self.log("Training finished.")
        emit(self.on_event, "completed", global_step=state.global_step)
