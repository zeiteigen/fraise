"""Coordinate checkpoints, validation protection, and run metadata."""

from .artifacts import RunArtifacts
from .checkpoints import CheckpointStore
from .logging import RunLog


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
        validation.report()
        self.checkpoints.prune(self.keep_last, validation.protected_paths)

    def finish(self, runtime, state, validation):
        validation.report(wait=True)
        self.checkpoints.prune(self.keep_last, validation.protected_paths)
        self.artifacts.finish(
            runtime.model,
            state,
            validation=validation.latest,
            validation_failures=validation.failures,
        )
        self.log("Training finished.")
