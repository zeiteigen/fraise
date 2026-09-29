"""Keep CPU checkpoint validation current without building a work backlog."""

from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path


class ValidationQueue:
    """Run one validation while retaining only the newest requested successor."""

    def __init__(
        self,
        submit: Callable[[Path], Future[float | None]],
        log: Callable[[str], None],
    ) -> None:
        self._submit = submit
        self._log = log
        self._active: tuple[int, Path, Future[float | None]] | None = None
        self._queued: tuple[int, Path] | None = None
        self.latest: tuple[int, float] | None = None
        self.failures: list[dict] = []

    def _start(self, step: int, checkpoint_path: Path) -> None:
        try:
            future = self._submit(checkpoint_path)
        except Exception as exc:
            self.failures.append({"step": step, "error": str(exc)})
            self._log(f"Could not queue validation at global step {step}: {exc}")
        else:
            self._active = (step, checkpoint_path, future)

    def queue(self, step: int, checkpoint_path: Path) -> None:
        if self._active is None:
            self._start(step, checkpoint_path)
        else:
            self._queued = (step, checkpoint_path)

    def report(self, *, wait: bool = False) -> None:
        while self._active is not None:
            step, _, future = self._active
            if not wait and not future.done():
                return
            try:
                validation_loss = future.result()
            except Exception as exc:
                self.failures.append({"step": step, "error": str(exc)})
                self._log(f"Validation failed at global step {step}: {exc}")
            else:
                if validation_loss is None:
                    self._log(f"Validation loss unavailable at global step {step}.")
                else:
                    self.latest = (step, validation_loss)
                    self._log(
                        f"Validation loss at global step {step}: {validation_loss:.6f}"
                    )
            self._active = None
            if self._queued is not None:
                queued_step, queued_path = self._queued
                self._queued = None
                self._start(queued_step, queued_path)

    def protected_paths(self) -> set[Path]:
        paths = set()
        if self._active is not None:
            paths.add(self._active[1])
        if self._queued is not None:
            paths.add(self._queued[1])
        return paths
