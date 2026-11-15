"""Train a registered language model over the prepared epoch plan."""

from pathlib import Path

from fraise.architectures import ModelConfigProtocol
from fraise.config.compatibility import check_compatibility
from fraise.config.training import TrainingSettings
from fraise.data.types import TrainingData
from fraise.tokenizer.loading import special_token_id

from .diagnostics import TrainingDiagnostics
from .events import emit
from .pipeline.batch_types import target_units
from .pipeline.batches import iter_dataset_packed_batches
from .pipeline.budgets import epoch_seed
from .pipeline.prefetch import prefetch_batches
from .pipeline.preparation import prepare_data
from .run.artifacts import inspect_run_destination
from .run.lifecycle import RunLifecycle
from .run.manifest import run_manifest
from .runtime import build_runtime, resolve_device
from .validation.measures import validation_measures
from .validation.runner import ValidationRunner


def pretrain_model(
    tokenizer,
    dataset: TrainingData,
    settings: TrainingSettings,
    model_config: ModelConfigProtocol,
    save_path: str | Path,
    model_name="fraise_model",
    *,
    resume_from: str | Path | None = None,
    config_paths: dict[str, Path] | None = None,
    device=None,
    stop_requested=None,
    on_event=None,
):
    check_compatibility(tokenizer, model_config, settings)
    save_path = Path(save_path)
    resume_from = Path(resume_from).expanduser() if resume_from is not None else None
    device = resolve_device(device, settings)
    manifest = run_manifest(dataset, tokenizer, model_config, settings, device)
    # Reject a reused run name or a mismatched resume before tokenizing and compiling.
    inspect_run_destination(save_path, model_name, manifest, resume_from)
    emit(on_event, "stage", phase="preparing")
    prepared = prepare_data(dataset, tokenizer, settings)
    if stop_requested is not None and stop_requested():
        emit(on_event, "stopped", phase="preparing")
        return None
    pad_id = special_token_id(tokenizer, "<|pad|>")
    runtime = build_runtime(
        model_config,
        settings,
        prepared.steps,
        device,
        initialize_weights=resume_from is None,
        tokenizer=tokenizer,
    )
    run = RunLifecycle(
        save_path,
        model_name,
        manifest,
        tokenizer,
        model_config,
        settings,
        resume_from=resume_from,
        config_paths=config_paths,
        on_event=on_event,
    )
    state = run.restore(runtime)
    runtime.log = run.log
    run.artifacts.update(training_status="running")
    emit(on_event, "stage", phase="training")
    total_train_tokens = prepared.total_targets
    if prepared.unit_lengths is not None:
        run.log(
            "Budgets count base tokens covered by macro targets; each source ends "
            f"with the macro that reaches its count ({total_train_tokens:,} exact)."
        )
    run.log(
        f"Training {settings.max_train_tokens:,} targets per epoch for {settings.num_epochs} "
        f"epochs ({total_train_tokens:,} total) from step {state.global_step}; "
        f"context {settings.sequence_length}, batch {settings.batch_size}, minibatch {settings.minibatch_size}, "
        f"accumulation {settings.gradient_accumulation_steps}, device {runtime.device}."
    )
    run.log(
        f"Execution: torch.compile {settings.use_compile}, "
        f"compiler workers {settings.compile_threads if settings.use_compile else 0}, "
        f"strict determinism {settings.deterministic}, "
        f"attention backend {'math' if settings.deterministic else 'auto'}."
    )
    with ValidationRunner(
        model_config,
        prepared.validation,
        settings,
        pad_id,
        run.log,
        measures=validation_measures(tokenizer),
        on_update=run.record_validation,
        stop_requested=stop_requested,
    ) as validation:
        diagnostics = TrainingDiagnostics(
            runtime,
            state,
            settings.log_interval_steps,
            run.log,
            on_event=on_event,
            total_tokens=total_train_tokens,
        )
        for epoch in range(state.epoch, settings.num_epochs):
            if stop_requested is not None and stop_requested():
                run.stop(runtime, state, validation)
                return runtime.model
            epoch_targets = sum(prepared.budgets[epoch].values())
            runtime.model.train()
            batches = iter_dataset_packed_batches(
                prepared.train,
                prepared.budgets[epoch],
                settings.sequence_length,
                settings.packing_stride,
                pad_id,
                batch_size=settings.batch_size,
                start_batch_index=state.next_batch_index,
                seed=epoch_seed(settings, epoch),
                shuffle_per_epoch=settings.shuffle_per_epoch,
                shuffle_data=settings.shuffle_data,
                unit_lengths=prepared.unit_lengths,
                window_counts=prepared.windows[epoch],
            )
            for batch_index, batch in prefetch_batches(
                batches, pin_memory=runtime.device.type == "cuda"
            ):
                if stop_requested is not None and stop_requested():
                    run.stop(runtime, state, validation)
                    return runtime.model
                loss, valid_tokens, grad_norm = runtime.step(batch)
                state.record_step(
                    batch_index,
                    loss,
                    valid_tokens,
                    target_units(batch, prepared.unit_lengths),
                )
                if stop_requested is not None and stop_requested():
                    diagnostics.record(state, loss, grad_norm, validation)
                    diagnostics.report(state, validation)
                    run.stop(runtime, state, validation)
                    return runtime.model
                if (
                    state.global_step % settings.checkpoint_interval_steps == 0
                    and state.epoch_valid_tokens < epoch_targets
                ):
                    run.checkpoint(runtime, state, validation)
                validation.report()
                diagnostics.record(state, loss, grad_norm, validation)
                if state.epoch_valid_tokens >= epoch_targets:
                    break
            if state.optimizer_steps_in_epoch == 0:
                raise RuntimeError(
                    f"epoch {epoch + 1} produced no valid training batches"
                )
            if state.epoch_valid_tokens != epoch_targets:
                raise RuntimeError(
                    f"epoch {epoch + 1} ended at {state.epoch_valid_tokens} / "
                    f"{epoch_targets} target tokens"
                )
            epoch_loss = state.epoch_loss
            epoch_tokens = state.epoch_valid_tokens
            state.finish_epoch()
            run.checkpoint(runtime, state, validation)
            diagnostics.report(state, validation)
            run.log(
                f"Epoch {epoch + 1}/{settings.num_epochs}: loss {epoch_loss:.6f}, "
                f"tokens {epoch_tokens}/{epoch_targets} this epoch, "
                f"total {state.tokens_seen}/{total_train_tokens}, latest validation {validation.latest}"
            )
        if state.tokens_seen != total_train_tokens:
            raise RuntimeError(
                f"training ended at {state.tokens_seen} / {total_train_tokens} target tokens"
            )
        # Ensure a resumed final-step checkpoint is validated in this session, too.
        if run.last_validation_step is None:
            run.checkpoint(runtime, state, validation)
        run.finish(runtime, state, validation, stop_requested=stop_requested)
    return runtime.model

