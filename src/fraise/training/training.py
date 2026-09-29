"""Train the Fraise transformer over the prepared epoch plan."""

from pathlib import Path

from fraise.config.model import ModelConfig
from fraise.config.training import TrainingSettings
from fraise.data.types import TrainingData
from fraise.tokenizer.loading import special_token_id

from .batches import iter_dataset_packed_batches
from .lifecycle import RunLifecycle
from .manifest import run_manifest
from .prefetch import prefetch_batches
from .preparation import prepare_data
from .runtime import build_runtime
from .validation_runner import ValidationRunner


def pretrain_transformer(
    tokenizer,
    dataset: TrainingData,
    settings: TrainingSettings,
    model_config: ModelConfig,
    save_path: str | Path,
    model_name="fraise_model",
    resume_from: str | Path | None = None,
    config_paths: dict[str, Path] | None = None,
    *,
    device=None,
):
    if tokenizer.get_vocab_size() != model_config.vocab_size:
        raise ValueError("tokenizer/model vocabulary mismatch")
    prepared = prepare_data(dataset, tokenizer, settings, model_config)
    pad_id = special_token_id(tokenizer, "<|pad|>")
    runtime = build_runtime(model_config, settings, prepared.steps, device)
    manifest = run_manifest(dataset, tokenizer, model_config, settings, runtime.device)
    run = RunLifecycle(
        Path(save_path),
        model_name,
        manifest,
        tokenizer,
        model_config,
        settings,
        resume_from=Path(resume_from).expanduser() if resume_from is not None else None,
        config_paths=config_paths,
    )
    state = run.checkpoints.restore(runtime.model, runtime.optimizer, runtime.scheduler)
    run.log(
        f"Training {settings.max_train_tokens:,} targets from step {state.global_step}; "
        f"context {settings.sequence_length}, batch {settings.batch_size}, minibatch {settings.minibatch_size}, "
        f"accumulation {settings.gradient_accumulation_steps}, device {runtime.device}."
    )
    with ValidationRunner(
        model_config, prepared.validation, settings, pad_id, run.log
    ) as validation:
        for epoch in range(state.epoch, settings.num_epochs):
            if state.tokens_seen >= settings.max_train_tokens:
                break
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
                seed=settings.seed + epoch,
                shuffle=settings.shuffle,
            )
            for batch_index, batch in prefetch_batches(
                batches, pin_memory=runtime.device.type == "cuda"
            ):
                loss, valid_tokens, grad_norm = runtime.step(batch, settings)
                state.record_step(batch_index, loss, valid_tokens)
                if (
                    state.global_step % settings.checkpoint_interval_steps == 0
                    and state.epoch_valid_tokens < epoch_targets
                ):
                    run.checkpoint(runtime, state, validation)
                    run.log(
                        f"Step {state.global_step}: loss {loss:.6f}, gradient norm {grad_norm:.6f}, "
                        f"tokens {state.tokens_seen:,}, learning rate {runtime.scheduler.get_last_lr()[0]:.8g}"
                    )
                validation.report()
                if state.tokens_seen >= settings.max_train_tokens:
                    break
            if state.optimizer_steps_in_epoch == 0:
                raise RuntimeError(
                    f"epoch {epoch + 1} produced no valid training batches"
                )
            epoch_loss = state.epoch_loss_sum / state.epoch_valid_tokens
            state.finish_epoch()
            run.checkpoint(runtime, state, validation)
            run.log(
                f"Epoch {epoch + 1}/{settings.num_epochs}: loss {epoch_loss:.6f}, "
                f"tokens {state.tokens_seen}/{settings.max_train_tokens}, latest validation {validation.latest}"
            )
        if state.tokens_seen < settings.max_train_tokens:
            raise RuntimeError(
                f"training ended at {state.tokens_seen} / {settings.max_train_tokens} target tokens"
            )
        # Ensure a resumed final-step checkpoint is validated in this session, too.
        if run.last_validation_step is None:
            run.checkpoint(runtime, state, validation)
        run.finish(runtime, state, validation)
    return runtime.model
