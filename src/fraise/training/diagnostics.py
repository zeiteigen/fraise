"""Periodic training throughput, optimizer rates, and resource diagnostics."""

from time import monotonic

import torch


class TrainingDiagnostics:
    def __init__(self, runtime, state, interval, log):
        self.runtime = runtime
        self.interval = interval
        self.log = log
        self.started = monotonic()
        self.tokens = state.tokens_seen
        self.compilation_seconds = runtime.compilation_seconds
        self.wait_seconds = 0.0
        self.reported_step = state.global_step
        self.latest = None

    def record(self, state, loss, gradient_norm, validation):
        self.latest = (loss, gradient_norm)
        if state.global_step % self.interval == 0:
            self.report(state, validation)

    def report(self, state, validation):
        if self.latest is None or state.global_step == self.reported_step:
            return
        now = monotonic()
        warmup = self.runtime.compilation_seconds - self.compilation_seconds
        elapsed = max(now - self.started - warmup, 1e-6)
        rate = (state.tokens_seen - self.tokens) / elapsed
        wait = validation.queue_wait_seconds - self.wait_seconds
        loss, norm = self.latest
        # Groups are ordered muon, adamw, adamw_no_decay; both AdamW groups share a rate.
        muon_lr, adamw_lr, *_ = self.runtime.scheduler.get_last_lr()
        message = (
            f"Step {state.global_step}: loss {loss:.6f}, gradient norm {norm:.6f}, "
            f"tokens {state.tokens_seen:,}, throughput {rate:,.0f} tokens/s "
            "(excluding compiled warmup), "
            f"next Muon learning rate {muon_lr:.8g}, "
            f"next AdamW learning rate {adamw_lr:.8g}, "
            f"validation queue wait {wait:.3f}s "
            f"(total {validation.queue_wait_seconds:.3f}s)"
        )
        if self.runtime.device.type == "cuda":
            device = self.runtime.device
            message += (
                f", GPU allocated {torch.cuda.memory_allocated(device) / 2**20:.1f} MiB, "
                f"reserved {torch.cuda.memory_reserved(device) / 2**20:.1f} MiB, "
                f"peak allocated {torch.cuda.max_memory_allocated(device) / 2**20:.1f} MiB"
            )
        self.log(message)
        self.started = now
        self.tokens = state.tokens_seen
        self.compilation_seconds = self.runtime.compilation_seconds
        self.wait_seconds = validation.queue_wait_seconds
        self.reported_step = state.global_step
