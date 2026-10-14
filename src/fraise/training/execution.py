"""Configure compilation and scope strict numerical execution settings."""

import os
from contextlib import contextmanager

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def configure_execution(settings):
    # Set before model placement or the BF16 device check initializes CUDA.
    if settings.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if settings.use_compile:
        import torch._inductor.config as compiler_config

        compiler_config.compile_threads = settings.compile_threads


@contextmanager
def deterministic_execution(enabled):
    if not enabled:
        yield
        return
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    precision = torch.get_float32_matmul_precision()
    benchmark = torch.backends.cudnn.benchmark
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.benchmark = False
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)
        torch.set_float32_matmul_precision(precision)
        torch.backends.cudnn.benchmark = benchmark
