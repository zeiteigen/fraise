"""Load the optional user-local Intel NPU runtime without system changes."""

import ctypes
import os
import sys
from pathlib import Path


def load_local_npu_runtime():
    if sys.platform != "linux":
        return
    directory = Path.home() / ".local/share/fraise/npu-runtime/lib"
    driver = directory / "libze_intel_npu.so.1"
    if not driver.exists():
        return  # Use the system-installed driver.
    os.environ.setdefault("ZE_ENABLE_ALT_DRIVERS", str(driver))
    for name in (
        "libze_loader.so.1",
        "libopenvino_intel_npu_compiler.so",
        "libopenvino_intel_npu_compiler_loader.so",
        "libze_intel_npu.so.1",
    ):
        ctypes.CDLL(str(directory / name), mode=ctypes.RTLD_GLOBAL)
