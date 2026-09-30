"""Convert each checkpoint to a static OpenVINO graph and execute it on NPU."""

import numpy as np
import torch
from torch import nn

from .validation_npu_runtime import load_local_npu_runtime


class OpenVinoNpuModel(nn.Module):
    def __init__(self, model, sequence_length, *, progress):
        super().__init__()
        try:
            import openvino as ov
        except ImportError as exc:
            raise RuntimeError(
                "NPU validation requires OpenVINO; run uv sync --extra npu."
            ) from exc

        load_local_npu_runtime()
        core = ov.Core()
        if not any(device.split(".")[0] == "NPU" for device in core.available_devices):
            raise RuntimeError(
                "OpenVINO cannot discover an NPU. Check the Intel NPU user-space driver "
                "and /dev/accel access. CPU fallback is disabled."
            )
        self.reference_model = model
        self.train(model.training)
        self.sequence_length = sequence_length
        shape = [1, sequence_length]
        progress(f"Converting checkpoint to OpenVINO, static input {shape}.")
        exported = torch.export.export(model, (torch.zeros(shape, dtype=torch.long),))
        graph = ov.convert_model(exported, share_weights=False)
        graph.reshape(shape)
        progress("Compiling checkpoint for NPU.")
        self.compiled = core.compile_model(graph, "NPU")
        devices = self.compiled.get_property("EXECUTION_DEVICES")
        if isinstance(devices, str):
            devices = [devices]
        if not devices or any(device.split(".")[0] != "NPU" for device in devices):
            raise RuntimeError(
                f"Expected NPU execution, OpenVINO reported {devices!r}."
            )
        self.output = self.compiled.output(0)
        progress(
            f"OpenVINO execution devices: {devices}; NPU forward passes, CPU FP32 loss reduction."
        )

    def forward(self, input_ids):
        if tuple(input_ids.shape) != (1, self.sequence_length):
            raise ValueError(
                "NPU validation requires one fixed-context sequence per batch"
            )
        outputs = self.compiled([input_ids.detach().cpu().numpy()])
        # OpenVINO owns and reuses its output buffer. Copy before the next request.
        return torch.from_numpy(
            np.array(outputs[self.output], dtype=np.float32, copy=True)
        )
