"""ONNX Runtime setup shared by the models: which device to run on, and session options for it."""

from __future__ import annotations

import onnxruntime as ort


def choose_providers(device: str) -> list[str]:
    """'auto' prefers DirectML (any Windows GPU, including integrated) and falls back to the CPU."""
    available = ort.get_available_providers()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "directml" and "DmlExecutionProvider" not in available:
        raise RuntimeError("DirectML requested but onnxruntime-directml is not installed")
    if device in ("auto", "directml") and "DmlExecutionProvider" in available:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def session_options(providers: list[str], threads: int = 0) -> ort.SessionOptions:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    if threads:  # 0 lets ONNX Runtime pick (best on hybrid P/E-core CPUs)
        opts.intra_op_num_threads = threads
    if providers[0] == "DmlExecutionProvider":
        opts.enable_mem_pattern = False  # required by DirectML
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return opts
