"""Dispatch matrices by their explicit persistent source schema."""

from __future__ import annotations

from .compressed_tensors import compressed_matrix_source
from .logical import EncodedRows, LogicalSource
from .modelopt import modelopt_nvfp4_source
from .safetensors import SafetensorsSource, tensor_source


def matrix_source(
    store: SafetensorsSource,
    name: str,
    shape: tuple[int, int],
    format: str | None = None,
) -> LogicalSource:
    """Resolve direct, compressed-tensors or ModelOpt storage lazily."""
    prefix = name.removesuffix(".weight")
    resolved: LogicalSource | None = None

    def resolve() -> LogicalSource:
        nonlocal resolved
        if resolved is not None:
            return resolved
        actual = format
        compressed_nvfp4 = store.has(prefix + ".weight_packed")
        modelopt_nvfp4 = store.has(prefix + ".weight_scale_2")
        if compressed_nvfp4 and modelopt_nvfp4:
            raise ValueError(f"{prefix}: conflicting NVFP4 source schemas")
        if actual == "nvfp4" and modelopt_nvfp4:
            resolved = modelopt_nvfp4_source(store, prefix, shape)
            return resolved
        if actual is None and compressed_nvfp4:
            actual = "nvfp4"
        if actual is None and store.describe(name).dtype == "F8_E4M3":
            scale = store.describe(prefix + ".weight_scale")
            if scale.dtype == "F32" and store.has(prefix + ".input_scale"):
                raise ValueError(
                    f"{prefix}: ModelOpt scalar FP8 requires value conversion; "
                    "it is not NInfer row-scaled FP8"
                )
            actual = "fp8_e4m3fn_row_bf16"
        resolved = (
            tensor_source(store, name, shape)
            if actual is None
            else compressed_matrix_source(store, prefix, shape, actual)
        )
        return resolved

    def encoded(begin: int, end: int) -> EncodedRows:
        reader = resolve().read_encoded
        if reader is None:
            raise ValueError(f"{name}: selected source does not provide encoded rows")
        return reader(begin, end)

    def divisor(which: str) -> bytes:
        reader = getattr(resolve(), which)
        if reader is None:
            raise ValueError(f"{name}: selected source does not provide {which}")
        return reader()

    return LogicalSource(
        shape,
        f"{store.path}:{name}",
        lambda begin, end: resolve().values(begin, end),
        encoded,
        lambda: divisor("weight_divisor"),
        lambda: divisor("input_divisor"),
    )
