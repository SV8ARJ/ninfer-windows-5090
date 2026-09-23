"""Interpret NVIDIA ModelOpt NVFP4 checkpoint tensors."""

from __future__ import annotations

import json
from math import prod
import struct

import torch

from tools.artifact.formats import valid_positive_fp32_word
from .logical import EncodedRows, LogicalSource
from .safetensors import SafetensorsSource


_E2M1_VALUES = torch.tensor(
    (
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ),
    dtype=torch.float32,
)


def _positive_f32(store: SafetensorsSource, name: str) -> float:
    info = store.describe(name)
    if info.dtype != "F32" or prod(info.shape) != 1:
        raise ValueError(f"{name}: expected a source FP32 scalar")
    raw = store.read_flat(name).view(torch.uint8).numpy().tobytes()
    word = struct.unpack("<I", raw)[0]
    if not valid_positive_fp32_word(word):
        raise ValueError(f"{name}: multiplier must be finite and positive")
    return struct.unpack("<f", raw)[0]


def _reciprocal_word(store: SafetensorsSource, name: str) -> bytes:
    raw = struct.pack("<f", 1.0 / _positive_f32(store, name))
    if not valid_positive_fp32_word(struct.unpack("<I", raw)[0]):
        raise ValueError(f"{name}: reciprocal is not finite positive FP32")
    return raw


def modelopt_nvfp4_source(
    store: SafetensorsSource, prefix: str, shape: tuple[int, int]
) -> LogicalSource:
    """Expose ModelOpt E2M1 words under NInfer's reciprocal-divisor contract."""
    n, k = shape
    if k % 16:
        raise ValueError(f"{prefix}: NVFP4 source K must be divisible by 16")
    packed = prefix + ".weight"
    scale = prefix + ".weight_scale"
    weight_multiplier = prefix + ".weight_scale_2"
    input_multiplier = prefix + ".input_scale"

    def signature(name: str, expected: tuple[int, ...], dtype: str) -> None:
        info = store.describe(name)
        if info.shape != expected or info.dtype != dtype:
            raise ValueError(
                f"{name}: expected {dtype}{expected}, got {info.dtype}{info.shape}"
            )

    signature(packed, (n, k // 2), "U8")
    signature(scale, (n, k // 16), "F8_E4M3")
    weight_divisor = _reciprocal_word(store, weight_multiplier)
    input_divisor = _reciprocal_word(store, input_multiplier)

    def encoded(begin: int, end: int) -> EncodedRows:
        if not 0 <= begin < end <= n:
            raise ValueError(f"{prefix}: invalid encoded rows [{begin},{end})")
        codes = store.read_flat(
            packed, begin * (k // 2), end * (k // 2)
        ).reshape(end - begin, k // 2)
        scales = (
            store.read_flat(scale, begin * (k // 16), end * (k // 16))
            .view(torch.uint8)
            .reshape(end - begin, k // 16)
        )
        if bool((scales > 0x7E).any()):
            raise ValueError(f"{scale}: expected nonnegative finite E4M3FN scales")
        return EncodedRows("nvfp4", codes, scales, weight_divisor)

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        first, last = begin // k, (end + k - 1) // k
        words = encoded(first, last)
        codes = torch.stack((words.codes & 15, words.codes >> 4), dim=-1).reshape(
            last - first, k
        )
        scales = (
            words.scales.view(torch.float8_e4m3fn)
            .float()
            .repeat_interleave(16, dim=1)
        )
        multiplier = _positive_f32(store, weight_multiplier)
        values = _E2M1_VALUES[codes.long()] * scales * multiplier
        return values.reshape(-1)[begin - first * k : end - first * k]

    return LogicalSource(
        shape,
        f"{store.path}:{prefix} (modelopt-nvfp4)",
        read,
        encoded,
        lambda: weight_divisor,
        lambda: input_divisor,
    )


def validate_modelopt_inventory(
    store: SafetensorsSource, expected: dict[str, str]
) -> None:
    """Require one exact ModelOpt mixed-precision module inventory."""
    path = store.root / "hf_quant_config.json"
    if not path.is_file():
        raise ValueError(f"{path}: ModelOpt source requires hf_quant_config.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    producer = data.get("producer", {})
    quantization = data.get("quantization", {})
    if str(producer.get("name", "")).lower() != "modelopt":
        raise ValueError(f"{path}: producer must be modelopt")
    if str(quantization.get("quant_algo", "")).upper() != "MIXED_PRECISION":
        raise ValueError(f"{path}: quantization algorithm must be MIXED_PRECISION")
    records = quantization.get("quantized_layers")
    if not isinstance(records, dict):
        raise ValueError(f"{path}: quantized_layers must be an object")
    actual = {}
    for name, record in records.items():
        if not isinstance(record, dict):
            raise ValueError(f"{path}: {name} record must be an object")
        algorithm = str(record.get("quant_algo", "")).upper()
        if algorithm not in ("NVFP4", "FP8"):
            raise ValueError(f"{path}: {name} has unsupported algorithm {algorithm!r}")
        if algorithm == "NVFP4" and record.get("group_size") != 16:
            raise ValueError(f"{path}: {name} NVFP4 group_size must be 16")
        actual[name] = algorithm
    missing = sorted(expected.keys() - actual.keys())
    extra = sorted(actual.keys() - expected.keys())
    changed = sorted(
        name
        for name in expected.keys() & actual.keys()
        if expected[name] != actual[name]
    )
    if missing or extra or changed:
        raise ValueError(
            "ModelOpt quantized layer inventory differs: "
            f"missing={missing[:4]}, extra={extra[:4]}, changed={changed[:4]}"
        )
    for prefix, algorithm in expected.items():
        suffixes = (
            ("weight", "weight_scale", "weight_scale_2", "input_scale")
            if algorithm == "NVFP4"
            else ("weight", "weight_scale", "input_scale")
        )
        for suffix in suffixes:
            name = prefix + "." + suffix
            if not store.has(name):
                raise ValueError(f"{store.path}: missing ModelOpt tensor {name!r}")
