"""Build Qwen3.8-27B-Uncensored's registered NVFP4 artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Iterable, Sequence

import torch

from tools.artifact.container import ArtifactIdentity, ArtifactWriter
from tools.artifact.layouts import encode_direct
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6_27b import convert as family_config
from tools.convert.qwen3_6_27b import draft_head
from tools.convert.qwen3_8_27b import convert_nvfp4 as qwen3_8_convert
from tools.convert.qwen3_8_27b import fp8_embedding
from tools.convert.qwen3_8_27b import recipe_nvfp4 as recipe

from . import inventory_nvfp4 as inventory
from .convert import load_resources
from . import dflash2


RECIPE_ID = "qwen3_8_27b_uncensored_nvfp4-dflash2-v1"
OUTPUT_BASENAME = "qwen3_8_27b_uncensored_nvfp4.ninfer"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _validate_quantized_config(config: dict[str, object]) -> dict[str, object]:
    summary = family_config.validate_config(config)
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        raise ValueError("quantized source is missing quantization_config")
    if quantization.get("quant_method") != "compressed-tensors":
        raise ValueError("quantized source must use compressed-tensors")
    if quantization.get("format") != "mixed-precision":
        raise ValueError("quantized source must use mixed-precision")
    groups = quantization.get("config_groups")
    if not isinstance(groups, dict) or set(groups) != {"group_0", "group_1"}:
        raise ValueError("quantized source must declare one FP8 and one NVFP4 group")
    return summary


def _preflight_quantized_sources(reader: ShardReader):
    """Validate the mixed export, whose BF16 output head is re-encoded locally."""
    requirements = dict(recipe.SOURCE_REQUIREMENTS)
    requirements.pop("lm_head.weight")
    requirements.pop("lm_head.weight_scale")
    missing = set(requirements).difference(reader.names)
    if missing:
        raise ValueError(f"quantized source is missing {sorted(missing)[0]}")

    for source in recipe.FP8_SOURCES:
        if source.name == "lm_head":
            continue
        for suffix in ("weight_packed", "weight_global_scale", "input_global_scale"):
            name = source.field(suffix)
            if reader.has(name):
                raise ValueError(f"FP8 source has forbidden field {name}")
    for source in recipe.NVFP4_SOURCES:
        if reader.has(source.field("weight")):
            raise ValueError(
                f"NVFP4 source has forbidden field {source.field('weight')}"
            )

    metadata = reader.metadata(reader.names)
    expected_quantized = frozenset(
        name
        for name, (_, dtype) in requirements.items()
        if dtype in ("F8_E4M3", "F32", "U8")
    )
    actual_quantized = frozenset(
        name
        for name, item in metadata.items()
        if item.dtype in ("F8_E4M3", "F32", "U8")
    )
    if actual_quantized != expected_quantized:
        detail = actual_quantized.symmetric_difference(expected_quantized)
        raise ValueError(f"quantized source allocation is not closed: {sorted(detail)[0]}")

    dtype_counts: dict[str, int] = {}
    shards: set[str] = set()
    for name, (shape, dtype) in requirements.items():
        actual = metadata[name]
        if actual.shape != shape or actual.dtype != dtype:
            raise ValueError(
                f"{name}: source signature {(actual.shape, actual.dtype)} "
                f"!= {(shape, dtype)}"
            )
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        shards.add(actual.shard)
    return recipe.family_recipe.SourcePreflight(
        recipe_count=(
            len(recipe.FP8_WEIGHT_RECIPES)
            + len(recipe.NVFP4_WEIGHT_RECIPES)
            + len(recipe.INPUT_DIVISOR_RECIPES)
            + len(recipe.QUANTIZED_DIRECT_RECIPES)
        ),
        source_tensor_count=len(requirements),
        source_shard_count=len(shards),
        source_dtype_counts=dtype_counts,
    )


def preflight_conversion(
    official_dir: str | Path,
    quantized_dir: str | Path,
    dflash2_model_dir: str | Path,
):
    official = Path(official_dir)
    quantized = Path(quantized_dir)
    qwen3_8_convert._validate_index(official)
    qwen3_8_convert._validate_index(quantized)
    official_config = family_conversion.load_json(official / "config.json")
    if official_config.get("quantization_config") is not None:
        raise ValueError("base source must not declare quantization_config")
    official_summary = family_config.validate_config(official_config)
    quantized_summary = _validate_quantized_config(
        family_conversion.load_json(quantized / "config.json")
    )
    if official_summary != quantized_summary:
        raise ValueError("base and NVFP4 source model configs do not match")
    dflash2_model = Path(dflash2_model_dir)
    dflash2_summary = dflash2.validate_config(
        family_conversion.load_json(dflash2_model / "config.json")
    )
    dflash2.validate_base_compatibility(official_summary, dflash2_summary)
    qwen3_8_convert.preflight_inventory()
    with ShardReader(official) as official_reader:
        official_source = recipe.preflight_official_sources(official_reader)
    with ShardReader(quantized) as quantized_reader:
        quantized_source = _preflight_quantized_sources(quantized_reader)
    dflash2_source = dflash2.preflight_sources(dflash2_model)
    resources = load_resources(official)
    resource_map = {resource.name: resource.data for resource in resources}
    object_plan = family_conversion.build_object_plan(inventory.OBJECT_SPECS, resource_map)
    draft = draft_head.compute_shortlist(_repo_root() / draft_head.DEFAULT_RANKING, official)
    return (
        official_summary,
        dflash2_summary,
        official_source,
        quantized_source,
        dflash2_source,
        resources,
        draft,
        object_plan,
    )


def convert(
    model_dir: str | Path,
    quantized_model_dir: str | Path,
    dflash2_model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> Path:
    output = Path(out_path)
    if output.name != OUTPUT_BASENAME:
        raise ValueError(f"output basename must be {OUTPUT_BASENAME!r}")
    started = time.perf_counter()
    resolved_device = pick_device(device)
    (
        summary,
        dflash2_summary,
        official_source,
        quantized_source,
        dflash2_source,
        resources,
        draft,
        object_plan,
    ) = preflight_conversion(model_dir, quantized_model_dir, dflash2_model_dir)
    print(
        f"preflight complete: {len(object_plan.objects)} objects, "
        f"{len(recipe.FP8_SOURCES)} FP8, {len(recipe.NVFP4_SOURCES)} NVFP4, and "
        f"{dflash2_source.source_tensor_count} DFlash2 source tensors, "
        f"device={resolved_device}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resource_map = {resource.name: resource.data for resource in resources}
    draft_ids = draft_head.materialize_draft_head_token_ids(draft)
    derived = {draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT: draft_ids}
    with (
        ShardReader(model_dir) as official_reader,
        ShardReader(quantized_model_dir) as quantized_reader,
        ShardReader.from_file(Path(dflash2_model_dir) / "model.safetensors") as dflash2_reader,
    ):
        with ArtifactWriter(
            output,
            ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
            object_plan.specs,
        ) as writer:
            for index, spec in enumerate(inventory.OBJECT_SPECS, start=1):
                payload: bytes | Iterable[bytes]
                if isinstance(spec, inventory.ResourceSpec):
                    payload = resource_map[spec.name]
                elif spec.name == "text/token_embedding":
                    payload = fp8_embedding.iter_reader_payload(
                        official_reader, recipe.OFFICIAL_EMBEDDING_SOURCE.name, spec.shape
                    )
                elif spec.name == "text/output_head":
                    payload = fp8_embedding.iter_reader_payload(
                        official_reader, "lm_head.weight", spec.shape
                    )
                elif spec.name.startswith("dflash2/"):
                    tensor = dflash2.materialize_tensor(spec.name, dflash2_reader)
                    payload = family_conversion.encode_tensor_payload(tensor, spec, resolved_device)
                    del tensor
                elif spec.name in recipe.FP8_WEIGHTS_BY_NAME:
                    payload = qwen3_8_convert._encode_fp8_weight(spec, quantized_reader)
                elif spec.name in recipe.NVFP4_WEIGHTS_BY_NAME:
                    payload = qwen3_8_convert._encode_nvfp4_weight(spec, quantized_reader)
                elif spec.name in recipe.INPUT_DIVISORS_BY_NAME:
                    scalar = recipe.materialize_input_divisor(
                        recipe.INPUT_DIVISORS_BY_NAME[spec.name], quantized_reader
                    )
                    payload = encode_direct(scalar, inventory.FP32)
                elif spec.name in recipe.QUANTIZED_DIRECT_BY_NAME:
                    tensor = qwen3_8_convert._materialize_direct(spec, quantized_reader)
                    payload = encode_direct(tensor, spec.format)
                    del tensor
                else:
                    tensor = qwen3_8_convert._materialize_official(
                        spec, official_reader, derived
                    )
                    payload = family_conversion.encode_tensor_payload(
                        tensor, spec, resolved_device
                    )
                    del tensor
                writer.write(spec.name, payload)
                del payload
                print(f"[{index}/{len(inventory.OBJECT_SPECS)}] {spec.name}", flush=True)

    elapsed = time.perf_counter() - started
    report = family_conversion.build_conversion_report(
        identity=ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        target_key=inventory.TARGET_KEY,
        recipe_id=RECIPE_ID,
        repo_root=_repo_root(),
        model_dir=model_dir,
        out_path=output,
        arguments={
            "model": str(model_dir),
            "quantized_model": str(quantized_model_dir),
            "dflash2_model": str(dflash2_model_dir),
            "out": str(out_path),
            "device": str(device),
        },
        config_summary={"base": summary, "dflash2": dflash2_summary},
        source_preflight=official_source,
        objects=object_plan.objects,
        elapsed_seconds=elapsed,
        final_bytes=output.stat().st_size,
        device=resolved_device,
        ranking_path=_repo_root() / draft_head.DEFAULT_RANKING,
    )
    report["source_preflight"] = {
        "official": report["source_preflight"],
        "quantized": {
            "recipes": quantized_source.recipe_count,
            "tensors": quantized_source.source_tensor_count,
            "shards": quantized_source.source_shard_count,
            "dtypes": dict(quantized_source.source_dtype_counts),
        },
        "dflash2": {
            "recipes": dflash2_source.recipe_count,
            "tensors": dflash2_source.source_tensor_count,
            "shards": dflash2_source.source_shard_count,
            "dtypes": dict(dflash2_source.source_dtype_counts),
        },
    }
    report_path = Path(str(output) + ".conversion.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"complete: {output.stat().st_size} bytes in {elapsed:.1f}s; report={report_path}")
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--quantized-model", required=True, type=Path)
    parser.add_argument("--dflash2-model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    convert(args.model, args.quantized_model, args.dflash2_model, args.out, device=args.device)


if __name__ == "__main__":
    main()
