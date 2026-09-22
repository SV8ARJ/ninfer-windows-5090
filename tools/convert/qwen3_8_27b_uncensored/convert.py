"""Convert Qwen3.8-27B-Uncensored into its registered groupwise artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import torch

from tools.artifact.container import ArtifactIdentity, ArtifactWriter
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6_27b import convert as family_config
from tools.convert.qwen3_6_27b import draft_head
from tools.convert.qwen3_8_27b import convert as qwen3_8_convert

from . import inventory


RECIPE_ID = "qwen3_8_27b_uncensored-v1"
OUTPUT_BASENAME = "qwen3_8_27b_uncensored.ninfer"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_resources(model_dir: str | Path) -> tuple[family_conversion.ResourcePayload, ...]:
    """Embed the checkpoint-owned frontend, not the official Qwen3.8 profile."""
    resources = family_conversion.load_resources(model_dir, inventory.RESOURCE_SPECS)
    tokenizer_config = json.loads(resources[1].data)
    chat_template = tokenizer_config.get("chat_template")
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("tokenizer_config.json must define a nonempty chat_template")
    return tuple(
        family_conversion.ResourcePayload(
            resource.name,
            chat_template.encode("utf-8")
            if resource.name == "frontend/chat_template.jinja"
            else resource.data,
        )
        for resource in resources
    )


def preflight_conversion(model_dir: str | Path):
    model = Path(model_dir)
    config_summary = family_config.validate_config(
        family_conversion.load_json(model / "config.json")
    )
    qwen3_8_convert.preflight_inventory()
    source = qwen3_8_convert.recipe.preflight_sources(model)
    resources = load_resources(model)
    resource_map = {resource.name: resource.data for resource in resources}
    object_plan = family_conversion.build_object_plan(inventory.OBJECT_SPECS, resource_map)
    draft = draft_head.compute_shortlist(_repo_root() / draft_head.DEFAULT_RANKING, model)
    return config_summary, source, resources, draft, object_plan


def convert(
    model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> Path:
    output = Path(out_path)
    if output.name != OUTPUT_BASENAME:
        raise ValueError(f"output basename must be {OUTPUT_BASENAME!r}")
    started = time.perf_counter()
    resolved_device = pick_device(device)
    config_summary, source, resources, draft, object_plan = preflight_conversion(model_dir)
    print(
        f"preflight complete: {len(object_plan.objects)} objects, "
        f"{source.source_tensor_count} source tensors, device={resolved_device}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resource_map = {resource.name: resource.data for resource in resources}
    with ShardReader(model_dir) as reader:
        with ArtifactWriter(
            output,
            ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
            object_plan.specs,
        ) as writer:
            for index, spec in enumerate(inventory.OBJECT_SPECS, start=1):
                if isinstance(spec, inventory.ResourceSpec):
                    payload = resource_map[spec.name]
                else:
                    tensor = qwen3_8_convert.materialize_tensor(spec, reader, draft)
                    payload = qwen3_8_convert.encode_tensor_payload(
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
        arguments={"model": str(model_dir), "out": str(out_path), "device": str(device)},
        config_summary=config_summary,
        source_preflight=source,
        objects=object_plan.objects,
        elapsed_seconds=elapsed,
        final_bytes=output.stat().st_size,
        device=resolved_device,
        ranking_path=_repo_root() / draft_head.DEFAULT_RANKING,
    )
    report_path = Path(str(output) + ".conversion.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"complete: {output.stat().st_size} bytes in {elapsed:.1f}s; report={report_path}")
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    convert(args.model, args.out, device=args.device)


if __name__ == "__main__":
    main()
