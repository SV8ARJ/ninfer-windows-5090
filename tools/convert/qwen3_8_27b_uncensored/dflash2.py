"""Fixed Qwen3.8 DFlash2 companion contract for the uncensored artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6.common import recipe as family_recipe
from tools.convert.qwen3_6.common.inventory import BF16, W8, TensorSpec, tensor_spec


_ROOT_CONFIG = {
    "architectures": ["DFlash2DraftModel"],
    "model_type": "qwen3",
    "dtype": "bfloat16",
    "hidden_act": "silu",
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_hidden_layers": 5,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "attention_bias": False,
    "is_causal": False,
    "layer_types": ["sliding_attention"] * 5,
    "use_sliding_window": True,
    "max_window_layers": 5,
    "sliding_window": 2048,
    "vocab_size": 248320,
    "num_target_layers": 64,
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
}
_ROPE_CONFIG = {"rope_theta": 10000000, "rope_type": "default"}
_DRAFT_CONFIG = {
    "block_size": 8,
    "conv_group_size": 16,
    "conv_kernel_size": 2,
    "mask_token_id": 248070,
    "selector_rank": 256,
    "selector_top_k": 16,
    "target_layer_ids": [5, 19, 33, 47, 61],
}


def _build_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        tensor_spec("dflash2/feature_projection", (5120, 25600), W8),
        tensor_spec("dflash2/context_norm", (5120,), BF16),
    ]
    for layer in range(5):
        prefix = f"dflash2/layers/{layer}/"
        specs.extend((
            tensor_spec(prefix + "input_norm", (5120,), BF16),
            tensor_spec(prefix + "attention_conv/base_kernel", (2, 2, 5120), BF16),
            tensor_spec(prefix + "attention_conv/kernel_projection", (1280, 5120), BF16),
            tensor_spec(prefix + "attention/query_key_value", (6144, 5120), W8),
            tensor_spec(prefix + "attention/query_norm", (128,), BF16),
            tensor_spec(prefix + "attention/key_norm", (128,), BF16),
            tensor_spec(prefix + "attention/output", (5120, 4096), W8),
            tensor_spec(prefix + "post_attention_norm", (5120,), BF16),
            tensor_spec(prefix + "mlp_conv/base_kernel", (2, 2, 5120), BF16),
            tensor_spec(prefix + "mlp_conv/kernel_projection", (1280, 5120), BF16),
            tensor_spec(prefix + "mlp/gate_up", (34816, 5120), W8),
            tensor_spec(prefix + "mlp/down", (5120, 17408), W8),
        ))
    specs.extend((
        tensor_spec("dflash2/final_norm", (5120,), BF16),
        tensor_spec("dflash2/candidate_selector/hidden_projection", (256, 5120), BF16),
        tensor_spec("dflash2/candidate_selector/predecessor_codebook", (248320, 256), BF16),
        tensor_spec("dflash2/candidate_selector/successor_codebook", (248320, 256), BF16),
    ))
    return tuple(specs)


DFLASH2_TENSOR_SPECS = _build_specs()


def _build_recipes() -> tuple[family_recipe.TensorRecipe, ...]:
    recipes: list[family_recipe.TensorRecipe] = [
        family_recipe.TensorRecipe("dflash2/feature_projection", family_recipe.source("fc.weight", (5120, 25600))),
        family_recipe.TensorRecipe("dflash2/context_norm", family_recipe.source("hidden_norm.weight", (5120,))),
    ]
    for layer in range(5):
        source = f"layers.{layer}."
        target = f"dflash2/layers/{layer}/"
        recipes.extend((
            family_recipe.TensorRecipe(target + "input_norm", family_recipe.source(source + "input_layernorm.weight", (5120,))),
            family_recipe.TensorRecipe(target + "attention_conv/base_kernel", family_recipe.source(source + "attention_conv.base_kernel", (2, 2, 5120))),
            family_recipe.TensorRecipe(target + "attention_conv/kernel_projection", family_recipe.source(source + "attention_conv.kernel_projection.weight", (1280, 5120))),
            family_recipe.TensorRecipe(target + "attention/query_key_value", family_recipe.Concat((
                family_recipe.source(source + "self_attn.q_proj.weight", (4096, 5120)),
                family_recipe.source(source + "self_attn.k_proj.weight", (1024, 5120)),
                family_recipe.source(source + "self_attn.v_proj.weight", (1024, 5120)),
            ), 0)),
            family_recipe.TensorRecipe(target + "attention/query_norm", family_recipe.source(source + "self_attn.q_norm.weight", (128,))),
            family_recipe.TensorRecipe(target + "attention/key_norm", family_recipe.source(source + "self_attn.k_norm.weight", (128,))),
            family_recipe.TensorRecipe(target + "attention/output", family_recipe.source(source + "self_attn.o_proj.weight", (5120, 4096))),
            family_recipe.TensorRecipe(target + "post_attention_norm", family_recipe.source(source + "post_attention_layernorm.weight", (5120,))),
            family_recipe.TensorRecipe(target + "mlp_conv/base_kernel", family_recipe.source(source + "mlp_conv.base_kernel", (2, 2, 5120))),
            family_recipe.TensorRecipe(target + "mlp_conv/kernel_projection", family_recipe.source(source + "mlp_conv.kernel_projection.weight", (1280, 5120))),
            family_recipe.TensorRecipe(target + "mlp/gate_up", family_recipe.Concat((
                family_recipe.source(source + "mlp.gate_proj.weight", (17408, 5120)),
                family_recipe.source(source + "mlp.up_proj.weight", (17408, 5120)),
            ), 0)),
            family_recipe.TensorRecipe(target + "mlp/down", family_recipe.source(source + "mlp.down_proj.weight", (5120, 17408))),
        ))
    recipes.extend((
        family_recipe.TensorRecipe("dflash2/final_norm", family_recipe.source("norm.weight", (5120,))),
        family_recipe.TensorRecipe("dflash2/candidate_selector/hidden_projection", family_recipe.source("candidate_selector.hidden_projection.weight", (256, 5120))),
        family_recipe.TensorRecipe("dflash2/candidate_selector/predecessor_codebook", family_recipe.source("candidate_selector.predecessor_codebook", (248320, 256))),
        family_recipe.TensorRecipe("dflash2/candidate_selector/successor_codebook", family_recipe.source("candidate_selector.successor_codebook", (248320, 256))),
    ))
    return tuple(recipes)


_RECIPES = _build_recipes()
_RECIPES_BY_NAME = {recipe.object_name: recipe for recipe in _RECIPES}


def validate_config(config: Mapping[str, object]) -> dict[str, object]:
    family_conversion.check_members("DFlash2 config", config, _ROOT_CONFIG)
    rope = config.get("rope_parameters")
    draft = config.get("dflash_config")
    if not isinstance(rope, Mapping) or not isinstance(draft, Mapping):
        raise ValueError("DFlash2 config.json must contain rope_parameters and dflash_config")
    family_conversion.check_members("DFlash2 rope_parameters", rope, _ROPE_CONFIG)
    family_conversion.check_members("DFlash2 dflash_config", draft, _DRAFT_CONFIG)
    return {"hidden_size": config["hidden_size"], "vocab_size": config["vocab_size"],
            "num_target_layers": config["num_target_layers"],
            "max_position_embeddings": config["max_position_embeddings"],
            "rope_theta": rope["rope_theta"]}


def validate_base_compatibility(base: Mapping[str, object], dflash2: Mapping[str, object]) -> None:
    text = base.get("text")
    rope = base.get("rope")
    if not isinstance(text, Mapping) or not isinstance(rope, Mapping):
        raise ValueError("base config summary is missing text or rope facts")
    pairs = (("hidden_size", text.get("hidden_size"), dflash2["hidden_size"]),
             ("vocab_size", text.get("vocab_size"), dflash2["vocab_size"]),
             ("num_target_layers", text.get("num_hidden_layers"), dflash2["num_target_layers"]),
             ("max_position_embeddings", text.get("max_position_embeddings"), dflash2["max_position_embeddings"]),
             ("rope_theta", rope.get("rope_theta"), dflash2["rope_theta"]))
    for name, primary, companion in pairs:
        if primary != companion:
            raise ValueError(f"base/DFlash2 {name} mismatch: {primary!r} != {companion!r}")


def preflight_sources(model_dir: str | Path) -> family_recipe.SourcePreflight:
    requirements = family_recipe.source_requirements(_RECIPES)
    with ShardReader.from_file(Path(model_dir) / "model.safetensors") as reader:
        actual = set(reader.names)
        required = set(requirements)
        if actual != required:
            missing = sorted(required - actual)
            extra = sorted(actual - required)
            raise ValueError(f"DFlash2 source inventory differs from its exact tensor contract: missing={missing[:1]!r}, extra={extra[:1]!r}")
        return family_recipe.preflight_source_reader(reader, _RECIPES)


def materialize_tensor(name: str, reader: ShardReader) -> torch.Tensor:
    return family_recipe.materialize_recipe(_RECIPES_BY_NAME[name], reader)
