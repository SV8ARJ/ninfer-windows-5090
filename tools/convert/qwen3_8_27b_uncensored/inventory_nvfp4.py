"""Persistent-object contract for Qwen3.8-27B-Uncensored NVFP4 weights."""

from tools.convert.qwen3_8_27b.inventory_nvfp4 import *  # noqa: F403

from .dflash2 import DFLASH2_TENSOR_SPECS


BASE_OBJECT_SPECS = OBJECT_SPECS
BASE_TENSOR_SPECS = TENSOR_SPECS
TENSOR_SPECS = BASE_TENSOR_SPECS + DFLASH2_TENSOR_SPECS
OBJECT_SPECS = BASE_OBJECT_SPECS + DFLASH2_TENSOR_SPECS


MODEL_ID = "qwen3.8-27b-uncensored"
WEIGHTS_ID = "nvfp4"
TARGET_KEY = "qwen3_8_27b"
