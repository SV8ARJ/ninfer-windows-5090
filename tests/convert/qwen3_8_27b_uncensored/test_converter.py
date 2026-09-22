from __future__ import annotations

import pytest

from tools.convert.qwen3_8_27b_uncensored import convert, convert_nvfp4, inventory
from tools.convert.qwen3_8_27b_uncensored import inventory_nvfp4


def test_artifact_identities_are_checkpoint_specific() -> None:
    assert (inventory.MODEL_ID, inventory.WEIGHTS_ID) == (
        "qwen3.8-27b-uncensored",
        "groupwise-int",
    )
    assert (inventory_nvfp4.MODEL_ID, inventory_nvfp4.WEIGHTS_ID) == (
        "qwen3.8-27b-uncensored",
        "nvfp4",
    )


def test_converters_reject_wrong_output_names_before_reading_sources(tmp_path) -> None:
    with pytest.raises(ValueError, match="output basename"):
        convert.convert(tmp_path / "missing", tmp_path / "wrong.ninfer", device="cpu")
    with pytest.raises(ValueError, match="output basename"):
        convert_nvfp4.convert(
            tmp_path / "missing-base",
            tmp_path / "missing-nvfp4",
            tmp_path / "missing-dflash2",
            tmp_path / "wrong.ninfer",
            device="cpu",
        )
