# Qwen3.8-27B-Uncensored Artifact Reference

This local checkpoint has two registered artifact identities:

```text
qwen3.8-27b-uncensored / groupwise-int
qwen3.8-27b-uncensored / nvfp4
```

They execute through the Qwen3.8-27B target package and retain its fixed model geometry, tensor
object names, layouts, and execution profiles. The identity distinguishes the checkpoint weights
and frontend from the official Qwen3.8-27B artifacts; it is not an alias for them.

## Groupwise artifact

`qwen3_8_27b_uncensored.ninfer` is converted from `models/Qwen3.8-27B-Uncensored` using the
Qwen3.8 groupwise inventory. Its six frontend resources are sourced from that checkpoint. The
embedded chat template is the `chat_template` value from `tokenizer_config.json`, serialized as
UTF-8 so the tokenizer configuration and standalone template are byte-identical after JSON parsing.

## NVFP4 artifact

`qwen3_8_27b_uncensored_nvfp4.ninfer` consumes the base checkpoint and
`models/Qwen3.8-27B-Uncensored-NVFP4`. The mixed source supplies the Qwen3.8 FP8 and NVFP4 matrix
words directly. Its `lm_head` is BF16 rather than row-scaled FP8, so the converter encodes that
single endpoint from the matching base checkpoint using `MAXABS_BF16S_RECIP_E4M3FN_RNE_V1`.

The NVFP4 artifact additionally contains the complete 66-object DFlash2 suffix from
`Qwen3.8-27B-DFlash2`. It is compatible with the uncensored target's fixed Qwen3.8 geometry and
is selected by a DFlash2-capable NInfer executable with `--spec dflash2 --draft-tokens 7`.
Target verification preserves the uncensored target output semantics; companion acceptance rate is
checkpoint-dependent because the companion was trained against the standard Qwen3.8 target.

`convertUncensored.cmd` regenerates the NVFP4+DFlash2 image from the three source directories with
the CUDA 13.0 PyTorch environment that supports the RTX 5090 `sm_120` architecture. Conversion
writes its descriptive `.conversion.json` report beside the artifact.
