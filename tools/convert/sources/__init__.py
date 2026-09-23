"""Checkpoint bytes, logical value views, and source-format interpretation.

logical owns source values and row/axis transforms; safetensors owns local file
access; matrix dispatches direct, compressed-tensors and ModelOpt storage;
compressed_tensors and modelopt own those quantized source contracts. User sources
can construct LogicalSource without using any checkpoint-specific reader.
"""
