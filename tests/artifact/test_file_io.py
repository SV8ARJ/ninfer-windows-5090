from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os

from tools.artifact.file_io import read_at, write_at


def test_positioned_io_is_complete_and_concurrent(tmp_path):
    path = tmp_path / "positioned.bin"
    payload = bytes(range(256)) * 4096
    fd = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_BINARY", 0),
    )
    try:
        assert write_at(fd, payload, 0) == len(payload)
        assert read_at(fd, len(payload) + 1, 0) == payload
        high_offset = (1 << 32) + 17
        assert write_at(fd, b"high", high_offset) == 4
        assert read_at(fd, 4, high_offset) == b"high"

        def read_block(index: int) -> bytes:
            offset = index * 4096
            return read_at(fd, 4096, offset)

        with ThreadPoolExecutor(max_workers=8) as executor:
            blocks = list(executor.map(read_block, range(len(payload) // 4096)))
        assert b"".join(blocks) == payload
    finally:
        os.close(fd)
