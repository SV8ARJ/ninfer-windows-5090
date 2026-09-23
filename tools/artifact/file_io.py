"""Portable positioned I/O with bounded Linux page-cache retention."""

from __future__ import annotations

import os

if os.name == "nt":
    import ctypes
    from ctypes import wintypes
    import msvcrt

    class _Overlapped(ctypes.Structure):
        _fields_ = (
            ("Internal", ctypes.c_size_t),
            ("InternalHigh", ctypes.c_size_t),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        )

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _read_file = _kernel32.ReadFile
    _read_file.argtypes = (
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(_Overlapped),
    )
    _read_file.restype = wintypes.BOOL
    _write_file = _kernel32.WriteFile
    _write_file.argtypes = (
        wintypes.HANDLE,
        wintypes.LPCVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(_Overlapped),
    )
    _write_file.restype = wintypes.BOOL

IO_CHUNK_BYTES = 8 * 1024 * 1024
WRITEBACK_BYTES = 64 * 1024 * 1024
_PAGE_BYTES = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def _windows_offset(offset: int):
    overlapped = _Overlapped()
    overlapped.Offset = offset & 0xFFFFFFFF
    overlapped.OffsetHigh = offset >> 32
    return overlapped


def _read_once(fd: int, count: int, offset: int) -> bytes:
    if hasattr(os, "pread"):
        return os.pread(fd, count, offset)
    buffer = ctypes.create_string_buffer(count)
    transferred = wintypes.DWORD()
    overlapped = _windows_offset(offset)
    if not _read_file(
        msvcrt.get_osfhandle(fd),
        buffer,
        count,
        ctypes.byref(transferred),
        ctypes.byref(overlapped),
    ):
        error = ctypes.get_last_error()
        if error == 38:  # ERROR_HANDLE_EOF
            return b""
        raise ctypes.WinError(error)
    return buffer.raw[: transferred.value]


def read_at(fd: int, count: int, offset: int) -> bytes:
    chunks = []
    done = 0
    while done < count:
        chunk = _read_once(fd, min(IO_CHUNK_BYTES, count - done), offset + done)
        if not chunk:
            break
        chunks.append(chunk)
        done += len(chunk)
    return b"".join(chunks)


def write_at(fd: int, data: bytes | memoryview, offset: int) -> int:
    if hasattr(os, "pwrite"):
        return os.pwrite(fd, data, offset)
    payload = bytes(data)
    transferred = wintypes.DWORD()
    overlapped = _windows_offset(offset)
    if not _write_file(
        msvcrt.get_osfhandle(fd),
        payload,
        len(payload),
        ctypes.byref(transferred),
        ctypes.byref(overlapped),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return transferred.value


def discard_cached_pages(fd: int, offset: int = 0, count: int | None = None) -> None:
    if not hasattr(os, "posix_fadvise"):
        return
    if count is None:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    elif count > 0:
        begin = offset // _PAGE_BYTES * _PAGE_BYTES
        end = (offset + count + _PAGE_BYTES - 1) // _PAGE_BYTES * _PAGE_BYTES
        os.posix_fadvise(fd, begin, end - begin, os.POSIX_FADV_DONTNEED)


class Writeback:
    """Bound dirty output across all open shards; release clean pages after writeback."""

    def __init__(self) -> None:
        self._bytes = 0
        self._fds: set[int] = set()

    def written(self, fd: int, count: int) -> None:
        self._fds.add(fd)
        self._bytes += count
        if self._bytes >= WRITEBACK_BYTES:
            self.flush()

    def flush(self) -> None:
        for fd in self._fds:
            (os.fdatasync if hasattr(os, "fdatasync") else os.fsync)(fd)
            discard_cached_pages(fd)
        self._fds.clear()
        self._bytes = 0
