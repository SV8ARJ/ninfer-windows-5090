#include "artifact/file_io.h"

#include "artifact/framing.h"
#include "artifact/schema.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ninfer::artifact {
namespace {

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
#ifdef _WIN32
    throw ArtifactError(path.string() + ": " + operation + ": Win32 error " +
                        std::to_string(GetLastError()));
#else
    throw ArtifactError(path.string() + ": " + operation + ": " + std::strerror(errno));
#endif
}

#ifndef _WIN32
off_t file_offset(std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        throw ArtifactError("file offset exceeds positional I/O range");
    }
    return static_cast<off_t>(offset);
}
#endif

} // namespace

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
#ifdef _WIN32
    // Match POSIX readers: another actor may replace or truncate the path while this handle remains
    // valid for positional reads of its opened file.
    const HANDLE handle = CreateFileW(path_.c_str(), GENERIC_READ,
                                       FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                       OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) { fail(path_, "open"); }
    LARGE_INTEGER size{};
    if (!GetFileSizeEx(handle, &size) || size.QuadPart < 0) {
        CloseHandle(handle);
        fail(path_, "size");
    }
    fd_    = handle;
    bytes_ = static_cast<std::uint64_t>(size.QuadPart);
#else
    fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) { fail(path_, "open"); }

    struct stat status {};

    if (::fstat(fd_, &status) != 0) {
        const auto error = errno;
        ::close(fd_);
        fd_   = -1;
        errno = error;
        fail(path_, "fstat");
    }
    if (status.st_size < 0 || !S_ISREG(status.st_mode)) {
        ::close(fd_);
        fd_ = -1;
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    bytes_ = static_cast<std::uint64_t>(status.st_size);
#endif
}

InputFile::~InputFile() {
#ifdef _WIN32
    if (fd_ != nullptr) { CloseHandle(static_cast<HANDLE>(fd_)); }
#else
    if (direct_fd_ >= 0) { ::close(direct_fd_); }
    if (fd_ >= 0) { ::close(fd_); }
#endif
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
#ifdef _WIN32
        OVERLAPPED request{};
        request.Offset     = static_cast<DWORD>(offset);
        request.OffsetHigh = static_cast<DWORD>(offset >> 32U);
        DWORD read         = 0;
        if (!ReadFile(static_cast<HANDLE>(fd_), destination.data(), static_cast<DWORD>(count), &read,
                      &request)) {
            fail(path_, "ReadFile");
        }
        if (!read) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
#else
        const auto read  = ::pread(fd_, destination.data(), count, file_offset(offset));
        if (read < 0) {
            if (errno == EINTR) { continue; }
            fail(path_, "pread");
        }
        if (!read) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
#endif
        offset += static_cast<std::uint64_t>(read);
        destination = destination.subspan(static_cast<std::size_t>(read));
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
#ifdef _WIN32
    read_exact(offset, destination);
    return destination.size();
#else
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
        throw ArtifactError(path_.string() + ": unaligned or oversized direct read");
    }
    if (destination.empty()) { return 0; }
    if (direct_fd_ < 0) {
        direct_fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (direct_fd_ < 0) { fail(path_, "open direct"); }
    }
    ssize_t read;
    do {
        read = ::pread(direct_fd_, destination.data(), destination.size(), file_offset(offset));
    } while (read < 0 && errno == EINTR);
    if (read < 0) { fail(path_, "direct pread"); }
    return static_cast<std::size_t>(read);
#endif
}

} // namespace ninfer::artifact
