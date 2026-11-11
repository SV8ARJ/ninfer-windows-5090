# Windows rebuild notes

This repository is the compiled Windows source tree. It contains one branch, `dev`, based on
`natpate/ninfer-windows` `dev` (`84e30457`) with the Windows build fixes committed on top.

## Layout

- Source and build tree root: this directory
- Build output: `build-windows/` (`apps/` and `bench/`)
- Deployment binaries: `bin/`
- Local dependency files: `windows-libs/` (headers and MSVC import libraries)
- CMake find modules: `build-support/`
- Model artifact: `L:\LLM_Models\Ninfer\models\qwen3_8_27b_nvfp4full.ninfer`

`windows-libs/`, `bin/`, and `build-windows/` are excluded locally through `.git/info/exclude`.

## Included Windows fixes

Commit `aa6f753d` (`build(windows): fix FP8 TMA descriptor passing and bench CUDA runtime`):

1. `src/ops/linear/fp8/fp8_a8_tma_mma.cuh`: MSVC rejects a by-value `alignas(128)` kernel
   parameter (C2719). Under `_WIN32` the descriptor block is copied to a pool-backed device
   buffer and the kernel receives a pointer; POSIX keeps the by-value `__grid_constant__`
   spelling. This mirrors the existing NVFP4 TMA workaround.
2. `bench/inference/benchmarks.cmake` and `bench/context_cost/benchmarks.cmake`: benchmark
   targets link `${NINFER_CUDART_TARGET}` instead of `CUDA::cudart`, avoiding a duplicate
   `cudaGetExportTable` against the statically linked CUDA runtime.

`build-support/FindFFMPEG.cmake` and `FindCURL.cmake` resolve FFmpeg and libcurl from
`windows-libs/` instead of vcpkg. They are committed for rebuild reproducibility.

## Toolchain

- Windows 11 x64
- Visual Studio 2022 Professional, MSVC 14.44.35207
- CUDA Toolkit 13.1.115
- CMake 3.31.0
- Ninja 1.11.1
- Target architecture: `sm_120a` (RTX 5090)

## Configure

Run from a shell where the MSVC environment is initialized (for example `vcvars64.bat`). The
`CL` value supplies the host-compiler flags the fork expects:

```powershell
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
set CL=/utf-8 /Zc:preprocessor

cmake -S L:\Temp\ninfer -B L:\Temp\ninfer\build-windows -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_CUDA_ARCHITECTURES=120a `
  -DNINFER_BUILD_APPS=ON `
  -DNINFER_BUILD_BENCHMARKS=ON `
  -DBUILD_TESTING=OFF `
  -DCMAKE_MODULE_PATH=L:/Temp/ninfer/build-support
```

## Build

```powershell
cmake --build L:\Temp\ninfer\build-windows -j
```

Outputs:

```text
build-windows/apps/ninfer.exe
build-windows/apps/ninfer-serve.exe
build-windows/apps/ninfer-perplexity.exe
build-windows/bench/ninfer_bench.exe
```

## Deploy

Copy the executables into `bin/`. The runtime DLLs already in `bin/` come from the same
`windows-libs` toolchain (FFmpeg 62/60/9, libcurl 8.13, and their transitive dependencies).

## Smoke test

```powershell
.\bin\ninfer-serve.exe L:\LLM_Models\Ninfer\models\qwen3_8_27b_nvfp4full.ninfer `
  --host 127.0.0.1 --port 8080 `
  --max-context 100000 --kv-capacity 100000 --max-concurrency 1 `
  --kv-dtype fp8 --spec dflash2 --draft-tokens 7 `
  --seed 123456 --greedy --no-prefix-reuse
```

A successful startup reports the engine, a 100,032-token FP8 KV capacity, and CUDA graph
readiness. A short chat request must complete with `dflash2 accepted` statistics.
