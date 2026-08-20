# NVFP4 Processing Reference

This document is a code-oriented reference for optimizing the current NVFP4 paths in NInfer.
It describes what happens to the input activation, how NVFP4 weights are laid out and consumed,
how the dot product is accumulated, and how results reach the output tensor.

The prioritized RTX 5090 improvement program is in
[`nvfp4_optimization.md`](nvfp4_optimization.md).

The persistent numeric contract is defined in [`docs/maintainer/tensor-formats.md`](docs/maintainer/tensor-formats.md).
The byte layout is defined in [`docs/maintainer/storage-layouts.md`](docs/maintainer/storage-layouts.md).
This document focuses on the implementation and performance implications of those contracts.

## Scope

The implemented exact NVFP4 matrix problems are:

| Name | N | K | Use |
|---|---:|---:|---|
| `AttnInput` | 14336 | 5120 | full-attention Q/K/gate/V parent |
| `GdnInput` | 16384 | 5120 | GDN Q/K/V/Z parent |
| `MlpGateUp` | 34816 | 5120 | MLP gate/up parent |
| `Residual6144` | 5120 | 6144 | attention/GDN output or residual projection |
| `Residual17408` | 5120 | 17408 | MLP down projection |

The geometry and admission rules are in `src/ops/linear/nvfp4/nvfp4_config.h`.

There are two fundamentally different activation policies:

- **A16**: BF16 activation is consumed directly by an NVFP4 weight-only GEMV path.
- **A4**: BF16 activation is quantized at runtime to E2M1 plus E4M3 scales, then consumed by
  a W4A4 Tensor Core path.

The A4 public operation includes both activation quantization and matrix execution. The benchmark
contract explicitly measures both phases (`docs/maintainer/linear-benchmark.md:100-102`).

## Persistent Weight Input

An NVFP4 matrix has logical shape `[N,K]`, with `N % 128 == 0` and `K % 64 == 0`.

For every group of 16 values along K:

```text
weight[n,k] = E2M1(code[n,k]) * E4M3FN(scale[n,floor(k/16)]) / d_w
```

`d_w` is one positive FP32 divisor for the complete matrix. The input activation divisor `d_x`
is separate model metadata and is not part of the persistent NVFP4 format.

The artifact payload is:

```text
code plane:   N * K / 2 bytes
padding:      zero bytes to the next 256-byte boundary
scale plane:  N * K / 16 bytes
divisor:      one FP32 word
```

The runtime `Weight` points directly into these planes. `validate_nvfp4_weight()` checks geometry,
alignment, positive finite divisors, and that `qdata` and `scales` point at the expected payload
offsets (`src/ops/linear/nvfp4/nvfp4_format.cpp:38-70`). No runtime weight repacking is performed.

### Code plane

Each byte contains two E2M1 values. The low nibble is the lower K coordinate and the high nibble
is the next K coordinate. A row therefore occupies `K/2` bytes.

### Scale plane

The scale plane is physically tiled for the kernel. For `row_tile = n/128`, `row_inner = n%128`,
`scale_tile = group/4`, and `scale_lane = group%4`, the byte offset is:

```text
(row_tile * (K/64) + scale_tile) * 512
+ (row_inner % 32) * 16
+ floor(row_inner / 32) * 4
+ scale_lane
```

This is `blockscale-k16-m128x4-v1`, documented in `docs/maintainer/storage-layouts.md:254-298`.
The layout is already suitable for the grouped scale accesses used by the CUDA kernels.

## A4 Input Phase

### Workspace

`nvfp4_dispatch()` selects A4 only when policy is `AllowA4` and the exact shape/token threshold
allows it (`src/ops/linear/nvfp4/nvfp4_dispatch.cpp:20-41`). It allocates two temporary device
planes through the caller-owned `WorkspaceArena`:

```text
activation codes:  T * K/2 bytes
activation scales: T * K/16 bytes
```

Both allocations are 256-byte aligned (`src/ops/linear/nvfp4/nvfp4_w4a4_plan.h:31-43`). The input
tensor remains BF16 and is not modified.

### Quantization kernel

`launch_nvfp4_w4a4_quantize()` launches one task per activation token and K16 group:

```text
tasks = T * (K/16)
```

The kernel is `nvfp4_w4a4_quantize_kernel()` in
`src/ops/linear/nvfp4/nvfp4_w4a4_mma.cuh:387-407`.

For each K16 group it:

1. Loads 16 BF16 values as two `uint4` vectors.
2. Expands the eight packed BF16 pairs to FP32 pairs.
3. Computes the maximum absolute value over all 16 values.
4. Computes `raw_scale = d_x * max_abs / 6`.
5. Converts `raw_scale` to E4M3FN with saturate-finite conversion.
6. Decodes the rounded E4M3 scale.
7. Normalizes every value as `value * d_x / decoded_scale`.
8. Converts the normalized values to E2M1x2 using CUDA conversion assembly.
9. Stores 8 packed code bytes and one scale byte.

The implementation is in `quantize_nvfp4_k16()` (`src/ops/linear/nvfp4/nvfp4_codec.cuh:63-92`).
The scale divisor is applied during normalization, while the matrix `d_w` is applied later in the
dot-product epilogue through `alpha`.

Important cost centers in this phase:

- one global BF16 read of all `T*K` activation values;
- FP32 expansion, max reduction, FP8 conversion, and 16 divisions per K16 group;
- one global write of both temporary planes;
- a mandatory kernel boundary before W4A4 execution.

The quantized result is a row-major temporary layout, not the persistent weight scale layout.
The W4A4 kernels remap it while staging.

## A4 Matrix Phase: MMA Path

For ordinary A4 linear execution, `launch_nvfp4_w4a4()` quantizes first and then selects a schedule
based on token count (`src/ops/linear/nvfp4/nvfp4_w4a4.cu:91-131`). The normal schedules use:

```text
BlockK = 256
BlockN = 64 or 128
BlockM = 32, 64, or 128
Stages = 1 or 2
```

The grid is output-row tiles by token tiles. `launch_gemm()` computes:

```text
grid.x = N / BlockN
alpha   = 1 / (d_x * d_w)
```

The kernel is `nvfp4_w4a4_mma_kernel()` in
`src/ops/linear/nvfp4/nvfp4_w4a4_mma.cuh:204-385`.

### Staging

Each CTA double-buffers or multi-buffers these objects in shared memory:

- activation E2M1 code rows;
- weight E2M1 code rows;
- activation E4M3 scales;
- weight E4M3 scales.

Activation and weight codes are copied in 16-byte units with `cp_async`. Rows are XOR-swizzled by
16-byte segment before the copy so that `ldmatrix` accesses are conflict-free. The helper is
`nvfp4_w4a4_swizzled_byte()` (`src/ops/linear/nvfp4/nvfp4_w4a4_mma.cuh:70-77`).

Activation tails beyond the active token count use zero-fill (`cp_async_zfill`), and output stores
explicitly suppress invalid tokens (`nvfp4_w4a4_mma.cuh:81-97`, `352-383`).

Weight scale loading differs from activation scale loading:

- contiguous weight rows use a packed 128-row tile/quartile mapping;
- non-contiguous row policies use the full physical scale offset calculation;
- activation scales are copied as raw four-byte scale groups.

See `stage_nvfp4_w4a4_activation()` and `stage_nvfp4_w4a4_weight()` in
`src/ops/linear/nvfp4/nvfp4_w4a4_mma.cuh:79-202`.

### Tensor Core operation

After staging, each warp loads E2M1 fragments with `ldmatrix.x4` for activations and `ldmatrix.x2`
for weights. Scales are loaded into registers and passed to:

```text
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X
m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
```

The wrapper is `mma_nvfp4_e4m3()` in `src/ops/common/mma.cuh:83-103`.
Each MMA produces four FP32 accumulators per output fragment. The hardware instruction performs
the E2M1 dot product using the supplied E4M3 activation and weight scales. The software scale
factor `alpha = 1/(d_x*d_w)` is applied after accumulation.

The kernel loops over `K/256` K tiles, with two K64 MMA operations per 256-wide stage. Between
stages, asynchronous copies and `cp_wait()` overlap global-memory traffic with MMA work.

## Large-T TMA Path

For ordinary Linear and LinearAdd, the TMA route is selected when `T >= 1024` and `T % 256 == 0`
(`src/ops/linear/nvfp4/nvfp4_w4a4.cu:52-86`,
`src/ops/linear_add/nvfp4/nvfp4_linear_add_w4a4.cu:57-69`).

The TMA schedule uses `BlockM=256`, `BlockN=128`, `BlockK=128`, four consumer warps groups,
and either two or three stages. MLP gate/up uses a two-stage variant; the other paths use three
stages where configured (`src/ops/linear/nvfp4/nvfp4_w4a4_tma.cu:19-20`).

The host creates four `CUtensorMap` descriptors for activation codes, weight codes, activation
scales, and weight scales. The kernel has one producer side and multiple consumer warps:

1. The producer waits on the stage's empty barrier.
2. It issues four TMA loads into shared memory.
3. It records the expected transaction byte count.
4. Consumers wait on the full barrier.
5. Consumers load fragments and execute the same NVFP4 MMA instruction.
6. The last consumer signals the stage empty barrier.

The implementation is `nvfp4_w4a4_tma_kernel()` in
`src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh:201-399`.

The TMA descriptors use 64-byte swizzle for code planes and no swizzle for scale planes
(`nvfp4_w4a4_tma.cuh:63-94`). On Windows, descriptor bytes are passed by value in grid-constant
memory so CUDA Graph replay does not retain a host-stack pointer (`nvfp4_w4a4_tma.cuh:24-35` and
`195-215`).

## A16 Input and Matrix Phase

A16 is not activation-quantized. `launch_a16()` chunks the public input into at most 32 tokens and
dispatches either decode GEMV for `T=1` or a specialized small-T kernel for `T=2..32`
(`src/ops/linear/nvfp4/nvfp4_dispatch.cpp:44-59`).

The A16 paths read:

- BF16 activation directly from the public input tensor;
- packed NVFP4 weight codes from the artifact;
- swizzled E4M3 weight scales from the artifact.

`nvfp4_gemv_kernel()` (`src/ops/linear/nvfp4/nvfp4_gemv.cuh:203-248`) assigns each warp a set of
output rows. For each K phase it:

1. Loads packed E2M1 codes for each assigned output row.
2. Loads or stages the corresponding E4M3 weight scales.
3. Loads BF16 activation pairs.
4. Decodes E2M1 pairs and E4M3 scales to FP32.
5. Performs scalar `fmaf` operations into multiple independent FP32 chains.
6. Reduces chains and then reduces across the warp.

The coefficient is `decode_e4m3(scale) / d_w`; unlike A4, `d_x` is not applied because the public
activation remains BF16. The source is `load_nvfp4_coefficients()` in
`src/ops/linear/nvfp4/nvfp4_gemv.cuh:114-146`.

Small-T schedules specialize token reuse, values-per-lane, activation access, scale access, and
warp count. For `T=2..4`, some geometries stage an activation phase in shared memory; other cases
load token-packed vectors directly (`src/ops/linear/nvfp4/nvfp4_small_t.cuh:53-205`). These choices
are measured RTX 5090 production schedules, not semantic requirements
(`src/ops/linear/nvfp4/nvfp4_config.h:163-234`).

## Output Phase

### Plain linear

The identity epilogue returns the FP32 accumulator unchanged. Values are converted to BF16 with
round-to-nearest and stored in token-major layout:

```text
output[token * N + row]
```

The common output policy is `Nvfp4ContiguousOutput` in
`src/ops/linear/nvfp4/nvfp4_output.cuh:17-30`.

### LinearAdd

The residual is read as BF16, converted to FP32, and added to the scaled NVFP4 result before the
single BF16 output store. The input and output point to the same residual tensor, so this fuses the
residual update without a second output kernel (`src/ops/linear_add/nvfp4/nvfp4_linear_add_epilogue.cuh:9-17`).

### SwiGLU

The fused gate/up route treats the 34816-row parent as two 17408-row branches. It loads and computes
both branches in one CTA, then applies:

```text
output = silu(gate * alpha) * (up * alpha)
```

and stores BF16 output. The non-TMA path pairs gate and up rows through `Nvfp4SwiGluRows`
(`src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4.cu:23-57`). The TMA path stages separate gate
and up code/scale tiles and performs the same fused epilogue
(`src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cuh:188-267`).

For token counts outside the fused routes, the implementation materializes the 34816-row projection
and runs a separate `silu_mul` operation (`src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp:131-142`).

## Route Matrix

### Linear

`AllowA4` thresholds in `nvfp4_dispatch.cpp` are:

| Problem | A16 below | A4 from |
|---|---:|---:|
| Attention input | 4 | 4 |
| GDN input | no A16 threshold in selector | all admitted T |
| MLP gate/up | 5 | 5 |
| Residual 6144 | 8 | 8 |
| Residual 17408 | 8 | 8 |

Within A4, `T >= 1024` and a multiple of 256 selects TMA. Smaller T selects one of the normal MMA
schedules. Within A16, `T=1` selects GEMV and `T=2..32` selects small-T kernels; larger public inputs
are chunked into 32-token pieces.

### Fused SwiGLU

`nvfp4_linear_swiglu_plan.cpp:26-41` selects:

| T | Route |
|---:|---|
| 1 | fused A16 decode |
| 2..4 | fused A16 small-T |
| 5..96 | fused W4A4 |
| positive multiples of 256 from 256 | fused W4A4 TMA |
| other A4 values | W4A4 linear followed by separate `silu_mul` |

This means non-M256 values above `T=96` do not use fused TMA and may materialize the large gate/up
projection.

## Optimization Progress

### 2026-08-20: Extend fused SwiGLU TMA to M256 token counts

**Status:** completed and retained.

The first optimization extended the existing fused W4A4 TMA LinearSwiGLU route from only `T=1024`
to every positive multiple of 256 from `T=256`. The fused route avoids materializing the
`[34816,T]` BF16 gate/up projection and avoids the separate `silu_mul` launch.

Changed implementation:

- `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp` now selects fused TMA when
  `T >= 256 && T % 256 == 0`;
- interval workspace planning includes the largest fused M256 point and retains baseline workspace
  for non-M256 values in the requested interval;
- `tests/ops/linear_swiglu/test_nvfp4.cpp` covers the newly admitted `T=256`, `512`, and `768`
  routes in addition to `T=1024`.

RTX 5090, CUDA 13.1, Release `sm_120a`, cold-cache public LinearSwiGLU benchmark with 256 MiB L2
flush, 10 warmups, and 50 measured launches:

| T | Materialized baseline | Fused TMA | Change | Useful TFLOP/s after |
|---:|---:|---:|---:|---:|
| 256 | 148.544 us | 136.256 us | -8.3% | 669.83 |
| 512 | 305.152 us | 230.720 us | -24.4% | 791.16 |
| 768 | 403.520 us | 322.624 us | -20.1% | 848.68 |

The baseline and candidate used the same executable benchmark contract, input generation, cold-cache
method, and token points. Raw local CSV outputs are under `profiles/bench/` and are intentionally not
repository documentation.

Verification:

- `ninfer_linear_swiglu_nvfp4_test` passed all A16 and A4 cases;
- the test directly qualifies outputs against the independent profile oracle;
- the test verifies workspace scope restoration and exact capacity/high-water agreement;
- a focused Release build of the benchmark and correctness target completed successfully.

Remaining gap: non-M256 values from `T=97` upward still use materialized Linear plus `silu_mul`.
The next route-level experiment should determine whether tail-capable fused TMA or an extended
cp.async fused schedule wins for those values.

### 2026-08-20: Reject tcgen05/TMEM mainloop for RTX 5090

**Status:** feasibility checked and rejected before implementation.

The planned SM120 `tcgen05`/TMEM prototype cannot run on RTX 5090. CUDA 13.1 exposes Tensor Memory
and `tcgen05` only for the `sm_100`, `sm_103`, and `sm_110` architecture families. The installed
CCCL wrappers explicitly omit `sm_120`, and CUDA 13.1 `ptxas` rejects `tcgen05.alloc` and
`.cta_group::1` when targeting `sm_120a` while accepting the same probe for `sm_100a`.

RTX 5090 does support the existing warp-level block-scaled instruction:

```text
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.
m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
```

Therefore no candidate kernel or production route was added. Large-T work on RTX 5090 must improve
the supported `mma.sync` mainloop, its TMA pipeline, or surrounding vertical dataflow. The next
architectural experiment is the quantized MLP handoff, which can remove the BF16 SwiGLU output
write/read and standalone down-projection quantization launch.

### 2026-08-20: Reject inline quantized MLP handoff

**Status:** implemented as an isolated candidate, measured, rejected, and removed.

The candidate changed the fused M256 SwiGLU TMA epilogue to preserve the semantic BF16 rounding in
shared memory and quantize those exact values directly into down-projection E2M1 codes and E4M3
scales. The existing down LinearAdd contraction then consumed those planes without writing or
rereading a global `[17408,T]` BF16 activation and without launching its standalone quantizer.

RTX 5090, CUDA 13.1, Release `sm_120a`, cold-cache complete post-mixer measurement with 256 MiB L2
flush, 10 warmups, and 50 measured launches:

| T | Existing two-Op path | Quantized handoff | Change |
|---:|---:|---:|---:|
| 256 | 256.864 us | 252.992 us | -1.5% |
| 512 | 329.824 us | 327.776 us | -0.6% |
| 768 | 495.712 us | 487.520 us | -1.7% |
| 1024 | 629.888 us | 637.056 us | +1.1% |

The candidate did not satisfy the primary `T=1024` acceptance criterion. Inline K16 quantization
extends every gate/up CTA's epilogue and residency, offsetting the removed BF16 traffic and causing a
full-prefill regression. The candidate kernel, composite route, workspace changes, and temporary
benchmark were removed. No production handoff remains.

The result narrows future work: dataflow fusion should not serialize down-input quantization onto the
gate/up CTA critical path. A future attempt would need an overlapped producer/consumer design or a
faster standalone quantizer rather than this inline epilogue architecture.

### 2026-08-20: Reject GDN Snapshot post-kernel tile retuning

**Status:** measured, rejected, and reverted.

The current NVFP4 materialized Snapshot post kernel assigns each warp an 8-token tile. Every tile
loads three preceding projected columns as convolution history. A 32-token candidate reduced this
duplicated projected-plane traffic by about 75% while preserving the width-4 convolution and state
publication formula.

RTX 5090, CUDA 13.1, Release `sm_120a`, complete public NVFP4 A4 Snapshot at `T=1024`, `B=1`, 256
MiB L2 flush, 10 warmups, and 50 measured launches:

| Execution | Existing T8 tile | Candidate T32 tile | Change |
|---|---:|---:|---:|
| CUDA Graph, cold | 239.392 us | 241.760 us | +1.0% |
| CUDA Graph, warm | 245.088 us | 246.560 us | +0.6% |
| eager, cold | 242.752 us | 244.832 us | +0.9% |
| eager, warm | 251.328 us | 254.080 us | +1.1% |

The independent Snapshot correctness test passed, but the larger per-warp serial tile reduced useful
latency hiding enough to outweigh the lower read duplication. The production 8-token tile was
restored.

The investigation also established that normal Engine prefill does not invoke the `T=1024`
Snapshot Op: it executes GDN projection followed by ordinary `causal_conv1d_silu`. Record admits
only `T=2..16`. A specialized Snapshot seam-buffer fusion could remove approximately 46.7 MiB of
intermediate traffic, but would optimize a public benchmark route rather than the main prefill path.
An impactful GDN fusion must instead target the actual prefill composition and solve cross-CTA
width-4 token dependencies there.

### 2026-08-20: Extend fused cp.async SwiGLU through T=96

**Status:** completed and retained.

The existing fused W4A4 cp.async LinearSwiGLU route previously ended at `T=48`. The materialized
Linear plus `silu_mul` route was compared against the fused route across the final-prefill tail
domain. The fused kernel remains faster through `T=96`; at `T=97`, its 48-token block geometry
requires a third token tile and latency jumps, while the materialized Linear route switches to a
much stronger schedule.

RTX 5090, CUDA 13.1, Release `sm_120a`, cold-cache public LinearSwiGLU, 256 MiB L2 flush, 10 warmups,
and 50 measured launches at the principal points:

| T | Materialized baseline | Retained fused route | Change |
|---:|---:|---:|---:|
| 49 | 144.480 us | 136.288 us | -5.7% |
| 64 | 144.512 us | 138.336 us | -4.3% |
| 80 | 171.072 us | 138.336 us | -19.1% |
| 95 | 173.152 us | 140.384 us | -18.9% |
| 96 | 173.216 us | 140.320 us | -19.0% |

At the rejected side of the boundary, the materialized route measured `83.040 us` at `T=97` and
`85.120 us` at `T=128`, so production deliberately returns to materialization from `T=97` except
for the separately qualified M256 fused TMA points.

Implementation and verification:

- `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp` selects fused W4A4 for `T=5..96` and
  plans workspace consistently across intervals;
- correctness cases now cover `T=96` and `T=97` directly;
- `ninfer_linear_swiglu_nvfp4_test` passed the independent numerical and workspace checks.

## Optimization Map

### Activation quantization

- Measure quantization separately from MMA, then measure the complete public operation.
- The quantizer reads every activation element and writes two temporary planes. Fusion with the
  consumer would remove one global write/read cycle and one launch, but would require integrating
  per-K16 max reduction with the MMA staging pipeline.
- The current quantizer performs FP32 division for every nonzero value. Any replacement must preserve
  the registered activation compute profile and be checked against the independent operator oracle.
- The quantizer uses one thread per K16 group. Check whether group-level occupancy, memory coalescing,
  and conversion instruction throughput are limiting before changing arithmetic.

### A16 decode

- A16 has no temporary activation plane and is likely sensitive to repeated weight-code and scale
  traffic at low T.
- `StagedRaw` versus `Direct` scale access and `Default` versus `Streaming` code cache are explicit
  schedule dimensions in `nvfp4_config.h`.
- The GEMV accumulator chain count, rows per warp, and values per lane trade register pressure
  against instruction-level parallelism and occupancy.
- Small-T activation access modes determine whether BF16 values are reused from shared memory,
  packed per token, or streamed directly.

### W4A4 MMA

- The main levers are BlockM/BlockN, stage count, producer/consumer split, shared-memory footprint,
  register repartitioning, and `MinBlocksPerSm`.
- Code swizzle and `ldmatrix` addressing are coupled. Changing layout requires changing both staging
  and fragment address calculations.
- Scale loads are a separate traffic path. Profile scale transactions and shared-memory bank
  behavior independently from code loads.
- The normal MMA path uses `cp_async`; the large-T path uses TMA and mbarriers. Do not infer that a
  TMA improvement transfers to low-T execution.

### Output and fusion

- Plain output writes one BF16 result per matrix element.
- LinearAdd already folds residual read, addition, conversion, and store into the epilogue.
- SwiGLU fused routes avoid a 34816-row BF16 intermediate and a second kernel. The fallback route
  does not; its workspace and extra memory traffic are visible in the plan.
- Output shared-memory staging adds a synchronization and a second shared-memory pass before global
  stores in the TMA kernel. It exists to make the epilogue and coalesced vector stores compatible;
  profile whether it remains optimal for each output geometry.

## Measurement Commands

The registered public Linear points are documented in `docs/maintainer/linear-benchmark.md:64-98`.
Representative commands are:

```bash
./build/bench/ninfer_linear_bench --qtype nvfp4 --policy a16 --n 14336 --k 5120 --t 1
./build/bench/ninfer_linear_bench --qtype nvfp4 --policy a4 --n 14336 --k 5120 --t 1024
./build/bench/ninfer_linear_bench --qtype nvfp4 --policy a4 --n 34816 --k 5120 --t 1024
```

Fused benchmarks:

```bash
./build/bench/ninfer_nvfp4_linear_add_bench --n 5120 --k 6144 --policy a4 --t-sweep 8,17,1024
./build/bench/ninfer_nvfp4_linear_swiglu_bench --policy a4 --t-sweep 5,48,49,128,1024
```

For kernel attribution, use `--profile` with one token point and launch Nsight Compute around the
benchmark. The benchmark flushes L2 before timed launches and includes the production A4 quantization
phase. Existing correctness tests cover A16 and A4 route boundaries, fused residual behavior, and
fused SwiGLU behavior:

- `tests/ops/linear/test_nvfp4_a16.cpp`
- `tests/ops/linear/test_nvfp4_a4.cpp`
- `tests/ops/linear_add/test_nvfp4.cpp`
- `tests/ops/linear_swiglu/test_nvfp4.cpp`

## Correctness Constraints

Optimization must preserve:

- E2M1 code nibble order and all 16 code meanings;
- E4M3FN scale decoding;
- `weight = code * block_scale / d_w`;
- A4 activation scale selection using `d_x * max_abs / 6`;
- BF16 output conversion and the semantic output layout;
- fused residual and SwiGLU operation ordering;
- valid-token handling for partial token tiles;
- CUDA Graph/TMA descriptor lifetime and replay behavior.

The mathematical NVFP4 weight oracle is in `docs/maintainer/tensor-formats.md:279-319` and the
operator-oracle boundary is in `docs/maintainer/tensor-formats.md:824-831`. Production kernels may
choose different private staging and reduction profiles, but every route must be checked against the
same represented-weight oracle.
