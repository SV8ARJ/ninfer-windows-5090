# NVFP4 RTX 5090 Optimization Plan

This document defines the optimization plan for NInfer's NVFP4 execution on NVIDIA GeForce RTX
5090 (`sm_120a`). It builds on the implementation trace in [`nvfp4.md`](nvfp4.md) and separates
existing measurements from optimization hypotheses.

The objective is maximum end-to-end inference performance for the registered NVFP4 artifacts,
without changing their model semantics or weakening numerical qualification.

## 1. Executive Summary

The highest-value optimization order is:

1. eliminate unnecessary MLP SwiGLU and down-projection intermediate traffic;
2. improve the large-T TMA contraction mainloop;
3. eliminate the materialized GDN A4 intermediate for Qwen3.6-27B;
4. tune the `T=4..48` A4 domain used by concurrent decode and MTP verification;
5. retune fused A16 MLP decode;
6. optimize activation quantization after measuring its contribution at every K/T regime.

The primary reason for this order is that activation quantization is not currently the dominant
cost at the measured `T=1024` attention-input point. Existing attribution estimates approximately
`11.52 us` for quantization in a `152.576 us` complete call. The effective TMA contraction takes
approximately `141 us`, reaches about `1066 TFLOP/s`, and reports substantial L2-tag pressure.

The strongest architectural opportunities therefore remove full-sized BF16 intermediates and
improve contraction throughput. Standalone quantizer arithmetic is secondary unless new profiling
shows that it dominates smaller token regimes.

## 2. Product-Relevant NVFP4 Workloads

### 2.1 Qwen3.6-27B NVFP4

The Qwen3.6-27B NVFP4 artifact contains 247 NVFP4 Text parents:

| Role | Parents |
|---|---:|
| attention input | 10 |
| attention output | 14 |
| GDN input | 48 |
| GDN output | 47 |
| MLP gate/up | 64 |
| MLP down | 64 |

This target makes MLP and GDN vertical fusion both important. Artifact assignment is documented in
`docs/maintainer/qwen3.6-27b-artifact.md`.

### 2.2 Qwen3.8-27B NVFP4

Qwen3.8-27B uses NVFP4 for MLP gate/up and down projections in Text layers `0..55`. Its attention
and GDN projections use row-scaled FP8. Consequently, an NVFP4 improvement shared by both targets
must first benefit:

- fused LinearSwiGLU `[34816,5120]`;
- MLP down LinearAdd `[5120,17408]`.

The allocation is documented in `docs/maintainer/qwen3.8-27b-artifact.md:69-100`.

### 2.3 Token regimes

The relevant token counts are:

| Regime | Representative T | Main concern |
|---|---:|---|
| single-request decode | 1 | A16 weight bandwidth and scalar FMA throughput |
| small concurrent decode | 2..8 | A16/A4 crossover and launch overhead |
| MTP3 verification | approximately `4 * active_requests` | A4 cp.async semantic Ops |
| aggregate decode | up to 48 in current measured surfaces | A4 reuse and fusion |
| final prefill tail | 49..1023 | cp.async/TMA coverage and tail efficiency |
| full prefill chunk | 1024 | TMA throughput and vertical fusion |

## 3. Existing RTX 5090 Evidence

This section records measurements already present in repository history or active documentation. It
does not claim that every value represents the latest source revision.

### 3.1 Complete T=1024 Linear

The active Linear benchmark reference reports complete public calls, including activation
quantization and contraction, on RTX 5090 with CUDA 13.1 and cold-cache measurement:

| Problem `[N,K]` | Median | Useful TFLOP/s | Dense-FP4 reference utilization |
|---|---:|---:|---:|
| `[14336,5120]` | 152.576 us | 985.24 | 58.79% |
| `[16384,5120]` | 174.784 us | 982.92 | 58.65% |
| `[34816,5120]` | 390.112 us | 935.81 | 55.84% |
| `[5120,6144]` | 72.992 us | 882.62 | 52.66% |
| `[5120,17408]` | 197.888 us | 922.42 | 55.04% |

Source: `docs/maintainer/linear-benchmark.md:482-496`.

The dense-FP4 percentage is a fixed comparison reference, not a measured ceiling for these kernels.

### 3.2 Attention-input attribution

Historical attribution for `[14336,5120]`, `T=1024`, reports:

```text
complete quantization + contraction: 152.576 us
activation quantization:              approximately 11.52 us
L2-tag throughput:                     79.89%
waves per SM:                          2.64
local/stack spill:                     none reported
```

This is the strongest retained indication that the contraction and its memory pipeline offer more
headroom than standalone quantization at this point.

### 3.3 A16 decode

Historical cold-cache attention-input decode measured `32.000 us` at `T=1`, corresponding to about
`1291.5 GB/s`, or 77.12% of the measured sustained-read reference used in that campaign.

The measured winning schedule remains encoded in
`src/ops/linear/nvfp4/nvfp4_config.h:163-168`:

```text
8 warps per CTA
2 rows per warp
16 values per lane
4 accumulator chains
staged raw scales
2 minimum blocks per SM
```

This suggests less headroom for plain A16 attention decode than for large-T contraction. Fused MLP
decode may still have different scale-traffic and register-pressure bottlenecks.

### 3.4 Fused SwiGLU

Historical `T=1024` measurements report:

```text
materialized Linear + silu_mul: 432.736 us
fused W4A4 TMA:                408.160 us
second fused sample:            404.736 us
fused improvement:              approximately 5.7%
materialized workspace:         approximately 70.8125 MiB
fused workspace:                approximately 2.8125 MiB
```

This demonstrates that eliminating the full gate/up intermediate improves both latency and memory
requirements. Current routing now uses fused cp.async through `T=96` and fused TMA at every positive
M256 point from `T=256`; other values above `T=96` use the materialized fallback.

### 3.5 End-to-end relevance

Qwen3.6-27B NVFP4 concurrent MTP3 decode scales from `202.4 tok/s` at one request to `1146.9 tok/s`
at eight requests (`docs/performance.md:113-162`). This makes the aggregate small-T A4 domain
important in addition to single-request A16 decode.

Short-context Qwen3.6-27B NVFP4 prefill reaches `11191.5 tok/s`, while longer contexts become
increasingly dominated by attention and KV work (`docs/performance.md:485-498`). Improvements to
linear paths should therefore be evaluated at both short and long context, with expectations scaled
to their actual phase contribution.

## 4. Current Route Gaps

### 4.1 LinearSwiGLU

Current `AllowA4` routing in
`src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp:26-40` is:

| T | Route |
|---:|---|
| 1 | fused A16 decode |
| 2..4 | fused A16 small-T |
| 5..96 | fused W4A4 cp.async |
| positive multiples of 256 from 256 | fused W4A4 TMA |
| all other A4 T | materialized Linear followed by `silu_mul` |

The first completed optimization admitted fused TMA for every positive M256 token count from
`T=256`; a second completed optimization extended fused cp.async through `T=96`. Measurements and
verification are recorded in [`nvfp4.md`](nvfp4.md). The remaining gap is non-M256 values above
`T=96`, which still materialize the full gate/up projection.

### 4.2 GDN

The A16 GDN path can fuse projection, convolution, activation, split outputs, and state publication.
The A4 context and batched routes materialize projection output and run post-processing kernels.
Qwen3.6 has 48 GDN input sites, making this a large repeated cost.

### 4.3 TMA architecture

The large-T kernel uses TMA for global-to-shared transfers but still executes warp-level
`mma.sync` through `mma_nvfp4_e4m3()` in `src/ops/common/mma.cuh:83-103`. It does not use a
Blackwell fifth-generation Tensor Core/TMEM mainloop.

The current TMA design also:

- operates at approximately one CTA per SM due to shared memory and register use;
- reloads some activation-scale bytes because a K128 stage consumes half of a 16-byte TMA box;
- stages FP32 results through shared memory before vectorized global stores;
- creates tensor-map descriptors on the host for every launch.

These are profiling targets, not established bottlenecks for every geometry.

## 5. Prioritized Improvement Plan

## Phase 0: Re-establish Attribution

Before changing kernels, capture a current baseline for the exact source revision and Windows CUDA
13.1 environment.

### Workloads

Measure these semantic operations:

- LinearSwiGLU `[34816,5120]`;
- LinearAdd `[5120,17408]`;
- LinearAdd `[5120,6144]`;
- attention input projection;
- GDN input projection and GDN Snapshot/Record;
- complete MLP post-mixer;
- one complete GDN layer transition.

Use these token points where admitted:

```text
1, 2, 4, 5, 7, 8, 16, 17, 32, 33, 48, 49,
128, 256, 512, 768, 1024
```

### Measurement modes

Use both:

- cold-cache operator benchmarks;
- CUDA Graph replay through the production execution route.

Collect:

- complete operation latency;
- per-kernel duration;
- tensor-pipe and SM utilization;
- L2 and DRAM bytes and throughput;
- registers per thread and shared memory per CTA;
- achieved occupancy and active waves;
- barrier, long-scoreboard, math-pipe, and memory-throttle stalls;
- Nsight Systems kernel count and launch gaps;
- `cuTensorMapEncodeTiled` call count and host time.

### Deliverable

Produce one route table containing current winners and one attribution summary for:

```text
quantization | contraction | epilogue/post | launch/API overhead
```

Do not tune from isolated kernel time when the public operation includes additional kernels.

## Phase 1: Extend Fused SwiGLU Coverage

This is the first implementation candidate because the fused design already exists and has measured
benefit.

### Experiments

1. Retain the completed fused TMA admission at `T=256`, `512`, and `768`.
2. Compare two-stage and three-stage TMA schedules at each point.
3. Compare against materialized Linear plus `silu_mul` using complete operation latency.
4. Add tail handling or a separate fused schedule for non-M256 token counts in `T=49..1023`.
5. Re-evaluate whether the cp.async fused route should extend beyond `T=48` before TMA wins.

### Expected effect

Avoid materializing `[34816,T]` BF16 gate/up output and avoid the separate `silu_mul` launch.

### Acceptance

- faster complete LinearSwiGLU at every newly admitted T;
- lower DRAM write/read traffic;
- lower workspace high-water mark;
- no regression at `T=5..96` or the qualified M256 TMA points;
- direct oracle qualification for each arithmetic profile.

## Phase 2: SM120-Native Large-T Mainloop Feasibility

**Outcome:** rejected on 2026-08-20. CUDA 13.1 and RTX 5090 `sm_120a` do not support Tensor Memory or
`tcgen05`; those facilities are exposed for `sm_100`, `sm_103`, and `sm_110`. Large-T kernel work
must use the supported warp-level NVFP4 `mma.sync` instruction. The feasibility evidence is recorded
in [`nvfp4.md`](nvfp4.md).

The original proposal was to investigate a `tcgen05`/TMEM-based NVFP4 mainloop designed specifically
for `sm_120a`. It is retained below as the rejected experiment definition.

### Development order

1. Confirm that the required block-scaled E2M1/E4M3 operation is exposed for the target toolchain.
2. Prototype one plain Linear geometry, preferably `[14336,5120]`, `T=1024`.
3. Preserve the current persistent code and scale layouts.
4. Compare complete quantize-plus-contraction latency, not contraction alone.
5. Tune CTA dimensions, TMEM allocation, TMA stages, producer/consumer organization, and register
   repartitioning.
6. Extend to `[34816,5120]` and `[5120,17408]` only after a clear public-operation win.
7. Generalize the winning mainloop to split-output, LinearAdd, and fused SwiGLU epilogues.

### Questions to answer

- Is the existing kernel limited by Tensor Core issue rate, scale traffic, L2 tags, barriers, or
  one-CTA-per-SM residency?
- Can TMEM accumulation reduce register pressure or output shared-memory traffic?
- Does a larger K or N tile improve reuse without reducing useful concurrency?
- Are two or three stages optimal per geometry rather than per broad kernel family?

### Acceptance

- material improvement in complete semantic operations;
- improved tensor-pipe utilization or reduced operation time with an explained mechanism;
- no increase in persistent artifact traffic;
- no CUDA Graph replay or Windows descriptor regression.

## Phase 3: Quantized MLP Handoff

**Outcome:** the first inline-epilogue architecture was rejected on 2026-08-20. It improved complete
post-mixer latency by 0.6–1.7% at `T=256..768` but regressed `T=1024` by 1.1%. The candidate was
removed; measurements and the architectural conclusion are recorded in [`nvfp4.md`](nvfp4.md).

The MLP sequence is:

```text
NVFP4 gate/up -> SwiGLU BF16 -> NVFP4 down -> residual BF16
```

The proposed handoff is:

```text
NVFP4 gate/up
    -> apply SwiGLU
    -> round to the semantic BF16 intermediate
    -> quantize that represented BF16 value to temporary E2M1/E4M3
    -> NVFP4 down projection consumes the temporary A4 planes
```

### Why BF16 rounding remains necessary

The current public down projection consumes the represented BF16 SwiGLU output. Directly quantizing
an unrounded FP32 SwiGLU value would change the operator input and therefore model semantics. The
producer may avoid writing BF16 globally, but it must preserve the same BF16 value before A4
quantization.

### Expected traffic reduction at T=1024

The `[17408,1024]` BF16 SwiGLU output is approximately 34 MiB. A producer-side A4 handoff can remove:

- approximately 34 MiB BF16 global write;
- approximately 34 MiB BF16 global read by the generic quantizer;
- one standalone activation-quantization launch.

The temporary A4 handoff is approximately 9.6 MiB for codes plus scales.

### Ownership constraints

- workspace lifetime must span both semantic operations;
- addresses must remain stable during CUDA Graph capture and replay;
- the Program workspace owner, not an individual short-lived Op scope, must own the handoff;
- the one-stream ordering contract must remain explicit.

### Acceptance

- faster complete post-mixer latency;
- faster short-prefill Engine throughput and TTFT;
- unchanged BF16 semantic boundary and output qualification;
- lower global traffic and fewer graph nodes;
- stable replay over repeated requests.

## Phase 4: Fused A4 GDN Pipeline

**Progress:** a low-risk Snapshot post-kernel tile experiment was rejected on 2026-08-20. Increasing
the warp tile from 8 to 32 tokens reduced duplicate projected reads but regressed complete
`T=1024` Snapshot latency by 0.6–1.1%. The original schedule was restored. Investigation also showed
that Engine prefill uses projection plus `causal_conv1d_silu`, not the Snapshot Op; future fusion
must target that actual composition. Details are recorded in [`nvfp4.md`](nvfp4.md).

Develop a complete A4 GDN route that avoids materializing the large projection parent before
convolution and publication.

Target sequence:

```text
NVFP4 projection
    -> width-4 causal convolution
    -> activation/control processing
    -> split output
    -> recurrent-state publication
```

### Design considerations

- width-4 convolution requires adjacent token values and correct boundary state;
- Snapshot and Record have distinct publication semantics;
- valid-column tails must remain zeroed where required;
- FP32 recurrent state and BF16 publication boundaries must remain unchanged;
- batched requests have independent sequence state and valid lengths.

### Expected effect

At `T=1024`, a `[10240,T]` BF16 projection is approximately 20 MiB. Fusion can remove its global
write and reread and reduce post-processing launches. Qwen3.6 repeats this path across 48 GDN
layers.

### Acceptance

- complete GDN Snapshot/Record improvement;
- lower DRAM traffic and kernel count;
- exact state-transition behavior at sequence boundaries;
- no regression in GDN output or subsequent layer behavior;
- measurable Engine prefill or MTP-round improvement.

## Phase 5: Concurrent and MTP A4 Tuning

Tune the `T=4..48` cp.async domain against actual semantic operations.

### Schedule dimensions

- `BlockM`, `BlockN`, and `BlockK`;
- one versus two pipeline stages;
- warps per CTA;
- minimum CTAs per SM;
- activation access mode;
- direct versus staged scale access;
- code cache policy;
- output shared-memory staging and vector-store strategy;
- row-pairing strategy for SwiGLU;
- split-output strategy for Attention and GDN.

### Route selection

Requalify complete-operation A16/A4 crossovers independently for:

- attention input;
- GDN input and Snapshot/Record;
- LinearSwiGLU;
- K6144 LinearAdd;
- K17408 LinearAdd.

Use ordinary batches `B=1..8` and MTP3 aggregate verification sizes near `T=4B`. CUDA Graph replay
must be part of crossover measurement because graph execution changes launch overhead but not
device work or graph-node count.

### Acceptance

- lower complete decode-round latency;
- higher aggregate committed tokens per second;
- MTP acceptance measured separately and unchanged within expected stochastic variation;
- no route-boundary discontinuity that regresses a supported batch size.

## Phase 6: A16 MLP Decode Retuning

Prioritize fused LinearSwiGLU and the down projection rather than plain attention decode.

### Candidates

- stage gate and up scale data together;
- change rows per warp and values per lane;
- retune accumulator-chain count;
- compare direct and staged scale access;
- compare default and streaming code cache policies;
- reduce registers without introducing long-scoreboard stalls;
- specialize paired branch loading for the physical gate/up row separation.

### Metrics

- cold-cache `T=1..4` complete operation latency;
- L2 and DRAM sectors;
- long-scoreboard stalls;
- register count and achieved occupancy;
- complete single-request decode-round latency;
- end-to-end `C=1` decode tokens per second.

## Phase 7: Activation Quantization

Optimize only after obtaining quantizer-only attribution for all production K values and token
regimes.

### Required baseline

Create or use a benchmark that isolates:

```text
BF16 input -> E2M1 code plane + E4M3 scale plane
```

Measure K=`5120`, `6144`, and `17408` over small-T, medium-T, and `T=1024` cases.

### Candidates

- replace repeated division with a reciprocal and multiplications;
- improve instruction scheduling around max reduction and conversions;
- process a K16 group cooperatively rather than with one thread;
- improve vectorized load/store coalescing;
- emit A4 directly from producer Ops that already own the semantic BF16 result;
- fuse quantization with the consumer only when it does not duplicate work across output CTAs.

### Numerical qualification

A reciprocal-multiply implementation is not assumed equivalent to repeated division near E2M1
rounding boundaries. A candidate must either:

- reproduce the current activation code and scale words exactly; or
- define a changed private arithmetic profile and qualify every production route directly against
  the independent FP64 operator oracle.

### Acceptance

- quantizer kernel improvement survives complete-operation measurement;
- no producer occupancy regression that removes the end-to-end gain;
- all zero-block, saturation, tie-boundary, and finite-input cases remain qualified.

## Phase 8: Host and TMA Pipeline Cleanup

These are secondary experiments after profiling identifies their contribution.

### Tensor-map descriptor caching

Stable Program weights and workspace addresses may permit precomputing tensor-map descriptors rather
than calling `cuTensorMapEncodeTiled` on every large-T launch.

Validate with Nsight Systems. Reject this work if encoding time is already hidden behind queued GPU
execution.

Windows must continue passing descriptor bytes by value in grid-constant storage. A host pointer is
not graph-replay safe.

### Activation-scale TMA traffic

A K128 stage consumes eight activation-scale bytes while TMA loads a minimum 16-byte box. Adjacent
K tiles may reload the same scale box. Test whether a K256 stage or revised scale pipeline removes
meaningful L2/TMA traffic without reducing occupancy.

### Output staging

The current TMA epilogue writes FP32-derived BF16 values to shared memory, synchronizes, reloads
vectors, and writes global output. Determine whether TMEM or a direct register-to-global mapping can
remove this pass while retaining coalesced stores and all output policies.

## 6. Persistent Layout Policy

Do not change `blockscale-k16-m128x4-v1` during the initial optimization phases.

A persistent-layout change would affect:

- converters and artifact contracts;
- published artifacts;
- binders and runtime validation;
- every A16 and A4 consumer;
- scale offset and code packing tests;
- zero-copy row views and fused-parent assumptions.

Consider a layout change only if profiling proves that the existing code or scale layout remains a
dominant bottleneck after kernel-local and vertical-fusion work. Any new layout must improve the
complete registered target enough to justify artifact regeneration and removal of the old path.

## 7. Correctness Requirements

Every optimization must preserve or explicitly requalify:

- E2M1 nibble order and all code meanings, including signed zero;
- E4M3FN scale decoding;
- persistent reconstruction `weight = code * block_scale / d_w`;
- A4 K16 maximum and scale selection using `d_x * max_abs / 6`;
- A16 behavior without applying `d_x`;
- the represented BF16 input and output boundaries of each semantic Op;
- LinearAdd residual ordering;
- SwiGLU gate/up row pairing and nonlinear ordering;
- GDN convolution, valid-tail, publication, and recurrent-state semantics;
- partial token-tile handling;
- caller-owned workspace lifetime and address stability;
- CUDA Graph capture/replay behavior;
- Windows by-value TMA descriptor semantics.

The independent oracle exact-decodes the persistent NVFP4 weight and evaluates from represented
public inputs. It does not reproduce a production kernel's quantization staging, reduction tree, or
private intermediate precision. Pairwise parity between old and new kernels is supplementary, not
the correctness oracle.

## 8. Performance Acceptance Gates

An optimization is accepted only when it improves the scope it claims.

### Kernel-level claim

Required evidence:

- same represented workload and route;
- complete kernel duration distribution;
- relevant profiler counters explaining the gain;
- unchanged observable numerical qualification.

### Semantic-Op claim

Required evidence:

- public complete operation, including quantization and post-processing;
- cold-cache and graph-replay measurements;
- workspace and traffic accounting;
- all supported route boundaries relevant to the change.

### End-to-end claim

Required evidence should include the affected cases:

- short prefill throughput and TTFT;
- long-context prefill when the claim applies;
- single-request decode;
- eight-request aggregate decode;
- MTP3 decode and acceptance;
- stable repeated request execution.

An isolated microbenchmark improvement is insufficient if complete semantic-Op or Engine performance
does not improve.

## 9. Recommended Experiment Order

Run the work in this order:

1. current-source Nsight Systems attribution for one `T=1024` Qwen3.6 prefill chunk and `C=1/C=8`
   decode rounds;
2. current-source NCU attribution for dominant LinearSwiGLU, down-projection, and GDN kernels;
3. admit fused SwiGLU TMA at `T=256`, `512`, and `768` as isolated candidates;
4. select fused coverage for the full `T=49..1023` range;
5. prototype an SM120 `tcgen05`/TMEM mainloop on one large-T geometry;
6. implement the quantized MLP handoff if traffic attribution supports it;
7. prototype the fused A4 GDN vertical path;
8. retune cp.async semantic Ops and A16/A4 crossovers for concurrency and MTP;
9. retune fused A16 MLP decode;
10. optimize standalone quantization only after route-specific attribution;
11. investigate descriptor caching, scale TMA duplication, and output staging where counters justify
    them;
12. consider persistent-layout changes only as a final, evidence-driven redesign.

## 10. Initial Success Criteria

The first optimization campaign is complete when it delivers all of the following:

- fused LinearSwiGLU coverage no longer has an unexplained materialization gap in `T=49..1023`;
- the dominant large-T contraction bottleneck is identified with profiler evidence;
- an SM120-native mainloop is either demonstrated as faster or rejected with sufficient evidence;
- complete MLP post-mixer traffic and latency are measured before and after any handoff design;
- concurrent and MTP route boundaries are based on complete graph-replay measurements;
- all accepted changes pass independent numerical qualification and repeated CUDA Graph replay;
- at least one affected Engine workload shows a reproducible end-to-end improvement.

Stop when these deliverables are supported. Do not expand into artifact-layout redesign or unrelated
kernel families without evidence that they limit the requested NVFP4 result.
