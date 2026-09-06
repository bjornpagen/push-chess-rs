# Applying bumblebench without turning it into dogma

Inspected 2026-09-06 at `../bumblebench`, commit
`30eb5eb29640932bf33e8d5530061bcc9856b5f7` (clean at inspection).
Current host read-only identification: Apple M2 Max, 12 physical/logical CPUs,
macOS 15.7.7, build 24G720. No benchmark, build, or workload was run.

The charter's unit of knowledge is a measured claim **in a regime**, with a
falsifier, emitted-code evidence and a verification stamp. Several dossier
verification logs retain PROVISIONAL qualifications even where the index says
VERIFIED. Those numbers are useful hypotheses; they are not fresh measurements
of Push Chess or guarantees about its hot loops.

## Transfer map

| Evidence in `../bumblebench/facts/` | Consequence for this engine | Boundary / negative control |
|---|---|---|
| `m2max.codegen.vec-header-aliasing.md` | Split disjoint slice borrows outside edge loops; inspect whether pointers/lengths hoist | Safe reborrows can help; no universal justification for unchecked indexing |
| `m2max.codegen.bounds-checks-structural.md` | Arrange ranges/iteration so bounds are evident; then inspect emitted code | Check count alone is not a performance model; stores and control-flow shape matter |
| `m2max.mem.gather-wall.md` | Prefer contiguous edge ranges and compact indices; remove pointer chasing before SIMD | Random gathers do not acquire streaming bandwidth just by being vectorized |
| `m2max.probe.batching-saturates.md` | Sweep batch width; measure diminishing returns | Larger batches add latency, memory and stale/pending work |
| `m2max.mem.prefetch-regime.md` | Consider prefetch only if an actual dependent-miss stall remains | L2-resident or already-saturated loops can slow down |
| `m2max.simd.minmax-universal.md` | Dense reductions are SIMD candidates | Its u64 min/max results are not predictions for short floating-point Gumbel loops |
| `m2max.codegen.slp-store-forward-trap.md` | Inspect vectorized per-item updates and reloads, not just vector instruction count | A SIMD-looking rewrite can introduce store-forward stalls and lose |
| `m2max.mem.bandwidth-core-scaling.md` | Measure worker scaling and CPU/GPU shared-memory contention | Its DRAM read streams plateau around four P-threads; this is not a hard four-worker limit for search |
| `m2max.mem.core-to-core-latency.md` | Worker-private mutable state, coarse messages, few ownership transfers | Cross-cluster/P↔E handoffs differ; avoid per-node shared atomics |
| `m2max.cache.l1d-line-64.md` | Separate genuinely contended queue/counter lines, measure padding | L1 versus outer-cache granularity differs; do not align every object to 128 bytes |
| `m2max.cache.16k-pitch-aliasing.md` | Check pitches only for large lockstep multi-stream buffers | Its small nonzero 16-KiB residues are pathological in the measured regime; do not pad tiny chess arrays by analogy |
| `m2max.predict.tage-memorizes-benchmarks.md` | Use varied positions, action widths and game phases | Repeating a tiny deterministic fixture can train the predictor, not measure the real workload |
| `m2max.method.sub-us-attribution.md` | Time aggregate repeated work with an attribution control | A sub-microsecond delta can be harness/code-shape noise |
| `m2max.method.interleaved-ab.md` | Same-session interleaved A/B, absolute numbers only under controlled load | This dossier is a protocol record, not a locally verified universal ±2% theorem |

The pitch dossier explicitly warns about an earlier **inverted** index entry.
The current index is corrected, but the historical warning remains in the
dossier. The correct lesson is to read measurements and controls, not copy
either an index slogan or an obsolete warning as a present-day fact.

## Data layout policy

- AoS when the same operation consumes an entity's fields together: a compact
  board cell or a tiny transition record can remain a simple value.
- SoA when the hot operation scans one/few columns over many entities: candidate
  example is edge visits/value/prior processing without reading moves/plans.
- Hot/cold split before a proliferation of arrays: edge selection should not
  drag full prepared moves or UI metadata through cache.
- AoSoA only if measured SIMD/cache behavior justifies a chunk width. Do not
  select a width from an unrelated database benchmark.
- Arena indices rather than scattered owning pointers; reusable scratch rather
  than fresh temporary vectors; explicit high-water/capacity handling.
- Avoid expanding a tiny record with padding or bitboards unless the repeated
  work saved repays the extra writes/cache footprint.
- Retain a straightforward scalar implementation as the correctness and speed
  control. Complexity must have an observable payoff.

## Worker-pool policy

Build persistent Rust workers with private game lanes, arena/scratch capacity,
repetition state and RNG streams. Batch requests to one Metal coordinator.
Have an explicit worker-count control and an all-core mode. Determine the
default from positions/second, completed games/second, latency and energy under
the real workload—not CPU-utilization percentage alone.

macOS does not give us portable, reliable hard pinning of each worker to a
chosen P/E core. A pool sized to the cores is not a promise of fixed placement.
QoS is a scheduling hint; record observed performance. Test P-oriented settings
and inclusion of E-core capacity; do not force slower workers into every
barrier. No nested pools or one Python process/model copy per core by default.

Benchmark spin barriers can make sense inside a controlled measurement. That
does **not** mean production workers should spin while awaiting GPU work or
while the user has paused compute. Use bounded queues, parking and cancellation.

## Measurement contract after permission

1. Use the sibling project's serialized measurement discipline and record chip,
   OS, toolchain, compiler flags, model, batch shape, seed and workload corpus.
2. Establish an interleaved identical/identical control before accepting small
   differences. Use varied traces and report distributions, not only best-case
   nanoseconds or an unrelated throughput microbenchmark.
3. Separate cold JIT/initialization, warm kernels, single-game latency, full
   self-play throughput, and tail batches. Record allocations and transferred
   bytes in a separate instrumented pass so probes do not distort timing.
4. Inspect emitted assembly for the proposed mechanism. Hardware counters are
   optional only if genuinely available; never fabricate cache-miss statistics.
5. Keep changes that simplify with nonregression or deliver a meaningful
   measured improvement at an acceptable memory/correctness cost. Keep a
   record of rejected hypotheses too.

No machine-specific percentage above is an acceptance result for the redesign.
