# Inference-first implementation record

The 2026-09-06 redesign was initially approved for coding only. The user then
released CPU capacity for execution, asked to work on `main`, and specified:
**commit and push before a training run; short test run before a heavy run**.
The existing paused checkpoint has not been resumed or modified.

Implementation sequence:

1. Inference shape buckets, bounded buffer reuse, exact-input revisioned cache,
   useful/padded-work and stage timing counters.
2. Configurable compact/global/effect-aware networks and matching training,
   checkpoint and experiment schemas.
3. Shared resolved transitions, allocation-conscious search slabs/scratch,
   direct encoding and generation-checked batched replies.
4. Persistent native worker capability and safe cancellation/bulk ownership.
5. Compact versioned replay, real-history restarts, reanalysis and averaging.
6. Evaluation/benchmark harnesses, regression tests and documentation.
7. Review, commit all project changes, push `main`, then a bounded training smoke test.

Conditional research projects (custom unsafe Metal buffer sharing, speculative
SIMD/prefetch, a shared search DAG, learned dynamics, alternative CPU evaluator)
remain gated on measurements. They must not be represented as verified
optimizations or silently enabled by this code-only pass.

## Implemented

- Inference: bounded batch/action/effect shape caches; tail buckets; reusable
  host staging; one combined policy/value readback; revision invalidation;
  optional exact-byte-identity LRU and duplicate suppression. Cache-off avoids
  constructing byte keys. Payload budgets do not pretend to measure Python RSS.
- Networks: checkpoint-compatible 96×6 reference; configurable compact trunk;
  optional mean/max global context and joint exact-effect policy tokens. The
  effect reduction is a portable differentiable one-hot/matmul baseline, not
  a claim of an optimal Metal kernel. Baseline board encoding stays version 2;
  optional effects have their own version 1 schema.
- Rules/search: generated moves retain resolved transitions; shared promotion
  application; board-only knight intermediate; touched-square undo mask;
  compact history prefix and local cursor; contiguous edge arenas with hot
  visit/value/prior columns and cold moves; reusable selection/generation scratch;
  direct final-buffer encoding. No legal-action truncation.
- Runtime: independently leased ready groups, whole-reply validation, cooperative
  cancellation, owned reusable arenas, a bounded shared worker queue, and explicit
  Idle/Working/Ready/Leased states. Device batch size is independent of actor,
  group, worker, and learner-batch counts. Returning one lease wakes native work
  while the device services other ready jobs. Explicit all-core capability.
  Workers park; no per-edge atomics, Python callbacks, or busy waiting. A sole
  ready worker transfers its output without a merge copy. Multi-worker replies
  copy into owned messages; no borrowed NumPy pointers escape. `workers=1`
  uses the same scheduler; `0` requests available logical parallelism,
  not hard core affinity. Python retains the authoritative games; workers own
  detached search lanes, not separate full game histories.
- Collection: persistent actor slots immediately replace completed games and
  retain unfinished trajectories across learning. Move-boundary checkpoints
  include compact, checksummed actor sidecars and per-target weight provenance.
  Quotas drain in-flight moves, not entire games; time limits invent no outcomes.
- Data: format-2 ragged legal IDs/policies and shared game+ply records; bounded
  exact-history reconstruction cache; format-1 reader; provenance and checkpoint
  shard checksums. Outcomes stay exact; truncations have no value labels.
- Learning: effect-aware batches/JIT; optional separate EMA weights and state;
  exact optimizer resume; sparse/standard/actual-history restart mixtures;
  bounded observed-error restart prioritization; configurable cheap-turn noise;
  history-correct reanalysis and distribution/calibration diagnostics.
- Evaluation: paired held-out openings, both colors, raw/EMA choice, equal-cap
  diagnostics or common requested wall-time; actual times and deadline overruns;
  conservative opening-pair confidence bounds with unresolved games kept as
  intervals. Deadlines are cooperative, not hard real-time guarantees.
- Experiments: compute-free manifest writer and opt-in interleaved A/B/identity
  inference probe with diverse fixtures/reachable positions. No automatic
  promotion of a trained model to the production engine.

## Verification

The first implementation pass passed the full Rust workspace suite and all
24 Python tests on Metal, including real forward/backward, checkpoint recovery,
effect permutation, native ownership, history reconstruction and pool shutdown.
Final checks and bounded-run results are recorded in [verification.md](verification.md).
Tests containing tiny gradient updates use temporary fixtures, not the paused
training run. No strength or speed improvement is inferred from passing tests.
After the implementation was committed and pushed, a short smoke run and its
resume passed, followed by a three-minute pilot (64 games, 85 updates, zero
pending updates). Warm inference probes and their limitations are recorded in
the verification report. The original paused experiment remains unchanged.

## Still gated / deliberately not claimed

Arena **capacity** reuse is implemented; played-child/subtree promotion and
inherited visit reuse are not. Exact inference caching avoids evaluations
without changing Gumbel evidence; persistent statistical tree reuse needs a
separate algorithm/parity gate. The edge limit is a node-count guard, not a
strict allocator-byte budget; reported arena bytes exclude cursor/queue memory.
Detailed hardware counters, graph/codegen attribution and native allocation
instrumentation remain measurement work. There is no custom Metal kernel,
unsafe shared-buffer bridge, mixed-precision switch, SIMD/prefetch rewrite,
shared search DAG, learned dynamics, learned regret head, CPU neural evaluator,
within-tree parallel leaves, or automatic production migration. Those were
conditional research projects, not prerequisites to the implemented system.
