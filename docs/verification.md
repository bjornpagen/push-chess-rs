# Redesign verification — 2026-09-06

Execution was initially prohibited. The user subsequently released CPU capacity
and required commit/push to `main` before any training run, followed by a short
test run before heavy training. The original paused experiment remains intact.

## Completed first pass

- Rust workspace, all features: 75 tests plus one compile-fail doctest passed.
  This includes prepared/reference rule parity, reachable traces, hashes and
  undo, exact effects, history-prefix parity, request IDs, stop/capacity behavior,
  worker/serial policy parity and worker restart/shutdown.
- Editable Python native extension rebuilt successfully.
- Python suite on Metal: **24 passed in 26.41 seconds**. Includes gradient updates
  in temporary fixtures, baseline/effect/EMA checkpoint recovery, old/new replay,
  action/effect permutation, cache revision invalidation and NumPy ownership.
- Initial strict adapter lint pass found one excessive-argument warning at the
  explicit Python bulk boundary; it has a local documented allowance. The
  deprecated atomic request-counter method was also replaced.

## Final pre-training checks

After the final runtime/data edits, the complete Rust suite passed again, the
actual `wasm32-unknown-unknown` target check passed, the native Python adapter
passed strict Clippy, and Rust format/diff checks passed. The final Python/Metal
rerun passed **24 tests in 28.46 seconds**. Next commit and push `main`; only then create a fresh,
bounded smoke-training directory and verify saved optimizer/replay resume.

Strict lint of the entire default core library is **not clean**: seven existing
findings remain in Cataclysm's network loop, Zobrist initialization loops and the
existing nine-argument move-generation helper. Historical engines contain more
pre-existing findings. Those strategy implementations were not rewritten to
make this separate learner's lint report look clean. No new self-play/runtime
finding appeared in that pass.

## Measurement and adoption limits

Passing tests is not evidence of speed or strength. Compare inference shapes,
plain/global/effect models, cache-off/on, and worker counts with warm interleaved
identity controls and separate absolute runs. Never infer GPU arithmetic time
from host timers. Use held-out paired matches with requested equal move time,
actual overrun reporting and a predeclared sample size before promotion.

Subtree/visit reuse, custom Metal memory sharing/kernels, mixed precision,
SIMD/prefetch, and production champion migration remain separate measured gates.

## Post-push smoke tests and bounded learning pilot

Implementation commit `a3e6f283f241d86fe29c1161bccfd49f1b614aa5` was pushed to
`origin/main` **before either run below**. These use new ignored experiment
directories, not the pre-existing paused experiment.

- `experiments/pushzero-redesign-smoke-20260906`: 64×4, global context every two
  blocks, effect width 16, two workers, four games × 16-ply cap, eight simulations,
  batch 16, EMA .995. First invocation: 19.74 s, 64 examples, four updates, clean
  checkpoint. Resume: 14.43 s, four more games plus four reanalysed policies,
  five further updates. Final: 132 replay entries, nine optimizer steps, zero
  pending updates. All eight games were truncated by the intentionally short
  cap and correctly supplied **zero value-loss weight**. This is pipeline proof,
  not a strength test.
- `experiments/pushzero-effects-pilot-20260906`: same 353,060-parameter model,
  32 actors/games per iteration, full/fast caps 64/16 at .25 full probability,
  256-ply cap, .25 sparse/.25 restart mixture, batch 64, reuse 4, cache 64 MiB,
  cheap-turn exploration disabled, EMA .995, reanalysis 16. Three-minute ceiling;
  completed in **175.30 s** with two committed iterations, **85 updates**, 64 games,
  1,346 replay entries and zero pending updates. 45 games reached real outcomes:
  18 decisive and 27 draws. The other 19 were deadline-truncated, not relabelled
  draws. Eight games used full-history restarts; archive retained 701 entries.
- Pilot prediction counts: 162,127 requested rows, 134,106 evaluated, 20,503
  exact-cache hits and 7,518 duplicate rows avoided. Inference-path timer 133.86 s;
  native-boundary timer 2.35 s. Forward/readback accounts for 121.12 s **including
  synchronization**, not a pure GPU hardware timer. No allocator/RSS claim is
  inferred from payload counters.
- Read-only compatibility check loaded the original 1,065,796-parameter paused
  checkpoint at 177 steps. Its pointer still records 50 pending updates; it was
  not resumed, rewritten, or replaced.
- All 19 local arXiv source archives/extracted text sets passed checksum
  verification again. Downloaded third-party source and training artifacts remain
  local/ignored; the source catalog, hashes, code and reports are committed.

## Warm inference probes

Apple M2 Max, macOS 15.7.7, tinygrad 0.14.0/Metal, cache off; diverse fixtures plus
seeded reachable positions. Each repetition evaluates batches of 1, 10 and 32
rows (43 useful rows total). Twelve interleaved fixed/tail/identity repetitions
per model, after warm-up; medians below exclude warm-up.

| Model | Fixed rows | Tail buckets | Independent fixed identity |
|---|---:|---:|---:|
| Plain 64×4, 333,188 parameters | 15.62 ms | 14.29 ms | 15.70 ms |
| Global + effects 64×4, 353,060 parameters | 21.57 ms | 19.79 ms | 22.02 ms |

Tail bucketing reduced this probe's median by about 8–9%. This is a specific
warm-workload observation, not a universal throughput promise. Architecture
probes were separate invocations, not an interleaved architecture comparison;
the combined global/effect candidate cost about 39% more here than the plain
candidate. That cost must earn its place through equal-time strength/data
efficiency tests. It cannot be attributed to effects alone without the global-only
and effect-only ablations. Cold timings have shared kernel-cache/order effects.

Raw probe records: `experiments/pushzero-profile-plain-20260906.json` and
`experiments/pushzero-profile-effects-20260906.json`. The latter's optional search
probe includes new shape compilation and must not be compared to historical
warm search timings as a regression claim. Long training and production
promotion remain unproven by these bounded tests.

## Evaluation-path smoke check

The pilot's final raw checkpoint played its frozen initial checkpoint over two
held-out opening pairs, with 25 ms requested per move and a 96-ply cap. Result:
**0 wins, 3 draws, 0 losses, 1 unfinished**; score interval [0.375, 0.625],
conservative paired 95% confidence interval [0, 1]. This provides **no evidence
of a strength gain**. Record: `experiments/pushzero-effects-pilot-20260906/eval-smoke.json`.

Actual mean move times were 41.89 / 37.61 ms and p95 times 112.98 / 100.11 ms,
with 155 overruns for each contestant. The root/new-shape compilation and
cooperative round deadlines matter substantially at this tiny budget. This is
not a strictly equal-consumed-time match; do not use it for promotion or Elo.
Broader shape warm-up, measured single-root latency, and an explicit overrun
acceptance gate are required before serious time-controlled comparisons.

All launched smoke/pilot/evaluation processes have completed; no unattended
long training or automatic restart was installed. The next useful experiment
is a predeclared equal-training-budget plain/global/effect ablation, alongside
the inference-latency work, before choosing an expensive long-run architecture.
