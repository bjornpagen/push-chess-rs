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
