# Ask for a capture, not a move transaction

## Evidence and change

The same live Run 2 sample described in move-ordering.md showed knight leg
resolution and composition at 87 and 73 of 3,144 worker leaf observations.
Core attack detection and Cataclysm's check detector asked the full knight
resolver to build a move transaction, then inspected only whether it captured.
That needlessly copied an intermediate board, resolved a second transaction,
and composed both displacement lists for a boolean answer.

Both callers now use one shared `knight_captures` predicate. Full move generation
and animation still use the authoritative transaction resolver. Route geometry
is shared between the predicate and full resolver; there is no second rules
implementation or alternate backend to select.

For a board-valid knight route, its two legs are perpendicular. Displacements
on the first ray cannot change the second ray beyond their common midpoint.
A capture on the final square requires an enemy there and an empty second-leg
interior: neither an intervening enemy nor a friendly push chain permits capture
through it. The second leg is at most two squares long, so at most one interior
square needs testing on the original board. First-leg feasibility still comes
from `resolve_push`, and a capture on that first leg remains forbidden.

This removes the intermediate board, second plan and composition from attack
queries, but deliberately retains the first plan. Avoiding even that plan would
need separate evidence and another correctness argument. No allocation, cache,
unsafe access, larger move record or per-node FFI is introduced.

## Correctness gates

`src/core/push_tests.rs` exhausts every board-valid origin/destination, both leg
orders and colors, and every empty/friendly/enemy assignment along the entire
first ray plus the second-leg interior. Each predicate result must equal the
full resolver's capture result. Additional cases cover empty/friendly targets,
empty sources and invalid coordinates.

All 15 engine profiles retain their pre-refactor fixed-node search fingerprints
in debug and release, including repeated searches with retained TT/history.
The normal rules, replay and native/Python tests are required as well. These
are behavior-preservation gates, not evidence of stronger play.

This change preserves the existing treatment of an empty attack target.
The separate [castling investigation](castling-transit-audit.md) now reproduces
transit exposure in both colors/wings and adds a read-only corpus audit.
It does not silently reinterpret the rules in this performance refactor.
Do not discard a corpus or claim its exposure before the actual audit.

## Measurements and limits

Run serially, preferably on an otherwise idle machine:

```sh
cargo test -p push-chess --release --lib knight_capture_probe -j 1 -- --ignored --nocapture --test-threads=1
```

Apple M2 Max / macOS 15.7.7 / rustc 1.100.0-nightly, release opt-level 2 with
thin LTO, 2026-09-06. The 12-worker tournament remained active, so these are
**contended kernel measurements, not whole-engine throughput claims**.

Four warmups precede 31 timed repetitions. Arm order reverses each repetition;
an identical full-resolver repeat brackets the predicate as an ambient control.
Fixture construction and correctness comparisons are outside timing.

| Query regime | Queries | Full median ns/query | Predicate | Full repeat |
|---|---:|---:|---:|---:|
| Reachable positions, small repeated stream | 1,478 | 20.55 | 7.39 | 20.61 |
| Varied adversarial occupancy | 32,768 | 33.44 | 21.47 | 33.70 |

The small stream can be memorized by the branch predictor; its roughly 2.8x
kernel improvement must not be generalized. The larger varied stream gives
about 1.56x, but contains synthetic boards, **not claimed legal game states**.
Both regimes check the exact full resolver. Neither reproduces the production
distribution of search positions, cache residency or branch history.

Keep the current tournament executable fixed. Whole-search measurements and
fresh paired games belong to the next verified, committed executable after
the current run seals its corpus.
