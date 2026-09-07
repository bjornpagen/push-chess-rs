# Whole-search cost: retain only the changes that win

## Protocol

`tests/search_cost.rs` is an ignored, bounded measurement harness, not a playing
strength test or another engine implementation. Select a registered engine:

```sh
PUSH_CHESS_PROBE_ENGINE=astra cargo test -p push-chess --release --test search_cost -- --ignored --nocapture --test-threads=1
```

For comparisons, build and save the before/after test executables separately,
then run them in alternating order. Never rebuild a live tournament executable.
Each process constructs the same 12 roots: four tactical/special-move fixtures
and eight positions on a deterministic reachable trace. Each root gets 8,192
nodes. One complete warmup pass precedes two measured passes. Position copying
and resetting TT/history lie outside timing; root setup, search and final PV
extraction lie inside. Some roots terminate before the node ceiling.

Every process checks restoration and repeatability. Its fingerprint includes
chosen moves, evaluations, depths, PVs and node/TT/quiescence/proof counters,
never clocks. The before/after fingerprints must match. The separate all-profile
fingerprint gate additionally exercises retained TT/history, at 2,048 nodes.

Measurements below: Apple M2 Max, macOS 15.7.7, rustc 1.100.0-nightly
(bff8e12ff 2026-08-26), release opt-level 2 / thin LTO, 2026-09-06. The 12-worker
corpus tournament remained active. These are **contended whole-search fixture
measurements**, not isolated NPS claims, proof of stronger play, or estimates
of total tournament throughput. Remeasure idle and at deeper budgets.

For each profile, 11 triplets alternate A/B/A and reversed-arm order. Both A
arms execute the **same saved baseline binary** as an ambient control. Each
reported process time averages its two measured passes. This discipline follows
bumblebench's insistence on actual code, interleaved controls, and working-set
costs instead of assuming that eliminating computation must help.

Baseline: `b2b3445` plus the harness committed in `45794b2`. Saved baseline executable
SHA256: `66d56bcd83e2f39e440f37dccbaf7ab857b878717394c0ef3b1930edbbf37a00`.

## Rejected: retaining every Astra push transaction

Astra's move generator already resolves push plans but discards them; making a
selected move resolves it again. The candidate retained `PreparedMove` vectors
per ply, reusing capacity. A u16 original-plan index occupied the scored move's
old padding, keeping ordering records at 12 bytes. Selected moves applied their
cached authoritative plans. Stalemate witnesses also reused prepared moves.
No rules, scores or move-ordering policies changed, and all fingerprints matched.

Median ms per 12-root pass:

| Profile | Baseline | Retained plans | Baseline repeat |
|---|---:|---:|---:|
| Astra | 99.50 | 101.25 | 101.51 |
| Meridian | 222.32 | 230.96 | 222.54 |
| Bedrock | 120.58 | 132.73 | 120.85 |

The extra copying, retained footprint and indexed access did not repay themselves
in this workload. Astra was inconclusive/slower; Bedrock lost about 10% against
both controls. **Do not deploy this candidate.** It was removed, including its
plan-index field and per-ply buffers. Cataclysm's different search/representation
is not evidence that copying its storage design into Astra will win.

Candidate executable SHA256:
`2b5b5a440474b2b852ca65da45f0f5e82e58e604cec3e469cf7c271bd0158955`.

## Kept: compile-time piece-square geometry

The original sample placed Astra's `placement` at 116 of 3,144 worker leaf
observations. It computed rank reflection, central-square distance, piece-type
branches and pawn geometry repeatedly: once per evaluated piece and twice per
move-ordering record. Those values never change for a given input.

Precompute both phases for seven piece types and 64 white-relative squares.
All entries fit i16, asserted during constant evaluation: 1,792 bytes total.
Reflect black squares by XOR 56. There is no runtime initialization, per-engine
table copy, allocation, network, or change to the evaluation function's values.
The independent historical formula stays only in tests; every piece/color/
square/phase combination is compared, not just positions reached by search.

The baseline release binary contains an out-of-line placement function with
rank/center arithmetic and a piece-type branch tree. In the new binary that
symbol and the per-move calls are absent. Actual move-ordering disassembly
contains the two signed halfword loads below, sharing the piece-table base:

```asm
eor   x8, x10, x26          ; color-relative destination
add   x9, x9, x21, lsl #7  ; 64 i16 entries per piece
ldrsh w28, [x9, x8, lsl #1]
eor   x8, x10, x17          ; color-relative origin
ldrsh w19, [x9, x8, lsl #1]
```

This replaces the runtime formula; it does not assert that a call instruction
alone was the bottleneck. Address setup and existing safety checks remain.

Median ms per 12-root pass, from a new complete interleaved comparison:

| Profile | Baseline | Table | Baseline repeat | Median within-triplet B/mean(A,A) |
|---|---:|---:|---:|---:|
| Astra | 101.77 | 94.09 | 100.24 | 0.935 |
| Sentinel | 125.16 | 119.82 | 125.59 | 0.955 |
| Bastion | 166.08 | 161.35 | 182.24 | 0.964 |
| Outrider | 126.25 | 122.06 | 129.21 | 0.959 |
| Tactician | 139.55 | 133.60 | 141.88 | 0.955 |
| Bedrock | 117.79 | 113.11 | 117.75 | 0.959 |
| Meridian | 197.39 | 185.15 | 200.43 | 0.952 |

The stable controls support a useful small improvement, especially Astra,
Bedrock and Outrider. Bastion's roughly 10% control imbalance makes its exact
gain inconclusive; some other profiles also have wide tails under contention.
The whole-pass improvement is not the isolated kernel's speedup. All 231
processes matched their profile's before/after search fingerprint.

Table-candidate executable SHA256:
`a9f51cdbefe8ba6a42165e7a91ac1c2866f578bb0518f9e198e2bc25857ffd23`.

The running corpus executable is unchanged. These mechanism-only changes
belong to the next committed executable, with fresh wall-time arenas still
required to measure their value in play.

## The reset guard caught a pre-existing state leak

When the whole-search harness was extended to Cataclysm-family comparisons,
Synthesis failed its within-process repeatability check in **both** the unchanged
baseline and a candidate. Per-root diagnostics isolated root 2:

`r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1`

At 8,192 nodes, the selected a1–a8 move, depth 5, selective depth 16, score
1,368, 5,710 quiescence nodes, 682 proof nodes and entire PV were identical.
TT hits differed: 651 in the first pass, 650 in the next. This was repeatable,
not clock noise. The benchmark now reports the exact root and full non-clock
signature when this guard fails, instead of only an aggregate hash.

`new_game` cleared history, killers and the counter-move table but retained
the `previous` per-ply action-context array. Tactical move ordering can consult
that context after the new game has populated counters, admitting prior-game
state into its choices. Clearing **only that array** at the existing reset
boundary made the original failing run repeatable again, with the first-pass
fingerprint `5652312666527311164`. All eight Cataclysm-family profiles then
passed the same 12-root, three-pass repeatability check.

A normal unit test deliberately poisons every slot with several valid prior
context indices, calls `new_game`, and requires cleared context plus identical
8,192-node search observations. This is an intentional cross-game state fix,
not a reason to relax fingerprint checks. The existing 2,048-node all-profile
fingerprints are not edited to hide it.

Within-search tactical/NULL parent-context handling is now a separate search
experiment, Waypoint (see engine-lab.md). It retains the compact scratch array
but supplies real edge context throughout search and rejects absent parents.
Do not mix that behavior change into a purported mechanism-only refactor.

The current tournament's binary remains fixed. Its saved moves and outcomes
are still the actual legal game facts; this discovery does not justify deleting
the corpus. Treat that run as exploratory and use fresh arenas for the repaired
implementation. A reset fix is not itself evidence of greater playing strength.

## Pending, not deployed: single-pass ray resolution

The next candidate scans each slider ray once. Before the first enemy/boundary,
the k-th empty square completes the non-capturing transaction for destination k:
with k empties and f friendly pieces in that prefix, its length is k+f, so
the f friends exactly fill the f slots after the destination. Capturing the
first enemy is possible only when no friend has been encountered. The candidate
retains only an ordered list of at most seven friendly sources, not every plan.

Exhaustive empty/friendly/enemy occupancy tests for every ray and both colors
matched the full single-destination resolver, including exact displacement order
and fused termination. The 15-profile fixed-node gate also matched. After the
Synthesis reset issue was isolated and fixed, comparison resumed against the
repaired `23665cc` baseline.

Inspecting emitted code mattered: both `#[inline]` and `#[inline(always)]`
inlined the ray cursor into `Enumerate::next`, but left 18 static calls to that
wrapper. The two annotation variants produced identical binaries. A direct
cursor returning `(stop, destination, plan)`, with linear square stepping and
without the enumerate/take adapters, removed those wrapper calls. That still
does not establish a useful whole-search improvement.

Eleven repetitions used four arms (baseline, adapter, direct, the identical
baseline again) and four profiles, reversing arm order: 176 processes, each
with the same 12 roots, one warmup and two measured passes. Every signature
matched the repaired baseline. Median ms per pass:

| Profile | Baseline | Adapter | Direct | Baseline repeat | Median direct/mean(baselines) |
|---|---:|---:|---:|---:|---:|
| Astra | 102.252 | 99.171 | 94.839 | 96.160 | 0.977 |
| Cataclysm | 97.354 | 95.862 | 97.095 | 96.051 | 0.981 |
| Synthesis | 115.483 | 129.580 | 111.905 | 116.212 | 0.986 |
| Bedrock | 114.554 | 119.734 | 112.437 | 113.112 | 0.974 |

The last column is the median within-repetition ratio, not a ratio of the
independent column medians. Wide tails and control imbalance under the live
tournament, other Rust builds and Spotlight activity make a 1–3% gain
inconclusive. **Neither ray candidate is deployed.** All its production changes
were removed; the current engine still uses the authoritative existing resolver.

An idle confirmation must rebuild a baseline from the latest committed source,
not compare newer context/search code to these old binaries. The proven ray
identity is not a license to assume its implementation is faster. The disposable
direct-cursor patch is in
`/tmp/push-chess-search-cost.F1XSll/ray-direct.patch` while that directory
survives; `ray-experiment.patch` is the older adapter version. No alternate
production backend remains.

Saved repaired-baseline executable SHA256:
`d5c8532a73f77806590c28c5655bbb9c5abc07ef7ac6aa799dd3c1f988574c69`.
Identical inline/always-inline adapter executable SHA256:
`cd2cf2c42ac0e5981b1ab0d5f453f84f2838fe375616937eb8a30e2f48b413e9`.
