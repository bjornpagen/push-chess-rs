# Remove a dependent load, preserve the search

## Evidence before changing representation

A 3-second, 10-ms live sample of Run 2 on 2026-09-06 placed Astra's
`ScoredMoves::pick_best` at 298 of 3,144 worker leaf observations (~9.5%).
Push resolution (401) and core attack detection (350) were also prominent.
These are leaf observations, not recursive frame totals. The single writer
was waiting for completed games in 193 of its 262 observations, so this sample
did not justify redesigning the database writer as the primary bottleneck.
One short sample is a direction, not a precise time allocation for the campaign.

The selector did not just scan scores. On every comparison it reloaded the
current maximum using its data-dependent index. The live release disassembly
contained this shape inside the loop (Astra record stride: 12 bytes):

```asm
ldr   w12, [x11], #12        ; next score
add   x13, x0, x0, lsl #1   ; current best index * 3
ldr   w13, [x9, x13, lsl #2]; reload best score, depends on prior comparison
cmp   w12, w13
csel  x0, x10, x0, gt
```

There was also a bounds-check edge on that dependent index inside the loop.
The indexed load creates a serial load/compare/index/load chain even if the
score remains L1-resident. This is not evidence of a DRAM-bandwidth problem.

## Change

Keep both the best index and best score. Read each remaining score once using
a sequential iterator; update the two scalar values together. Swap the entire
record only after selection. No new buffer, allocation, unsafe access, SIMD
intrinsic, prefetch, SoA split or full sort is required.

The resulting actual Astra selector and benchmark specialization were merged
to the same machine-code address in the release test binary. Its hot loop has
one sequential score load followed by register compares/selects; the best-score
gather and its in-loop bounds check are absent. Bounds checks still protect
the public input index and final swap. Cataclysm retains its **last**-maximum
tie rule; Astra retains its **first**-maximum rule. This is not a stable sort:
preserving the exact swap sequence, not merely descending scores, matters.

The measurement discipline follows the adjacent bumblebench dossiers on
structural bounds-check costs, interleaved comparisons, and predictor-memorized
microbenchmarks. No assumption that SoA or SIMD must be faster is needed here.

## Probe and limits

Run serially, preferably on an otherwise idle machine:

```sh
cargo test -p push-chess --release --lib ordering_probe -- --ignored --nocapture --test-threads=1
```

The ignored probes live inside the two engine modules so they use the actual
records and real preparation/order scores: 12-byte Astra scored moves and
56-byte Cataclysm actions. The deterministic reachable-position stream produced
1,024 lists / 36,595 records for each family. It covers random-play positions,
not the full production distribution of TT/history-conditioned search nodes.
Four warmup passes precede 15 timed passes. Arm order reverses each repetition;
the original loop is repeated as an identical-code ambient control. Reset/copy
and allocation occur outside timing. Caps of one, eight and all selections
cover early exits and full consumption. A two-pass max-then-find candidate
stays test-only, not as a selectable production backend.

Apple M2 Max / macOS 15.7.7 / rustc 1.100.0-nightly (bff8e12ff), release opt-level
2 with thin LTO, 2026-09-06. The tournament was active: absolute timings below
are **contended**, and are not a whole-engine throughput claim.

| Family | Selected per list | Original median ns/pick | New median ns/pick | New/original |
|---|---:|---:|---:|---:|
| Astra | 1 | 59.37 | 22.01 | 0.371 |
| Astra | 8 | 69.71 | 18.83 | 0.270 |
| Astra | all | 52.21 | 13.57 | 0.260 |
| Cataclysm | 1 | 73.45 | 27.87 | 0.379 |
| Cataclysm | 8 | 86.01 | 19.63 | 0.228 |
| Cataclysm | all | 63.82 | 14.01 | 0.219 |

The repeated original arm's medians were within 0.3%. The two-pass candidate
was slightly faster only for Astra's single selection (21.61 vs 22.01 ns),
but slower for repeated selection and for Cataclysm. That ~2% corner is not
grounds for a dispatch threshold or extra representation under this load.
The first exploratory build likewise favored the cached maximum overall.

## Behavior gates

- Exhaust every list of length 1–8 drawn from `i32::MIN`, zero and `i32::MAX`,
  for both tie rules. Compare entire record permutations after every selection
  and at each arbitrary suffix, not just scores or the first chosen record.
- Verify one score read per suffix element; retain the 600-move spill test.
- Before changing the selector, capture fixed-node fingerprints for all 15
  existing profiles. Twelve fixture/reachable roots, twice each (fresh then
  retained TT/history), at 2,048 nodes record move, score, depth, selective
  depth, PV, node/TT/quiescence/proof counters. Every fingerprint matches after
  the change in both debug and release; clocks are deliberately excluded.
- Run the full rules, replay, candidate and native/Python integration checks.

These gates protect a mechanism-only refactor; they do not establish stronger
play. The running campaign's executable is unchanged. Measure whole-search
cost and paired playing strength on the next committed executable after the
current owner has sealed its corpus.
