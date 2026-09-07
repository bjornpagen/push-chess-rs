# Bounded refinement of the existing NNUE

> Historical record: old-rules Runs 1–4 and their training data were purged in
> the [single-rules corpus cutover](corpus-cutover.md). Only Runs 5–6 remain.
> Old replay commands and training reproductions below require data no longer
> present in the live corpus; they are not current operating instructions.

## Authorized scope and fixed comparison plan — 2026-09-07

The user authorized NNUE improvement, cleanup/bug fixes, committing and pushing
verified code on main, then small comparison games. No new built-in engines,
large tournaments, repeated campaigns, or automatic model promotion.

Train one refinement from the existing `residual-r2.safetensors` weights:
4,000 maximum Metal updates, batch 256, learning rate 0.0003, seed 1,
validation every 250 updates, patience four checks. Keep the best exact-export
validation checkpoint, including the initialization. Only sealed Run 2 supplies
data; Runs 1, 3 and 4 remain excluded. Same 10,000/2,000 game limits and 16
positions/game as before; known castling-transit-affected games are excluded
without altering their historical results. No search-score teacher mixing.

After verification and the implementation push, compare the before/after
weights in the **same existing Cataclysm search**, using distinct artifact aliases
`aurora-r2` and `aurora-r3` for provenance. The embedded control is unchanged.
All games use the corrected `push-chess-history-v2` rules and one frozen binary.

Two fixed, small arena checks, each 64 color-swapped opening pairs (128 games):

| Check | Budget/move | Opening seed |
|---|---:|---:|
| Short | 25 ms | 20260907051 |
| Deeper | 100 ms | 20260907052 |

Both use 12 workers, six opening plies, 512-ply cap, 900-second admission
ceiling and 20-GiB store ceiling. Total planned: **256 games**, no adaptive
extension based on score. Audit the first before starting the second. Store
every trajectory, result and search observation as normalized bumbledb facts
in `data/corpus`. Neither check becomes training data. Report missing/capped
games separately and pair/family uncertainty honestly; these small samples
may not resolve a modest strength difference. Do not call loss improvement
or an encouraging small-match score proof of stronger play.

## Training result

The single planned refinement completed 4,000 updates. Exact-export validation
improved at every 250-step check, so the selected checkpoint is step 4,000;
the step ceiling, not patience, ended the run. Updates, selection checks and
checkpoint preparation took 19.708 seconds **after** corpus loading and the
initial reference evaluations. This is not an end-to-end loader timing.

| Weights / reference | Game-balanced validation BCE |
|---|---:|
| Handwritten only | 0.4059665536 |
| Unchanged embedded control | 0.4003572695 |
| Aurora-r2 initialization | 0.3928291002 |
| Refined Aurora-r3 | **0.3884641343** |

All four rows use the same **2,000 games / 31,553 positions**. Relative loss
reduction: 1.11% against r2, 2.97% against embedded, 4.31% against handwritten.
Excluding affected games changes the seeded reservoir, so these are freshly
rescored references: do not compare against the different sample in the
original r2 report. Repeated validation selects a candidate; it is not an
independent strength test or proof that further training would help.

Training retained 10,000 games / 157,718 positions from 16,756 scanned games;
89 duplicate trajectories were skipped and Run 2/game 14,797 excluded.
Validation scanned 2,070 games, skipped eight duplicates and excluded
Run 2/game 12,764. The parent weights remain v1-derived; filtering this pass
does not erase what an earlier parent learned. Historical results are untouched.

Reproduction:

```sh
uv run --no-sync pushzero nnue train --db data/corpus --runs 2 \
  --init models/residual-r2.safetensors --output models/residual-r3.safetensors \
  --steps 4000 --batch-size 256 --learning-rate 0.0003 \
  --validation-interval 250 --patience 4 --seed 1 --device METAL
uv run --no-sync pushzero nnue export models/residual-r3.safetensors \
  --output models/residual-r3.bin
```

Artifact identities (checkpoint and export are preserved in
[models/](../models/README.md), not a second game store):

- Export, 49,280 bytes: `a936e29830ffde362a94c4321deb19e7f216b4bc86285bc8c5edde2f350ae759`.
- Checkpoint: `0847cb50db2fc560c71c5011684a705a240da818e15f9178e9e73ef6459ea94c`.
- Training sample: `b36f78a79b087b06574e523c23a3981b44539493fe89f6e35f29aaecf313f11f`.
- Validation sample: `7813d4712427231d3c2c23e609b1570e796a95ad446375e7a1348706b2289c9a`.
- Trainer source: `4aab2f460331324d7d8b11c8ad5f4f756a8a420a168b51146ca90dcf575b4d4e`.
- Native training extension: `697736ac2c3d3afc313f69c10892b8ceef2e26a93e3774cf8e723aed03809430`.

The exported r3 model matches the quantized Metal forward exactly on 256
deterministic reachable positions. All 118 Rust and 55 Python tests pass;
workspace/native Clippy and formatting pass. The historical all-profile
fixed-node gate is unchanged. Full replay verified all 31,230 previous games
under their original rules. [Search measurements](search-cost.md) record the
modest cleanup speed gain and the remaining small shared-refactor cost.

Exact-history checks on the previously reviewed Run 4/game 12 used v1 rules
and 524,288-node ceilings, never FEN-only reconstructions. Both r2 and r3
choose the collateral-sensitive knight route `10006` at ply 51, rather than
the game's shallow `5910`. At ply 67 they choose different alternatives
(`3388` / `18155`); this is a changed evaluation/search result, not a proven
improvement or a regression target with a known winning move. Both retain
the independently verified mate-in-five at ply 94: action `5899`, proven
in 376 nodes. No search heuristic was patched toward these preferred moves.

## Playing checks

Implementation and the fixed comparison plan were committed and pushed to main
as `13efc6d` **before** either new run. Both use executable SHA256
`c36e55384af8578b69ce2be8ec2e25efa219f8368c598b96731905226ed63c9c`,
preserved at `data/bin/lab-c36e55384af8578b`, and the unchanged r2 / exported
r3 weight identities above. No built-in engine or default weight was replaced.

Run 5 (25 ms) finished all 128 terminal games in 22.49 seconds: **r3 won 66,
drew 4 and lost 58**, scoring 53.125%. All 64 pairs have distinct opening
families. Its pair-points histogram, in r3 orientation for 0/0.5/1/1.5/2 points,
is `[8, 3, 39, 1, 13]`. The conservative 95% family bound is
**36.15–70.10%**: encouraging direction, inconclusive strength evidence.

Full replay audited 9,132 moves and 8,364 search observations in 1.172 seconds;
all outcomes verified, no missing/capped games and no transit warnings (one
castle). At this short budget, 81 r3 searches and 64 r2 searches had no completed
depth/proof; those are incomplete searches, not unfinished games.

Run 6 (100 ms) finished all 128 terminal games in 84.21 seconds: **r3 won 64,
drew 10 and lost 54**, scoring 53.90625%. All 64 pairs again have distinct
opening families. The r3 pair-points histogram is `[6, 6, 36, 4, 12]` and its
conservative 95% family bound is **36.93–70.88%**. Replay verified 11,623 moves
and 10,855 search observations in 1.728 seconds, with no missing/capped games
and no transit warnings (two castles). Only four r3 / two r2 searches lacked
a completed depth/proof; the games themselves all reached terminal outcomes.

Summary, always from the refined r3 weight version's perspective:

| Budget | Games | Wins | Draws | Losses | Score |
|---|---:|---:|---:|---:|---:|
| 25 ms | 128 | 66 | 4 | 58 | 53.125% |
| 100 ms | 128 | 64 | 10 | 54 | 53.90625% |
| Descriptive total | **256** | **130** | **14** | **112** | **53.515625%** |

**Conclusion:** prediction loss improved, and both small playing checks lean
in the same favorable direction. The samples do not establish a reliable
strength gain or a new champion. The quoted intervals are per-check bounds,
not a joint two-check guarantee; the pooled score is descriptive, not a
promotion test. Keep r3 as a promising candidate and preserve r2 and embedded
controls. Do not extend the sample, tune against these arena games or launch
another campaign without a new instruction.

The complete store now has 31,486 games / 2,385,428 moves / 2,196,516 search
observations, consuming 11,327,979,520 bytes. The sole nonterminal saved game
is still the original Run 1 cap. Both new runs are sealed and audited; all
training and match processes have exited. No new engine architecture or
built-in entry was added, no old run was changed, and no heartbeat was installed.
