# Retain query work across games

The first full Round 2 replay audit exposed an application-level bottleneck.
A one-second macOS sample of its release process on 2026-09-07 observed
701 of 759 main-thread samples (92.4%) in the two piece-delta queries inside
`relational::read_game`. Most of that path was selection-index construction
and allocation, not chess replay or inference. Sampling one interval does
not establish the whole audit's time distribution.

The reader prepared and discarded the `PieceRemoved` and `PiecePlaced`
queries for every game. The pinned bumbledb version
`3ed3303eb226de6c98955ab75c296887eaf7704b` retains relation images but keeps
reusable selection state on each `PreparedQuery`. Discarding the query threw
away that work. A logical `(run,game)` filter was not a physical prefix seek.

One lazily initialized `GameReader` now owns both queries for the `Corpus`
lifetime. Each read still uses its current snapshot and fresh operation budget.
bumbledb checks relation epoch and source identity at execution; no snapshot
is retained across writes. `close()` drops the queries first. No dependency,
schema, stored fact, replay check or Python game-transfer format changes.

## Structural regression

`retained_query_work_is_flat_across_game_count_and_fresh_snapshots` builds
isolated stores containing 16 and 128 copies of the same eight-ply trajectory
under distinct game keys. After warming game 0, it reads the final game with
retained queries and then fresh queries against already-cached relation images.
Every read uses a new snapshot and operation context, including across games.
Both paths must return exactly the same trajectory. Measured bumbledb work units:

| Stored games | Retained queries | Rebuilt queries |
|---:|---:|---:|
| 16 | 5,722 | 7,240 |
| 128 | 5,722 | 18,602 |

The test checks that retained work grows slowly while rebuilding exposes
relation-sized extra work. An initial blanket 2× total-work threshold failed
at 16 games: fixed per-game reads dominate that fixture. The replacement tests
scaling, not an unjustified universal speed ratio. Work units are not elapsed
time, and this synthetic result is not a full-corpus speedup claim.

The corruption regression warms both queries before separately deleting a
required removal or placement fact. The next audit must reject the altered
transition, proving that retained state cannot conceal either changed relation.
Existing page, reopen, exact-history and special-move tests use the same reader.

Only returned trajectories are page-bounded. Query indexes can scale with their
source relations within resource limits; do not claim constant total read
memory or duplicate these indexes in another application cache. Measure the
full audit and Python loading path before making further storage changes.

## Full Run 2 replay

The release audit built from `5d19553` completed successfully in **126.07 s
wall / 113.76 s user / 1.36 s system** on the same M2 Max. It verified all
21,000 games, 1,551,085 moves and 1,425,085 analyses. Executable SHA256:
`f072dbca075b1775f5f007cf54bc19c850ea8e713290238e3004b3db9b0c905f`.
Disposable output is `data/round-0002-audit.log`; facts remain in bumbledb.

The previous read-only audit was still unfinished when stopped after 27m58s
elapsed (10m27.56s CPU). It is not a completed, interleaved baseline; do not
turn the elapsed ratio into a claimed precise speedup. Other application work,
cache state and time without scheduled CPU differed.

A one-second sample 34 seconds into the new audit placed 661/727 main-thread
observations in game reconstruction, now dominated by point reads (including
PV facts), plus 61/727 in the separate trajectory validation. Selection-index
construction was absent from this sample. Physical footprint was 335.7 MiB;
the larger RSS includes mapped database pages and is not an application-heap
measurement. This removes the diagnosed repeated work without claiming that
all remaining read costs are optimal.
