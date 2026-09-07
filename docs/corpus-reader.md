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
