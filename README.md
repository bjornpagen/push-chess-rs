# Push Chess

A Rust engine-development arena for a chess variant where pieces can push
friendly pieces along their movement paths. Includes shared rules, experimental
engines, automated matches, a playable terminal interface, and replay tools.
This is not an ordinary-chess engine or a neural-network training project.

## Rules in brief

Friendly pieces can be displaced in a chain, provided there is room beyond the
destination. Pushing a chain off the board or into an enemy piece is illegal.
Enemy pieces are captured directly, not pushed. Knights follow either of two
L-shaped paths and can push along those paths. A pushed pawn can promote.
Moves must leave the moving side's king safe.

## Project map

- `src/core/`: board state, move generation, push resolution, move undo, and
  position hashing.
- `src/engine.rs`: the common engine interface and factory entry type.
- `src/candidates/`: 42 experimental engine implementations; 24 are selectable
  through `ENGINE_REGISTRY` in `mod.rs`. Unregistered engines are retained as
  historical experiments and still compiled.
- `src/bin/showdown.rs`: matches, round-robin tournaments, standings, and SQLite
  logging of games, positions, moves, and search telemetry.
- `src/bin/play.rs`: human versus engine in a terminal interface.
- `src/bin/replay.rs`: browse and replay saved games in a terminal interface.
- `src/bin/dump.rs`: textual tournament/match reports with detailed telemetry.
- `tests/core_tests.rs`: 23 tests of the shared rules and board operations.

## Run

The repository selects nightly Rust (edition 2024) through `rust-toolchain.toml`,
including rustfmt and Clippy. Run from the repository directory. Release builds
are important for meaningful engine timing. To update the selected channel:

```sh
rustup update nightly
```

```sh
# Test all targets.
cargo test --all-targets

# Play as White against Void, with 1 second of engine thinking per move.
cargo run --release --bin play -- void white 1000

# Browse saved games or list recorded tournaments.
cargo run --release --bin replay
cargo run --release --bin dump -- list

# Detailed report for the latest recorded match or its enclosing tournament.
cargo run --release --bin dump -- latest

# Run eight games, alternating colors, at 1 second per move.
cargo run --release --bin showdown -- void chronos 8 1000000

# Run a four-engine round robin, with eight games per pairing.
cargo run --release --bin showdown -- tournament elite4 1000000 8 void chronos oblivion abyss
```

`play` takes milliseconds; `showdown` takes microseconds. Matches and tournaments
write to `pushchess.db` by default. Plain `play` defaults to Oblivion, not the
latest tournament winner.

## Previous elite tournament: Void

Snapshot inspected before Astra's debut on September 5, 2026. Tournament #12,
`elite4f`, started at database timestamp `2026-04-02 00:41:13`, with a one-second
move budget and eight games per pairing. No new tournament was run for this
snapshot.

| Engine | Wins | Draws | Losses | Score |
| --- | ---: | ---: | ---: | ---: |
| Void | 15 | 5 | 4 | 72.9% |
| Oblivion | 15 | 3 | 6 | 68.8% |
| Chronos | 11 | 5 | 8 | 56.3% |
| Abyss | 0 | 1 | 23 | 2.1% |

Score counts a draw as half a win. Chronos leads the saved all-time score table
at 63.9% over 216 games; these are different comparisons, not Elo ratings.

The current Void implementation (`src/candidates/void_engine.rs`, labelled
`void_002`) builds on Chronos's search architecture:

- Evaluate positions cheaply using material and piece-square tables: how much
  material each side has and whether its pieces occupy favorable squares.
- Search progressively deeper with alpha-beta/principal-variation search,
  spending less effort on branches unlikely to affect the chosen move.
- Reuse results in a 4,194,304-entry position cache, with packed 12-byte entries
  (48 MiB of entry storage).
- Prioritize captures, promotions, and promising pushes; track which pushes
  worked during search to guide subsequent searches.
- Select the next promising move only when needed instead of sorting every
  remaining move up front.
- Extend tactical continuations through captures and promotions at the search
  boundary, up to ten additional plies.

In plain language: Void trades elaborate positional judgment for cheaper
calculation and selective lookahead. That is its design, not proof of why it
won. In tournament #12, its reported average search depth was 13.3 plies versus
Chronos's 13.0, but its measured nodes per second were roughly the same. Reported
depth can include an interrupted iteration and is not uniform full-width depth.

### Limits of the evidence

Recent tournaments have different winners. Games start from the same initial
position; the current Void and Chronos implementations ignore their game seeds.
The database stores engines by name with empty source hashes, so historical
results cannot be reliably tied to exact source revisions. All-time percentages
also mix different opponents and budgets. Treat Void as the latest recorded
winner of that event, not a conclusively established strongest engine.

## Latest elite winner: Astra

`src/candidates/astra.rs` introduces generation 13. Its search uses pseudo-legal
move generation and checks legality only when visiting a move, check-aware
quiescence (including quiet evasions), ply-normalized mate scores in a full-key
two-way position cache, repetition detection, guarded null-move pruning, and
iterative deepening with aspiration windows. Its evaluation includes phase-aware
king placement, pawn shelter, passed pawns, and latent attack lines. Push ordering
recognizes friendly pieces along an entire path, including knight paths and
moves to empty destinations.

It respects time, node, and depth budgets, reports only completed search depths,
and does not use opponent identities or saved tournament results to choose moves.

```sh
cargo run --release --bin play -- astra white 1000
cargo run --release --bin showdown -- tournament astra_debut_001 1000000 8 astra void chronos oblivion
```

The debut uses the same one-second move budget and eight games per pairing as
the previous elite tournament. Its results must be interpreted separately:
the shared push-cascade fixes landed after the older tournaments.

Astra won the completed 48-game tournament #13, finishing its own 24 games at
**17W / 2D / 5L (75%)**, ahead of Void (52.1%), Oblivion (45.8%), and Chronos
(27.1%). It won the head-to-head match against each incumbent.
See [the debut report](docs/astra-debut.md)
for match scores, verification, and source fingerprints.

## Representation-first refactor

- Push resolution returns `Option<PushPlan>`: illegal paths cannot masquerade
  as partially initialized moves. Captures are optional squares, not sentinel
  numbers plus a separate flag. Bounded displacement and undo collections own
  their lengths.
- Pushes apply simultaneously from a snapshot, including the intermediate
  knight board. This fixes overwritten pieces and mistaken identities when
  composing two push cascades. Historical tournament results predate this fix.
- `core::children` uses a generic associated type (GAT) for a lending iterator.
  One borrowed child position exists at a time; dropping it undoes the move.
  Legal-move generation uses this interface, including automatic restoration
  on early exit or panic. GATs themselves do not require unstable features.
- Fourteen engines share a const-generic inline move collection that can spill
  to the heap. Moves and scores are stored together, eliminating duplicated
  length fields, unchecked overflow, and Abyss's uninitialized-array bug.
- Thirty-nine engines retain actual enum values in their caches rather than
  converting unchecked bytes. Twenty-seven engines share one bounded SIMD
  ordering implementation. Void and Abyss now forbid unsafe code locally.
- Engine evaluation/search policies and historical source variants remain
  separate; the refactor does not collapse different experiments into one
  configurable engine.

Verification includes core-rule tests, cascade regressions, deterministic
playout/hash invariants, move-buffer spillover, short searches from every
selectable engine, and a compile-fail test for overlapping child borrows:

```sh
cargo fmt --all -- --check
cargo test --all-targets
cargo test --doc
cargo test --release --all-targets
cargo clippy --all-targets -- -D warnings -A clippy::needless_range_loop -A clippy::too_many_arguments
```

The lint exceptions retain explicit indexed numerical loops and established
recursive-search signatures. They do not suppress compiler or safety warnings.
ARM builds use the shared NEON path; other architectures use the scalar fallback.

These checks do not establish playing strength or prove every search heuristic
correct. Rerun competitive tournaments before declaring a new winner.

## What is safe to clean?

- `target/` contains generated binaries and build caches. `cargo clean` removes
  them; later builds regenerate them and take longer.
- Keep `Cargo.lock`: it pins dependency versions for reproducible builds.
- Keep `pushchess.db`: the inspected database contains 884 game rows and 62,872
  move records, not disposable cache data. It is ignored by Git, so back it up
  separately. SQLite `-wal` and `-shm` sidecars are not arbitrary junk either.
- Keep historical candidate source files unless deliberately retiring those
  experiments. A low score or absence from the selectable roster is not enough
  reason to delete an implementation.

Warning: `showdown purge-db` deletes the database and its sidecars immediately.
It is not a build-cache cleanup command.
