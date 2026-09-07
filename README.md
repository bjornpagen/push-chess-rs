# Push Chess training ground

A native engine laboratory and a fresh training corpus. Cataclysm and Astra
are controls; fifteen independently named challengers test evaluation and
search hypotheses. Results—not names—determine which experiments survive.

**bumbledb is the only durable source for tournament runs, entrants, complete
games, move histories, search analysis and outcomes.** There is no legacy
schema, importer, alternate database backend, or file-shard replay store.
Games are decomposed into typed relations for setups, pieces, actions,
positions, exact piece changes, principal-variation steps and results—no
stored JSON/FEN blobs. See [the relational model](docs/engine-lab.md#one-bumbledb-schema).

## Engine laboratory

Current research constraint: Run 4 is finished and audited; launch no further
tournaments (including smoke or arena rounds). Continue code/model improvement instead;
see [the current authorization](docs/research-loop.md#current-authorization--2026-09-07).
The commands below document the interface, not permission to start another run.

```sh
cargo test --workspace
cargo build --release -p push-chess-lab
target/release/lab list
mkdir -p data
target/release/lab init --db data/corpus
target/release/lab tournament --db data/corpus --engines all \
  --pairs 500 --workers 12 --time-ms 100 --purpose corpus \
  --opening-plies 6 --max-plies 512 --seed 20260906 \
  --max-seconds 86400 --max-gib 50
```

That 17-engine roster has 136 matchups: 500 color-swapped pairs per matchup
schedules 136,000 games. Matchups interleave. Time, disk and game caps are explicit;
the first reached ends admission. SIGINT/SIGTERM safely stop at move boundaries.
The disk cap is checked between commits and may overshoot by in-flight games.

While running, query the owning process, which reads the same bumbledb store:

```sh
target/release/lab status --db data/corpus
target/release/lab status --db data/corpus --run 1
```

After stopping, use `summary` or `report` instead of `status`.
Use `lab verify --db data/corpus --run 1` for a full relational/rules audit.
bumbledb has one process owner; do not open another process against a live
store. The local control socket is not a second datasource. Progress logs are
disposable; every reported committed game is in bumbledb.

## NNUE training

The current learning loop improves Cataclysm's small residual without replacing
classical search. One NNUE interface selects the appropriate implementation:
native incremental inference in search, tinygrad differentiation for training.
The deployed model owns the format and feature contract. Training reads only
explicit, sealed bumbledb source runs. Replace `CORPUS_RUN_ID` below with a new,
audited training run ID; the old-rules corpus has been purged and the remaining
256 games are evaluation-only ([cutover](docs/corpus-cutover.md)):

```sh
uv sync --group dev
uv run maturin develop --release
uv run pushzero nnue train --db data/corpus --runs CORPUS_RUN_ID \
  --output models/residual-candidate.safetensors --steps 1000 --batch-size 256
uv run pushzero nnue export models/residual-candidate.safetensors \
  --output models/residual-candidate.bin
```

Run this **after** the tournament releases the store and passes replay audit. Training is opt-in,
not part of tournament generation. Model/optimizer tensors are checkpoint
artifacts, not game stores. Search scores are retained as observations;
only rules-verified endings supply value labels. A game cap is not a draw.
Candidate artifacts do not replace the embedded control. Require fresh paired
arenas before promotion. Full-policy Metal/self-play tools remain available
for later work; see [training](training/README.md).

## Layout

- `src/core/`: authoritative rules, lossless moves and prepared transitions.
- `src/engine/`: common clock/node limits, compact table storage and tie policy.
- `src/engines/`: active control families and const-specialized experiments.
- `crates/lab/`: bounded worker pool, bumbledb schema, writer, reports and live status.
- `src/selfplay/`: reusable Rust neural-search runtime and history representation.
- `training/`: tinygrad/Metal training and direct corpus reader.
- `sources/`: retained research sources.
- `data/`: ignored local bumbledb corpus.
- `models/`: [preserved r2/r3 models and checkpoints](models/README.md); other candidates stay ignored.

See [the engine lab](docs/engine-lab.md) and [training](training/README.md).
Retired engines and interactive/browser clients are in Git history.
