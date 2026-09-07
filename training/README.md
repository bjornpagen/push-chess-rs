# Metal learning and research primitives

This directory contains source code, not the dataset. Durable game, position,
action and analysis facts live exclusively in the bumbledb directory `data/corpus`.
Model/optimizer tensors are separate checkpoint artifacts. No file-shard replay
store or legacy game importer is supported.

The Rust `CorpusReader` transfers typed metadata and owned NumPy action/analysis
arrays in whole-game pages. Initial/final FENs are derived at the boundary, not
stored in the database. Exact histories preserve repetition semantics. Completed
teacher searches supply one-hot policy targets; only rules-verified terminal
results supply WDL targets. Raw engine scores are uncalibrated observations.
The bounded sampler deduplicates trajectories and reservoir-samples the training
split. Capped games have zero value weight, never fabricated draw labels.

```sh
cargo build --release -p push-chess-lab
uv sync --group dev
uv run maturin develop --release
uv run --no-sync pytest -q
```

The retained tinygrad model, learner, optimizer, inference caches, Rust batched
search, rolling actor slots, restart archive and reanalysis are available for
experimentation. Rolling snapshots and replay are RAM-only. A future durable
neural collector must add typed policy facts to bumbledb. The current research
loop focuses on classical search plus NNUE, not replacing search with a purely
neural player. No neural training starts automatically with a tournament.

The optional full-model teacher baseline remains available after the tournament
releases bumbledb's single-process ownership:

```sh
uv run pushzero pretrain --db data/corpus --output models/student.safetensors \
  --steps 1000 --batch-size 128 --capacity 100000 --max-games 10000
```

Metal is required by default. Explicit CPU diagnostics need `DEV=CPU` and
`--allow-cpu`. `--resume CHECKPOINT` restores weights and optimizer for a new
bounded teacher pass; it is not a bit-exact sampler continuation. Tiny update
tests verify integration, not playing strength. Do not promote on training loss.
