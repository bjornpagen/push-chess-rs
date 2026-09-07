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

## Small NNUE refinement

This is the intended neural component of the classical-engine research loop:
the existing 768-feature, 32-hidden-unit Cataclysm residual, with its fixed
handwritten evaluation and search left intact. It is not a policy network.

After the tournament has stopped and the corpus has passed `lab verify`:

```sh
uv run pushzero nnue --db data/corpus --runs 2 \
  --output models/residual-r2.safetensors --steps 1000 --batch-size 256
uv run pushzero nnue-export models/residual-r2.safetensors \
  --output models/residual-r2.bin
```

Source runs must be explicit. Both training and validation need eligible
terminal games; there is no fallback to training-set validation. Test/arena
families are never read. Whole trajectories are deduplicated; eligible games
are reservoir-sampled, with up to 16 positions retained per game. Batches
sample a game uniformly, then a position within it. Non-check positions with
completed searches and non-mate-range scores are eligible. Search scores are
not training labels: the verified final result supplies expected-score targets
(win=1, draw=0.5, loss=0) relative to the side to move. This predicts expected
score, not separate win/draw/loss probabilities. Validation averages losses
within each game before averaging games. Similar strategies can still cross
opening-hash boundaries, so this holdout alone is not a strength test.

The Rust boundary emits owned `[plies,2,64]` u16 feature IDs plus baseline and
check-status arrays once per game page. Sparse replay costs 256 bytes/position
for features. Only a sampled batch expands to dense float32 features for Metal
matrix multiplication; no Python per-piece FFI loop is needed.

Quantization-aware forward arithmetic reproduces the deployed accumulator:
shared color/rank-reflected features; integer feature sums; hidden clipping to
`[0,256]`; signed residual division by 8192 truncated toward zero; fixed
handwritten score; side-relative tempo +14; final clipping to ±28000. Weights
use straight-through ties-to-even rounding. Output weights are constrained to
±2047 so every integer output sum fits exactly in float32, and reduced-precision
tensor-core multiplication is disabled. Rust/NumPy/Metal parity and checkpoint
optimizer recovery are regression-tested. All replay stays in RAM.

Each checkpoint records source identities, sampling digest, optimizer state,
control/export hashes and before/after validation loss. `--resume` restores
optimizer-matched weights for a new seeded pass, not a bit-exact replay cursor.
Export produces the separate 49,280-byte i16 candidate. It does **not** install
the network, edit the embedded control, or promote it. Require fresh paired
playing-strength tests at multiple budgets before adopting a candidate.
