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

The optional full-model teacher baseline requires an audited, sealed training
corpus. The old-rules data was purged; the 256 remaining games are evaluation-only,
so there is currently no training data. No training run is authorized.
After a future training corpus exists and its writer releases ownership:

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

After a future corpus run has stopped and passed `lab verify`, replace
`CORPUS_RUN_ID` with that run's ID (the purged Run 2 is not available):

```sh
uv run pushzero nnue train --db data/corpus --runs CORPUS_RUN_ID \
  --output models/residual-candidate.safetensors --steps 1000 --batch-size 256
uv run pushzero nnue export models/residual-candidate.safetensors \
  --output models/residual-candidate.bin
```

Source runs must be explicit. The native page reader validates every requested
run before yielding its first game: unknown, unsealed and failed sources are
errors, even when another requested source is valid. An interrupted sealed run
can still supply its complete games. Run selection happens before relational
reconstruction, rules replay and NNUE feature construction, not after Python
has received unrelated trajectories. Both training and validation need eligible
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

`pushzero nnue` is the sole NNUE command entry point (`train`, `export`,
`benchmark`); Python uses `pushzero.nnue.Model`. The deployed Rust model owns
the format/dimensions, feature IDs, clipping/division constants and final score
conversion. Python imports that contract, not a parallel constant set.
Tinygrad provides differentiable forward/backward and optimizer updates.
Frozen inference/validation calls the actual Rust search accumulator in
whole batches; the learning-only Rust evaluator and production NumPy evaluator
are removed. An independent integer oracle exists only in tests. Each
validation pass freezes current weights once, so optimizer changes cannot
leave a stale prediction cache. No per-node Python/GPU calls or backend fallback.
Repeated inference can call `frozen = model.freeze()` once, followed by
`frozen.scores(ids, baselines, sides)` with contiguous u16/i32/u8 arrays (at
most 4096 rows per call). That object owns immutable weights, unaffected by
later training; freeze again explicitly to adopt new weights. This avoids
export, weight transfer and decoding on every prediction request.

Choose backends by measured workload, not language. A bounded comparison is:

```sh
uv run pushzero nnue benchmark --batches 1 32 256 1024 --repeats 15
```

It compares native sparse inference with the same tinygrad forward on CPU and
Metal, including host-input preparation and synchronized host scores. Separate
resident-input timings bound the tensor-only case. It also times complete
temporary QAT updates on both tinygrad devices. Inputs rotate, order alternates,
JIT is warmed and inference parity is asserted. Nothing opens the corpus,
persists games or saves trained weights. Active all-core tournaments contaminate
absolute timings; repeat idle before acting on small differences. This measures
full sparse rebuilds, not the cheaper changed-piece updates inside tree search.
The [measured backend decision](../docs/nnue-backends.md) keeps native inference
and Metal at the production batch sizes. Tiny batches favored CPU; select
`nnue train --device CPU` explicitly when that matches the measured workload.
`--device METAL` is also explicit; omission uses DEV (normally METAL).

Quantization-aware forward arithmetic reproduces the deployed accumulator:
shared color/rank-reflected features; integer feature sums; hidden clipping to
`[0,256]`; signed residual division by 8192 truncated toward zero; fixed
handwritten score; side-relative tempo +14; final clipping to ±28000. Weights
use straight-through ties-to-even rounding. Output weights are constrained to
±2047 so every integer output sum fits exactly in float32, and reduced-precision
tensor-core multiplication is disabled. Rust/NumPy/Metal parity and checkpoint
optimizer recovery are regression-tested. All replay stays in RAM.

Each checkpoint records source identities, sampling digest, optimizer state,
control/export hashes and before/after validation loss. Validation includes the
old embedded network and a zero-network handwritten
baseline, both scored by the same deployed native kernel. The zero network is
only a validation reference, not the training initialization or an installed
engine. Beating the old residual's loss alone does not beat the no-NNUE baseline.

Training validates the exact exported integer model every 250 steps by default
(`--validation-interval`) and stops after four non-improving checks (`--patience`).
It selects the lowest game-balanced held-out loss, including the initial weights;
the final update is not automatically better. Metadata records the complete
validation curve, attempted and selected steps, and early stopping. These are
selection-set measurements, not an independent estimate of playing strength.

`--resume` restores weights, optimizer and random sampler at the same selected
step, and rejects changed sample digests or batch size. `--init CHECKPOINT`
explicitly starts a fresh refinement from weights only; it resets optimizer and
sampler and optionally accepts `--learning-rate`. Use `--init` for older
checkpoints without a sampler or for an intentional change of data/learning rate.
Only checkpoints matching the current model/rules contract can initialize or
resume training. The archived r2 checkpoint is no longer accepted by the current
trainer; its unchanged exported inference weights remain available as a control.

Core, lab and Python implement one corrected ruleset. The normalized database
has no per-run rules column, and the active engine has no historical replay
switches or special legacy sampling path. Old schema stores fail to open rather
than being silently interpreted. Checkpoint rules/encoding identities remain
as compatibility guards, not selectable game modes. See
[the verified corpus cutover](../docs/corpus-cutover.md).

Export produces the separate 49,280-byte i16 candidate. It does **not** install
the network, edit the embedded control, or promote it. Require fresh paired
playing-strength tests at multiple budgets before adopting a candidate.

An exported candidate can enter those real-search comparisons without editing
the control or recompiling weights into the engine. Build/verify/commit the
runner first, and wait for the active database owner to finish:

```sh
target/release/lab tournament --db data/corpus --engines cataclysm,aurora-r2 \
  --nnue aurora-r2=models/residual-r2.bin --pairs 100 --workers 12 \
  --time-ms 100 --purpose arena --seed 2026090701 --max-seconds 3600 --max-gib 50
```

Use a fresh seed for each independent confirmation. Changed network bytes need
a new generation name; the corpus rejects relabelling an existing binary/name
identity. Multiple `--nnue NAME=PATH` options are allowed. Every worker owns its
search scratch and borrows shared immutable weights; no Python or GPU calls
occur inside classical search.

For root-level investigation, `Opponent("aurora-r2", network_bytes)` loads an
isolated candidate. `new_game()` clears search history; `analyse(state,
time_ms=0, nodes=4096)` returns a legal selected move, score, node/depth/cost
observations, proof observations and an owned u32 PV array. It releases Python
during CPU search. No candidate may use a built-in name. No game is persisted
by these diagnostic calls; tournament generation is still the durable path.

## Typed forensic access

`games(path, split, runs=[2,3])` and `CorpusReader.page(..., runs=[2,3])` share
the native source-selection contract. IDs are sorted/deduplicated; cursor
pagination visits only those runs. Omitting `runs` selects all eligible sealed
runs and retains quarantine filtering; an explicit empty list is an error.
Split/outcome restrictions remain in force. This changes no database schema
and creates no copied dataset or persistent analysis file.

`CorpusReader.state(run,game,ply)` restores the actual prefix, including draw
history, rather than constructing a history-free state from a FEN. This is an
explicit forensic API, including quarantined runs. Do not use it to bypass
training splits; NNUE training always uses eligible `page()` results.

Besides `moves` and the six existing `analysis` columns, a game page supplies:

- `search_details`: i64 columns `seldepth, wall_us, reported_us, qnodes, tt_hits`;
  use `analysis[:,2]` to distinguish actual searches from opening rows.
- `pv_offsets` and `pv_actions`: owned u32 arrays; the PV for ply p is the
  actions slice `[pv_offsets[p]:pv_offsets[p+1]]`.
- `proof_searches` and `mate_proofs`: sparse u64 `(ply,nodes)` and `(ply,plies)`
  rows. Absence is not a fabricated zero-node proof.

PVs/proofs are observations reported by the engine, not independently verified
mate certificates. All these arrays are derived from normalized bumbledb facts
and remain in memory; they do not create another replay or analysis datastore.
