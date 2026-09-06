# PushZero

A rules-only self-play learner for Push Chess: tinygrad **0.14.0**, Apple **Metal**,
and the existing Rust rules. It does not use Cataclysm scores, piece values,
human games, engine demonstrations, or shaped strategic rewards. The existing
engine strategies and deployed network are unchanged; shared rules internals
have been refactored with parity tests. This is a new training system, not
a claim that a newly trained checkpoint is already stronger than Cataclysm.

## Run it

Requires an Apple Silicon Mac, Rust, Python 3.12+, and `uv`. From the repository root:

```sh
uv sync
uv run pushzero doctor
uv run pushzero train --run experiments/my-pushzero --minutes 60
uv run pushzero train --run experiments/my-pushzero --resume --minutes 60
uv run pytest -q
cargo test --workspace --all-features
```

The command-line entry point defaults to `DEV=METAL` and refuses silent fallback.
For tinygrad 0.14, **`DEV=METAL` is correct; `METAL=1` is deprecated and errors**.
Direct Python programs must set `DEV` before importing tinygrad. Explicit CPU
diagnostics use `DEV=CPU uv run pushzero --allow-cpu ...`.

After changing Rust sources, rebuild the editable extension:

```sh
uv sync --reinstall-package pushzero
```

`doctor` checks real inference, backpropagation, a saved/loaded checkpoint, and
search throughput. It does not produce strategic training data. The default
network has **1,065,796 parameters** (96 channels, six residual blocks). Use
`--channels 64 --blocks 4` for a 333,188-parameter alternative.

## What is learned

The network sees 32 board planes, with ranks flipped and colors swapped for
Black so that the current player always moves in the same canonical direction:

- 12 piece planes for the current board, and 12 for the previous board;
- four own/opponent castling-right planes;
- en-passant target, normalized halfmove clock, current repetition count,
  and absolute side to move.

The previous board is reconstructed from the last undo record without cloning
the history. The search retains the **complete** history for adjudication;
the network's finite observation is not a complete encoding of that history.

A pre-activation, GroupNorm residual CNN produces square features and a pooled
global feature. The policy head scores **each actual legal action**, using its
source/destination features plus learned route, stop, promotion, and special-move
embeddings. All six fields of a move's identity survive encoding. Different
knight routes and pushed-pawn underpromotions are never collapsed into one label.
Action arrays are padded to power-of-two widths, but legal sets are never cut off.

A second head predicts win/draw/loss probabilities. The scalar search value is
`P(win) - P(loss)`, always relative to the player to move. Both output heads start
at zero logits, so the initial predictions are uniform, not seeded by an engine.

## How self-play improves the policy

1. The network evaluates each root. Gumbel noise samples root candidates.
2. Rust allocates the search budget using sequential halving. It traverses exact
   legal moves, pauses on new nonterminal leaves, and asks Python to evaluate a
   batch of leaves from different games. Terminal leaves use the rules' outcome.
3. Values alternate sign during backup. Unvisited actions receive a completed-Q
   estimate mixing the network value and prior-weighted visited Q values.
4. The improved policy is `softmax(network_logits + transformed_completed_Q)`.
   Interior nodes use deterministic improved-policy deficit selection. Root
   Gumbel noise affects exploration and the selected move, **not** the target
   distribution directly. There is no additional Dirichlet noise.
5. Finished games attach exact win/draw/loss targets to sampled positions.
   AdamW trains policy cross-entropy plus WDL cross-entropy, with weight decay
   and global gradient clipping. The model then generates the next games.

This is a compact AlphaZero/Gumbel AlphaZero-style system, not MuZero: the rules
are known, so learning a dynamics model would spend scarce compute approximating
something Rust already does exactly. It incorporates later ideas where useful:
Gumbel low-budget policy improvement, mixed playout caps, replay, and optional
reanalysis. It does not claim to implement every post-AlphaGo innovation.

Default self-play uses 64 full simulations, 16 fast simulations, and a 25%
full-search probability. Training retains full-search positions, with a final
fallback for games that never got a full search. There are 32 concurrent games,
64 games per iteration, a 100,000-position replay window, and four sampled
training examples per newly stored position on average.

The optional sparse-position curriculum (`--curriculum`, default 0.25) samples
valid boards with both kings and 1–4 additional pieces. It gives an untrained
network more opportunities to encounter decisive terminal outcomes. No engine
chooses these boards or labels them. It changes the starting-position
distribution, so performance must still be measured from normal openings.
Set it to zero for strictly standard-start self-play. `--reanalysis N` refreshes
up to N replay policies per iteration using the current network and exact
reconstructed game histories; it preserves the original outcome labels.

## The Rust/Python boundary

`src/selfplay/` is a normal, independently testable Rust module. It has no Python,
NumPy, or neural-runtime dependency. Its `State` owns validated position/history
and a sorted legal-move proof. `BatchSearch` owns trees, schedules, and evaluation
phases. Terminal, unevaluated, and expanded nodes are different enum variants.

Each tree has one reusable board cursor. Simulations make/unmake moves against
that cursor instead of repeatedly cloning a growing history. Arena node indices
use compact optional child links. Previous-board features copy only 64 pieces.

`training/native/` is a thin PyO3/NumPy ownership adapter. The hot loop sends:

- Rust → Python: contiguous board/action arrays for all pending leaves;
- Python → Rust: contiguous logits/value arrays, borrowed and validated once;
- search completion: legal move IDs, normalized targets, visits, and node counts.

Rust transfers its batch buffer allocations to NumPy ownership without copying
or boxing individual elements. This is **zero-copy at the Rust→NumPy boundary**,
not end-to-end zero-copy: feature packing, padded Python staging, and tinygrad's
host/Metal transfers still exist. Expensive native search releases the GIL;
submission keeps it while borrowing Python-owned data, avoiding concurrent
mutable aliases. There are no Python callbacks inside tree traversal.

The new `advance` protocol combines submission and the next request in one
boundary per neural round, plus the initial request. Replies carry globally
distinct request IDs, lengths and optional effects; stale/duplicate replies are
rejected before tree mutation. The older two-call API remains available.
Terminal-only rounds need no neural call. Root setup and result packaging happen
once per search. Native-boundary timers include waiting when workers are enabled;
they are not measurements of CPU instructions or device arithmetic.

Before this redesign, the first local measurements were roughly 2,000–3,200 network positions/second
for the default model on an M2 Max (32-position batches; concurrent work affects
the rate). One measured 512-simulation batch took 0.236 s, of which 0.0061 s was
native search. That suggests optimizing the neural path matters more next than
adding CPU worker complexity. These are throughput measurements, not Elo.

## Reliability and resuming

The same Rust state and search are directly callable from Python:

```python
import os
os.environ.setdefault("DEV", "METAL")  # before importing tinygrad
import numpy as np
from pushzero import State
from pushzero.learning import load_checkpoint
from pushzero.model import Predictor
from pushzero.search import search

model, metadata = load_checkpoint("experiments/my-pushzero")
games = [State() for _ in range(32)]
with Predictor(model, batch_size=32) as predictor:
    results = search(games, predictor, np.random.default_rng(42), simulations=64)
for game, result in zip(games, results):
    game.play(result.move)  # full Rust history survives the next search
```

For another neural runtime, implement `packed(boards, actions)` returning
contiguous float32 `(logits, values)` arrays, plus the metrics counters used in
`search.py`. Or use the public `SearchBatch` request/submit interface directly.
Rust never needs to know which framework supplied the evaluations.

### Checkpoint transactions

Every checkpoint contains model weights, Adam moments, learning rate, bias
correction state, NumPy RNG state, configuration, iteration, and committed replay
shard names/checksums, optional restart archive, and optional separate EMA state.
Checkpoints use safetensors. Replay format 2 stores ragged IDs/targets and one
game log with per-sample ply references; only selected minibatches reconstruct
inputs, through a bounded cache. The legacy format-1 reader remains available.
Writes use unique
temporary files and atomic replacement. `latest.json` advances only after a
checkpoint is complete. Orphaned files from an interrupted iteration are not
silently consumed on resume. An exclusive run lock rejects a second writer.

Resume restores the **saved configuration**, not new architecture/hyperparameter
flags supplied on that invocation. `--minutes` and `--iterations` apply to the
new session. Unfinished optimizer updates are checkpointed and completed before
generating new games. Ctrl-C/SIGTERM requests a graceful stop at the next search/update
boundary. Hard crashes leave the last committed checkpoint usable. Time limits
are soft at those boundaries; compilation or a current GPU operation can overrun.

Stopping the process releases the CPU/GPU work; no background monitor or automatic
restart is installed. Restart only with an explicit `train --resume` invocation.

Games stopped by a time/ply limit are **not draws**. Their policy targets remain
usable, but their WDL target is zero and value-loss weight is zero. Genuine
stalemate, repetition, and fifty-move outcomes are draws; mate takes precedence.

An important tinygrad-specific invariant is tested: the inference graph cache
belongs to a model revision. Learning increments that revision and invalidates
old inference graphs, because transformed weights can otherwise remain stale.
Tests also verify matching optimizer updates after saving and resuming.

Run files are under the ignored `experiments/` directory. `status.json` contains
the last event; `metrics.jsonl` records progression; `initial.json` identifies the
frozen untrained checkpoint; `latest.json` identifies the newest committed one.
No training result automatically replaces the deployed Cataclysm network.

## Measure strength, not just loss

Pass an immutable checkpoint filename, a run directory (latest checkpoint), or
`initial.json` (frozen untrained checkpoint):

```sh
uv run pushzero evaluate experiments/my-pushzero \
  --opponent cataclysm --pairs 16 --simulations 64 --opponent-ms 50 \
  --output experiments/my-pushzero/vs-cataclysm.json
uv run pushzero analyse experiments/my-pushzero \
  --simulations 128
```

`--opponent` also accepts `random`, `astra`, or another checkpoint/run/pointer.
For example, use `--opponent experiments/my-pushzero/initial.json` to measure
progress against the frozen initial network at the same search budget.
Evaluation uses held-out random legal opening prefixes with both colors for
each opening. It stores complete move histories and reports unfinished games
separately, with score bounds and actual average/p95 move times. Checkpoint opponents
receive equal simulation caps; native opponents have their own time/node caps.
**These native matches are diagnostic, not time-equalized Elo estimates.**
Add `--move-ms 50` to request the same wall-time budget for each contestant.
Neural searches stop at evaluation-round boundaries, so root inference or new
shape compilation can overrun; the report includes overrun counts. Confidence
bounds group both colors by opening and use conservative Hoeffding bounds.
They assume a fixed sample size and independent opening draws, not repeated
peeking until a model happens to look good.
The `analyse` FEN interface has no pre-FEN repetition history; the Python `State`
interface can retain it by replaying moves.

An excellent player will need many more games than a bootstrap run. Increase
training duration only after checking throughput, terminal-outcome coverage,
held-out WDL/policy behavior, and matches against frozen checkpoints and native
engines. Then compare higher search budgets, architecture sizes, reanalysis,
and curriculum schedules in controlled runs. Tree reuse, within-tree parallel
leaves, mixed precision, and broader history encodings are possible next steps,
not implemented features. A larger network alone is not evidence of strength.

Algorithm references and the adapted mctx schedule license: [THIRD_PARTY.md](THIRD_PARTY.md).

## Verification scope

See [the current verification record](../docs/verification.md). The redesign's
first pass passed the Rust workspace tests and 24 Python/Metal tests. These
include real effect-head gradients/EMA recovery and mock interrupted-update
orchestration, not evidence of playing strength. A host build of the WASM crate
does not substitute for an actual `wasm32-unknown-unknown` target check.
Strict Clippy for **all historical engines** is not clean: those retained
experimental implementations have hundreds of pre-existing lint findings. They
are preserved as comparison baselines instead of receiving a blanket lint-only
rewrite in this training change.

## Inference-first architecture experiments

Defaults preserve the old 96×6 model and optimizer recovery. New architectures
require a new run; resume uses the saved configuration. The first hypothesis is
`--channels 64 --blocks 4 --global-every 2 --effect-channels 16`. Effects encode
joint `(action+1, square, before-piece, after-piece)` changes, canonicalized to
the current player. Action zero is a padding segment, not a real action. Exact
effects add rules information, not strategic labels.

Useful independent controls:

- `--global-every 0`, `--effect-channels 0`: plain trunk/reference ablations.
- `--inference-cache-mb 64`: bounded exact-input prediction cache (default off).
- `--workers 1`: serial baseline; `2/4/8`: persistent coarse native workers;
  `0`: available logical parallelism. More workers are not assumed to be faster.
- `--curriculum 0 --restart-fraction .25`: standard starts plus previously visited
  states with their full repetition/previous-board prefix. The bounded archive
  uses observed prediction error as a sampling priority, not as a value target.
- `--no-fast-explore`: remove Gumbel noise on cheap turns.
- `--ema-decay .995`: checkpoint a separate averaged model. Evaluate it with
  `--weights ema`; training and optimizer resume always use raw weights.
- `--reconstruction-cache-mb`, `--max-graphs`, `--max-nodes`: explicit capacity
  controls. Exceeding search capacity reports an error, never a synthetic draw.

Plan-only and opt-in measurement commands (nothing launches automatically):

```sh
uv run pushzero plan --output experiments/design.json
uv run pushzero profile --channels 64 --blocks 4 --actors 32 \
  --output experiments/inference-64x4.json
uv run pushzero profile --channels 64 --blocks 4 --global-every 2 \
  --effect-channels 16 --actors 32 --workers 2 --search-simulations 16 \
  --output experiments/inference-effects.json
```

The plan writer imports neither tinygrad nor the native extension. Profiling
warms bounded shapes and interleaves fixed/tail/identity controls; report cold
time separately. Cache-on deliberately tests repeated corpus hits, so run
cache-off too. Host staging reuse is not an unsafe NumPy-to-Metal pointer trick.
