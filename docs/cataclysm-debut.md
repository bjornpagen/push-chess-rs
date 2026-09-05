# Cataclysm 001

An independent generation-14 engine for Push Chess. The rules and push resolver
remain shared with every opponent; Cataclysm has its own move expansion,
transactional search board, search, learned evaluator, and forcing-mate solver.
It does not wrap an incumbent, consult opponent identities, read the game
database during play, or choose moves from an opening book.

## What the entire history said

Before development, both databases contained **974 game rows and 69,904 moves**.
The [complete audit](history-audit-001.md) visits every row and provides a ledger
for every game, including unfinished and empty records. This is exhaustive
automated reconstruction, not a claim that every position received a deep
engine search or a human-style annotation.

Three recurring features motivated the design:

- **965 of 1,454 promotions** promoted a pushed pawn. A rook or bishop can be a
  delivery mechanism, not merely a piece seeking a good square.
- **6,021 of 9,027 reconstructed checks** were neither captures nor promotions.
  Tactical search cannot focus exclusively on exchanges.
- **3,665 pushes displaced a friendly king**, and **10,380 pushes landed on an
  empty destination**. Checking only the nominal mover/destination misses much
  of what a move does.

Not all historical transitions survive the shared-rule corrections: 69,750
saved moves are legal today and 68,530 exactly reproduce the recorded next
board. Older outcomes are therefore imperfect learning evidence, not ground
truth for the current game. The audit records these differences explicitly.

## The implementation

### Pushes as transactions

Move expansion retains the resolved displacement plan. Once search selects a
move, it applies that transaction without resolving its push again. Mailbox,
piece bitboards, hash, king locations, material, piece-square scores, and both
neural accumulators change together and restore from a bounded snapshot.
Search does not mutate the caller's board or accumulate shared undo records.

Before competitive testing, differential verification checked **55,577 distinct
saved FENs and 2,143,533 transactions** against the shared rules. It compared
pseudo-legal move sets, resulting FENs, hashes, king locations, check detection,
and rollback. Later checks also compare incrementally updated neural features
against freshly reconstructed features.

### A small network, not a giant runtime

The model has 768 piece/square inputs and 32 clipped-ReLU units, evaluated from
both colors' perspectives. Subtracting the two outputs enforces color symmetry.
Its **49,280-byte** quantized weights are embedded in the binary. Inference uses
integer arithmetic and incremental feature updates; no GPU, Python, external
model download, or machine-learning library is needed to play.

The trainer is also Rust: sparse forward propagation, analytic backpropagation,
Adam updates, checkpoint selection, and quantization. Tests compare gradients
with finite differences and check color symmetry.

Training uses final game outcomes, softened for positions far from the ending.
It does not use saved search evaluations or engine identities as labels. Of
880 eligible distinct finished games, 709 are assigned to training and 171 to
validation by a deterministic whole-game signature. Duplicate games are removed;
1,937 exact positions shared with training are removed from validation. This
leaves 43,960 training and 9,923 validation positions. Position identity for this
split includes board, side, castling, and en-passant fields; related positions
can still share patterns, so this is not a fully independent tournament test.

The selected checkpoint is epoch 2. Validation cross-entropy is **0.513698**,
against **0.521660** for the fixed material baseline. Later epochs overfit and
are rejected. The [machine-readable report](cataclysm-training.json) preserves
every epoch's measurements. These prediction metrics do not establish strength.

Live games showed that the network should not dominate evaluation. The retained
version blends half its learned residual with incremental piece placement,
bishop-pair value, and explicit passed-pawn urgency. Several parameters changed
together during development, so no isolated causal claim is justified.

### Search and a separate siege solver

Iterative deepening uses principal-variation search, aspiration windows, a
four-way full-key 32 MiB transposition table, history/countermove ordering,
guarded null moves, and selective reductions. Ordinary pushes are protected from
late quiet-move pruning. Tactical leaves search quiet check evasions and include
checking moves in their first two layers. Capture-only and check-inclusive
tactical cache entries have different depth classes.

Before ordinary search, a separate bounded AND/OR solver looks for forcing
checking sequences: the attacker may choose a checking move, but **every legal
defender reply** must be refuted. An incomplete or interrupted proof makes no
claim. It receives at most roughly one-twelfth of the time/node allowance and
searches at most 11 plies. A successful proof reports its length separately;
`depth_reached = 0` in that case is not an incomplete principal-variation search.

Search respects time, node, and depth budgets, including the mate prover's work.
Tests cover terminal mate, quiet evasions, stalemate at a static cutoff, and
checkmate taking precedence over the fifty-move counter.

## Runner correction

The audit found **10,047 friendly pushes mislabeled as captures**. The runner
previously counted any occupied destination as a capture and reset its own draw
clock. It now distinguishes enemy captures (including en passant) from friendly
pushes and uses the authoritative board's halfmove counter. Regression tests
cover a friendly rook push reaching 100 halfmoves, a genuine capture, en passant,
and a pawn-initiated push. Existing saved rows remain unchanged.

`SHOWDOWN_JOBS` can cap concurrent games. A new `gauntlet` mode plays one entrant
against each other engine without scheduling irrelevant opponent/opponent games.
A gauntlet is **not** a full round robin, and its all-entrant percentage ranking
is not a fair comparison of equal schedules.

## Development evidence, including failures

All results below are Cataclysm's wins/draws/losses against Astra. They are small,
adaptive development tests, not independent confirmation matches.

| Match | Version | Move budget | W / D / L |
|---:|---|---:|---:|
| 102 | Initial hand-written evaluation | 100 ms | 2 / 0 / 6 |
| 103 | Network-dominant evaluation | 200 ms | 1 / 0 / 7 |
| 104 | Blended evaluation, protected pushes | 200 ms | 6 / 1 / 1 |
| 105 | Mate prover and stalemate checks added | 200 ms | 5 / 1 / 2 |

Match 102 ran eight games concurrently and includes one timeout loss for each
engine. Subsequent development matches used two concurrent games and finished
normally. Source/model revisions were frozen within each match.

## Reproduce

```sh
# Analyze the databases without modifying them. The report must not exist yet.
cargo run --release --bin study -- new-audit.md pushchess.db src/candidates/pushchess.db
cargo run --release --bin verify_history -- pushchess.db src/candidates/pushchess.db

# Regenerate the model from the pre-development snapshot (game IDs <= 932).
cargo run --release --bin train_cataclysm -- src/candidates/cataclysm/network.bin docs/cataclysm-training.json 40 932 pushchess.db src/candidates/pushchess.db

# Four games, alternating colors, against every other selectable engine.
SHOWDOWN_JOBS=6 cargo run --release --bin showdown -- gauntlet cataclysm_roster_001 250000 4 cataclysm

cargo run --release --bin play -- cataclysm white 1000
```

The historical databases are Git-ignored; reproducing training requires the
same local data. Building and playing use the embedded model and do not require
either database. Floating-point training results can vary slightly across
architectures/compiler versions; the checked-in quantized model fixes inference.

## Full-roster gauntlet

Tournament #14, `cataclysm_roster_001`, finished on September 5, 2026:
**81 wins, 3 draws, 12 losses (85.9%)** across 96 games, four against each of
the 24 other selectable engines, with 250 ms per move and six workers.
Cataclysm won 20 matchups, tied two, and lost two. All games completed normally.

| Opponent | Cataclysm W / D / L |
|---|---:|
| Astra | 3 / 0 / 1 |
| Void, Eternity (each) | 1 / 1 / 2 |
| Chronos, Leviathan (each) | 2 / 0 / 2 |
| Oblivion, Vortex, Terminus (each) | 3 / 0 / 1 |
| Zenith | 3 / 1 / 0 |
| Remaining 15 opponents (each) | 4 / 0 / 0 |

This is broad evidence of competitiveness, not universal dominance or an
equal-schedule round-robin ranking. Source and model were frozen within the
event. The executable SHA-256 was
`9a36fa8c3a7af47fd859f790e8fd5edde83f6eb5c22613b1b3bb59862e9a63ed`.

## Deployment choice and self-play

The deployed engine retains that tested search and blended evaluator. Model
SHA-256: `49edc822ebaedc91f7cf567468dfe6ef218dd93c45369fed9edbed203b765f26`;
telemetry FNV-1a ID: `89d502491ca870eb`. No self-play candidate replaced it.
An unfinished search-label bootstrap prototype was removed before release.

Release verification passed all 51 tests, the compile-fail documentation test,
formatting and strict Clippy checks, and builds of every release binary. A final
history differential scan passed **62,680 distinct FENs / 2,426,265 transactions**,
including incremental neural feature reconstruction, with no malformed king
layouts skipped.

The Rust `evolve` tool completed three generations: **384 self-play games** at
6,000 nodes per move, followed by warm-start training. Generation 1 selected
the unchanged checkpoint; generations 2 and 3 scored **24W/1D/23L (51.0%)** and
**25W/0D/23L (52.1%)** respectively in separate 48-game, color-paired gates at
20,000 nodes. Neither met the unchanged promotion rule: at least 62.5% plus an
approximate one-sided 95% normal paired lower bound above 50%. This is a
conservative empirical screen, not a formal multiple-testing guarantee.

Self-play did not show a reliable improvement here. Shallow search, correlated
positions, and noisy outcome-only targets limit the signal; this experiment
does not establish that self-play cannot work. The rejected models and games
remain in the ignored `experiments/cataclysm-evolution-001/` directory.
Gate games never enter the loop's training replay, and the loop never overwrites
the embedded model automatically.

```sh
cargo build --release --bins
target/release/evolve experiments/new-run src/candidates/cataclysm/network.bin 3 128 48

# All 25 selectable engines, four games per pairing: 1,200 games.
SHOWDOWN_JOBS=6 SHOWDOWN_OPENING_PLIES=6 SHOWDOWN_OPENING_SEED=20260905 \
  target/release/showdown tournament cataclysm_full_001 250000 4
```

The separate `cataclysm-reference` and `cataclysm-candidate` aliases load explicit
models from `CATACLYSM_REFERENCE_MODEL` and `CATACLYSM_CANDIDATE_MODEL` for
experiments only. They are excluded from the normal tournament roster. Normal
`cataclysm` always loads its embedded model, unaffected by those variables.
