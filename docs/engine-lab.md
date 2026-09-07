# Engine laboratory

## Population

All profiles have independent public names and explicit lineage metadata.
Const specialization shares mechanisms without branching on profile names
inside search. None of the new hypotheses is claimed to be stronger yet.

| Engine | Lineage | Experiment |
|---|---|---|
| cataclysm | Cataclysm | Control: prepared push transactions, NNUE and mate proof |
| abacus | Cataclysm | Neural score off; accumulator still maintained (ablation only) |
| resonance | Cataclysm | Double neural score contribution |
| perimeter | Cataclysm | Mobility, king-ring pressure and exposed material |
| convoy | Cataclysm | Pawn corridors and push support |
| kinetic | Cataclysm | Phase-aware transaction ordering |
| flashpoint | Cataclysm | Critical pushes unreduced; wider early quiescence |
| synthesis | Cataclysm | Combined evaluation/order/tactical changes |
| astra | Astra | Neural-free control |
| sentinel | Astra | Mate/draw precedence and stalemate witnesses |
| bastion | Astra | Sentinel plus obstruction-aware king pressure |
| outrider | Astra | Sentinel plus promotion races and rear push support |
| tactician | Astra | Sentinel plus wider checking search and unreduced pushes |
| bedrock | Astra | Sentinel without speculative pruning or reductions |
| meridian | Astra | Sentinel plus pressure, races and tactical changes |

All seven Astra-lineage engines load no network and maintain no accumulator.
Compare their experimental features against Sentinel to isolate effects of
its correctness guards. Abacus is deliberately not described as neural-free.

Outcome-refined NNUE artifacts enter as separately named candidates using the
unchanged Cataclysm control search and baseline. `--nnue NAME=PATH` may repeat;
the name must occur in `--engines`, or `--engines all` includes all supplied
candidates alongside the built-ins. Built-in names cannot be overridden.
Resolve and decode artifacts before opening the store. Malformed, missing,
duplicate or unused candidates are errors; there is no substitute network.

## Shared mechanisms and memory

The two search families share SearchLimits, fixed-width Buckets, and explicit
ordering tie policy. Replacement, pruning, mate normalization and evaluation
remain family-specific. Cataclysm keeps four 16-byte table entries per bucket,
Astra two; each engine has a 32 MiB table. Ordering pairs stay together because
move and score are consumed/reordered together. This is a deliberate AoS use,
not a blanket assertion that SoA is faster.
The shared selector now retains its current maximum in a register instead of
reloading it through a dependent index. Exact first/last tie behavior and all
15 fixed-node fingerprints are preserved; see the [measured change](move-ordering.md).
Astra's constant piece-square geometry is now a compile-time table, exhaustively
equivalent to the original formula. A separate experiment retaining all Astra
push plans lost and was removed; see the [whole-search comparisons](search-cost.md).

Cataclysm reuses per-ply action and quiet-history buffers and root history
capacity. Astra uses inline scored moves with safe overflow. There are no
database locks or Python calls in search. Teacher generation needs no GPU.

Each worker lazily retains one engine per entrant, warms each once, and resets
search history/table at each game boundary. A full 15-engine, 12-worker pool
uses about 5.6 GiB for transposition tables, plus bounded search buffers and
database pages. The embedded network is shared once per process. There is one
thread per selected worker; macOS places work across heterogeneous cores.
Hard affinity, unsafe SIMD, custom allocators and speculative prefetching are
not introduced without measurements.

Candidate weights likewise have one immutable shared owner per artifact.
Each root borrows a network through its board; recursive node search performs
no reference counting. A search's model cannot be swapped under its TT/history.
Candidates retain the same 32-MiB worker-owned tables as controls.

NNUE has one format contract and one deployed accumulator/scoring kernel.
The Python `pushzero.nnue.Model` interface uses that kernel for frozen batch
inference and tinygrad for differentiation. The separate learning-only Rust
evaluator and NumPy production scoring path are gone. `pushzero nnue benchmark`
compares complete inference and training workloads before backend decisions;
training and per-node inference need not select the same device.

A bounded channel (two game slots per worker) hands owned trajectories to one
writer. A failed move or worker panic stops the campaign; no replacement move
is invented. The writer drains workers on shutdown and seals the run status.

## One bumbledb schema

The dependency pins the tested commit of the adjacent bumbledb project
(`3ed3303eb226de6c98955ab75c296887eaf7704b`) for reproducible builds,
rather than following unrelated live edits in that checkout.

Every fact is a typed tuple; each relation is a set, not an ordered bag.
There are no stored FEN strings, JSON documents, serialized configurations,
move/PV arrays, or untyped diagnostic payloads. Optional observations are
absent facts, not nulls or sentinel scores. Hashes identify content; they do
not replace its decomposed facts.

| Subject | Relations |
|---|---|
| Campaign and budget | Run, NodeBudget or TimeBudget, RunEnd, RunFailure |
| Engine identity | Engine, EngineNetwork; Entrant associates a run slot with an engine |
| Pairing and game | Pair, Game, GameResult |
| Shared initial board | Setup, SetupPiece, SetupCastle, SetupEp |
| Position state | Position, CastlingRight, EnPassant, including the final ply |
| Action vocabulary | Action (origin, destination, knight route, stop, special kind), ActionPromotion |
| Played transitions | Move, PieceRemoved, PiecePlaced |
| Search evidence | Analysis, ordered PvStep facts, ProofSearch, MateProof |

An engine's binary/name identity is shared between campaigns. Candidate engine
IDs additionally bind the full SHA256 of the supplied network bytes. The
existing `Engine(binary,name)->Engine` dependency rejects reusing one name
with different weights under the same executable, atomically, including the
attempted run insert. Use a fresh generation name for changed weights. The
EngineNetwork checksum comes from the actual loaded model, not the embedded
control. This requires no schema change or migration. Setups and
actions are shared dictionaries. Pair stores entrant slots; Game stores the
pair and color orientation, not duplicated player descriptions. Position
stores side, clocks and a check hash. Board occupancy is the initial piece
relation plus exact square-level transition differences:

`Pieces(p+1) = (Pieces(p) − Removed(p)) ∪ Placed(p)`

This works for multi-piece pushes, off-board losses, captures, promotions,
castling and en passant. Unchanged pieces create no transition tuples.
The game key is `(run,index)`, position/move/analysis key `(run,game,ply)`,
piece-delta key `(run,game,ply,square)` and PV key `(run,game,ply,step)`.
Ordering is explicit in ply/step, never implied by retrieval order.

The schema declares functional dependencies (`->`), containment (`<=`),
projected-set equality (`==`), closed domains and cardinality ceilings.
For example, terminal games and result-bearing games are exactly the same
set; node-budget and time-budget runs each have exactly their required
budget fact; only promotion actions have promotion facts, and kings/pawns
are excluded from that target domain. Orphan analysis and PV facts fail.

bumbledb does not support dependent lower bounds: exact contiguous counts,
valid coordinate ranges, legality and final results are verified at the
application's atomic replay boundary. Readers also check action identities,
piece differences, castling/en-passant facts, position hashes and the complete
trajectory digest against rules replay. `lab verify --db DIRECTORY --run N`
audits every saved game in a run with bounded per-game snapshots, including
quarantined data. Schema constraints alone are not a proof of chess legality.
There are no migrations or compatibility schemas.

Each game and all its facts commit together with bumbledb's durable LMDB
defaults. Work/memory budgets are renewed per operation. The writer owns no
snapshot across a write. Readers use bounded pages; they never materialize
the whole corpus merely to read it. A schema/storage refusal is a failure,
not an invitation to use another backend.

bumbledb has exclusive process ownership. The tournament exposes a private
Unix socket for read-only status/report queries through its own database
handle. Its short `/tmp/push-chess-lab-<SHA256>.sock` name derives from the
canonical database path, supports long macOS paths and uses mode 0600.
When stopped, the same store opens directly for reporting/training.
A crash can leave an unfinished run and a stale socket; these are not silently
repaired or admitted to training. Preserve evidence and inspect explicitly.
JSON is used only for disposable command/status presentation, never as a
database field or a game transfer format. Native Python pages contain typed
metadata and owned NumPy action/analysis arrays; presentation FENs are derived
from relational facts when crossing that boundary.

Pages also expose search cost columns, CSR-encoded PV actions, and sparse
`(ply,value)` proof observations. `CorpusReader.state(run,game,ply)` explicitly
reconstructs a saved position with its exact preceding history for forensic
analysis. It can inspect quarantined data; it is deliberately not a training
eligibility shortcut. The split-filtered page API remains the training source.

## Data and experimental hygiene

The first four full action IDs define an opening family, with stable 80/10/10
train/validation/test hashing. Color-swapped games use the identical opening.
Corpus campaigns sample all families; arena campaigns sample only test
families and label them evaluation. Train and validation cannot overlap arena
families under this rule. Similar-but-nonidentical openings still need a
downstream leakage audit; hashing does not prove strategic independence.

Matchups interleave round-robin, with distinct reproducible opening seeds.
Fixed nodes compare search behavior per expanded node; fixed wall time also
charges evaluation/move generation. Wall overruns are recorded, not forfeits.
OS contention and heterogeneous cores mean fixed time is not deterministic.

Move caps and interruption have unknown values, never fabricated draw labels.
Random openings have no Analysis fact. Unfinished searches retain their raw
observations but are excluded from teacher-policy training. PVs are exactly
as reported, not independent proofs or multi-PV distributions. Winning a
game does not certify every move as optimal.

Failed or unsealed runs are quarantined from training. Safely interrupted
runs retain their completed, verified games; individually interrupted games
are excluded. The Python reader deduplicates full trajectories within its
bounded scan, reservoir-samples completed search examples, learns a one-hot
teacher move and uses only terminal WDL for the value target. Teacher scores
are not assumed calibrated across evaluators.

Reports contain wins/draws/losses and score bounds accounting for unknown or
missing games, plus paired opening-family fixed-sample Hoeffding intervals.
The intervals assume independent families and are not sequential stopping rules.
Promotion requires paired opening-family analysis, multiple-testing care,
a separate held-out replication, and deeper-budget confirmation.

## Launch protocol

1. Build and run correctness tests, including corpus rejection/round-trip tests.
2. Commit and push the exact code.
3. Run a bounded all-roster smoke tournament on evaluation families.
4. Verify live status, full replay, search completion and database growth.
5. Start the larger corpus campaign with explicit CPU/time/disk/game limits.

A practical first campaign: all 15 engines, 100 pairs per matchup (21,000
scheduled games), 12 workers, 100 ms/move, six opening plies, 512-ply cap,
eight-hour cap and 50-GiB store cap. Actual throughput/quality must be measured;
the number scheduled is not a promise that all games finish within that cap.
No automatic engine promotion or neural training occurs.
