# Engine laboratory

## Population

All profiles have independent public names and explicit lineage metadata.
Const specialization shares mechanisms without branching on profile names
inside search. None of the new hypotheses is claimed to be stronger yet.

| Engine | Lineage | Experiment |
|---|---|---|
| cataclysm | Cataclysm | Control: prepared push transactions, NNUE and mate proof |
| abacus | Cataclysm | Neural score off; accumulator still maintained (ablation only) |
| granite | Cataclysm | Abacus-equivalent search with no neural weights, updates or undo bytes |
| resonance | Cataclysm | Double neural score contribution |
| perimeter | Cataclysm | Mobility, king-ring pressure and exposed material |
| convoy | Cataclysm | Pawn corridors and push support |
| kinetic | Cataclysm | Phase-aware transaction ordering |
| flashpoint | Cataclysm | Critical pushes unreduced; wider early quiescence |
| synthesis | Cataclysm | Combined evaluation/order/tactical changes |
| waypoint | Cataclysm | Actual parent-action context in tactical/proof search; absent context after NULL |
| astra | Astra | Neural-free control |
| sentinel | Astra | Mate/draw precedence and stalemate witnesses |
| bastion | Astra | Sentinel plus obstruction-aware king pressure |
| outrider | Astra | Sentinel plus promotion races and rear push support |
| tactician | Astra | Sentinel plus wider checking search and unreduced pushes |
| bedrock | Astra | Sentinel without speculative pruning or reductions |
| meridian | Astra | Sentinel plus pressure, races and tactical changes |

All seven Astra-lineage engines load no network and maintain no accumulator.
Compare their experimental features against Sentinel to isolate effects of
its correctness guards. Granite is also neural-free; Abacus is deliberately
not described that way and remains the unchanged mechanism control.

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
search history/table at each game boundary. A full 17-engine, 12-worker pool
would use 6.375 GiB for transposition tables, plus bounded search buffers and
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
snapshot across a write. Readers return bounded game pages. The two prepared
piece-delta queries survive pages and fresh snapshots: their first execution
can build a relation-sized selection index, retained within bumbledb's resource
limits. Returned trajectories remain page-bounded; index memory is not constant
in corpus size. Relation epochs invalidate the indexes after writes, and closing
the reader drops query state before closing the store. This is not a physical
prefix seek or a second dataset. See [reader measurements](corpus-reader.md).
A schema/storage refusal is a failure, not an invitation to use another backend.

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

A practical first campaign used the original 15 engines, 100 pairs per matchup
(21,000 scheduled games), 12 workers, 100 ms/move, six opening plies, 512-ply cap,
eight-hour cap and 50-GiB store cap. Actual throughput/quality must be measured;
the number scheduled is not a promise that all games finish within that cap.
No automatic engine promotion or neural training occurs.

The current registry includes Waypoint and Granite; `--engines all` now means
17 built-ins and 27,200 games at 100 pairs, before any supplied NNUE candidates.
Run 4 retains its fixed five-entrant roster. The user's 2026-09-07 instruction
prohibits another tournament, including a smoke or arena round, after Run 4.
The launch protocol above describes the interface, not current authorization.

## Waypoint: exact search-edge context

This is an isolated search hypothesis, not a new evaluator or proven upgrade.
Waypoint keeps Cataclysm's network, evaluation, pruning thresholds and table
layout. Its counter-move key is still the compact `(side, mover piece type,
destination)` tuple, not a claim to encode the complete action losslessly.
The change is where that key comes from:

- Main search already records the real parent action before descending.
- Quiescence and mate-proof search now do so too, reading the mover **before**
  applying the move (including promotion). An earlier sibling's context cannot
  stand in for that edge.
- A NULL edge has no actual move: clear its context during that search and
  restore it even when the node budget interrupts the child.
- Root and absent-parent contexts do not read or reward a counter-move entry.

One per-node lookup supplies move ordering. Existing controls retain their
within-search policy through const specialization, while all profiles keep the
cross-game context-reset fix. The 15 existing 2,048-node fingerprints remain
unchanged; Waypoint has its own new fingerprint. Tests cover absent-context
rejection, the 70,000 ordering bonus, capture/promotion and proof edges, and
restoration after interrupted NULL search.

Verification on 2026-09-06: 104 Rust tests and 47 Python tests pass, as do both
Clippy gates, formatting and the release all-profile fingerprint gate. The
12-root, 8,192-node whole-search harness also passes Waypoint's three-pass
repeatability/restoration check (signature `6867009960455420266`). Its contended
timing is not a claim of greater speed or playing strength.

After the current store owner exits and its corpus is audited, compare
`--engines cataclysm,waypoint --purpose arena` on fresh color-swapped families,
then confirm at a deeper wall-time budget. Keep NNUE refinement as a separate
candidate so its effect is not confounded with this ordering change.
This earlier comparison plan is now paused: inspect the already-running Run 4,
but do not launch another arena under the current authorization.

## Granite: remove unused neural state

Granite shares Abacus's evaluation, pruning, ordering, mate proof and table
policy. It changes representation only. `Board<R>` has a compile-time residual
policy: neural profiles retain the existing network/accumulator kernel, whereas
Granite uses a zero-sized handwritten policy. Its undo payload is `()`, not an
accumulator or a copied model reference. No per-node enum or virtual dispatch
selects between these policies. Granite owns no model; supplying one violates
the profile invariant rather than silently selecting another evaluator.

On this Apple M2 Max target, board size falls from 600 to 336 bytes; move undo
size falls from 560 to 304 bytes. Transposition tables and the main position
representation are unchanged. The learning feature loader also uses the same
handwritten board baseline, avoiding neural accumulation that it never consumed.
It still emits exactly the existing feature IDs and deployed baseline scores.

Abacus retains its accumulator and network identity. Granite reports neither;
Rust and Python search interfaces and bumbledb engine/network relations test
this distinction. No database schema changes or alternate game stores are needed.

Verification on 2026-09-07: 114 Rust tests and 50 Python tests pass, both Clippy
gates and formatting are clean, and all previous fixed-node fingerprints are
unchanged. Granite matches Abacus's complete non-clock signature at 1, 257 and
8,192-node limits, including repeated searches with retained table/history,
special-move fixtures and reachable roots. Its 2,048-node all-profile fingerprint
is also identical to Abacus (`18076246204620320795`). Handwritten and neural boards
both pass reference-rule transition checks.

The first bracketed whole-search comparison is too noisy to establish a speed
gain; see [the measurements](search-cost.md#granite-neural-free-representation).
Keep Granite as an isolated experimental candidate and repeat idle measurements
after Run 4. Do not promote it, change controls, or launch games to test it under
the current no-more-tournaments instruction.

## Audit search cost by entrant

`lab verify --db DIRECTORY --run N` derives `search_cost` and
`search_by_engine` during its existing full replay audit. There is one fixed-size
accumulator per entrant, no retained position sample, second game scan, alternate
analysis store or extra database facts. White/black ownership follows each
game's actual entrants, including color-swapped games; an unplayed entrant
reports zero searches and absent means, not a zero-time performance observation.

Counts include completed and incomplete searches, excluding opening moves.
The report gives total external wall time, engine-reported time, maximum wall
time, overruns against the configured time budget, mean/max depth, quiescence
nodes, table hits and optional proof-search observations. An absent proof
search differs from a reported zero-node proof search. Mate-proof reports are
not independently verified certificates.

Node rate is `sum(nodes) / sum(search_wall_seconds)`, not an average of individual
rates or games per campaign second. Summed search time across concurrent workers
is not elapsed campaign time. These descriptive rates reflect actual positions,
engine node-count conventions and machine contention; they do not establish
equal-work speedups or isolate inference time. Use them to select the next
controlled profile/benchmark, alongside paired playing results.
