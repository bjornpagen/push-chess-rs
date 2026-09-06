# PushZero redesign: exact effects, little waste, strong self-play

**Proposal dated 2026-09-06; implementation approved.** See
[implementation status](implementation-status.md) for what is implemented,
verified, and still measurement-gated. The document below preserves the design
agenda; it is not a claim that every research alternative has been adopted.

This document is the full proposed scope, implementation order, experiment
agenda, correctness contract, and stopping criteria. Research/source ingestion
and static inspection were completed before implementation. The user has now
released CPU capacity for validation and requested commit/push to `main` before
any training run, followed by a small test run before a heavy run. Automatic
deployment and automatic restarts remain outside this work.

## 1. The recommendation

Build a small, purpose-built self-play system around one strong representation:
**an exact, reusable description of what a legal move does**.

Use it twice: the Rust engine applies it without resolving the same push again;
the learned policy can reason about the pieces that actually move, not only
the move's source and destination. Keep exact rules, a small learned board
network, Gumbel policy improvement, and outcome-based self-play. Organize the
runtime around worker-owned memory and persistent inference buffers. Optimize
useful decisions and learning per second, not abstraction count, parameter
count, or the appearance of full CPU utilization.

The recommended first new model candidate is a **64-channel, four-block residual
trunk with inexpensive global context and an exact-move-effect policy head**.
The current 96-channel, six-block network remains the reference. The new model
is a deliberate hypothesis, not something the papers have already validated.
We will test its two additions separately before keeping the combination.

**Execution priority is inference-first**, following the user's clarification.
First improve the existing model's inference path without changing its learned
function, then compare cheaper models and avoided evaluations. The deeper
engine work follows where it enables better inputs or is supported by new
profiles. Section order below describes the architecture, not work priority;
the execution sequence is in section 14.

This is ambitious about changing representations and conservative about
unmeasured complexity. Papers inform the design; they do not limit it. A
novelty must beat the simpler alternative under our actual compute budget.

## 2. What the evidence currently says

The working tree is based on commit
`9b5032ba032b667402a145d211a70bb7436a100f`, with pre-existing uncommitted learner
work. Those edits and the paused run must be preserved.

| Existing observation | What it means—and does not mean |
|---|---|
| Current network: 1,065,796 parameters, 96 channels × 6 blocks; tinygrad 0.14.0, Metal | A functioning starting point, not a proven strength/compute optimum |
| Doctor probe: 32 roots × 16 simulations, 236.4 ms total, 6.1 ms in its Rust-search timer, 35 hot-loop FFI calls | In this workload the measured Rust region was about 2.6%; raw FFI call count is already batched |
| Paused run: 547.63 s total; 493.37 s inference-path timer; 9.87 s Rust-search timer | About 90.1% in the inference path and 1.8% in the measured native search region; inference time includes host staging/dispatch/synchronization, not only GPU arithmetic |
| 818,261 inferred positions but only 7,212 retained replay positions | Search queries and training examples are different quantities; both throughput and learning yield matter |
| Early Cataclysm probe lost all four games; four-game probe against initial weights scored 1W/3D/0L | Too little evidence to claim strength; Cataclysm comparison also used unequal time budgets |
| Early sparse starts produced many draws | Investigate the curriculum; this does not establish that sparse training is always bad |

Sources: `experiments/pushzero-doctor-metal.json`,
`experiments/pushzero-metal-001/metrics.jsonl`, and the evaluation/paused records.
These are existing measurements, not fresh isolated benchmarks. The timers do
not comprehensively attribute every Rust/Python operation. Root preparation,
packing, bookkeeping and waiting need finer attribution.

Amdahl's law matters: making only the 1.8%-timed region infinitely fast would
improve that recorded total by only about 1.8%. That is not a ceiling on the
whole redesign: avoiding evaluations, shrinking the model, reusing buffers and
changing the pipeline affect the other regions. As inference improves, the
native engine's share can become much larger.

## 3. Principles and explicit non-goals

1. Preserve useful work as data: resolve once, carry the result, apply it many
   times. Parse/check at boundaries; keep trusted internal states well-formed.
2. A compact flat loop over the right representation is the default. No virtual
   dispatch, trait-object graph, or per-node heap object framework in hot paths.
3. Data layout follows access: AoS, SoA and hot/cold splits are tools, not teams.
4. Own mutable state locally. Cross a language or thread boundary for a chunk
   of useful work, not for an edge, piece or simulation callback.
5. Reuse capacity; make growth, memory budgets and overflow visible. Never
   silently truncate the legal action set.
6. Measure emitted code and complete workloads. All-core capability is useful;
   all-core saturation is not itself the objective.
7. Keep rules and learned strategy separate. No Cataclysm evaluation labels,
   inherited piece values, hand-written tactical rewards or draw contempt.
8. Preserve a small correct reference and rollback path until its replacement
   has earned adoption. No simultaneous rewrite of every historical engine.

Not in the first implementation: a custom allocator framework, mandatory
manual SIMD, unconditional prefetch, packed/unaligned structs, a fully shared
lock-free search DAG, a learned rules simulator, or a large transformer.
No claim of a guaranteed 100% win rate: draws, stronger opponents and adversarial
weaknesses make that a different—and generally unattainable—target.

## 4. Target ownership and module boundaries

The intended runtime sequence is:

1. A Rust worker owns several game lanes, their root/search cursors, history,
   arenas, scratch, and RNG streams.
2. It traverses exact rules until it has unresolved neural leaves. It writes
   their board/action/effect inputs into a leased batch buffer.
3. A single inference coordinator groups compatible ready rows, invokes the
   tinygrad Metal predictor, and returns a result lease tagged with model version.
4. Workers consume their rows, back up values, continue search, choose moves,
   and append compact trajectory records. Completed games supply exact W/D/L.
5. Python controls experiments, replay selection, optimization and checkpoints;
   it is not the inner tree-search scheduler.

Prefer modules inside the existing Rust library before new crates. Expected
boundaries, with names finalized during implementation:

| Area | Owns | Must not own |
|---|---|---|
| `core` position/transition/history | Exact board state, prepared transitions, make/unmake, adjudication | Neural weights, strategic material scores, Python objects |
| `selfplay` arena/search/cache/runtime | Tree statistics, scheduling, capacities, rule-safe state identities | A second implementation of push rules |
| `selfplay` encoding | Versioned board/action/effect representation, direct buffer filling | Per-element Python boxing |
| `training/native` | Checked Python boundary and explicit buffer leases | Search algorithms or rule duplication |
| `training/pushzero` | tinygrad models, training objectives, experiment/run control | Per-edge native callbacks |
| Production/search evaluators | Learned or legacy strategy over the shared exact rules kernel | Competing rule semantics |

UI IDs, FEN, saved-game versions, native CLI and WASM interfaces stay compatible
through adapters. Do not introduce a generic game-engine framework to achieve
these boundaries.

## 5. Rules engine: compiled legal transitions

### 5.1 Separate board state from its uses

Today `Position` combines a board with a growing `Vec<UndoInfo>`. Introduce a
compact position value, a game-history owner, and a reusable search cursor.
The cursor overlays a reversible search path on the immutable game prefix.
The current and previous board needed for encoding must be accessible without
cloning the complete undo history at each root or leaf.

Candidate board layout: one compact piece code per square, side/castling/EP/
clocks, king squares and hash; add occupancy/piece bitboards only where the
avoided scans repay their update and copy cost. Keep a simple mailbox reference.
No claim that a 64-square board copy is inherently slow: compare copying with
changed-square deltas using real push-length distributions.

Repetition gets its own representation. A compact key history and a cursor
overlay may remove repeated scans of large undo objects. First preserve the
exact existing repetition semantics; compare linear key scans with a maintained
count structure. Do not import orthodox-chess irreversible-move pruning without
proving it valid for these push rules. A hash is an acceleration mechanism, not
a proof that complete histories are interchangeable.

### 5.2 Generate once, retain the result

Current waste is concrete:

- `core/movegen.rs` resolves pushes, then emits only `Move`.
- `core/children.rs` allocates a temporary pseudo-legal list.
- Legal checking makes each move and resolves its push again.
- Search traversal later resolves the selected move yet again.

Replace this with a rules-only **PreparedTransition**, produced once per move
candidate, used for legality checking, and retained for subsequent application.
It holds stable move identity and a compact exact effect: moved/captured pieces,
promotion, state-field changes, and phase information where needed. The format
must represent all six move-ID fields and both knight routes.

Cataclysm's `Action { mv, plan, ... }` already demonstrates reusable plans.
Extract the **concept**, not its entire `Board`: that board also owns handcrafted
evaluation and an incumbent neural accumulator. The shared kernel must not
accidentally import either into pure self-play.

Prepared moves are origin-bound. External callers choose a move at a checked
revision; internally, a node owns its move batch and the cursor must be at that
node's origin. Use private construction, typed handles, arena generations and
debug exact-state assertions. Do not expose an arbitrary stale transition that
can be applied to any position just because a 64-bit hash happens to match.

### 5.3 Make the effect representation do the work

- Resolve straight push rays once where common prefixes can genuinely be
  shared across candidate endpoints; preserve stop/promotion distinctions.
- Compose knight legs using a lightweight board view and bounded scratch,
  without constructing a semantically unrelated full `Position` object.
- Preserve simultaneous-source movement. Applying a sequential series of
  overwrites can silently duplicate or erase pushed pieces.
- Produce a unique changed-square set during compilation. A 64-bit touched
  mask is a candidate to replace linear duplicate detection, subject to cost.
- Store small effects inline in generation scratch; retain larger persisted
  effects in a contiguous cold arena. Benchmark full inline plans versus
  offsets: avoiding recomputation must not explode edge cache footprint.
- Use fixed castling paths and geometry tables for pure rule constants; no
  tiny heap vectors for two or three known squares.
- Derive authoritative limits from rules or retain checked growth. The observed
  maximum legal count is not a valid hard cap of 128.

### 5.4 Migration and correctness

Migrate neural search first through a compatibility facade. Differentially
compare sorted complete move IDs, resulting positions, check status, clocks,
hashes and outcomes against the reference on curated cases and reachable
random trajectories. Compare every make/unmake round trip. Then migrate
production Cataclysm and UI consumers without changing their evaluation.

Keep the incumbent/reference behavior available until both rule parity and
equal-budget strength/nonregression gates pass. Historical engines remain
available; do not delete them as a cleanliness exercise.

## 6. Search memory: hot/cold first, SoA where it earns it

The current arena still contains a `Vec<Edge>` per expanded node, and selection
creates temporary transformed-Q/policy vectors. Replace these with:

- A contiguous node arena holding node tags, edge ranges, raw value and compact
  indices. Terminal, unevaluated, pending and expanded states are explicit.
- An edge slab with contiguous ranges per node. Selection statistics are hot;
  move identity, child links and prepared effects are separate when their access
  pattern justifies it. Children remain compact integer handles, not pointers.
- Per-worker reusable path/selection/generation scratch. Reserve from configured
  budgets and grow only at explicit safe points, with high-water telemetry.
- Bounded arena reclamation on root advance/game completion. Stale handles are
  rejected via generation/epoch; indices must not wrap silently.

| Data | First candidate | Alternative to measure |
|---|---|---|
| Board squares | Compact AoS value array | Mailbox plus incrementally maintained bitboards |
| Selection fields (`visits`, `sum`, `prior`, `logit`) | Hot contiguous storage; compare a small stats AoS and SoA | AoSoA only after a useful SIMD reduction is demonstrated |
| Move IDs / child handles | Compact parallel cold/warm arrays | Co-locate with stats if traversal dominates and separation loses locality |
| Prepared effects | Variable-length records in a contiguous cold slab | Tiny inline representation if lower indirection wins |
| Pending inference rows | Dense board inputs, ragged logical action/effect metadata, explicit offsets | A few padded GPU shape buckets at the boundary |
| Worker state | One private owner; independently owned queues/counters | Padding only for demonstrated false sharing |

Fuse repeated scalar reductions where semantics permit. Compute total visits,
visited prior mass and value numerator without heap temporaries; calculate
completed-Q normalization and stable softmax using reusable scratch or passes.
Do not maintain dozens of fragile incremental aggregates before measuring
whether simple linear scans over short action lists are already cheap.

Preserve NaN rejection, zero-visit behavior, value signs, stable tie breaking,
legal masking and Gumbel schedules. Reassociation/SIMD can change close floating
ties: require an explicit numerical tolerance and decision/strength check; keep
a deterministic reference mode. Do not enable fast-math globally.

Acceptance target: **zero heap allocation on warmed selection/traversal/backup
within reserved capacity**, no redundant push resolution for a retained selected
edge, and no per-node owning heap allocation for edge lists. This is an allocation
contract, not a promised throughput factor.

## 7. Rust/Python/Metal: fewer boundaries and fewer bytes

This is the first optimization workstream, ahead of a broad engine rewrite.
The initial target is the measured inference path, not an assumed GPU arithmetic
bottleneck. Break it into staging, transfer, dispatch, device execution and
synchronization before deciding which mechanism to optimize.

### 7.1 Preserve what already works

PyO3 transfers Rust vector ownership to NumPy without per-element boxing.
The native request path releases the GIL; incoming NumPy arrays are borrowed
carefully. There is no evidence that replacing PyO3 with a handwritten C ABI
would fix the current bottleneck. Keep the thin adapter unless measurements
show a real problem.

### 7.2 A persistent batch state machine

Replace separately rebuilding requests and responses with a coarse operation
conceptually like `advance(previous_reply) -> NeedEvaluation(batch_lease) |
Completed(records) | Stopped(state)`. The first call has no reply. Rust advances
ready work until a neural boundary or explicit stop condition.

Typed request IDs, row-to-lane mappings, model revisions, lengths and epochs
must detect duplicate, stale, missing and mismatched replies. One boundary per
evaluation round is the target; round count still depends on the search.
Construct roots and produce trajectory results in bulk too: the current hot-loop
FFI count does not include every outer per-game call.

Encode directly into final leased staging slices. Eliminate temporary `Encoded`
collections followed by a second packing allocation. Reuse row maps and legal
lengths; initialize padding only where required. Never allow a padded action
to enter selection or a policy-training softmax.

### 7.3 Buffer lifetime, not wishful zero-copy

Baseline: a small pool of owned input/output arrays with explicit leases.
Python cannot retain a writable alias to a buffer that Rust will refill, and
Rust cannot refill while Metal reads it. Holding the GIL alone is not a complete
cross-thread/GPU ownership protocol. If safe reuse is complicated, retain one
bounded bulk copy as the correctness baseline.

Define the cycle: **free → CPU filling → GPU in flight → result ready → consumed
→ free**, with cancellation/failure draining or retaining the in-flight owner.
Use immutable inference model versions. No optimizer mutation of weights used
by an outstanding inference graph; first alternate inference and training,
then consider double-buffered models only if overlap pays.

Apple unified memory does not automatically make NumPy→tinygrad zero-copy.
In tinygrad 0.14.0's Metal allocator, `external_ptr` is interpreted as an
**MTLBuffer object handle**, not an ordinary NumPy data pointer. Therefore
`Tensor.from_blob(array.ctypes.data, device="METAL")` is not a valid shortcut.

Later candidate: persistent Metal-owned shared buffers, Rust writes their mapped
CPU memory under a lease, and tinygrad wraps the correct Metal buffer handles.
This requires a tiny platform-specific adapter with explicit retention, offset,
alignment and GPU-completion semantics. Isolate it behind a tested boundary;
do not spread unsafe pointers throughout search. Adopt only if copies or
synchronization are measured to justify the complexity.

### 7.4 Fix the tail and audit the actual graph

The predictor currently rebuilds fixed-size batch-32 arrays and pads small tails.
Add a bounded shape family, initially batch sizes 1/2/4/8/16/32 plus action-width
buckets, subject to profiling. Do not compile every possible shape. Keep enough
game lanes live to refill completed games, but drain promptly at a stop/deadline.

Inspect the warm tinygrad graph for normalization/reduction launches, repeated
gathers/embeddings, weight transforms, synchronizing `.numpy()` calls, copies
and concatenations. Distinguish cold compilation from warm performance.
Invalidate all relevant compiled graphs/caches on model revision: transformed
weights have already caused one real stale-inference bug here.

Start with built-in scheduling/fusion and persistent buffers. Test mixed
precision with FP32 losses/reductions and a parity/strength gate. Write a custom
Metal kernel only for a proven dominant small operation whose extra maintenance
cost is warranted. Keep a pinned known-working tinygrad version; upgrade to a
newer version in a separate compatibility experiment, not in the middle of an
engine comparison.

## 8. Thread-per-core-capable runtime

Use a **persistent Rust worker pool**, not one Python process and neural model
per core. Each worker owns several whole game/search lanes and private arena,
scratch and RNG state. A coarse bounded channel carries ready inference work;
one coordinator owns Metal submission. Avoid nested parallel libraries.

Expose explicit worker counts and an all-core mode. This host reports 12 cores;
we can size a pool across that capacity, but macOS scheduling/QoS does not promise
hard one-worker-to-one-physical-core placement. Account for the coordinator,
training and P/E asymmetry; do not blindly run 12 CPU-heavy workers plus several
other pools.

Start with independent roots and one pending descent per tree. Different games
can proceed without changing each other's sequential Gumbel evidence. Dispatch
batches by fullness **or a latency deadline**, so an E-core worker or nearly
finished game cannot hold everyone at a global barrier. Stable lane IDs and
per-lane RNG streams keep ordering changes from changing exploration noise.

Idle workers park. Pause/cancel is cooperative at bounded work boundaries and
does not spin or silently restart. On failure, stop new work, preserve committed
games/checkpoints, and release buffers only after their consumers are done.

Future work stealing is whole lanes/coarse tasks, not individual edges or nodes.
Start with simple mutex/channel synchronization: message rate and contention
must justify a more elaborate queue. Avoid a shared mutable tree with an atomic
increment at every node. Per-worker caches are the first cache topology; shard
or share only when duplicate neural work exceeds lookup/coordination cost.

Implement/expand the pool when profiles show that native preparation limits
inference supply or a cheaper model exposes CPU work. Do not build a complex
pool just to keep cores busy while a single coordinator is inference-bound.
Test 1/2/4/8/all-core settings after CPU permission. Choose the default from
end-to-end throughput, tail latency, memory, energy and responsiveness. If four
workers keep Metal busy, eight busy workers are not automatically an improvement.
The all-core path should nevertheless exist for CPU-heavy search configurations
and smaller/faster evaluators where it pays.

## 9. New architecture: show the model what a push actually does

### 9.1 Why change the policy representation

The present head gathers source/destination square features and adds route,
stop, promotion, special-move and global embeddings. A push may move several
other pieces. A small network must infer all collateral changes implicitly.
The rules engine already computes them exactly while finding legal moves.

Our central new hypothesis: **reuse those exact effects as policy input**, so
the network learns whether an effect is good rather than spending as much
capacity rediscovering what happened. This is rules-derived representation,
not teacher advice. No effect receives a handcrafted strategic score.

### 9.2 The first candidate, concretely

1. Encode the current board, previous board, rights, clocks, side and repetition
   information with a versioned input schema. Keep the existing representation
   as a parity baseline; assess its history limitations explicitly.
2. Run a compact 64-channel four-block residual trunk over 8×8 squares.
3. In two blocks, pool a small channel subset and broadcast a learned global
   bias. On a fixed board, a duplicate board-size-scaled mean is unnecessary.
   Compare simple mean/max pooling before attention-based alternatives.
4. For every legal action, retain its six-field identity and produce exact
   transition-effect tokens: before/after piece and square information or
   origin/destination displacement information, with captures/promotions and
   knight-route/phase metadata preserved as necessary.
5. A small learned effect encoder combines projected board features at affected
   squares with these tokens. Aggregate per action and combine with mover,
   destination and global features to produce one legal-action logit.
6. A W/D/L head predicts outcome from the board trunk; expected value is P(win)
   minus P(loss), in a consistently defined side-to-move perspective.

Keep effect tokens logically ragged with offsets; compare their dense/bucketed
GPU representation. Do not create an enormous `[batch, actions, squares,
channels]` tensor by default. Project square features to a small effect width
once before gathering, and measure whether pooling/gather kernels erase the
benefit. Exact effects are available from the rules work, but transporting and
processing them is **not free**.

Start with fixed small pooling of effect embeddings. An action-conditional
attention head is a later alternative if simple pooling loses critical
piece-to-square relationships. Preserve joint token fields: independently
pooling piece types and destinations can erase which piece moved where.

### 9.3 Bolder follow-on ideas, with adoption gates

- **Lightweight relation mixing:** one small 64-square attention/mixing block,
  or learned row/column communication, to capture long-range interactions.
  Compare to global pooling under measured Metal time. This is not a mandate
  for an autoregressive language-model architecture.
- **Counterfactual effect learning:** auxiliary prediction of masked transition
  effects or future board changes from self-play. Labels come from exact rules
  or future trajectories, never an incumbent evaluator. Avoid a trivial
  copy-the-input objective and measure whether gradients help policy/value.
- **Action-conditioned value/uncertainty:** a cheap head predicts which actions
  warrant expensive continuation. Train on completed outcomes and explicitly
  identified search-derived auxiliary targets. Calibrate on held-out searches;
  do not equate softmax confidence with actual uncertainty.
- **Adaptive search budget:** spend more on uncertain/high-consequence decisions
  and less on clear ones, with a minimum exploration floor. Validate at equal
  total inference time and preserve unbiased definitions of training targets.
- **CPU-native learned evaluator:** an incremental learned feature accumulator
  paired with exact alpha-beta/search may dominate very low-latency deployment.
  Train it on our own outcomes/search, not Cataclysm labels. This is a separate
  system contender, not a concealed replacement of the pure-learning boundary.
- **All-GPU environment/search experiment:** only if many thousands of tiny
  games make native handoffs dominant. Exact push legality is branchy and must
  be differentially verified; shared memory alone is not a reason to port it.

These are our design hypotheses. The papers do not need to have proposed the
same architecture; their evidence helps choose cheap, informative experiments.

## 10. Search reuse, caching, and controlled algorithm changes

### 10.1 Neural inference cache first

Key results by complete network input identity, encoding version, legal action
ordering/effect inputs and model revision. If caching the trunk separately,
document that its key excludes action inputs only because the trunk does not
consume them. Resolve hash collisions by verifying identity. Include a memory
cap, replacement policy, hit-rate and avoided-query counters.

Never reuse a terminal outcome solely because the network input/hash matches:
rules adjudication uses actual history. Sharing identical network predictions
is safe even when two rule histories differ; sharing their search statistics
or terminal labels is not automatically safe.

### 10.2 Tree reuse next, with Gumbel semantics made explicit

Promote the played child and retain safe reusable structure/evaluations.
First reset search statistics/new-root noise as required by the reference
algorithm. This can reuse evaluations without quietly changing visit schedules.
Inherited visits require a separate definition of additional-search budgets,
completed Q and sequential halving, plus tests and equal-time strength evidence.

Tag reusable neural state by model revision; start conservatively by discarding
stale evaluated statistics on a model change. Do not blend values from multiple
weight versions without an explicit algorithm.

### 10.3 Single-game latency is a distinct problem

Cross-game batching improves self-play, but a tournament agent has one root.
After the throughput baseline, compare small-batch inference, a lightweight
CPU evaluator, and multiple pending leaves with virtual-mean/loss accounting.
The latter is an algorithm experiment: record duplicate work, stale selections
and policy/strength changes, not just higher batch fullness.

Exact terminal solving can back up proven wins/losses/draws without inference.
More advanced solver/tablebase work is a bounded later option; it is not a
promise to solve full Push Chess or an excuse to ignore history-dependent draws.

## 11. Self-play data and learning efficiency

Preserve the main objective:

**legal-policy cross entropy against Gumbel-improved targets + W/D/L outcome
cross entropy + controlled regularization**, with optional clearly separated
auxiliary tasks. Exact game outcomes remain authoritative. A truncated game is
not a draw label. Self-play's policy target is not simply the exploratory move
that happened to be played.

First experiments:

1. Standard starts versus a modest mixture of **real history-preserving visited
   states**. Disable arbitrary sparse starts in one comparison, not by assertion
   that they can never help. Keep standard-start strength as the objective.
2. Search-cap randomization versus fixed effort at equal total cost. Audit noise
   on cheap turns separately; KataGo's cheap turns remove exploration settings,
   which is not automatically the behavior of our current Gumbel loop.
3. Tune data reuse/training steps against new independent games. Track replay
   age and outcome diversity instead of treating lower training loss as progress.
4. A small policy-reanalysis fraction versus spending that time on fresh games.
   Preserve source policy/model version and actual outcome provenance.
5. Weight averaging/EMA and auxiliary heads only after the representation and
   baseline are stable. Do not run a full population for the first tuning pass.
6. Then test regret-prioritized restarts, beginning with a cheap measured-error
   version and comparing learned RGSC heads only if the cheaper version plateaus.

Replay should store one compact game record plus ply references, not a complete
move history and expanded float board for every sample. Keep periodic snapshots
or a bounded decode cache so space savings do not require replaying a long game
for every minibatch. Legal IDs, target distributions, model version and terminal
outcome remain lossless. Re-encode dense tensors only for selected minibatches.
Version shards, checksums and encodings; keep old readers for the paused run.

Track distribution by opening/ply, board density, outcome, repeated state,
action count, policy entropy and value calibration. These are diagnostics, not
handcrafted strategic rewards. Prove every data augmentation against the rules;
Go's eight board symmetries do not transfer to directional pawns/castling.

For meaningful comparisons, predeclare small experiment arms, run equal
wall-clock budgets, use independent seeds when affordable, and eliminate clear
losers early using development evaluations. Final held-out results must not be
the same games used to choose architecture/hyperparameters.

## 12. Checkpoints, pause/resume, and reproducibility

The existing paused run must remain usable:

- checkpoint `checkpoint-000004-0f7e4af0.safetensors`;
- 177 optimizer updates, 256 recorded games, 7,212 replay positions;
- 50 pending updates recorded for completion before new self-play on an
  explicitly requested resume.

Keep atomic commit order for replay, optimizer, weights, RNG, model/config
schema and pending work. New worker RNG streams derive from explicit seed/lane
identities. Define reproducibility levels: exact single-worker reference,
schedule-stable independent-lane mode, and explicitly nondeterministic
throughput mode if later needed.

Do not silently resume old weights into a changed input/head shape. Architecture
experiments use a new run, with any warm start or transfer explicitly recorded.
No automatic deployment or champion replacement follows from successful training.

## 13. Measurement and correctness gates

### Workload matrix

Measure separately: long and short pushes, both knight routes, promotions,
castling/EP, king displacement/check evasion, high action count, repetition-rich
late games, shallow/wide and deep/narrow trees, batch-1 latency, full batches,
tail batches and longer steady-state self-play. Use reachable diverse traces
as well as targeted synthetic adversaries. Do not benchmark one memorized FEN.

### Counters and timing

- Total decision/game/iteration wall time; cold versus warm JIT.
- Move generation, resolution, legality, apply/undo, repetition, selection,
  encoding, packing, queue wait, host copies, inference and training.
- Allocations and bytes by stage; live/high-water arena memory, growth events,
  feature bytes, effect-token counts and active/padded rows/actions.
- FFI calls, batches/second, useful inferred positions, cache hits, duplicate
  pending leaves, simulations, retained examples and completed games.
- Worker utilization and queue occupancy; GPU/host timing only where accurately
  available. Do not infer device arithmetic time from a broad host timer.

Follow [the bumblebench application guide](../sources/BUMBLEBENCH.md): same-session
interleaved A/B, identity controls, isolated absolute timings, codegen inspection
and regime labels. Do allocation instrumentation separately if it distorts the
timed loop. No speed claim from source aesthetics or one noisy minimum.

### Must-pass correctness battery

Complete move IDs; simultaneous pushes; both knight routes; pushed promotions;
castling paths/rights; EP; displaced kings and king attacks; reversible state;
hash recomputation; previous-board encoding; full repetition history;
mate-before-draw precedence; no legal-action truncation; finite legal-policy
normalization; side-to-move sign parity; stable reference ties; Gumbel candidate
and halving schedules; no pending/terminal inference confusion; buffer lease
and cancellation safety; stale model/JIT/cache invalidation; replay and
checkpoint round trips; old saved games/CLI/WASM compatibility.

Unsafe code, if any, is isolated with documented invariants and targeted tests.
Use sanitizers/Miri on supported CPU-only boundaries after compute permission;
they cannot by themselves prove Metal driver lifetime correctness.

### Strength gates

Evaluate both colors over paired, held-out reachable openings, standard starts,
tactical counterexamples, draw/repetition traps, and historical opponents.
Report both equal-time deployment matches and equal-search diagnostic matches.
Use win/draw/loss and score confidence intervals, preferably paired/grouped by
opening rather than assuming every game independent. Predeclare promotion
criteria and sample size or a valid sequential-testing protocol; do not peek
at a few wins and declare a champion.

Maintain a diverse checkpoint/opponent pool to detect non-transitivity. Once a
strong baseline exists, train bounded adversaries from our own self-play and
retain their failures as regression cases. Tournament strength, adversarial
robustness and mathematical perfect play are distinct goals.

## 14. Implementation sequence and deliverables

Every phase ends in a reviewable change with a rollback path. No phase bypasses
the compute pause; code can be prepared while measurements remain deferred.

| Phase | Deliverable | Acceptance / decision |
|---|---|---|
| 0 — This task | Local 19-paper source library, research notes, static audit, this proposal | Review the entire design before implementation |
| 1 — Reference and instrumentation | Rule/search differential harness, diverse trace corpus, allocation/timing counters, compatibility tests | Known behavior and timer attribution; baseline measurements only after permission |
| 2 — Existing-model inference | Attribute the warm graph; remove redundant staging/copies/concatenations; persistent safe buffers; tail-aware shapes; consider exact-input revisioned caching | Same model outputs within declared tolerance; actual end-to-end inference improvement; no shape/JIT explosion or stale caches |
| 3 — Less inference work | Compare 64×4 and global-context trunks to 96×6; audit head gathers/normalization/fusion; bounded precision/kernel experiments where measured | Better strength per elapsed training/deployment time, not merely fewer parameters; no training until permission |
| 4 — Exact transition kernel | Reusable prepared moves and effects; state/history/cursor separation where it removes duplication; generation scratch; neural-search adapter | Exact rule parity; enables the effect-aware input without repeated resolution or excessive memory |
| 5 — Effect-aware network | Versioned effect tokens; effect-head-only and global-plus-effect arms against the selected plain trunk | Equal-time learning and deployment comparisons; reject if extra gathers/bytes erase the strength benefit |
| 6 — Native/runtime efficiency | Arena/slab storage; hot/cold AoS/SoA comparison; coarse advance API; direct encoding; persistent all-core-capable workers when CPU supply becomes limiting | Allocation contract and safe ownership; measured end-to-end gain; correct stop; no needless workers or scheduling drift |
| 7 — Better data | Compact replay; standard/visited-state starts; cap/reuse/reanalysis experiments | More useful learning per second; no history loss, target contamination or checkpoint breakage |
| 8 — Reuse and advanced search | Versioned inference cache; safe structure reuse; later inherited-visits/single-root batching arms | Avoided queries outweigh cache cost; semantics and equal-time strength validated |
| 9 — Production consolidation | Migrate additional consumers to shared rules; robust eval/champion gate; remove only superseded duplicate code | CLI/UI/WASM compatibility and strength nonregression; explicit deployment approval |

Phase 2 preserves the current network and search semantics as far as possible,
so a throughput win is interpretable. A changed floating-point kernel or precision
mode has a separate parity gate. If staging/synchronization dominates, prioritize
buffer/dispatch work; if device arithmetic dominates, prioritize the trunk/head
and useful-query count. Re-profile after each win.

Manual SIMD, prefetch, custom Metal sharing, all-core scaling and special
allocator work are conditional subprojects, not prerequisites. Do not wait for
a complete engine cleanup to improve the dominant inference path. Phases 3,
5 and 7–8 use explicit experimental configurations so a failed idea does not
leave permanent complexity in the main path.

## 15. Priority, rejection rules, and alternatives

| Candidate | Priority | What would make us reject/defer it? |
|---|---|---|
| Tail-aware inference/direct staging/graph audit | First implementation priority | Shape/JIT explosion or a lifetime protocol more costly than one bulk copy |
| Cheaper trunk/head and avoided evaluations | First model/compute priority | Lower per-query cost fails to improve strength under equal time |
| Reusable exact transitions and hot-path scratch | Next; enables effect inputs and removes proven duplication | A representation that preserves work but increases memory/copies enough to lose in situ; choose a smaller encoding |
| Exact-effect policy head | Highest original architecture hypothesis | Extra feature/gather cost exceeds equal-time strength improvement |
| Small trunk plus global context | First architecture experiment | Smaller trunk loses more playing strength than extra search can recover |
| Real-state restarts / search-cap tuning | First learning-efficiency experiments | More short games but worse standard-start learning or biased coverage |
| Per-core-capable worker pool | Conditional on CPU supply/search bottleneck | Extra workers contend with Metal, add tail barriers, or mostly wait |
| Inference cache / safe reuse | Next | Low hit rate, oversized identities or memory pressure dominate avoided queries |
| Shared Metal buffer bridge | Conditional systems optimization | Host copy time is too small, sync dominates, or backend lifetime contract is unstable |
| Learned regret/budget heads | Later | Added network work or selection bias exceeds learning benefit |
| Small attention model | Contender | tinygrad kernel efficiency/data budget favor convolutions/global pooling |
| CPU incremental learned evaluation | Deployment contender | Inferior strength/time or an incompatible self-play training signal |
| Full graph search | Research option | History identity/multi-parent backup complexity offers little beyond the inference cache |
| MuZero/EfficientZero stack | Low priority for this exact game | Learning cheap known dynamics creates more computation and approximation error |
| Search-contempt / population training | Exploratory | Objective drift, weak evidence, or opportunity cost of extra runs |

For a pure systems change, retain a meaningful measured speed/memory win, or a
clear simplification with nonregression. For an algorithm change, throughput
alone is insufficient: require learning/playing-strength evidence under equal
resources. Predeclare the tolerated noise band from the baseline controls;
do not invent a universal percentage threshold or promised 10× result.

## 16. Approval boundary

The proposed direction is: **one exact rules/transition kernel; worker-local
flat memory; a thin leased Python/Metal boundary; an effect-aware small network;
measured self-play learning and robust evaluation.**

Before executing, the user reviews this plan. Implementation approval permits
coding in this scope. It does not automatically authorize training, benchmarks,
heavy builds/tests, all-core runs, or deployment while the CPU pause remains.
Resume of any expensive work will be explicit, with a resource/time budget.

Research backing: [source catalog](../sources/README.md),
[paper notes](../sources/NOTES.md), and
[bumblebench application guide](../sources/BUMBLEBENCH.md).
