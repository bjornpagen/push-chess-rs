# Cataclysm training corpus

The deployed Cataclysm stays frozen. Collecting more data does not automatically
retrain it, prove improvement, or replace its embedded network. All code is Rust;
collection adds no dependencies, GPU runtime, Python, or CI runner.

## Initial collection plan

The full-roster `cataclysm_full_001` round robin is evaluation-only. After that
event finishes, collect **8,192 new games** in 128 batches of 64, with six workers.
The default schedule mixes Cataclysm self-play, Cataclysm versus Astra, Void,
Eternity, Chronos and Oblivion, and Astra versus Void and Eternity.

Each set of twelve batches cycles through the opponent pool. Opening lengths
cycle through 4, 8 and 12 random legal plies. Games between different engines
use identical openings with colors reversed; same-engine self-play uses a fresh
opening for each game. Opening moves, subsequent searches, and outcomes are all
saved. Training seeds are in a separate range from the evaluation tournament
and the earlier promotion gates.

Most batches request **50,000 nodes / 250 ms per move**; every fourth pass through
the pool requests **200,000 nodes / 500 ms**. Both limits apply to Cataclysm and
Astra; legacy engines may ignore the node limit and remain time-limited. This
is a varied data-collection schedule, not an equal-compute strength comparison.
The initial target counts generated games, not necessarily unique games or
accepted training examples. Reaching it is a bounded first tranche, not a claim
that there is a meaningful maximum possible corpus.

## Quality and separation

The collector freezes both its playing executable and itself inside the run
directory. Its SQLite catalog records the source revision/dirty state, executable
and model FNV-1a fingerprints, budget, seed, opponents, and progress. Those hashes
identify artifacts, not cryptographic authenticity. `manifest.json` is an atomic,
human-readable snapshot of the same information.

After each batch, the collector checks SQLite integrity and replays each eligible
game from the starting board under the current shared rules. It verifies every
saved move, intermediate FEN, continuity, search/opening telemetry, final board,
and terminal outcome. Timeouts, illegal moves, unfinished games and 300-ply
adjudications cannot supply outcome labels. They remain in the raw database
with rejection reasons. Collection stops for investigation if more than half
of a batch's labels are unusable.

Each accepted game receives a `corpus_audit` record. The board after the random
opening (including side, castling and en-passant rights, excluding counters)
determines its training/validation group by FNV modulo five. Identical opening
positions, including reversed-color engine pairs, stay in the same group.
The trainer additionally deduplicates whole games, averages repeated positions,
removes exact validation positions shared with training, and skips random-opening
and zero-node samples in these audited shards. Related but nonidentical positions
can still share patterns: this is not proof of statistical independence.

The raw `search.eval_cp` values are side-to-move-relative, not uniformly calibrated
win probabilities. Search depth and node semantics vary by engine; mate-prover
successes in Cataclysm report proof length separately rather than ordinary depth.
Those fields are retained for later research, not silently treated as perfect
teacher labels. The existing trainer remains outcome-only.

## Run, resume, and train later

```sh
cargo build --release --bins

# Freeze a plan without starting games (e.g. while a tournament is running).
target/release/collect experiments/cataclysm-corpus-001 8192 2305843009233954857 6 --prepare

# Start, or resume the same immutable plan. The OS lock prevents two collectors.
experiments/cataclysm-corpus-001/collect experiments/cataclysm-corpus-001

# Later, train a separate experimental model using only completed audited shards.
# This is not run automatically and does not deploy the resulting model.
target/release/train_cataclysm experiments/candidate.bin experiments/training.json \
  24 2147483647 --warm-start src/candidates/cataclysm/network.bin \
  experiments/cataclysm-corpus-001
```

Passing a run directory to the trainer resolves only finished shards in its
catalog. Do not manually glob every `.db`: that would also select the catalog
and interrupted attempts. Legacy history databases are still supported, with
their original whole-game split, so the deployed network remains reproducible.

A resumed collection skips completed batches. An interrupted attempt is kept
and a new uniquely named database is used; nothing is overwritten. Before
resuming after a process crash, ensure an orphaned playing subprocess is not
still running. A failed run needs diagnosis, not a blind retry loop. A completed
run resumes as a no-op. To expand the corpus, create a new run directory with a
fresh seed; never delete the first run to reuse its name.

All game databases, copied binaries, logs, rejected attempts and experimental
models stay under Git-ignored `experiments/`. Back that directory up separately;
pushing the source repository does **not** back up these games. The original
history databases remain intact.

## Verification before the first production run

The suite includes 55 automated tests plus the compile-fail documentation test,
release builds, formatting and strict Clippy checks. The collector-specific
checks cover opening grouping, invalid move encodings, rejected/truncated outcomes,
and the trainer's filtering and preservation of audited splits. An end-to-end
two-game smoke test replayed all 164 saved moves exactly and accepted 156 searched
positions. Resuming that finished run added no games or attempts. These are
implementation checks, not evidence that a future model will be stronger.
