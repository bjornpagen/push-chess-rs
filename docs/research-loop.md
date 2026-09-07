# Engine research loop

The user has authorized ongoing local engine research until stopped. This is
an open-ended objective, not a promise of an unbeatable engine. Work on main;
commit and push verified implementation before production tournaments.

## Current generation

The running Round 2 has fifteen entrants: Cataclysm and Astra controls, seven
Cataclysm experiments, and six Astra experiments. Seven entrants are entirely
neural-free; Abacus disables the NNUE score but still maintains its accumulator. See engine-lab.md
for the precise hypotheses. These are candidates, not proven upgrades.

The next executable adds Waypoint, an isolated actual-parent counter-move
context experiment. The registry now has 16 built-ins; `all` therefore schedules
24,000 games at 100 pairs, not 21,000. Do not alter the live round. Test Waypoint
against Cataclysm in fresh arenas before drawing any strength conclusion.

The initial cutover uses fresh normalized bumbledb facts only. The Python/Metal
source is retained in training/; runtime data belongs in data/corpus. No neural
self-play campaign or full-model training is part of this research loop.
Cataclysm NNUE refinement is in scope as a component of classical search.

## Round protocol

1. Verify rules, all engine profiles, relational rejection/round-trip tests,
   direct Python page reading and tiny Metal update/checkpoint correctness.
2. Commit and push main. Keep the binary fixed throughout each tournament;
   bumbledb records its SHA256 and each engine/network identity.
3. Run an all-roster smoke test: one color-swapped pair per matchup, 12 workers,
   25 ms/move, six opening plies, 256-ply cap and ten-minute ceiling. Audit all
   saved facts against replay. Smoke uses evaluation families, never training.
4. Run the first broad research round: 100 pairs per matchup = 21,000 scheduled
   games, 12 workers, 100 ms/move, six opening plies, 512-ply cap, eight-hour
   ceiling and 50-GiB store ceiling. Measure actual throughput; ceilings can
   interrupt admission, so scheduled games are not completed-game promises.
5. When the process has actually stopped, audit the run. Read paired results,
   draw/unknown rates, opening asymmetry, search completion, tactical mistakes,
   nodes/time and corpus growth. Data, not a roster name, determines priority.
   Inspect `castling_audit` before any rules/version decision; see
   castling-transit-audit.md. Its warnings do not reclassify saved v1 outcomes.
6. Choose a small next generation of falsifiable changes. Preserve controls
   and diverse high-quality opponents. Test changes individually before combining.
7. Run fresh paired comparisons and a second corpus tournament. Repeat with
   increasing useful evidence and stronger opponents until the user stops it.

Corpus rounds are exploratory and trainable. Arena rounds use held-out opening
families and are excluded from training. Opening-family hashes keep exact
prefixes together, but do not establish strategic independence. Do not reuse
one fixed arena seed as a perpetual selection target. Investigate results on
fresh opening families and at deeper search budgets before promotion.

## Improvement priorities

- Tactics: extract real missed mates, unsafe king pushes, reduction failures,
  repetition/draw errors, promotion races and horizon traps from saved games.
  Convert them into exact-history regression cases and selective search tests.
- Search: measure expensive work and its decision value; compare ordering,
  quiescence, pruning and budget allocation under both nodes and wall time.
  Keep mutable scratch local to workers and do not assume SoA is always faster.
- Evaluation: improve push-specific king safety, coordination, mobility and
  endgames only when ablations and games support the added cost.
- NNUE: train/refine the small residual on deduplicated training-family data;
  balance by game/family and retain exact teacher/label provenance. Keep terminal
  WDL separate from uncalibrated engine scores. Validate quantized inference
  against training arithmetic and preserve an unchanged network control.
- Diversity: keep complementary strong styles; do not discard a useful training
  opponent solely because its aggregate win rate is lower in one population.

## Evidence and promotion

Treat a color-swapped opening pair as the elementary comparison, not every ply
as an independent trial. Report missing/truncated games separately. Use complete
pair scores with uncertainty, examine matchup-specific weaknesses and account
for trying many hypotheses. A selected candidate needs a new independent
comparison and deeper-budget confirmation. Training loss, a tiny sample or a
single lucky tournament is insufficient. No automatic champion replacement.

Reports now retain the five-bin color-swapped pair score histogram (0, 0.5,
1, 1.5, 2 points for the first entrant), count incomplete pairs explicitly,
and separate missing games from saved games without outcomes. Repeated
four-ply opening families are clustered before calculating uncertainty.
The family-balanced mean has conservative fixed-sample Hoeffding bounds,
including a Bonferroni correction across the run's matchups. These bounds
assume independent opening families; they are not sequential stopping rules.
Looking repeatedly, selecting hypotheses, incomplete-result selection and
strategic similarity all require fresh independent confirmation. Overall
score bounds still account pessimistically/optimistically for missing results.

The `pushzero nnue` path refines the small existing residual from explicit,
sealed bumbledb source runs. It keeps terminal outcomes separate from raw
search scores and exports an isolated candidate. See training/README.md for
the arithmetic, sampling, strict holdout and preservation-of-control contract.
Explicit run IDs are validated together at the native boundary; invalid sources
cannot silently become a partial training request. Unselected runs are skipped
before trajectory replay or feature construction.

## Process ownership and resource policy

The 30-minute task heartbeat follows this same task and checkout. Before
starting any run, inspect the actual process and `lab status --db data/corpus`.
Never start a second writer, infer death from silence, or reopen the live store
from Python. The running owner answers status using committed bumbledb snapshots.
When stopped, `summary`, `report` and `verify` open the store directly.

Only local compute is authorized by this workflow; do not purchase cloud compute
or alter adjacent repositories. Time/disk caps remain explicit. A later user
pause takes precedence immediately. SIGINT/SIGTERM stops at a move boundary,
commits verified partial progress and seals run status. Failed/unsealed runs
are quarantined, and interrupted games never supply made-up outcomes.

Keep the machine and app running for local scheduled follow-ups. Notify only
meaningful results, failures and decisions; otherwise continue useful research
or wait on a verified live process. Logs and these notes are disposable evidence
or plans, not alternative sources of game truth.
