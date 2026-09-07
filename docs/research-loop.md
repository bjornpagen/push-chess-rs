# Engine research status

## Current instruction — 2026-09-07

Clean up the old-rules corpus and mixed-rules schema, update bumbledb to the
latest verified revision, then commit and push. **Do not start training,
self-play, data generation or another tournament.** There is not enough
eligible data to train the full model yet.

The active code implements one corrected ruleset. The database no longer
stores a rules choice on each run, and historical replay modes have been
removed. Checkpoints retain a single compatibility identity so incompatible
weights are rejected, not silently relabeled.

## Data and candidates

The old-rules Runs 1–4, including all of the previous training corpus, were
removed from the active database. Only Runs 5 and 6 survive: 256 evaluation games,
20,755 moves and 19,219 search observations. Their IDs, results, exact position
hashes, model/binary identities, timestamps and other stored facts are preserved.
See [the cutover record](corpus-cutover.md) for verification and final status.

These are held-out games, not replacement training data. A future training
phase needs a separately authorized, audited corpus run. Do not change the
survivors' purpose or splits to make a training command succeed.

R2 and r3 model/checkpoint files remain in [models](../models/README.md).
Neither is promoted; the embedded control and 17 built-in engines are unchanged.
R3 improved same-sample prediction loss 1.11% versus r2 and scored 53.52% in
the two bounded checks. This is encouraging but inconclusive strength evidence.

## Operating boundaries

- bumbledb is the only durable store of game, position, action, search and result
  facts. No JSON/FEN blobs, file-shard replay store or old-schema importer.
- One process owns a corpus. Never open a second owner against a live run;
  the existing local status connection serves live queries.
- Tests use disposable fixtures, never production games for training.
- Keep complete opening families together; evaluation games remain held out.
- Use new model output names and preserve controls. Training loss and search
  throughput do not prove stronger play.
- Work locally; no paid cloud compute or changes to adjacent repositories.
- No heartbeat is installed and no automatic continuation is authorized.

## Historical records

[Round 2 NNUE](nnue-round-2.md), [Run 4 review](run-4-review.md),
[bounded r3 refinement](nnue-round-3.md), [castling correction](castling-transit-audit.md)
and [search measurements](search-cost.md) describe completed historical work.
Reports referring to purged games are not evidence those trajectories remain
available, nor permission to resume their former protocols.
