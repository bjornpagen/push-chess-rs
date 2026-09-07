# First isolated Run 2 residual

The sealed corpus passed its complete current-rules audit before learning.
The production code was verified and pushed before this run; no embedded
network was changed. The checkpoint is a candidate, not a promoted engine.

```sh
uv run --no-sync pushzero nnue train --db data/corpus --runs 2 \
  --output models/residual-r2.safetensors --steps 1000 --batch-size 256 --device METAL
uv run --no-sync pushzero nnue export models/residual-r2.safetensors \
  --output models/residual-r2.bin
```

Defaults: seed 1, at most 10,000 retained training games / 2,000 validation
games, at most 16 sampled positions per game. Each batch samples a game
uniformly then a position within it. Whole-game deduplication and opening-family
separation remain enforced; test/arena games are not read. Targets are verified
terminal expected scores, not cross-engine raw search evaluations.

## Result, 2026-09-07

Apple M2 Max, tinygrad 0.14.0, Metal training and native frozen inference:

| Validation reference | Game-balanced binary cross-entropy |
|---|---:|
| Embedded control / before training | 0.3989457154 |
| Handwritten baseline, zero network | 0.4040046430 |
| Candidate after 1,000 updates | 0.3914221763 |

All three use the same 2,000 games / 31,554 positions and exact deployed integer
scoring. Lower is better: the candidate reduces this loss by 1.89% relative to
the control and 3.11% relative to the handwritten-only reference. This is one
seeded bounded refinement, not a hyperparameter sweep or strength measurement.
In particular, the old network itself beats handwritten-only loss while Abacus
led the exploratory tournament: these objectives are not interchangeable.

The training scan visited 16,756 games; 16,578 supplied eligible positions,
89 trajectories were duplicates, and the reservoir retained 10,000 games /
157,744 positions. Validation scanned 2,070 games, with 2,055 eligible.
Checkpoint metadata contains source engine/binary/rules identities and the
full sampling configuration. The corpus has the two documented v1 castling
warnings; no result was silently reclassified, no rule corrected mid-run.

End-to-end elapsed time was **88.30 s** (88.95 user / 1.44 system). The learner's
timer, beginning after both datasets and baseline validations, reports **4.943 s**
for updates, final validation and preparation of checkpoint metadata. It is not
the end-to-end loader time or a pure-kernel benchmark. Separate backend timing
is in nnue-backends.md. Sampling/replay dominates this one short pass.

Artifact identities:

- Checkpoint SHA256: `8d9b245b5810ad2576754c594e610d1f20f463653d9e0e45cf27d82ff1cdffb9`.
- Export: 49,280 bytes; SHA256 `155eb342e0af7ec7694694158f8eecce6e454bd117c99374e2e3a8d3383505c4`.
- Unchanged control SHA256: `49edc822ebaedc91f7cf567468dfe6ef218dd93c45369fed9edbed203b765f26`.
- Training sample SHA256: `92e28aea5e56297bb68740970e6c54ac25afba88b3f8291801bdd6adf93bb39c`.
- Validation sample SHA256: `2ac4595a2b68c937b73de846dcde657889c12b1db761e8320ea1999972cb0199`.

Artifacts belong in ignored `models/`; game facts remain only in bumbledb.
Source implementation is `5d19553` (HEAD `f836b3e` added audit documentation).

## Next playing test

Use the candidate alias `aurora-r2`, never the built-in Cataclysm name. Smoke
the five entrants Cataclysm, Abacus, Astra, Waypoint and Aurora-r2 on one
color-swapped pair per matchup (20 games), 12 workers, 25 ms/move, six opening
plies, 256-ply cap, 600-second and 50-GiB ceilings. Audit before continuing.

Then compare the same fixed binary/weights on a fresh arena seed: 500 pairs
per matchup (10,000 games), 12 workers, 100 ms/move, six opening plies, 512-ply
cap, three-hour and 50-GiB ceilings. These are held-out-family evaluation games,
not additional training labels. This is a fixed exploratory comparison; do not
stop early on a favorable score or automatically promote the selected engine.
Modest gaps can still remain inconclusive with 500 families, especially after
accounting for ten matchups. A promising selection needs a fresh replication
and deeper-budget confirmation before the next production corpus generation.

The smoke completed as Run 3: all 20 games terminal, 1,280 moves, 1,160 analyses,
and a successful full replay audit (no castling warnings). All ten matchups
have both colors and no missing/unknown outcomes. Twelve workers were requested;
the runner capped this tiny workload at its ten available opening pairs. The
larger arena has enough pairs to use all twelve. No strength conclusion is
drawn from this one-pair smoke test.

The full arena started as **Run 4**, seed `2026090703`, with 10,000 scheduled
games and all 12 workers. Its fixed executable SHA256 is
`f072dbca075b1775f5f007cf54bc19c850ea8e713290238e3004b3db9b0c905f` and candidate
bytes have the export digest above. Live progress must come from
`lab status --db data/corpus --run 4` through the owning process; never reopen
the live store in Python. The disposable log is `data/arena-r2.log`. Audit after
the owner exits and inspect the sealed report before choosing any follow-up.
