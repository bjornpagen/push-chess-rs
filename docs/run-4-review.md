# Run 4: completed comparison and first actual game review

## Result and audit — 2026-09-07

The original tournament owner exited normally. Run 4 is sealed as `finished`:
10,000/10,000 games, all terminal, 798,273 moves, 808,273 positions and 738,273
search observations. There were 373 draws (3.73%). The runner recorded
5,072.12 seconds; this is its elapsed timer, not a reconstruction of macOS sleep
or calendar time. All ten matchups have 500 complete color-swapped pairs.

The exact producing executable's full replay audit passed in 54.24 seconds:
all games and moves verified, 241 castles, **zero castling-transit warnings**.
This does not correct the documented v1 rule defect or erase Run 2's warnings.
737,094 searches completed a depth or reported mate proof; 1,179 did not.
No truncated game was converted into a draw.

| Engine | Wins | Draws | Losses | Score over 4,000 games |
|---|---:|---:|---:|---:|
| Aurora-r2 | 2,130 | 143 | 1,727 | 55.0375% |
| Abacus | 2,039 | 135 | 1,826 | 52.6625% |
| Astra | 1,967 | 132 | 1,901 | 50.8250% |
| Waypoint | 1,741 | 180 | 2,079 | 45.7750% |
| Cataclysm | 1,750 | 156 | 2,094 | 45.7000% |

Score is `(wins + draws/2) / games`. Aurora-r2 scored 57.8% against Cataclysm,
51.1% against Abacus, 52.8% against Astra and 58.45% against Waypoint.
The family-balanced score against Abacus is 51.1%, with conservative unadjusted
95% family bounds of approximately 45.03–57.17%; it is not a proven advantage.
The simultaneous bounds against Cataclysm and Waypoint just exclude 50%, but
this was an exploratory, repeatedly observed comparison, not independent
post-selection confirmation. No model is automatically promoted. Waypoint's
results provide no reason to replace Cataclysm's existing search policy.

All data remain in `data/corpus`, the sole bumbledb source. Run 4 is evaluation
data and remains excluded from training. Across runs 1–4 there are 31,230 saved
games and 2,364,673 moves; the one nonterminal saved game is outside Run 4.
These notes are derived review evidence, not an alternate game/replay dataset.

## One actual game: Abacus–Aurora-r2, 1–0

Source: Run 4, **game index 12** (zero-based), Abacus White, Aurora-r2 Black.
This was selected as the first Abacus/Aurora game in storage order, not a
representative sample or a highlight chosen to prove superiority. Its paired
color-swapped game is index 13; that also ended in an Abacus win, but the full
matchup remained nearly even. The first six plies were a shared randomized
opening, not decisions made by either engine. The game ended on White's move 50.

Identity:

- Rules: `push-chess-history-v1`.
- Producing binary SHA256:
  `f072dbca075b1775f5f007cf54bc19c850ea8e713290238e3004b3db9b0c905f`.
- Trajectory SHA256:
  `7db41eed2a6e7cdcec1941208bc9907bf9a41a04bc882850fb7280222190533a`.
- Aurora weights SHA256:
  `155eb342e0af7ec7694694158f8eecce6e454bd117c99374e2e3a8d3383505c4`.

The entire saved action sequence replayed through the authoritative native
rules to the stored final position: Black has no legal moves and White wins
by checkmate. Reanalysis used the exact preceding history, not FEN-only roots.
No new games or tournament were generated.

### What the game actually shows

**12...Rc8–c2:** Aurora's rook travels down the c-file and pushes its own queen
from c6 to c1. One move brings both heavy pieces deep into White's position.
This is not conventional chess notation omitting a second turn: both effects
belong to the same saved Push Chess action.

**16...Qe4xe2 17.Re1xe2 Bc4xe2:** Aurora gives up its queen for White's bishop
and rook. It looks alarming without the full position, but it is not enough
to call it a queen blunder: Aurora, Abacus and Astra all still select `Qxe2`
when reanalyzing the exact pre-move history at 524,288 nodes. Their valuations
disagree, and none is an oracle, but the sacrifice has a concrete tactical basis.

**26...Ng3–e4:** the route matters as much as the destination. The played action
(`5910`, long leg first) displaces Black's f3 pawn to d3. At the same exact root,
both Aurora and Abacus with 524,288 nodes instead choose action `10006`: the
same knight endpoints but the other route, leaving the f3 pawn in place. Astra
chooses a different knight move entirely. The historical search reached only
depth 2 with 2,304 nodes. This is a useful search-horizon / collateral-effect
valuation case, not proof of a move-encoding bug or of a forced loss. Engine
scores from different evaluators must not be treated as one calibrated curve.

**34...Rd6xd4 35.Ne5–c6:** Aurora takes a bishop, then faces a knight check and
loses the rook to `36.Nc6xd4`. At the pre-capture root, deeper Aurora and Abacus
analysis prefer `...Pe8–e7` instead; Astra prefers a king move. That is a second
concrete defensive decision worth investigating with exact-history variants,
not a demonstrated universal best move or a quantified blunder rate.

**45.Qa2–a7, promoting the pushed pawn:** Abacus's queen passes through its own
pawn on a3 and pushes it all the way to a8, where it becomes a second queen.
This is genuine push-specific conversion technique, not an ordinary pawn march.

**48.Nd2–e4:** the knight's long-first route pushes White's queen from d4 to d5,
giving a discovered queen check against the king on f7. Abacus has a forced
mate within three White moves. The actual finish was:

`48.Nd2–e4 Kf7–e8 49.Qg2–g6 Ke8–e7 50.Qg6–e6#`

The full five-ply winning continuation was checked separately from the engine's
reported proof: use native search only to select a White witness move, enumerate
**every legal Black reply**, and require each leaf to be an actual White checkmate
under the rules/history API. Four terminal branches passed (14 visited states,
seven White choices, six Black reply edges). The other defensive branches were:

- `48...Kf7–e7 49.Qg2–g5 Ke7–e8 50.Qg5–d8#`.
- `48...Kf7–e8 49.Qg2–g6 Pf8–f7 50.Qg6xf7#`.
- `48...Kf7–e8 49.Qg2–g6 Ke8–f8 50.Qd5–d8#`.

This independently checks the search claim against the same authoritative rules;
it is not a second independent rules implementation. Aurora and Abacus both
found the root mate in 376 nodes on a fresh diagnostic search; Astra also found
the mating move through its ordinary search. These are tactics they really can
execute, not strength inferred from architecture alone.

## Interpretation and next work

This sample supports a more concrete description than the standings did:
strong, variant-specific tactical ideas and real conversion technique, mixed
with choices sensitive to shallow search and evaluator disagreement. One game
does not establish a human rating, a population blunder rate or strategic mastery.

Continue without another tournament:

1. Inspect more existing wins **and losses** from both leading engines, with a
   declared sample rule. Separate opening-induced trouble from engine decisions.
2. Investigate route-sensitive collateral moves and the knight-fork defense in
   this game. Use exact history and bounded deeper analysis; derive a regression
   only once its expected property is defensible. Do not patch toward one evaluator's
   unsupported preferred move.
3. The post-tournament identical-node comparison is now complete: Granite took
   5.37% less time than old Abacus, but the shared refactor slowed existing
   controls by about 2–3%. Investigate and remove that overhead before claiming
   the implementation is performance-complete. See [search-cost.md](search-cost.md).
4. Consider bounded residual improvements using eligible audited corpus data
   only. Do not train on this arena or claim new playing strength without games.
5. Keep the v1 castling issue separate: any correction must preserve historical
   rules identity and replay interpretation, not silently invalidate this corpus.

The user's no-more-tournaments instruction remains in force, including smoke,
arena and self-play campaigns. No replacement automatic follow-up is installed.
