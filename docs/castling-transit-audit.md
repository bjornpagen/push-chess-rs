# Castling through a knight-controlled transit square

## Correction — 2026-09-07

New play now uses `push-chess-history-v2`, shared by core, lab and Python.
The castling test places the king on its occupied transit square and vacates
its origin, leaving the rook in place. The normal legal-child filter checks
the final position after the rook moves. Both knight routes retain the actual
push blockers; there is no geometric substitute.

Four attacked color/wing cases, four knight-free controls, four blocked-route
controls, a vacated-origin case and a final-rook-blocker case are regression
tested against core and specialized search. Rules-specific position hashes
keep transposition/repetition identities separate; the old hash values are
unchanged. The historical rules survive copies and full-history reconstruction.

The existing normalized `Run.rules` fact selects replay semantics. No schema
migration, relabelling, deletion or stored game blob is involved. New saves
reject game/run rules mismatches. The old nine-game bumbledb regression still
verifies all v1 castles and reports the same four warnings. Python pages expose
the warnings as typed run/game-local observations; NNUE sampling excludes
affected whole games while retaining their original results in the corpus.
Full policy learning rejects historical rules rather than mixing legal targets.

The rebuilt executable `c36e55384af8578b69ce2be8ec2e25efa219f8368c598b96731905226ed63c9c`
passed full offline replay of **all 31,230 historical games / 2,364,673 moves**.
Run 1 retains its one capped game; all other saved games are terminal. Audits
still report exactly two warnings, both the original Run 2 references below.
Run 1/2/3/4 audit times were 16.654 / 107.540 / 0.614 / 60.407 seconds on the
local machine. The original executables, model bytes and game facts remain intact.

## Confirmed behavior, not a silent rules change

On 2026-09-06 the native rules API accepted castling in all four positions
below while rejecting the ordinary king move onto the transit square. Under
the standard prohibition on castling through check, these are rule defects.
They are not evidence that one search challenger plays better than another.

| Side / wing | FEN | Illegal ordinary step | Accepted castle |
|---|---|---|---|
| White / king | `k7/8/8/8/8/8/7n/4K2R w K - 0 1` | e1–f1 | e1–g1 |
| White / queen | `7k/8/8/8/8/8/1n6/R3K3 w Q - 0 1` | e1–d1 | e1–c1 |
| Black / king | `4k2r/7N/8/8/8/8/8/K7 b k - 0 1` | e8–f8 | e8–g8 |
| Black / queen | `r3k3/1N6/8/8/8/8/8/7K b q - 0 1` | e8–d8 | e8–c8 |

The native API also successfully played each castle to its final position.
Thus final-position king safety does not catch the skipped transit exposure.

The cause is the mismatch between castling's empty-square attack query and
the knight capture query's occupied-enemy-target contract. The former calls
`is_attacked_by` on the empty transit square; the latter returns false for an
empty target. The earlier full-transaction attack implementation likewise
required a capture. The measured knight-predicate refactor preserved this
behavior; reverting that optimization would not fix castling.

Pure L-shaped geometry is not an adequate replacement in push chess. For
example, `k7/8/8/8/8/8/6Pn/4K2R w K - 0 1` has the same h2 knight but the
g2 pawn blocks one route and the h1 rook blocks the other. Both e1–f1 and
castling are legal there. The transit audit includes this negative control.

## Audit without changing the corpus

`Trajectory::validate` now returns a small `ReplayAudit`. It reuses the legal
move set already generated at each recorded ply. For a recorded castle, it
checks whether the ordinary one-square king transit is in that exact set.
There is no extra rules replay, geometric attack approximation, alternate
engine implementation, or per-position database write.

After the live owner exits, the usual command includes these observations:

```sh
target/release/lab verify --db data/corpus --run 2
```

`castling_audit` reports total castles, transit anomalies, and at most 32 exact
`game`/`ply`/`action` references. Counts cover the entire audited run; the example
list is explicitly bounded. The output is a disposable derived report, not
another durable source of game facts. Missing outcomes remain missing and
warnings do not become losses, draws, quarantines or automatic label changes.

The regression saves and rereads nine synthetic games through bumbledb:
all four exposures, their four knight-free controls, and the blocked-route
control. It requires nine verified v1 games, nine castles and four warnings.
This pins the diagnostic under the current rules, not an endorsement of the
defect as a future rule.

Verification: all 107 Rust tests and 48 Python tests pass, with workspace and
native-extension Clippy gates and formatting clean. The production tournament
executable is unchanged; its recorded SHA256 remains
`b14e186c181fe92b0d308fecb477e50bce1a2e2e712cde89b7d3ca062ce89320`.

## Original cutover requirements (now implemented above)

Keep the running executable and rules fixed. Measure the actual corpus's
exposure before choosing a cutover. A correction needs an explicit new rules
identity shared by core, lab and Python/checkpoints; castling checks must use
the king's actual transit occupancy, including the vacated origin, while
respecting knight route blocking. Test both colors/wings and both attacked
and blocked-route cases.

Do not silently replay v1 games as if they were generated under corrected
rules, erase the run, or claim the affected proportion before the audit.
Historical facts and new-rules training eligibility are separate questions.
The diagnostic added here changes neither move generation nor the schema.

## Actual Round 2 exposure

The complete 2026-09-07 audit verified 21,000 games under unchanged v1 rules
and found **571 castles, two transit warnings**:

| Run | Game | Ply (zero-based) | Action ID |
|---:|---:|---:|---:|
| 2 | 12,764 | 24 | 262276 |
| 2 | 14,797 | 19 | 265916 |

This is two distinct games (0.00952% of the run), not evidence that the defect
is impossible elsewhere or strategically harmless. Stored outcomes remain the
actual v1 results. The audit does not relabel, erase or silently migrate them;
any refinement using this run is explicitly v1-derived. A future rules cutover
still needs the shared identity and occupied-transit tests described above.
