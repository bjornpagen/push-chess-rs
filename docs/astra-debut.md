# Astra 001: elite debut

Astra is a new generation-13 engine, not a renamed incumbent. It uses only the
position and the supplied search budget, with no opponent-specific code or
access to saved game results.

## Design

- Generate pseudo-legal moves, then test legality as each move is searched.
  This avoids the incumbents' preliminary make/unmake pass over every move.
- Search quiet check evasions at tactical leaves instead of evaluating an
  in-check position as though the side could pass. Search checking quiet moves
  at the first quiescence layer too.
- Store full position keys in a two-way, 32 MiB cache. Mate scores fit their
  representation and are normalized for distance from the search root.
- Recognize pushes along the full movement path, including empty destinations
  and both knight paths. Keep move-ordering history separate for those paths.
- Use principal-variation search, aspiration windows, selective late-move
  reductions, guarded null-move pruning, repetition detection, and completed
  iterative-deepening results. Honor node and depth limits as well as time.
- Evaluate material, centralization, phase-dependent king placement, pawn
  shelter, pawn structure, passed pawns, and latent lines toward the enemy king.

## Tournament

Tournament #13, `astra_debut_001`, started at database timestamp
`2026-09-05 22:44:46` (UTC). Four engines, eight games per pairing, alternating
colors: 48 games total, one second per move, up to 12 games in parallel.
The tournament finished at `2026-09-05 22:50:18` UTC, after 5 minutes 32 seconds.
All 48 games completed. **Astra won the tournament.**

| Place | Engine | Wins | Draws | Losses | Score |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Astra | 17 | 2 | 5 | 75.0% |
| 2 | Void | 11 | 3 | 10 | 52.1% |
| 3 | Oblivion | 11 | 0 | 13 | 45.8% |
| 4 | Chronos | 6 | 1 | 17 | 27.1% |

```sh
cargo run --release --bin showdown -- tournament astra_debut_001 1000000 8 astra void chronos oblivion
```

Astra completed its 24 games with **17 wins, 2 draws, and 5 losses**, a **75%**
score (draws count as half a win).

| Opponent | Astra wins | Draws | Astra losses | Astra score |
| --- | ---: | ---: | ---: | ---: |
| Void | 5 | 2 | 1 | 75% |
| Chronos | 6 | 0 | 2 | 75% |
| Oblivion | 6 | 0 | 2 | 75% |

The field consists of the three strongest saved incumbents, not the entire
24-engine selectable roster. Games start from the standard initial position;
the four engines do not randomize their play using game seeds. These are
experimental results, not independent randomized trials or proof of universal
dominance. The shared push-rule fixes apply equally to all four engines and
make comparisons with pre-refactor tournaments less direct.

Full moves and search telemetry are saved in `pushchess.db`; use tournament 13
in the replay browser or `cargo run --release --bin dump -- 13` for a report.

## Verification and source identity

All 36 release-mode tests passed, including four Astra-specific tests for mate
score encoding, push recognition, check evasions, mate in one, and node limits.
The full-roster smoke test also checks legal moves and position restoration.
Formatting and the repository's strict Clippy invocation passed. No engine code
was changed during the tournament.

Base commit for shared rules and incumbent engines:
`249690564e4911a190a81d7660f4ac9cadc34673`.

SHA-256 fingerprints of the sources and binary used for this run:

| File | SHA-256 |
| --- | --- |
| `src/candidates/astra.rs` | `49eaa696241a5ebc80b75d14cab6ca94d258b6e4cacfc22330ec7fb1d106cef6` |
| `src/candidates/void_engine.rs` | `47a02d3d86ebbb3c73ad6804ca2c970d091d540305f909e33b88cfeea1511475` |
| `src/candidates/chronos.rs` | `96a54c2dbf3dc7ccbb4ab8eaf664bc663fa2b3e83cbfc5e28149eb09970bd677` |
| `src/candidates/oblivion.rs` | `c22e56aec33069bcfe604d0d79e5f25c373683f5782f3cca7805e5b113472c6f` |
| `target/release/showdown` | `e36ff63634c5043a0a4fbc4d0c610dd1377204371b6f17d786eacb2ce50dfbcf` |

The game database remains a local, Git-ignored artifact; this report preserves
the result and exact source identities separately.
