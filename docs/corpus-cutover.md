# Single-rules corpus cutover — 2026-09-07

The user authorized purging all old-rules database data, removing mixed-rules
support, upgrading bumbledb, and committing/pushing the cleanup. No training or
data-generation run is authorized.

## Retention boundary

Remove Runs 1–4: 31,230 games, 2,364,673 moves and 2,177,297 search observations.
Preserve Runs 5–6: 256 terminal evaluation games, 20,755 moves, 21,011 positions
and 19,219 search observations. Run IDs remain 5 and 6; the next ID is 7.
There is no remaining train/validation corpus. Existing models are not deleted
or promoted, and evaluation games do not become training data.

## Verification

A one-use native tool read the original schema and copied only surviving facts
and their referenced engine, setup and action facts into a fresh normalized
bumbledb store. Only the redundant Run.rules field was projected away.
No game serialization, stored blobs, relabeling or legacy importer was added.

All 27 open-relation fact sets were compared before/after using sorted SHA256
digests of their typed values. The aggregate retained-fact checksum is
`f1394f37c1ec4392dad8de04915670e3f788a88c28eaab0c9ff7535a1907f630`.
This covers run/binary provenance, timestamps, network fingerprints, pairs,
game identities, outcomes, every position hash, transitions, analysis, PVs
and proof observations. Unreferenced old setup/action/engine facts were omitted.

Full replay of the copied store verified Run 5's 128 games / 9,132 moves /
8,364 analyses, and Run 6's 128 games / 11,623 moves / 10,855 analyses. All
games retain terminal outcomes. The one-use old-schema tool was then removed.

## Active implementation

Run has no rules field. Board state and Python constructors have no historical
mode; all play/replay uses the corrected occupied-transit castling rule.
The current position-hash salt is unchanged, preserving the surviving facts.
Unsafe saved castles fail validation rather than becoming sampling warnings.
Run allocation uses the greatest existing ID, not the number of runs.

Model checkpoints still carry the current rules/encoding compatibility identity.
This is a guard against incompatible weights, not a choice of game rules.
The archived r2 checkpoint is rejected by current training; its unchanged
inference export remains a comparison control. R3 can initialize a future pass
from weights, but its purged source data precludes exact sampling resume.

## Dependency and publication

Upgraded from bumbledb 0.20.3 to **1.0.1**, pinning upstream main at
`5e83ee60c4e5d88e8ca395daa3de4d92a0c03186` in both Rust dependency locks.
The upstream head was checked again after the upgrade tests.

All 118 Rust and 54 Python tests pass, including current-rules castling
rejection/roundtrip and run-ID-gap regression tests. Workspace/native Clippy
and formatting pass. The all-engine fixed-node fingerprints are unchanged;
they were verified on current rules before and after this cleanup.

The replacement was published at `data/corpus` and both surviving runs passed
full replay again with bumbledb 1.0.1. The Python reader sees exactly zero
train/validation/test games and 256 evaluation games. The active store is now
81,723,392 bytes, down from 11,327,979,520 bytes. All five preserved model/control
artifact hashes are unchanged, and r3 still loads and exports identically.

The retired store was moved to macOS Trash at
`push-chess-old-corpus-20260907-M4x466/corpus`. It is recoverable until Trash is
emptied, but is no longer an active source and requires the historical schema
to read. Git never contained those game records. No training or production
game-generation run was started; automated tests used disposable fixtures.
