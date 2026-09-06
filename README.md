# Push Chess

Rust rules and engines for Push Chess, with native and browser clients and a
Metal-accelerated, rules-only self-play learner.

## Self-play training

The inference-first redesign and its research/experiment plan are documented in
[docs/implementation-status.md](docs/implementation-status.md). New architecture
options are experimental; this is not a claim of improved playing strength.

```sh
uv sync
uv run pushzero doctor
uv run pushzero train --run experiments/my-pushzero --minutes 60
uv run pushzero train --run experiments/my-pushzero --resume --minutes 60
uv run pushzero evaluate experiments/my-pushzero --opponent cataclysm \
  --output experiments/my-pushzero/evaluation.json
```

See [training/README.md](training/README.md) for the exact algorithm, Metal setup,
Rust/Python memory boundary, checkpointing, evaluation budgets, and limitations.
Training does not replace the deployed engine automatically.

## Code organization

- `src/core/`: board representation, validated FEN parsing, move generation,
  push resolution, make/unmake, and position hashes.
- `src/game.rs`: shared legality, outcome/history rules, and UI transactions.
- `src/session.rs`: the production Cataclysm analysis session.
- `src/candidates/`: Cataclysm and retained historical engine experiments.
- `src/selfplay/`: Python-independent Rust state, encoding, and batched neural
  search. Its input is network evaluations, not handcrafted strategic scores.
- `training/native/`: thin PyO3/NumPy adapter; no duplicate rule implementation.
- `training/pushzero/`: tinygrad network, self-play orchestration, replay,
  optimization, checkpointing, evaluation, and command-line entry point.
- `crates/native/`: native game/tournament tools; `crates/wasm/` and `web/`:
  browser integration.

## Tests

```sh
cargo test --workspace --all-features
uv run pytest -q
```

After editing Rust code used by Python, rebuild the extension with
`uv sync --reinstall-package pushzero`. The Python extension is an independent
Cargo package so Python build dependencies do not enter the WASM workspace.
