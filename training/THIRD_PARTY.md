# Algorithm attribution

The sequential-halving visit schedule in `../src/selfplay/search.rs` is adapted from
DeepMind's [mctx](https://github.com/google-deepmind/mctx),
`mctx/_src/seq_halving.py` (Copyright 2021 DeepMind Technologies Limited), under
the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
The complete license is included in `licenses/mctx-APACHE-2.0.txt`.
The adaptation replaces JAX array scheduling with a Rust-owned search machine;
the visit schedule and algorithm attribution are retained.

The completed-Q transformation and deterministic interior selection are
implementations of *Policy improvement by planning with Gumbel*,
Danihelka et al., ICLR 2022, using mctx as the algorithmic reference.

The adapted schedule is provided WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied, subject to the Apache License 2.0.
