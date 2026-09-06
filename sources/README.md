# PushZero research library

Collected 2026-09-06 for the [proposed redesign](../docs/pushzero-redesign.md).
This is preparation, not approval to change the engine or restart compute.

19 version-pinned arXiv source submissions are stored locally in `arxiv/`.
Each directory contains the original archive (including figures), safely
extracted LaTeX/bibliography/style text in `tex/`, the arXiv metadata response,
and `metadata.json` with authors, title, abstract, dates, URLs, and SHA-256 hashes.
The repository-level [manifest](manifest.json) retains pinned archive hashes and
provenance even when the ignored download directory is absent. Original archives
total 45,578,297 bytes (about 43.5 MiB).
No paper was converted from a PDF and presented as original LaTeX. No downloaded
TeX, script, or code was compiled or executed.

## Reading map

The links below open the primary source entry points. Some include other files.
Read [NOTES.md](NOTES.md) for inspected sections, implications, and limitations;
read [BUMBLEBENCH.md](BUMBLEBENCH.md) for hardware evidence and its boundaries.

| Paper | Local LaTeX | Why keep it |
|---|---|---|
| [AlphaZero, 1712.01815v1](https://arxiv.org/abs/1712.01815v1) | [main](arxiv/1712.01815v1-alphazero/tex/main.tex) | Exact rules, learned policy/value, self-play foundation |
| [KataGo, 1902.10565v5](https://arxiv.org/abs/1902.10565v5) | [main](arxiv/1902.10565v5-katago/tex/Accelerating_Self_Play_Learning_In_Go_2020.tex) | Search-cap randomization, global context, auxiliary tasks |
| [MCTS as Regularized Policy Optimization, 2007.12509v1](https://arxiv.org/abs/2007.12509v1) | [main](arxiv/2007.12509v1-mcts-policy-optimization/tex/grill2020monte-carlo.tex) | Why low-budget targets need more than visit fractions |
| [Batch MCTS, 2104.04278v1](https://arxiv.org/abs/2104.04278v1) | [main](arxiv/2104.04278v1-batch-mcts/tex/main.tex) | Neural-result cache separate from tree statistics |
| [Monte-Carlo Graph Search, 2012.11045v1](https://arxiv.org/abs/2012.11045v1) | [main](arxiv/2012.11045v1-graph-search/tex/main.tex) | Transposition/DAG alternative and backup hazards |
| [MuZero Reanalyse, 2104.06294v1](https://arxiv.org/abs/2104.06294v1) | [main](arxiv/2104.06294v1-muzero-reanalyse/tex/main.tex) | Refresh search targets on existing trajectories |
| [Scaling Scaling Laws with Board Games, 2104.03113v2](https://arxiv.org/abs/2104.03113v2) | [main](arxiv/2104.03113v2-board-game-scaling/tex/main.tex) | Joint training/search compute allocation |
| [AlphaZero Neural Scaling and Zipf's Law, 2412.11979v2](https://arxiv.org/abs/2412.11979v2) | [main](arxiv/2412.11979v2-neural-scaling-zipf/tex/main.tex) | State-distribution problems and inverse scaling |
| [Go-Exploit, 2302.12359v2](https://arxiv.org/abs/2302.12359v2) | [main](arxiv/2302.12359v2-go-exploit/tex/sample.tex) | Restarts from real visited/search states |
| [Regret-Guided Search Control, 2602.20809v1](https://arxiv.org/abs/2602.20809v1) | [body](arxiv/2602.20809v1-regret-search-control/tex/camera-ready.tex), [appendix](arxiv/2602.20809v1-regret-search-control/tex/appendix_camera-ready.tex) | Recent learned restart prioritization |
| [Population-Based AlphaZero, 2003.06212v1](https://arxiv.org/abs/2003.06212v1) | [directory](arxiv/2003.06212v1-population-based-alphazero/tex/) | Adaptive hyperparameters; compute-heavy alternative |
| [Adversarial Policies Beat Superhuman Go AIs, 2211.00241v4](https://arxiv.org/abs/2211.00241v4) | [main](arxiv/2211.00241v4-adversarial-go/tex/main.tex) | High average strength does not imply robustness |
| [MuZero, 1911.08265v2](https://arxiv.org/abs/1911.08265v2) | [directory](arxiv/1911.08265v2-muzero/tex/) | Learned-dynamics alternative |
| [EfficientZero, 2111.00210v2](https://arxiv.org/abs/2111.00210v2) | [method](arxiv/2111.00210v2-efficientzero/tex/03_method.tex) | Consistency, value prefixes, stale-data correction |
| [EfficientZero V2, 2403.00564v2](https://arxiv.org/abs/2403.00564v2) | [method](arxiv/2403.00564v2-efficientzero-v2/tex/sections/methods.tex) | Sampled Gumbel search, search-based value estimation |
| [Amortized Chess Transformers, 2402.04494v2](https://arxiv.org/abs/2402.04494v2) | [method](arxiv/2402.04494v2-amortized-chess-transformer/tex/02_methodology.tex) | Strong no-search inference, but enormous teacher dataset |
| [Sampled MuZero, 2104.06303v1](https://arxiv.org/abs/2104.06303v1) | [main](arxiv/2104.06303v1-sampled-muzero/tex/main.tex) | Search when enumerating actions is impractical |
| [Search-contempt, 2504.07757v1](https://arxiv.org/abs/2504.07757v1) | [main](arxiv/2504.07757v1-search-contempt/tex/main.tex) | Speculative opponent/search distribution changes |
| [Apple Firestorm/Oryon Branch Predictors, 2411.13900v1](https://arxiv.org/abs/2411.13900v1) | [main](arxiv/2411.13900v1-apple-branch-predictors/tex/asplos25-paper-template.tex) | Branch-predictor measurement; not an M2-specific result |

## Important non-arXiv references and gaps

- **Policy improvement by planning with Gumbel**, Danihelka et al., ICLR 2022:
  [canonical OpenReview entry](https://openreview.net/forum?id=bERaNdoegnO).
  No matching arXiv submission was found in this search. OpenReview returned a
  browser-verification challenge; no public LaTeX source was obtained. This is
  an explicit gap, not a missing file silently substituted with another paper.
- [Google DeepMind mctx](https://github.com/google-deepmind/mctx): primary
  executable reference for Gumbel scheduling, action selection, and Q transforms.
  Existing attribution and license live in [training/THIRD_PARTY.md](../training/THIRD_PARTY.md).
  A future parity test should pin a particular upstream revision, not `main`.
- Casey Muratori, [Semantic Compression](https://caseymuratori.com/blog_0015):
  verified public article entry; a design influence, not a hardware benchmark.
- The user's pasted representation-first note is an additional design input:
  preserve information in the data/types rather than repeatedly rediscovering it.

## Reproduce and verify

```sh
python3 sources/fetch.py
python3 sources/fetch.py --verify
python3 sources/fetch.py --only 2302.12359v2
```

The fetcher uses only Python's standard library, spaces network requests, bounds
archive/text sizes, refuses unsafe member paths and non-regular members, and
never overwrites an existing paper directory. Incomplete entries require manual
inspection; changed pinned archive hashes also require review. Only selected
text extensions and license/readme notices are
extracted; all other assets remain in the original archive. Its purpose is a
readable research library, not a buildable TeX environment.

The downloaded `arxiv/` tree is intentionally ignored by Git. Papers and style
files retain their original rights; public download does not establish a
redistribution license. The authored catalog, fetcher, and notes are the
reproducible repository record. Inspect upstream terms before vendoring any
third-party source into a public repository or release. Metadata does not
pretend to know a license that arXiv's API did not supply.
