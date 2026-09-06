# Research findings and decision record

2026-09-06. These are targeted readings of source sections, not independent
replications or claims to have checked every proof. Links in [README](README.md)
resolve to the actual version-pinned LaTeX. Experimental results from another
game are evidence for a hypothesis, not promised Push Chess speed or Elo gains.

## What belongs in the main design

### AlphaZero: keep the exact-model policy-iteration loop

`1712.01815v1`, algorithm and training description. Search produces an improved
policy; completed games supply outcome targets; a shared network predicts both.
Rules remain exact. Our six-field push moves, directional pawns, history and
draw rules must be represented faithfully. An encoder is not automatically
Markov just because the rules engine is exact.

Decision: keep this foundation, using our existing Gumbel implementation at low
search budgets. The paper's compute scale and results do not establish that a
small local implementation will reach the strongest opponent.

### KataGo: the highest-priority collection of transferable efficiency ideas

`1902.10565v5`, **Major General Improvements**, **Ablation Studies**, architecture,
training and game-initialization appendices. Read source around lines 87–164,
319–341, and the relevant appendices.

- Randomly give a minority of turns a large search and the others a small one;
  train policy targets on the high-search turns. This balances games/outcome
  diversity against quality of policy improvement. The paper's initial settings
  were 25% full searches, 600 versus 100 visits—not our 64/16 settings.
- Global pooling inside selected residual blocks conditions local computation
  on global state. An inexpensive candidate for nonlocal push tactics.
- Auxiliary opponent-policy prediction regularizes the trunk. Go ownership and
  score heads provide stronger domain-specific signals, but have no direct
  chess analogue. Material rewards would change our objective and are excluded.
- Forced playouts plus policy-target pruning decouple exploration from the
  training target. These were developed with PUCT; do not layer them blindly
  onto Gumbel completed-Q targets.
- Weight averaging is a candidate for stabilization. Gate it on actual strength
  and preserve unaveraged training state separately.

The ablation table reports approximate factors of 1.37 (cap randomization),
1.60 (global pooling), and 1.30 (auxiliary policy) in its Go setup. They are
short-run estimates, not multipliers we can multiply into a Push Chess forecast.
The run began at 6 blocks × 96 channels but still used up to 28 V100 GPUs and
19 days overall. Matching its first architecture does not match its resources.

### Regularized policy optimization: preserve the meaning of the search target

`2007.12509v1`, **Algorithmic benefits**, especially source lines 472–534.
Raw visit fractions quantize targets severely at small budgets and can lag
newly discovered Q information. The paper formulates a regularized improved
policy and distinguishes acting, searching, and learning with it.

Decision: retain completed-Q policy targets and validate low-budget behavior
against the Gumbel reference. This paper explains the design direction; it is
not the exact Gumbel algorithm. Parallel scheduling is an algorithm change
when it changes which evidence is available at selection time.

### Batch MCTS: cache evaluations, not necessarily search statistics

`2104.04278v1`, **Trees and Transposition Table**, **Virtual Mean**, algorithms,
and experimental setup; source lines 97–121 and 399 onwards.
The paper separates neural inferences in a transposition table from tree-local
statistics, and considers virtual means while building pending batches. Its
experiments use Go and a MobileNet evaluator.

Decision: first cache exact neural inputs/results under a model revision while
keeping repetition-sensitive paths distinct. Batch across independent games
before issuing multiple unresolved descents in a single tree. The latter needs
explicit pending-statistics semantics and strength tests, not just a faster loop.

### Reanalyse: spend existing compute on better targets, selectively

`2104.06294v1`, **Reanalyse**, **Reanalyse for Data Efficiency**; source lines
137–171 and 206–239. Repeated search with new network weights refreshes targets
on old observations. The paper varies data collection versus reanalysis at
fixed overall compute, and discusses stale/off-policy trajectories.

Decision: retain exact-history reconstruction and benchmark a capped policy
reanalysis fraction. The main value target stays the actual completed outcome.
Old outcomes are still samples from an older behavior policy; a retained label
is not magically a fresh current-policy value. A separate bootstrapped value
experiment would need explicit provenance, bias analysis, and an approval gate.

## Distribution and compute allocation

### Go-Exploit: simplest promising curriculum replacement

`2302.12359v2`, **Go-Exploit**, visited/search-state archive variants, experiments
and **Go-Exploit vs. KataGo** (source lines 157–170, 211 onwards, 270 onwards).
Start some self-play trajectories from previously encountered nonterminal
states. Shorter trajectories yield more independent outcomes and broaden the
states on which the value function is trained. Visited-state circular archives
are simpler than separate search-state archive actors.

Decision: compare standard starts against a bounded mixture of real visited
states. Store their entire rules history, not FEN alone. Keep a substantial
standard-start component and evaluate on standard starts. Evidence is Connect
Four/9×9 Go sample efficiency, not a measured local wall-clock advantage.

### Regret-Guided Search Control: promising, not free

`2602.20809v1`, `camera-ready.tex`: method; `appendix_camera-ready.tex`:
**Computational Cost Analysis**. It uses discrepancies between selected-action
search values and eventual outcomes, accumulated along the remaining trajectory,
to learn a regret value/ranking signal. Prioritized buffers select restart
states from played trajectories and search nodes.

The paper reports average gains of 77 Elo over AlphaZero and 89 over Go-Exploit
across 9×9 Go, 10×10 Othello and 11×11 Hex. Crucially, its appendix reports
**1.35× inference time and 1.25× iteration time for a 3-block model**, versus
about 1.03× for a 15-block model. Its experiments used four RTX A6000 GPUs.

Decision: test plain visited-state restarts first; then simple observed-error
prioritization; only then compare the actual learned ranking method. Observed
error prioritization is an inspired simplification, not a reproduction of RGSC.
Any implementation must resolve terminal indexing and value perspective in the
paper's regret equation explicitly, and must avoid repeatedly sampling noise.

### Scaling laws: choose a model/search pair, not a parameter-count winner

`2104.03113v2`, **Train-test trade-off**, discussion and implementation appendix.
Hex experiments show training and inference compute can substitute over a
bounded regime. Very small networks make non-network overhead important.
Do not extrapolate its fitted Elo/compute slopes or perfect-play thresholds to
Push Chess. Measure our own equal-time frontier.

`2412.11979v2`, **Connecting inverse scaling to the game structure**, source
lines 466 onwards. Checkers/Oware exhibit inverse scaling associated with
frequent, strategically less useful late-game states. The proposed explanation
is not a universal causal law, and the paper distinguishes these games from
ordinary chess. Our sparse-start draw rate is a reason to investigate the
training distribution—not evidence that this mechanism already occurs here.

Decision: compare 64×4, 96×6 and only later larger models at equal elapsed
training time and equal deployment time. Track outcome, turn, repetition and
state-frequency distributions. Do not enlarge the model because loss fell.

### Population-based training

`2003.06212v1`: metadata/abstract reviewed; retained as a later tuning reference.
A population adapts hyperparameters during training rather than fixing them
once. It increases concurrent training cost and complicates attribution.
Decision: start with a small predeclared sequential experiment grid; defer
population-based tuning until multiple affordable runs exist.

## Alternatives we should retain, not build all at once

### Graph search

`2012.11045v1`, **Monte-Carlo Graph Search**, data structure, backup and
discussion. The source incorporates a step counter to avoid cycles and notes
that history-dependent neural inputs can violate Markov assumptions even when
empirically tolerated in that implementation.

Decision: useful alternative, but not permission to merge our board-only hashes.
Repetition history, NN history, draw clocks, multiple-parent backups and solved
states require separate proofs. Reusing inferences captures a simpler subset
of the potential benefit without sharing search evidence across paths.

### MuZero / EfficientZero / EfficientZero V2 / Sampled MuZero

`1911.08265v2`: abstract/algorithm overview; `2111.00210v2`: **Method**;
`2403.00564v2`: **Method**; `2104.06303v1`: abstract/overview.
MuZero learns latent dynamics used during planning. EfficientZero adds
self-supervised temporal consistency, value-prefix prediction and correction
for stale replay. V2 extends sampled Gumbel planning to continuous actions and
uses search-based value estimation. Sampled MuZero addresses action spaces too
large to enumerate.

Decision: exact Rust dynamics are available and cheap relative to today's
inference path. Learning them adds model error and recurrent evaluation work.
Push Chess legal actions are enumerable, so sampled continuous-action machinery
is not the default. Self-supervised auxiliary learning can be tested separately
without importing learned dynamics. Do not confuse low environment sample count
with low total computation.

### Amortized transformer chess

`2402.04494v2`, **Data Preprocessing and Training**, predictors and discussion.
The method uses Stockfish 16 annotations, about 15.3 billion action-value data
points and models up to about 270 million parameters. Its FEN representation
explicitly omits repetition history. These results establish the possibility of
strong amortized decisions, not inexpensive pure self-play learning.

Decision: retain a small square-token transformer as a controlled architecture
alternative trained on our own data. No Stockfish or Cataclysm teacher labels.
It competes under the same wall-clock budget, including its Metal kernel cost.

### Search-contempt

`2504.07757v1`, **Experimental results**, **Compute-Efficient AlphaZero**,
**Plausible training schedule**. It changes assumptions about opponent search,
examines odds-chess strength, decisive/draw ratios and duplicate games. Some
distribution experiments use 100 games per setting. Its strong from-zero cost
claim is a proposed extrapolation, not a controlled demonstration of a full
cheap-from-zero replacement for AlphaZero.

Decision: exploratory only. More decisive games need not mean more informative
targets, and exploiting bounded opponents is not equivalent to minimax strength.
Do not add contempt or alter draw rewards in the main learner.

## Robustness and hardware

`2211.00241v4`, introduction, **Threat Model**, attack evaluations and limitations.
An adversary can beat a very strong Go system without being generally strong.
Some attacks persist at extremely large search budgets. The lesson is an
evaluation requirement: diverse opponents and targeted counterexample suites,
not a claim that Go's particular attack transfers to Push Chess.

`2411.13900v1`, abstract/introduction and predictor overview. Reverse-engineers
Firestorm and Oryon branch predictors and illustrates source-code/benchmark
pitfalls. Firestorm is not the M2 Max's performance-core microarchitecture.
For local allocation, layout and threading decisions use the regime-qualified
[bumblebench record](BUMBLEBENCH.md) and future in-situ measurements.

## Overall recommendation

Exact rules + small residual CNN + Gumbel improvement remains the best first
bet. First reduce data movement and per-query overhead, then improve the data
distribution and compute allocation. Global-context blocks and real-state
restarts are early learning experiments. Our additional original hypothesis is
an exact-move-effect policy head: share the rules engine's resolved collateral
piece movements with the network. It is not a result established by these
papers. The execution order is inference-first; see the full redesign for
separate baseline, global-context and effect-head comparisons. Learned dynamics, large
transformers, DAG statistics, regret heads and population training are separate
options with specific adoption gates, not a single giant architecture.
