# Reusable rolling self-play runtime

The inference-first runtime remains available for later neural experiments.
This phase's durable producer is the native engine laboratory; its sole game
store is bumbledb. The old file-based training runner and replay formats have
been retired.

## Ownership and scheduling

Rust owns board history, prepared moves, search arenas and persistent worker
threads. Python owns Metal execution, network/optimizer tensors, and the
bounded in-memory replay sampler. Work crosses the boundary in batches:
board/action/effect arrays transfer native allocation ownership to NumPy;
there is no Python callback for each search node.

A fixed population of rolling actor slots advances independently. Finished
games are replaced immediately rather than draining the population at every
collection quota. Targets retain their actual policy-step provenance.
Collection quotas stop admitting new work at move boundaries. They do not
turn unfinished games into draws or discard their pending targets.

Independent lanes lease tokenized inference requests. Replies may arrive out
of order; duplicate, stale and foreign tokens are refused. Node/edge capacity
exhaustion is explicit. A failure shuts down the runtime rather than silently
reducing search or dropping legal actions.

Actor snapshots are owned in-memory representations that reconstruct and
validate exact action history. They have no durable file format. When a
production neural collector is added, game and policy facts must enter the
same bumbledb model used by the engine tournament.

## What to measure

Measure completed moves/second, legal-action width distributions, request
sizes, useful Metal rows, cache hits, native worker time, arena capacity and
the slow tail. Profile inference kernels only when they dominate end-to-end
time; optimizing CPU search while Metal is dominant is not the objective.
Use fixed-weight A/B/identity controls and distinguish warmup from steady
state. No hardware-saturation or playing-strength claim follows from unit
tests alone.
