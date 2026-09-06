# Rolling self-play and inference scheduling

## Representation and invariants

Games, searches, and device batches have different lifetimes. A game occupies a
stable actor slot until an actual outcome or the configured ply cap. A search
job owns a small group of actors for **one move**. A device batch leases ready
rows from independent jobs, and owns an exact reply-routing table. Neither a
training collection boundary nor a GPU batch boundary ends a game.

- One bounded Rust worker pool; owned reusable search arenas move through its
  work queue. No per-game threads, per-edge locks, or spinning.
- Native lane states are Idle, Working, Ready, and Leased. A unique request ID
  names a reply lease. Validate all reply rows before resuming any lane.
- Actor count, search-group size, inference batch size, and learner batch size
  are independent. Ready work beyond one device batch remains queued. Returning
  a batch wakes its CPU searches while the GPU serves other independent jobs.
- One Python/tinygrad device owner. Do not use competing training processes or
  claim asynchronous zero-copy from NumPy. Metal's current copy-in/out paths
  synchronize; no undocumented buffer aliasing or disabled M2 safety workaround.
- Collection stops admitting new **moves** at its quota, then drains only the
  in-flight moves. It retains unfinished games, targets, and complete histories
  across learning/checkpoint boundaries. All searches use one weight revision;
  targets record their originating optimizer step, even in mixed-revision games.
- Pending targets store ply references, legal IDs, policy, and prediction—not
  full board/action tensors. A pickle-free, checksummed actor sidecar is committed
  with model, optimizer, replay, RNG, and pending-update metadata.
- Errors fail explicitly. There is one production search scheduler; the serial
  tree implementation remains a correctness reference, not a runtime fallback.

## Implementation and acceptance

1. Stop and checkpoint the old trainer (done: iteration 8, step 358; 26 pending
   updates preserved).
2. Replace the round-barrier runtime with independently leased search jobs and
   a thin bulk Python interface. Test stale/reordered replies, ownership, bounded
   capacity, failure, shutdown, and serial-policy parity.
3. Implement persistent actor slots and compact unfinished targets; checkpoint
   and restore them exactly. Test no population drain, no invented terminal
   labels, weight provenance, interruption, and resume-before-new-data ordering.
4. Measure frozen-model workloads at 32/64/128/256 actors and independent device
   batch sizes. Separate cold compilation from warm useful-row/move throughput;
   include identity controls and report actual memory, batching, and GPU counters
   where available. Higher utilization alone is not success.
5. Commit and push to main before any training smoke test, then verify resumed
   learning and checkpoint recovery. Push measured results before the long run.

This changes scheduling, not the policy/value network or search targets. The
previous immutable checkpoints remain usable through an explicit resume config
override; old runs have no unfinished-actor sidecar to restore.
