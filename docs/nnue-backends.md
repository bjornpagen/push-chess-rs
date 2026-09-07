# NNUE backend decision

One model contract, different work: the differentiable trainer must calculate
gradients and optimizer updates; search needs tiny, exact, changed-piece integer
updates. A language label does not decide either workload's winner.

## Measurement, 2026-09-06

Apple M2 Max, 12 cores, macOS 15.7.7, tinygrad 0.14.0,
rustc 1.100.0-nightly (bff8e12ff 2026-08-26). The 12-worker corpus tournament
was active throughout. These are contended observations, **not isolated
throughput claims or promised whole-engine speedups**.

`uv run --no-sync pushzero nnue benchmark --batches 1 32 256 1024 --repeats 15`
was followed by a 51-repetition confirmation of batches 256 and 1024.
Four warmups precede timing; arm order alternates and four input banks rotate.
Frozen weights are decoded/uploaded once outside the timed region for all arms.
Inference returns synchronized host scores and asserts exact agreement. CPU
inference rebuilds sparse accumulators using the actual engine kernel; it does
not time the even smaller changed-piece update used in recursive search.
Tinygrad CPU used its configured 12 tensor threads; native batch inference uses
one thread. Neither arm is scaled to an invented equal FLOP count.

Median microseconds per batch (51-repetition confirmation):

| Work | Batch | Native CPU | tinygrad CPU | tinygrad Metal |
|---|---:|---:|---:|---:|
| Frozen inference, host inputs/output | 256 | 109 | 6,195 | 7,604 |
| Frozen inference, tensor inputs resident, host output | 256 | 109 | 3,205 | 1,327 |
| Frozen inference, host inputs/output | 1024 | 258 | 23,193 | 11,234 |
| Frozen inference, tensor inputs resident, host output | 1024 | 258 | 12,993 | 1,427 |
| Complete QAT update | 256 | — | 10,638 | 6,589 |
| Complete QAT update | 1024 | — | 29,957 | 7,547 |

Native inference uses the same host sparse inputs in both rows; the resident
row is a favorable tensor lower bound, not a claim that native executes on GPU.
Training includes forward, backward, gradient clipping, AdamW, tensor transfer
and synchronized loss/norm readback. Common batch sampling/dense preparation is
outside that training timer. There is no separate handwritten Rust optimizer to
compare; the native column is deliberately absent rather than fabricated.

The first sweep's singleton inference medians were 7.3 µs native versus 961 µs
for resident-input Metal (5,145 µs including tensor preparation). Small training
batches reversed the device preference: CPU/Metal medians were 4.0/5.3 ms at
batch 1 and 4.4/7.3 ms at batch 32. Do not extrapolate those thresholds to an idle
machine: at batch 256 in the confirmation, training p10–p90 spanned 6.5–16.1 ms
on CPU and 4.9–11.0 ms on Metal. At 1024 it spanned 20.1–38.6 and 5.9–13.2 ms.
Remeasure when the machine is idle, particularly before acting on small gaps.

## Decision and removal of duplicates

- Keep the actual Rust accumulator for tree inference and frozen batch scoring.
  It wins even against the resident-input tinygrad forward in these measurements.
- Use tinygrad Metal for the planned 256/1024-position training work. CPU training
  remains explicitly selectable for tiny batches; no failing backend silently
  changes to another one.
- Expose one `pushzero.nnue.Model` and one `pushzero nnue` command namespace.
  Training uses its differentiable forward; `freeze()` creates an immutable
  deployed backend once for repeated inference, while `scores()` freezes once
  per complete validation pass. Exported artifacts use one Rust-owned contract.
- Remove the learning-only Rust evaluator, production NumPy integer evaluator,
  duplicated format constants, and separate `nnue-export` command. The NumPy
  arithmetic oracle remains only in tests so a shared-constant bug can be caught.

This is a choice among measured implementations, not a theorem about the fastest
possible kernel. The comparison is kept runnable to challenge this decision.
It opens no corpus, creates no persistent games, and saves no training weights.

## Confirmation after Round 2, 2026-09-07

The tournament had exited before this measurement, and no audit, training run
or arena was active. Other application/background work, including a Rust build
in another project, remained; this is lower contention, not machine isolation.
Same hardware/runtime and interleaved harness, 31 repetitions at the default
12 CPU tensor threads:

| Work | Batch | Native CPU µs | tinygrad CPU µs | tinygrad Metal µs |
|---|---:|---:|---:|---:|
| Frozen inference, host inputs/output | 256 | 95.3 | 4,233.0 | 3,429.6 |
| Frozen inference, tensor inputs resident, host output | 256 | 95.3 | 1,858.9 | 980.9 |
| Frozen inference, host inputs/output | 1024 | 238.8 | 11,610.7 | 6,058.8 |
| Frozen inference, tensor inputs resident, host output | 1024 | 238.8 | 6,766.9 | 1,006.7 |
| Complete QAT update | 256 | — | 5,926.8 | 3,876.0 |
| Complete QAT update | 1024 | — | 16,364.8 | 4,607.9 |

Separate 15-repetition processes with `NUM_CPU_THREADS=1`, `4`, and `8` all
retained the same backend winner. CPU update medians ranged 5.84–5.94 ms at
256 and 15.61–15.71 ms at 1024; corresponding Metal medians were 3.82–3.93
and 4.51–4.53 ms. Those processes were sequential, not a bracketed thread-count
comparison, so small differences do not establish an optimal CPU thread count.

The production decision remains native inference and Metal training at these
batch sizes. Metal's complete update is about 1.5× faster at 256 and 3.6× at
1024 in the default-thread confirmation. Native wins even against the favorable
resident-tensor inference bound. Training performance and playing strength
remain separate questions; larger batches are not interchangeable optimizer
regimes merely because they process more examples per second.
