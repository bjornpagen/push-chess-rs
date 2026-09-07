# Preserved NNUE candidates

These are weight versions for the existing Cataclysm search, not new engines.
Neither is promoted or replaces the embedded control at
`src/engines/cataclysm/network.bin`.

| Candidate | Inference weights | Training checkpoint | Record |
|---|---|---|---|
| Aurora-r2 | [residual-r2.bin](residual-r2.bin) | [residual-r2.safetensors](residual-r2.safetensors) | [Round 2](../docs/nnue-round-2.md) |
| Aurora-r3 | [residual-r3.bin](residual-r3.bin) | [residual-r3.safetensors](residual-r3.safetensors) | [Round 3](../docs/nnue-round-3.md) |

Each inference export is 49,280 bytes for the existing 768-feature,
32-hidden-unit residual. The checkpoints contain model/optimizer tensors and
training provenance; the linked reports record their SHA256 identities.
All four files retain the exact bytes used for the reported comparisons.

The r2 checkpoint is an archived historical artifact and is not accepted by
the current trainer. Its exported inference weights remain usable as a control.
R3 matches the current checkpoint contract; its ancestry is still historical.
The original training corpus was purged, so exact sampling resume/reproduction
is no longer available. A future pass may initialize from r3 weights using
`--init`, with newly collected data. Use new output names; preserved files
must not be overwritten.

R3 improved same-sample validation loss by 1.11% versus r2 and scored 53.52%
across the two fixed small playing checks (256 games). That is encouraging,
not statistically conclusive evidence of stronger play. No additional games
or automatic promotion are implied by preserving these files.

Only these selected models, checkpoints and this README are allowlisted in
Git. Other experimental artifacts remain ignored. The local `data/corpus`
bumbledb database remains the sole game-fact store and is not included.
