//! One current rules contract for play, search and model checkpoints.
//! No historical replay modes are supported by the active codebase.
pub const RULES_VERSION: &str = "push-chess-history-v2";

/// Preserve the hashes already used by the corrected-rules corpus.
pub(crate) const POSITION_HASH_SALT: u64 = 0x4528_21e6_38d0_1377;
