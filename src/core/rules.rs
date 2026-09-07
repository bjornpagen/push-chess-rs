//! One rules identity for play, search, corpus replay and Python checkpoints.
//! Historical rules are explicit, never inferred from a FEN or silently upgraded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Rules {
    HistoryV1,
    #[default]
    HistoryV2,
}

pub const RULES_VERSION: &str = Rules::HistoryV2.name();

impl Rules {
    pub const fn name(self) -> &'static str {
        match self {
            Self::HistoryV1 => "push-chess-history-v1",
            Self::HistoryV2 => "push-chess-history-v2",
        }
    }

    pub fn parse(name: &str) -> Result<Self, &'static str> {
        match name {
            "push-chess-history-v1" | "push-chess-v1-history-castling" => Ok(Self::HistoryV1),
            "push-chess-history-v2" => Ok(Self::HistoryV2),
            _ => Err("unknown rules identity"),
        }
    }

    /// Disjoint transposition/repetition namespaces; v1 hashes stay unchanged.
    pub const fn hash_salt(self) -> u64 {
        match self {
            Self::HistoryV1 => 0,
            Self::HistoryV2 => 0x4528_21e6_38d0_1377,
        }
    }
}
