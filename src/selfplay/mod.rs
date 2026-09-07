//! Rules-only neural search. No Python, GPU runtime, or engine evaluation here.
//! The caller supplies evaluations at explicit batch boundaries.
mod cursor;
mod encoding;
mod runtime;
mod search;
mod state;

pub use encoding::{ACTION_FIELDS, BOARD_FLOATS, EFFECT_FIELDS, Encoded, Features};
pub use runtime::SearchRuntime;
pub use search::{
    BatchSearch, SearchMetrics, SearchOptions, SearchResult, SearchRoot, considered_visits,
};
pub use state::State;

pub use crate::core::rules::RULES_VERSION;
pub const ENCODING_VERSION: u32 = 2;
pub const EFFECT_ENCODING_VERSION: u32 = 1;
