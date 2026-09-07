pub mod control;
mod corpus;
mod relational;
mod roster;
mod runner;
pub mod schema;
mod statistics;

pub use corpus::{Corpus, CorpusGame, CorpusPage, digest_file};
pub use roster::{Candidate, Roster};
pub use runner::{RunConfig, generate, generate_controlled};

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;

use sha2::{Digest, Sha256};

pub use push_chess::core::rules::{RULES_VERSION, Rules};

/// Core generation appends; lab consumers need a replacement set each ply.
/// Keep that ownership contract in one boundary instead of every caller.
fn legal_moves(
    pos: &mut push_chess::core::position::Position,
    out: &mut Vec<push_chess::core::types::Move>,
) {
    out.clear();
    push_chess::core::movegen::generate_legal_moves(pos, out);
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ply {
    pub action: u32,
    /// Opening moves are not teacher recommendations.
    pub origin: &'static str,
    pub side: i32,
    pub score: Option<i32>,
    pub score_perspective: &'static str,
    pub search_complete: bool,
    pub nodes: u64,
    pub depth: u32,
    pub seldepth: u32,
    pub wall_us: i64,
    pub reported_us: i64,
    pub pv: Vec<u32>,
    pub diagnostics: push_chess::core::types::SearchDiagnostics,
    pub pieces: u32,
    pub halfmove_clock: u16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Trajectory {
    pub rules: Rules,
    pub index: usize,
    pub pair: Option<usize>,
    pub white: String,
    pub black: String,
    pub initial_fen: String,
    pub final_fen: String,
    pub opening_key: String,
    pub split: String,
    pub termination: String,
    /// Only rules-verified terminal outcomes. Caps/interruption/forfeits have
    /// no value label, not a synthetic draw or loss target.
    pub white_value: Option<i32>,
    pub plies: Vec<Ply>,
}

/// Read-only observations from the same exact-history validation pass.
/// An anomaly does not reinterpret a v1 move, result or training label.
#[derive(Default, Debug)]
pub struct ReplayAudit {
    pub castles: u64,
    /// Castling accepted while its ordinary one-square king transit is illegal.
    pub castling_transit_anomalies: Vec<usize>,
}

impl Trajectory {
    pub fn validate(&self) -> Result<ReplayAudit> {
        use push_chess::core::position::Position;
        use push_chess::game::{Outcome, adjudicate};
        if self.plies.len() > 4096 || self.pair != Some(self.index / 2) {
            return Err("invalid game identity or length".into());
        }
        let mut pos = Position::try_from_fen_with_rules(&self.initial_fen, self.rules)?;
        let mut legal = Vec::new();
        let mut audit = ReplayAudit::default();
        for (ply, record) in self.plies.iter().enumerate() {
            legal_moves(&mut pos, &mut legal);
            if adjudicate(&pos, &legal) != Outcome::Playing {
                return Err("moves after terminal position".into());
            }
            if record.side != pos.side_to_move as i32
                || record.pieces != pos.board.iter().filter(|p| !p.is_empty()).count() as u32
                || record.halfmove_clock != pos.halfmove_clock
            {
                return Err(format!("invalid observation at ply {ply}").into());
            }
            if record.score_perspective != "stm"
                || record.wall_us < 0
                || record.reported_us < 0
                || !matches!(record.origin, "opening" | "search")
                || (record.origin == "search") != record.score.is_some()
                || (record.origin == "opening" && record.search_complete)
            {
                return Err("invalid analysis provenance".into());
            }
            let mv = legal
                .iter()
                .find(|m| m.id() == record.action)
                .ok_or("illegal stored action")?;
            if mv.special == push_chess::core::types::SpecialMove::Castle {
                audit.castles += 1;
                let transit = push_chess::core::types::Move {
                    from: mv.from,
                    to: (mv.from + mv.to) / 2,
                    ..Default::default()
                };
                // Reuse the authoritative legal set already generated for
                // validation: no geometric substitute or second rules engine.
                if !legal.contains(&transit) {
                    audit.castling_transit_anomalies.push(ply);
                }
            }
            pos.make_move(mv);
        }
        legal_moves(&mut pos, &mut legal);
        let (ending, value) = terminal(&adjudicate(&pos, &legal));
        if pos.to_fen() != self.final_fen
            || value != self.white_value
            || (self.termination != ending
                && !(ending == "ply_limit" && self.termination == "interrupted"))
        {
            return Err("trajectory outcome or final position mismatch".into());
        }
        if self.termination != "interrupted" {
            let key = opening_key(&self.initial_fen, self.plies.iter().map(|p| p.action));
            if key != self.opening_key
                || (self.split != split(&key)
                    && !(self.split == "evaluation" && split(&key) == "test"))
            {
                return Err("opening family or split mismatch".into());
            }
        }
        Ok(audit)
    }
}

/// Group by the first four actual moves (including full knight routes), not
/// row IDs, teacher identity, or independent game seeds. Keep complete games
/// and their color-swapped pair together. Finer opening similarity remains
/// an explicit downstream split audit, not a claim made by this key.
pub(crate) fn opening_key(initial: &str, actions: impl Iterator<Item = u32>) -> String {
    let mut hash = Sha256::new();
    hash.update(initial.as_bytes());
    hash.update([0]);
    for action in actions.take(4) {
        hash.update(action.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

pub(crate) fn split(key: &str) -> &'static str {
    match u32::from_str_radix(&key[..8], 16).expect("SHA256 prefix") % 100 {
        0..=79 => "train",
        80..=89 => "validation",
        _ => "test",
    }
}

pub(crate) fn terminal(outcome: &push_chess::game::Outcome) -> (&'static str, Option<i32>) {
    use push_chess::core::types::Color;
    use push_chess::game::Outcome;
    match outcome {
        Outcome::Playing => ("ply_limit", None),
        Outcome::Checkmate {
            winner: Color::White,
        } => ("checkmate", Some(1)),
        Outcome::Checkmate {
            winner: Color::Black,
        } => ("checkmate", Some(-1)),
        Outcome::Stalemate => ("stalemate", Some(0)),
        Outcome::FiftyMove => ("50_move_rule", Some(0)),
        Outcome::Repetition => ("threefold_repetition", Some(0)),
    }
}
