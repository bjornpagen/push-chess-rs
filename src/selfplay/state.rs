use crate::core::position::{Position, start_position};
use crate::core::prepared::{MoveScratch, PreparedMove, generate_prepared};
use crate::core::types::{Color, Move};
use crate::game::{Outcome, adjudicate};

/// A validated position plus the legal-action proof used by all its consumers.
#[derive(Clone)]
pub struct State {
    pub(super) pos: Position,
    pub(super) legal: Vec<Move>,
    pub(super) prepared: Vec<PreparedMove>,
    scratch: MoveScratch,
    outcome: Outcome,
}

impl Default for State {
    fn default() -> Self {
        Self::from_position(start_position())
    }
}

impl State {
    pub fn from_fen(fen: &str) -> Result<Self, String> {
        Position::try_from_fen(fen)
            .map(Self::from_position)
            .map_err(|e| e.to_string())
    }
    fn from_position(pos: Position) -> Self {
        let mut state = Self {
            pos,
            legal: Vec::new(),
            prepared: Vec::new(),
            scratch: MoveScratch::default(),
            outcome: Outcome::Playing,
        };
        state.refresh();
        state
    }
    fn refresh(&mut self) {
        self.legal.clear();
        self.prepared.clear();
        generate_prepared(&mut self.pos, &mut self.prepared, &mut self.scratch);
        self.prepared.sort_unstable_by_key(|m| m.mv().id());
        self.legal
            .extend(self.prepared.iter().map(PreparedMove::mv));
        self.outcome = adjudicate(&self.pos, &self.legal);
        if self.outcome != Outcome::Playing {
            self.legal.clear();
            self.prepared.clear();
        }
    }
    pub fn position(&self) -> &Position {
        &self.pos
    }
    pub fn legal_moves(&self) -> &[Move] {
        &self.legal
    }
    pub fn outcome(&self) -> &Outcome {
        &self.outcome
    }
    pub fn white_value(&self) -> Option<f32> {
        white_value(&self.outcome)
    }
    pub fn play(&mut self, id: u32) -> Result<(), &'static str> {
        let index = self
            .legal
            .binary_search_by_key(&id, |m| m.id())
            .map_err(|_| "move ID is not legal in this position")?;
        self.prepared[index].apply(&mut self.pos);
        self.refresh();
        Ok(())
    }
}

pub(super) fn white_value(outcome: &Outcome) -> Option<f32> {
    match outcome {
        Outcome::Playing => None,
        Outcome::Checkmate {
            winner: Color::White,
        } => Some(1.0),
        Outcome::Checkmate {
            winner: Color::Black,
        } => Some(-1.0),
        _ => Some(0.0),
    }
}
