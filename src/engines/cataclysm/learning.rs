//! Batched learning boundary for the SMALL residual inside classical search.
//! Shares feature identities and integer inference with the deployed engine.
//! No game storage and no changes to the immutable embedded control network.
use super::{
    board::Board,
    eval::baseline_white,
    network::{Accumulator, Network},
};
use crate::core::{position::Position, types::Color};

pub const FEATURES: usize = super::network::FEATURES;
pub const SLOTS: usize = 64;
pub const WIDTH: usize = super::network::WIDTH;
pub const CONTROL_BYTES: &[u8] = include_bytes!("network.bin");

pub struct Input {
    /// White and color/rank-reflected perspectives; FEATURES is zero padding.
    pub ids: [[u16; SLOTS]; 2],
    /// Unclamped, white-relative, handwritten score without the tempo bonus.
    pub baseline: i32,
}

pub fn input(pos: &Position) -> Input {
    let mut ids = [[FEATURES as u16; SLOTS]; 2];
    for (slot, (square, piece)) in pos
        .board
        .iter()
        .enumerate()
        .filter(|(_, p)| !p.is_empty())
        .enumerate()
    {
        for (perspective, row) in ids.iter_mut().enumerate() {
            row[slot] = super::network::feature_index(*piece, square as u8, perspective) as u16;
        }
    }
    Input {
        ids,
        baseline: baseline_white(&Board::new(pos)),
    }
}

/// A separate candidate, never installed into the running search implicitly.
pub struct Evaluator(Network);
impl Evaluator {
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        Network::decode(bytes).map(Self).map_err(|e| e.to_string())
    }
    pub fn evaluate(&self, pos: &Position) -> i32 {
        let mut acc = Accumulator::new(&self.0);
        for (square, piece) in pos.board.iter().enumerate().filter(|(_, p)| !p.is_empty()) {
            acc.update(*piece, square as u8, 1, &self.0);
        }
        let white = baseline_white(&Board::new(pos)) + acc.white_residual(&self.0) / 2;
        ((if pos.side_to_move == Color::White {
            white
        } else {
            -white
        }) + 14)
            .clamp(-28_000, 28_000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selfplay::State;
    #[test]
    fn learning_input_and_integer_reference_match_live_control() {
        let model = Evaluator::decode(CONTROL_BYTES).unwrap();
        assert!(Evaluator::decode(&CONTROL_BYTES[..100]).is_err());
        let words: Vec<_> = CONTROL_BYTES
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| i16::from_le_bytes(*b) as i32)
            .collect();
        let mut state = State::default();
        for ply in 0..96 {
            let pos = state.position();
            let row = input(pos);
            let hidden: [[i32; WIDTH]; 2] = std::array::from_fn(|p| {
                std::array::from_fn(|h| {
                    let mut value = words[FEATURES * WIDTH + h];
                    for &feature in &row.ids[p] {
                        if feature < FEATURES as u16 {
                            value += words[feature as usize * WIDTH + h];
                        }
                    }
                    value.clamp(0, 256)
                })
            });
            let sum: i64 = (0..WIDTH)
                .map(|h| {
                    i64::from(hidden[0][h] - hidden[1][h])
                        * i64::from(words[(FEATURES + 1) * WIDTH + h])
                })
                .sum();
            let white = row.baseline + (sum / 8192) as i32;
            let score = ((if pos.side_to_move == Color::White {
                white
            } else {
                -white
            }) + 14)
                .clamp(-28000, 28000);
            assert_eq!(score, model.evaluate(pos));
            assert_eq!(score, super::super::eval::evaluate::<0>(&Board::new(pos)));
            if state.legal_moves().is_empty() {
                state = State::default();
            }
            let mv = state.legal_moves()[(ply * 13 + 3) % state.legal_moves().len()].id();
            state.play(mv).unwrap();
        }
    }
}
