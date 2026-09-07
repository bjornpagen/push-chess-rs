//! Batched learning boundary for the SMALL residual inside classical search.
//! Shares feature identities and integer inference with the deployed engine.
//! No game storage and no changes to the immutable embedded control network.
use super::{
    board::Board,
    eval::{baseline_white, relative_score},
    network::{Accumulator, Model},
};
use crate::core::{position::Position, types::Color};

pub use super::network::{
    FEATURES, HIDDEN_CLIP, RESIDUAL_DIVISOR, SCORE_LIMIT, SLOTS, TEMPO, WIDTH,
};

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
        baseline: baseline_white(&Board::handwritten(pos)),
    }
}

impl Model {
    /// Sparse whole-batch inference uses the deployed accumulator kernel and
    /// score conversion. The caller supplies the shared handwritten baseline.
    /// This is the same backend as search, not a separate integer evaluator.
    pub fn score_features(
        &self,
        ids: &[[u16; SLOTS]; 2],
        baseline: i32,
        side: Color,
    ) -> Result<i32, String> {
        // Bounds also make externally supplied baselines safe to add/negate.
        if baseline.unsigned_abs() > (1 << 23) {
            return Err("baseline outside exact NNUE input range".into());
        }
        let model = self.network();
        let mut acc = Accumulator::new(model);
        for (perspective, row) in ids.iter().enumerate() {
            let mut seen = [0u64; FEATURES / 64];
            for &id in row {
                let id = usize::from(id);
                if id == FEATURES {
                    continue;
                }
                if id > FEATURES {
                    return Err("invalid NNUE feature ID".into());
                }
                let mask = 1u64 << (id % 64);
                if seen[id / 64] & mask != 0 {
                    return Err("duplicate NNUE feature ID".into());
                }
                seen[id / 64] |= mask;
                acc.update_feature(perspective, id, 1, model);
            }
        }
        Ok(relative_score(baseline + acc.white_residual(model), side))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selfplay::State;
    #[test]
    fn learning_input_and_integer_reference_match_live_control() {
        let model = Model::embedded();
        assert!(Model::decode(&Model::CONTROL_BYTES[..100]).is_err());
        let words: Vec<_> = Model::CONTROL_BYTES
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
            assert_eq!(
                score,
                model
                    .score_features(&row.ids, row.baseline, pos.side_to_move)
                    .unwrap()
            );
            assert_eq!(score, super::super::eval::evaluate::<0>(&Board::new(pos)));
            if state.legal_moves().is_empty() {
                state = State::default();
            }
            let mv = state.legal_moves()[(ply * 13 + 3) % state.legal_moves().len()].id();
            state.play(mv).unwrap();
        }
    }
}
