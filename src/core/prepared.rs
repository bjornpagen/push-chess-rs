//! Rules-only moves whose push resolution survives generation and legality.
//! Construction/application stay inside the rules/search implementation: a
//! plan is valid at its generating origin, not at arbitrary caller positions.
use super::movegen::generate_into;
use super::position::Position;
use super::push::PushPlan;
use super::types::{Move, Piece, PieceType, SpecialMove, file_of, make_square, rank_of};

#[derive(Clone, Debug)]
pub struct PreparedMove {
    pub(crate) mv: Move,
    pub(crate) plan: Option<PushPlan>,
}

impl PreparedMove {
    pub fn mv(&self) -> Move {
        self.mv
    }

    pub(crate) fn apply(&self, pos: &mut Position) {
        pos.make_resolved(&self.mv, self.plan.as_ref());
    }

    /// Exact board effect with no legality/hash/history work. This delegates
    /// ordinary simultaneous movement/promotion to the same primitive as make.
    pub(crate) fn board_after(&self, pos: &Position) -> [Piece; 64] {
        let mut board = pos.board;
        match self.mv.special {
            SpecialMove::Castle => {
                let rank = rank_of(self.mv.from);
                let (from, to) = if file_of(self.mv.to) == 6 {
                    (7, 5)
                } else {
                    (0, 3)
                };
                let (rook_from, rook_to) = (make_square(rank, from), make_square(rank, to));
                board[self.mv.to as usize] = board[self.mv.from as usize];
                board[self.mv.from as usize] = Piece::default();
                board[rook_to as usize] = board[rook_from as usize];
                board[rook_from as usize] = Piece::default();
            }
            SpecialMove::EnPassant => {
                board[self.mv.to as usize] = board[self.mv.from as usize];
                board[self.mv.from as usize] = Piece::default();
                board[make_square(rank_of(self.mv.from), file_of(self.mv.to)) as usize] =
                    Piece::default();
            }
            _ => self.plan.as_ref().expect("generated push").apply_promoting(
                &mut board,
                pos.side_to_move,
                self.mv.promo_piece,
            ),
        }
        board
    }

    /// Generate canonical, joint (action+1, square, before, after) tokens.
    /// There are at most 64 changed squares, a proven board-size bound.
    pub(crate) fn effects(&self, pos: &Position, action: usize, out: &mut Vec<[i32; 4]>) {
        let after = self.board_after(pos);
        let us = pos.side_to_move as usize;
        let flip = if us == 0 { 0 } else { 56 };
        let code = |p: Piece| {
            if p.piece_type == PieceType::None {
                0
            } else {
                1 + (p.color as usize ^ us) as i32 * 6 + p.piece_type as i32 - 1
            }
        };
        for (sq, (&before, &after)) in pos.board.iter().zip(&after).enumerate() {
            if before != after {
                out.push([
                    (action + 1) as i32,
                    (sq ^ flip) as i32,
                    code(before),
                    code(after),
                ]);
            }
        }
    }
}

/// Reusable pseudo-legal scratch. No fixed action-count assumptions.
#[derive(Clone, Default)]
pub struct MoveScratch {
    candidates: Vec<PreparedMove>,
}

pub fn generate_prepared(
    pos: &mut Position,
    out: &mut Vec<PreparedMove>,
    scratch: &mut MoveScratch,
) {
    scratch.candidates.clear();
    generate_into(pos, &mut scratch.candidates);
    let us = pos.side_to_move;
    for candidate in scratch.candidates.drain(..) {
        candidate.apply(pos);
        let legal = !pos.in_check_color(us);
        pos.unmake_move();
        if legal {
            out.push(candidate);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{
        movegen::{generate_legal_moves, generate_legal_moves_reference},
        position::start_position,
    };

    fn check(pos: &mut Position, scratch: &mut MoveScratch) -> Vec<PreparedMove> {
        let fen = pos.to_fen();
        let key = pos.zobrist;
        let depth = pos.undo_stack.len();
        let (mut reference, mut direct, mut prepared) = (Vec::new(), Vec::new(), Vec::new());
        generate_legal_moves_reference(pos, &mut reference);
        generate_legal_moves(pos, &mut direct);
        generate_prepared(pos, &mut prepared, scratch);
        assert_eq!(reference, direct);
        assert_eq!(
            reference,
            prepared.iter().map(PreparedMove::mv).collect::<Vec<_>>()
        );
        assert_eq!(pos.to_fen(), fen);
        assert_eq!(pos.zobrist, key);
        assert_eq!(pos.undo_stack.len(), depth);
        for (a, m) in prepared.iter().enumerate() {
            let expected = m.board_after(pos);
            let mut effects = Vec::new();
            m.effects(pos, a, &mut effects);
            assert!(effects.len() <= 64);
            assert!(effects.iter().all(|t| t[0] == (a + 1) as i32));
            let mut ordinary = pos.clone();
            ordinary.make_move(&m.mv());
            m.apply(pos);
            assert_eq!(pos.board, expected);
            assert_eq!(pos.board, ordinary.board);
            assert_eq!(pos.to_fen(), ordinary.to_fen());
            let incremental = pos.zobrist;
            pos.compute_zobrist();
            assert_eq!(pos.zobrist, incremental);
            pos.unmake_move();
            assert_eq!(pos.to_fen(), fen);
            assert_eq!(pos.zobrist, key);
        }
        prepared
    }

    #[test]
    fn prepared_matches_reference_on_fixtures_and_reachable_trace() {
        let mut scratch = MoveScratch::default();
        for fen in [
            "7k/P7/R7/8/8/8/8/K7 w - - 0 1",
            "7k/8/8/4R3/4N3/8/8/K7 w - - 0 1",
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "7k/8/8/3pP3/8/8/8/K7 w - d6 0 1",
        ] {
            check(&mut Position::try_from_fen(fen).unwrap(), &mut scratch);
        }
        let mut pos = start_position();
        let mut seed = 0xcafe_f00du64;
        for _ in 0..192 {
            let moves = check(&mut pos, &mut scratch);
            if moves.is_empty() {
                pos = start_position();
                continue;
            }
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            moves[seed as usize % moves.len()].apply(&mut pos);
        }
    }

    #[test]
    fn exact_effect_tokens_include_both_ends_of_a_simple_move() {
        let mut pos = Position::try_from_fen("7k/8/8/8/8/8/8/K7 w - - 0 1").unwrap();
        let mut moves = Vec::new();
        generate_prepared(&mut pos, &mut moves, &mut MoveScratch::default());
        let m = moves
            .iter()
            .find(|m| m.mv.from == 0 && m.mv.to == 1)
            .unwrap();
        let mut tokens = Vec::new();
        m.effects(&pos, 0, &mut tokens);
        assert_eq!(tokens, [[1, 0, 6, 0], [1, 1, 0, 6]]);
    }
}
