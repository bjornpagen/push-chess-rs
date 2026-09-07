//! Authoritative game outcomes. No UI, animation, session or browser state.
use crate::core::position::Position;
use crate::core::types::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum Outcome {
    Playing,
    Checkmate { winner: Color },
    Stalemate,
    FiftyMove,
    Repetition,
}

/// Mate/stalemate take precedence over the move clock or repetition rule.
pub fn adjudicate(pos: &Position, legal: &[Move]) -> Outcome {
    let repeats = pos
        .undo_stack
        .iter()
        .filter(|u| u.zobrist == pos.zobrist)
        .take(2)
        .count();
    adjudicate_with_repetitions(pos, legal.is_empty(), repeats)
}

pub(crate) fn adjudicate_with_repetitions(
    pos: &Position,
    no_legal_moves: bool,
    repeats: usize,
) -> Outcome {
    if no_legal_moves {
        return if pos.in_check() {
            Outcome::Checkmate {
                winner: opponent(pos.side_to_move),
            }
        } else {
            Outcome::Stalemate
        };
    }
    if pos.halfmove_clock >= 100 {
        return Outcome::FiftyMove;
    }
    if repeats >= 2 {
        Outcome::Repetition
    } else {
        Outcome::Playing
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::movegen::generate_legal_moves;
    #[test]
    fn mate_precedes_fifty_move_rule() {
        let mut p = Position::try_from_fen("7k/6Q1/5K2/8/8/8/8/8 b - - 100 1").unwrap();
        let mut moves = Vec::new();
        generate_legal_moves(&mut p, &mut moves);
        assert_eq!(
            adjudicate(&p, &moves),
            Outcome::Checkmate {
                winner: Color::White
            }
        );
    }
    #[test]
    fn nonterminal_and_stalemate_are_distinct() {
        let mut p = Position::try_from_fen("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1").unwrap();
        let mut moves = Vec::new();
        generate_legal_moves(&mut p, &mut moves);
        assert_eq!(adjudicate(&p, &moves), Outcome::Stalemate);
    }
}
