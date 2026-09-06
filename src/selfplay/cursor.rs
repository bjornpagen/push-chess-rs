//! A compact immutable game prefix plus one reversible search path.
use super::State;
use crate::core::position::Position;
use crate::core::types::Piece;

pub(super) struct Cursor {
    pub pos: Position,
    prefix_keys: Vec<u64>,
    previous: Option<[Piece; 64]>,
}

impl Cursor {
    pub fn from_state(state: &State) -> Self {
        Self {
            pos: state.pos.without_history(),
            prefix_keys: state.pos.undo_stack.iter().map(|u| u.zobrist).collect(),
            previous: state.pos.previous_board(),
        }
    }

    pub fn repetitions(&self) -> usize {
        self.prefix_keys
            .iter()
            .copied()
            .chain(self.pos.undo_stack.iter().map(|u| u.zobrist))
            .filter(|&key| key == self.pos.zobrist)
            .take(2)
            .count()
    }

    pub fn previous_board(&self) -> Option<[Piece; 64]> {
        self.pos.previous_board().or(self.previous)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn detached_prefix_preserves_history_and_local_undo() {
        let mut state = State::from_fen("7k/8/8/8/8/8/8/K7 w - - 0 1").unwrap();
        for (from, to) in [(0, 1), (63, 62), (1, 0), (62, 63)] {
            let id = state
                .legal_moves()
                .iter()
                .find(|m| m.from == from && m.to == to)
                .unwrap()
                .id();
            state.play(id).unwrap();
        }
        let mut cursor = Cursor::from_state(&state);
        assert!(cursor.pos.undo_stack.is_empty());
        assert_eq!(cursor.repetitions(), 1);
        assert_eq!(cursor.previous_board(), state.pos.previous_board());
        let m = state.prepared[0].clone();
        m.apply(&mut cursor.pos);
        state.play(m.mv().id()).unwrap();
        assert_eq!(cursor.previous_board(), state.pos.previous_board());
        assert_eq!(
            cursor.repetitions(),
            state
                .pos
                .undo_stack
                .iter()
                .filter(|u| u.zobrist == state.pos.zobrist)
                .take(2)
                .count()
        );
        cursor.pos.unmake_move();
        assert_eq!(cursor.repetitions(), 1);
    }
}
