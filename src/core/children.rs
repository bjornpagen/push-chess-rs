//! Borrowed child positions: the board is restored when each child is dropped.
use std::ops::Deref;

use super::movegen::generate_pseudo_legal_moves;
use super::position::Position;
use super::types::Move;

/// Unlike `Iterator`, each item may borrow the cursor for just one call.
pub trait LendingIterator {
    type Item<'a>
    where
        Self: 'a;
    fn next(&mut self) -> Option<Self::Item<'_>>;
}

/// Visits pseudo-legal children without cloning the position. The GAT prevents
/// advancing while a child still borrows the same board.
///
/// ```compile_fail
/// use push_chess::core::children::{LendingIterator, PseudoLegalChildren};
/// use push_chess::core::position::start_position;
/// let mut position = start_position();
/// let mut children = PseudoLegalChildren::new(&mut position);
/// let first = children.next().unwrap();
/// let second = children.next(); // cannot borrow the board twice
/// println!("{}", first.to_fen());
/// ```
pub struct PseudoLegalChildren<'p> {
    position: &'p mut Position,
    moves: std::vec::IntoIter<Move>,
}

impl<'p> PseudoLegalChildren<'p> {
    pub fn new(position: &'p mut Position) -> Self {
        let mut moves = Vec::new();
        generate_pseudo_legal_moves(position, &mut moves);
        Self {
            position,
            moves: moves.into_iter(),
        }
    }
}

impl LendingIterator for PseudoLegalChildren<'_> {
    type Item<'a>
        = ChildPosition<'a>
    where
        Self: 'a;

    fn next(&mut self) -> Option<Self::Item<'_>> {
        let mv = self.moves.next()?;
        self.position.make_move(&mv);
        Some(ChildPosition {
            position: self.position,
            mv,
        })
    }
}

/// Read-only access keeps callers from unmaking the guard's move themselves.
#[must_use = "dropping the child immediately restores its parent position"]
pub struct ChildPosition<'p> {
    position: &'p mut Position,
    mv: Move,
}

impl ChildPosition<'_> {
    pub fn mv(&self) -> Move {
        self.mv
    }
}

impl Deref for ChildPosition<'_> {
    type Target = Position;
    fn deref(&self) -> &Position {
        self.position
    }
}

impl Drop for ChildPosition<'_> {
    fn drop(&mut self) {
        self.position.unmake_move();
    }
}
