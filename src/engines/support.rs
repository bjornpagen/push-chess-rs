//! Move-ordering storage shared by the active Astra family.
use crate::core::types::Move;
use smallvec::SmallVec;
use std::ops::{Deref, DerefMut};

/// A move and its score travel together, including when reordered.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScoredMove {
    pub mv: Move,
    pub score: i32,
}

/// Inline storage for typical positions, with safe spillover for larger ones.
/// There is no separately maintained length or uninitialized element access.
pub(crate) struct ScoredMoves<const INLINE: usize = 256>(SmallVec<[ScoredMove; INLINE]>);

impl<const INLINE: usize> ScoredMoves<INLINE> {
    pub fn new() -> Self {
        Self(SmallVec::new())
    }

    pub fn push(&mut self, mv: Move) {
        self.0.push(ScoredMove { mv, score: 0 });
    }

    /// Select the first maximum, preserving the engines' existing tie policy.
    pub fn pick_best(&mut self, from: usize) {
        crate::engine::shared::pick_best::<_, false>(&mut self.0, from, |m| m.score);
    }

    #[cfg(test)]
    pub fn selection_sort(&mut self) {
        for i in 0..self.len() {
            self.pick_best(i);
        }
    }
}

impl<const INLINE: usize> Deref for ScoredMoves<INLINE> {
    type Target = [ScoredMove];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const INLINE: usize> DerefMut for ScoredMoves<INLINE> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moves_keep_scores_and_spill_safely() {
        let mut moves = ScoredMoves::<2>::new();
        for i in 0..600 {
            moves.push(Move {
                from: (i % 64) as u8,
                ..Move::default()
            });
            moves[i].score = i as i32;
        }
        assert_eq!(moves.len(), 600);
        moves.selection_sort();
        for (i, entry) in moves.iter().enumerate() {
            assert_eq!(entry.score, (599 - i) as i32);
            assert_eq!(entry.mv.from, ((599 - i) % 64) as u8);
        }
    }

    #[test]
    fn ties_select_the_first_maximum() {
        let mut moves = ScoredMoves::<4>::new();
        for from in 0..4 {
            moves.push(Move {
                from,
                ..Move::default()
            });
        }
        moves[1].score = 10;
        moves[2].score = 10;
        moves.pick_best(0);
        assert_eq!(moves[0].mv.from, 1);
    }
}
