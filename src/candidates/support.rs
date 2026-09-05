//! Shared storage and ordering primitives, not shared engine policy.
use std::ops::{Deref, DerefMut};

use smallvec::SmallVec;

use crate::core::types::Move;

/// Shared scalar/SIMD ordering primitive. The slice boundary proves every load
/// in the architecture-specific implementation is within initialized storage.
pub(crate) fn find_max_index(scores: &[f32], start: usize, len: usize) -> usize {
    if len <= start {
        return start;
    }
    let remaining = &scores[start..len];
    #[cfg(target_arch = "aarch64")]
    if remaining.len() >= 4 {
        // SAFETY: AArch64 has baseline NEON; the slice contains at least 4 items.
        return start + unsafe { neon_max_index(remaining) };
    }
    let mut best = 0;
    for i in 1..remaining.len() {
        if remaining[i] > remaining[best] {
            best = i;
        }
    }
    start + best
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_max_index(scores: &[f32]) -> usize {
    use std::arch::aarch64::*;
    // SAFETY: The caller supplies >=4 initialized f32 values. Full-vector
    // loads cover only complete chunks; tail loads use checked slice indexing.
    unsafe {
        let chunks = scores.len() / 4;
        let ptr = scores.as_ptr();
        let mut maximum = vld1q_f32(ptr);
        for i in 1..chunks {
            maximum = vmaxq_f32(maximum, vld1q_f32(ptr.add(i * 4)));
        }
        let mut max_value = vmaxvq_f32(maximum);
        for &value in &scores[chunks * 4..] {
            if value > max_value {
                max_value = value;
            }
        }
        let target = vdupq_n_f32(max_value);
        for i in 0..chunks {
            let mask = vceqq_f32(vld1q_f32(ptr.add(i * 4)), target);
            let mut lanes = [0u32; 4];
            vst1q_u32(lanes.as_mut_ptr(), mask);
            if let Some(lane) = lanes.iter().position(|&value| value != 0) {
                return i * 4 + lane;
            }
        }
        scores[chunks * 4..]
            .iter()
            .position(|&v| v == max_value)
            .map_or(0, |i| chunks * 4 + i)
    }
}

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
        let mut best = from;
        for i in from + 1..self.len() {
            if self[i].score > self[best].score {
                best = i;
            }
        }
        self.0.swap(from, best);
    }

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
    fn max_index_handles_offsets_ties_and_simd_tails() {
        let scores = [100.0, -9.0, 2.0, 8.0, 8.0, 1.0, 10.0, 99.0];
        assert_eq!(find_max_index(&scores, 1, 6), 3);
        assert_eq!(find_max_index(&scores, 1, 7), 6);
        assert_eq!(find_max_index(&scores, 2, 3), 2);
        assert_eq!(find_max_index(&scores, 3, 3), 3);
    }

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
