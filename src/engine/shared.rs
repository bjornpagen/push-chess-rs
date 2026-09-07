//! Shared search mechanisms, not shared strategy. Engines choose margins,
//! reductions, replacement policy and evaluation independently.
use crate::core::types::SearchBudget;
use std::ops::{Index, IndexMut};
use std::time::Instant;

pub struct SearchLimits {
    pub started: Instant,
    pub micros: u128,
    pub limit: u64,
    pub nodes: u64,
    pub seldepth: usize,
    pub stopped: bool,
}

impl Default for SearchLimits {
    fn default() -> Self {
        Self {
            started: Instant::now(),
            micros: 0,
            limit: 0,
            nodes: 0,
            seldepth: 0,
            stopped: false,
        }
    }
}
impl SearchLimits {
    pub fn begin(&mut self, budget: &SearchBudget, usable_percent: u128) {
        assert!(usable_percent > 0 && usable_percent <= 100);
        self.started = Instant::now();
        self.micros = if budget.max_time_us > 0 {
            (budget.max_time_us as u128 * usable_percent / 100).max(1)
        } else {
            0
        };
        self.limit = budget.max_nodes.max(0) as u64;
        self.nodes = 0;
        self.seldepth = 0;
        self.stopped = false;
    }
    /// The two search families used this same 128-node clock cadence. Keep
    /// the counter and its stop transition together, including node-only runs.
    pub fn tick(&mut self, ply: usize) -> bool {
        if self.stopped {
            return true;
        }
        if (self.limit > 0 && self.nodes >= self.limit)
            || (self.micros > 0
                && self.nodes & 127 == 0
                && self.started.elapsed().as_micros() >= self.micros)
        {
            self.stopped = true;
            return true;
        }
        self.nodes += 1;
        self.seldepth = self.seldepth.max(ply);
        false
    }
}

/// Contiguous fixed-width buckets. The policy for probing/replacement stays
/// in each engine. No alignment/prefetch trick is assumed to be a win.
pub struct Buckets<E: Copy, const WAYS: usize>(Vec<[E; WAYS]>);
impl<E: Copy + Default, const WAYS: usize> Buckets<E, WAYS> {
    pub fn new(bytes: usize) -> Self {
        let width = std::mem::size_of::<[E; WAYS]>();
        assert!(width > 0 && bytes.is_multiple_of(width));
        let count = bytes / width;
        assert!(count.is_power_of_two());
        Self(vec![[E::default(); WAYS]; count])
    }
}
impl<E: Copy, const WAYS: usize> Buckets<E, WAYS> {
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn fill(&mut self, value: [E; WAYS]) {
        self.0.fill(value);
    }
}
impl<E: Copy, const WAYS: usize> Index<usize> for Buckets<E, WAYS> {
    type Output = [E; WAYS];
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
impl<E: Copy, const WAYS: usize> IndexMut<usize> for Buckets<E, WAYS> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

/// Tie policy is explicit: historical Cataclysm chooses the last maximum,
/// Astra the first. Sharing storage must not silently change search order.
/// Scores must be stable during selection. Each remaining score is read once;
/// keep the running maximum in a register, not behind a data-dependent index.
/// Panics if `from` is not a valid item index.
pub fn pick_best<T, const LAST: bool>(items: &mut [T], from: usize, score: impl Fn(&T) -> i32) {
    let mut best = from;
    let mut high = score(&items[from]);
    for (i, item) in items.iter().enumerate().skip(from + 1) {
        let value = score(item);
        if value > high || (LAST && value == high) {
            best = i;
            high = value;
        }
    }
    items.swap(from, best);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn node_limit_is_exact_and_stop_is_sticky() {
        let mut limit = SearchLimits::default();
        limit.begin(
            &SearchBudget {
                max_nodes: 2,
                ..SearchBudget::default()
            },
            94,
        );
        assert!(!limit.tick(1));
        assert!(!limit.tick(2));
        assert!(limit.tick(3));
        assert_eq!(limit.nodes, 2);
        assert_eq!(limit.seldepth, 2);
        assert!(limit.tick(0));
    }
    #[test]
    fn tie_policy_is_preserved() {
        let mut first = [(4, 0), (4, 1)];
        let mut last = first;
        pick_best::<_, false>(&mut first, 0, |x| x.0);
        pick_best::<_, true>(&mut last, 0, |x| x.0);
        assert_eq!(first[0].1, 0);
        assert_eq!(last[0].1, 1);
    }

    #[test]
    fn selection_preserves_exact_suffix_permutations_for_both_tie_policies() {
        fn check<const LAST: bool>() {
            // Exhaust tie-heavy lists, signed boundaries, arbitrary suffixes
            // and repeated selections. Comparing scores alone misses ties.
            for length in 1..=8 {
                for code in 0usize..3usize.pow(length as u32) {
                    let mut digits = code;
                    let original: Vec<_> = (0..length)
                        .map(|id| {
                            let score = [i32::MIN, 0, i32::MAX][digits % 3];
                            digits /= 3;
                            (score, id)
                        })
                        .collect();
                    let mut expected = original.clone();
                    let mut actual = original.clone();
                    for from in 0..length {
                        crate::engine::ordering_probe::historical::<_, LAST>(
                            &mut expected,
                            from,
                            |x| x.0,
                        );
                        pick_best::<_, LAST>(&mut actual, from, |x| x.0);
                        assert_eq!(actual, expected);
                        let mut suffix_expected = original.clone();
                        let mut suffix_actual = original.clone();
                        crate::engine::ordering_probe::historical::<_, LAST>(
                            &mut suffix_expected,
                            from,
                            |x| x.0,
                        );
                        pick_best::<_, LAST>(&mut suffix_actual, from, |x| x.0);
                        assert_eq!(suffix_actual, suffix_expected);
                    }
                }
            }
        }
        check::<false>();
        check::<true>();
    }

    #[test]
    fn selection_reads_each_suffix_score_once() {
        let calls = std::cell::Cell::new(0);
        let mut data = [9, 2, 5, 3, 5];
        pick_best::<_, false>(&mut data, 1, |x| {
            calls.set(calls.get() + 1);
            *x
        });
        assert_eq!(calls.get(), 4);
        assert_eq!(data, [9, 5, 2, 3, 5]);
    }
}
