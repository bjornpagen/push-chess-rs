//! Descriptive paired evidence, derived from committed outcome facts.
//! No Elo fitting, automatic promotions, or treating plies as independent trials.
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[derive(Default)]
pub(super) struct PairedScores {
    /// Number of complete pairs scoring 0, 0.5, 1, 1.5, 2 points for A.
    counts: [u64; 5],
    /// A repeated four-ply family is one uncertainty unit, not new evidence.
    families: BTreeMap<[u8; 32], (u64, u64)>,
}

impl PairedScores {
    pub fn record(&mut self, opening: [u8; 32], outcomes: [i32; 2]) {
        debug_assert!(outcomes.iter().all(|v| (-1..=1).contains(v)));
        let quarters = (outcomes[0] + outcomes[1] + 2) as usize;
        self.counts[quarters] += 1;
        let family = self.families.entry(opening).or_default();
        family.0 += quarters as u64;
        family.1 += 1;
    }

    pub fn report(&self, planned: usize, comparisons: usize) -> Value {
        let complete = self.counts.iter().sum::<u64>();
        let points = self
            .counts
            .iter()
            .enumerate()
            .map(|(q, n)| q as f64 * *n as f64 / 4.)
            .sum::<f64>();
        let pair_mean = (complete > 0).then(|| points / complete as f64);
        let n = self.families.len();
        let family_mean = (n > 0).then(|| {
            self.families
                .values()
                .map(|(quarters, pairs)| *quarters as f64 / (4. * *pairs as f64))
                .sum::<f64>()
                / n as f64
        });
        let interval = |alpha: f64| {
            family_mean.map(|mean| {
                // Bounded [0,1] family means: conservative fixed-sample
                // Hoeffding interval, no zero-width all-win small samples.
                let radius = ((2. / alpha).ln() / (2. * n as f64)).sqrt();
                [(mean - radius).max(0.), (mean + radius).min(1.)]
            })
        };
        json!({
            "complete_pairs":complete,"incomplete_or_missing_pairs":planned as u64-complete,
            "pair_points_histogram":self.counts,"pair_score":pair_mean,
            "opening_families":n,"family_balanced_score":family_mean,
            "family_hoeffding_95":interval(0.05),
            "all_matchups_family_hoeffding_95":interval(0.05/comparisons.max(1) as f64),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_colors_form_one_score_and_missing_pairs_are_not_draws() {
        let empty = PairedScores::default().report(3, 1);
        assert_eq!(empty["complete_pairs"], 0);
        assert!(empty["pair_score"].is_null());
        assert!(empty["family_hoeffding_95"].is_null());
        let mut scores = PairedScores::default();
        for (index, outcomes) in [[-1, -1], [-1, 0], [-1, 1], [1, 0], [1, 1]]
            .into_iter()
            .enumerate()
        {
            scores.record([index as u8; 32], outcomes);
        }
        let report = scores.report(8, 10);
        assert_eq!(report["pair_points_histogram"], json!([1, 1, 1, 1, 1]));
        assert_eq!(report["pair_score"], 0.5);
        assert_eq!(report["complete_pairs"], 5);
        assert_eq!(report["incomplete_or_missing_pairs"], 3);
        assert!(
            report["all_matchups_family_hoeffding_95"][0]
                .as_f64()
                .unwrap()
                <= report["family_hoeffding_95"][0].as_f64().unwrap()
        );
    }

    #[test]
    fn repeated_families_do_not_manufacture_confidence() {
        let mut scores = PairedScores::default();
        for _ in 0..100 {
            scores.record([0; 32], [1, 1]);
        }
        let report = scores.report(100, 1);
        assert_eq!(report["pair_score"], 1.0);
        assert_eq!(report["opening_families"], 1);
        assert_eq!(report["family_hoeffding_95"], json!([0., 1.]));
        scores.record([1; 32], [-1, -1]);
        let report = scores.report(101, 1);
        assert_eq!(report["family_balanced_score"], 0.5);
        assert!(report["pair_score"].as_f64().unwrap() > 0.99);
    }
}
