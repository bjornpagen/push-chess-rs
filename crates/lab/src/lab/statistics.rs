//! Descriptive paired evidence, derived from committed outcome facts.
//! No Elo fitting, automatic promotions, or treating plies as independent trials.
use serde_json::{Value, json};
use std::collections::BTreeMap;

/// Bounded descriptive totals, never a second copy of position observations.
/// Only call after the trajectory's side/timing/provenance checks have passed.
#[derive(Default)]
pub(super) struct SearchCost {
    pub searches: u64,
    pub complete: u64,
    pub nodes: u64,
    pub wall_us: u64,
    pub overruns: u64,
    reported_us: u64,
    max_wall_us: u64,
    depth: u64,
    max_depth: u32,
    seldepth: u64,
    qnodes: u64,
    tt_hits: u64,
    proof_searches: u64,
    proof_nodes: u64,
    mate_proof_reports: u64,
}

impl SearchCost {
    pub fn record(&mut self, ply: &super::Ply, budget_us: Option<u64>) {
        if ply.score.is_none() {
            return; // Opening moves are not zero-cost searches.
        }
        self.searches += 1;
        self.complete += u64::from(ply.search_complete);
        self.nodes += ply.nodes;
        self.wall_us += ply.wall_us as u64;
        self.reported_us += ply.reported_us as u64;
        self.max_wall_us = self.max_wall_us.max(ply.wall_us as u64);
        self.overruns += u64::from(budget_us.is_some_and(|us| ply.wall_us as u64 > us));
        self.depth += u64::from(ply.depth);
        self.max_depth = self.max_depth.max(ply.depth);
        self.seldepth += u64::from(ply.seldepth);
        self.qnodes += ply.diagnostics.qnodes;
        self.tt_hits += ply.diagnostics.tt_hits;
        if let Some(nodes) = ply.diagnostics.proof_nodes {
            self.proof_searches += 1;
            self.proof_nodes += nodes;
        }
        self.mate_proof_reports += u64::from(ply.diagnostics.mate_proof_plies.is_some());
    }

    pub fn report(&self) -> Value {
        let ratio = |numerator: u64, denominator: u64| {
            (denominator > 0).then(|| numerator as f64 / denominator as f64)
        };
        json!({
            "searches":self.searches,"completed_searches":self.complete,
            "completion_fraction":ratio(self.complete,self.searches),
            "nodes":self.nodes,"wall_us":self.wall_us,"reported_us":self.reported_us,
            "max_wall_us":(self.searches>0).then_some(self.max_wall_us),
            "mean_wall_us":ratio(self.wall_us,self.searches),
            "nodes_per_wall_second":ratio(self.nodes,self.wall_us).map(|n|n*1_000_000.),
            "time_overruns":self.overruns,
            "mean_depth_all_searches":ratio(self.depth,self.searches),
            "max_depth":(self.searches>0).then_some(self.max_depth),
            "mean_seldepth_all_searches":ratio(self.seldepth,self.searches),
            "qnodes":self.qnodes,"qnode_fraction":ratio(self.qnodes,self.nodes),
            "tt_hits":self.tt_hits,"tt_hits_per_node":ratio(self.tt_hits,self.nodes),
            "proof_searches":self.proof_searches,"proof_nodes":self.proof_nodes,
            "mean_nodes_per_reported_proof_search":ratio(self.proof_nodes,self.proof_searches),
            "mate_proof_reports":self.mate_proof_reports,
        })
    }
}

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
    fn search_cost_uses_total_time_and_preserves_absence_and_incomplete_work() {
        let mut cost = SearchCost::default();
        let empty = cost.report();
        assert_eq!(empty["searches"], 0);
        assert!(empty["completion_fraction"].is_null());
        assert!(empty["nodes_per_wall_second"].is_null());
        assert!(empty["max_wall_us"].is_null());
        assert!(empty["mean_nodes_per_reported_proof_search"].is_null());
        let mut ply = super::super::runner::tests::fixture().plies.pop().unwrap();
        ply.score = None;
        cost.record(&ply, Some(100));
        assert_eq!(cost.report(), empty);
        ply.score = Some(0);
        ply.search_complete = true;
        ply.nodes = 100;
        ply.wall_us = 100;
        ply.reported_us = 90;
        ply.depth = 4;
        ply.seldepth = 8;
        ply.diagnostics.qnodes = 60;
        ply.diagnostics.tt_hits = 10;
        ply.diagnostics.proof_nodes = None;
        ply.diagnostics.mate_proof_plies = None;
        cost.record(&ply, Some(100));
        ply.search_complete = false;
        ply.nodes = 200;
        ply.wall_us = 300;
        ply.reported_us = 290;
        ply.depth = 0;
        ply.seldepth = 12;
        ply.diagnostics.qnodes = 120;
        ply.diagnostics.tt_hits = 20;
        ply.diagnostics.proof_nodes = Some(0);
        cost.record(&ply, Some(100));
        let result = cost.report();
        assert_eq!(result["searches"], 2);
        assert_eq!(result["completed_searches"], 1);
        assert_eq!(result["completion_fraction"], 0.5);
        assert_eq!(result["nodes_per_wall_second"], 750_000.);
        assert_eq!(result["wall_us"], 400);
        assert_eq!(result["reported_us"], 380);
        assert_eq!(result["mean_wall_us"], 200.);
        assert_eq!(result["max_wall_us"], 300);
        assert_eq!(result["time_overruns"], 1);
        assert_eq!(result["mean_depth_all_searches"], 2.);
        assert_eq!(result["max_depth"], 4);
        assert_eq!(result["mean_seldepth_all_searches"], 10.);
        assert_eq!(result["qnode_fraction"], 0.6);
        assert_eq!(result["tt_hits_per_node"], 0.1);
        assert_eq!(result["proof_searches"], 1);
        assert_eq!(result["mean_nodes_per_reported_proof_search"], 0.);
        assert_eq!(result["mate_proof_reports"], 0);
        let mut node_budget = SearchCost::default();
        node_budget.record(&ply, None);
        assert_eq!(node_budget.report()["time_overruns"], 0);
    }

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
