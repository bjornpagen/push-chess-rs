use super::legal_moves as generate_legal_moves;
use super::{Corpus, Ply, Result, Roster, Trajectory, opening_key, split, terminal};
use push_chess::core::position::start_position;
use push_chess::core::types::*;
use push_chess::engine::Engine;
#[cfg(test)]
use push_chess::engines::find_engine;
use push_chess::game::{Outcome, adjudicate};
use serde::{Deserialize, Serialize};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc::sync_channel,
};
use std::time::Instant;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    pub engines: Vec<String>,
    pub pairs: usize,
    pub workers: usize,
    pub nodes: u64,
    pub time_ms: u64,
    pub max_plies: usize,
    pub opening_plies: usize,
    pub seed: u64,
    /// `arena` reserves test-opening families. `corpus` retains all splits.
    pub purpose: String,
    pub max_seconds: u64,
    pub max_bytes: u64,
}

impl RunConfig {
    pub fn matchups(&self) -> Vec<(usize, usize)> {
        (0..self.engines.len())
            .flat_map(|a| (a + 1..self.engines.len()).map(move |b| (a, b)))
            .collect()
    }
    pub fn total_pairs(&self) -> usize {
        self.pairs
            .saturating_mul(self.engines.len() * self.engines.len().saturating_sub(1) / 2)
    }
    pub fn matchup(&self, pair: usize) -> (usize, usize) {
        let matchups = self.matchups();
        matchups[pair % matchups.len()]
    }
    /// Scheduling/resource checks. The separate immutable Roster resolves
    /// names and models before a run can be admitted or any worker spawned.
    pub fn validate(&self) -> Result<()> {
        if self.engines.len() < 2
            || self.engines.len() > 32
            || self.engines.iter().any(|e| e.is_empty() || e.len() > 64)
            || self
                .engines
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
                != self.engines.len()
        {
            return Err(
                "require 2..=32 distinct nonempty entrant names of at most 64 bytes".into(),
            );
        }
        let available = std::thread::available_parallelism()?.get();
        if self.workers == 0 || self.workers > available {
            return Err(format!("workers must be 1..={available}").into());
        }
        if self.pairs == 0 || self.total_pairs() > 1_000_000 {
            return Err("pairs must be positive; at most 1000000 total pairs".into());
        }
        if (self.nodes == 0) == (self.time_ms == 0)
            || self.nodes > i64::MAX as u64
            || self.time_ms > 3_600_000
        {
            return Err(
                "choose exactly one positive budget: nodes or time-ms (at most one hour)".into(),
            );
        }
        if !(4..=20).contains(&self.opening_plies)
            || self.max_plies <= self.opening_plies
            || self.max_plies > 4096
        {
            return Err(
                "opening-plies must be 4..=20; max-plies must be greater and at most 4096".into(),
            );
        }
        if self.max_seconds > 7 * 24 * 3600 || self.max_bytes < 64 * 1024 * 1024 {
            return Err("require a disk cap >=64 MiB and time cap <=7 days".into());
        }
        if self.purpose != "arena" && self.purpose != "corpus" {
            return Err("purpose must be arena or corpus".into());
        }
        Ok(())
    }
}

fn random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut x = *state;
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

fn opening(config: &RunConfig, pair: usize) -> Result<(Vec<u32>, String)> {
    let mut seed = config
        .seed
        .wrapping_add((pair as u64).wrapping_mul(0x100000001b3));
    let mut legal = Vec::new();
    for _ in 0..10_000 {
        let mut pos = start_position();
        let initial = pos.to_fen();
        let mut actions = Vec::with_capacity(config.opening_plies);
        for _ in 0..config.opening_plies {
            generate_legal_moves(&mut pos, &mut legal);
            if adjudicate(&pos, &legal) != Outcome::Playing {
                break;
            }
            legal.sort_unstable_by_key(|m| m.id());
            let chosen = legal[random(&mut seed) as usize % legal.len()];
            actions.push(chosen.id());
            pos.make_move(&chosen);
        }
        let key = opening_key(&initial, actions.iter().copied());
        if config.purpose == "corpus" || split(&key) == "test" {
            return Ok((actions, key));
        }
    }
    Err("could not select a held-out opening family".into())
}

fn play(
    config: &RunConfig,
    pair: usize,
    swapped: bool,
    engines: (&mut dyn Engine, &mut dyn Engine),
    opening: &[u32],
    key: &str,
    stop: &AtomicBool,
) -> Result<Trajectory> {
    let (a, b) = config.matchup(pair);
    let (white, black) = if swapped {
        (&config.engines[b], &config.engines[a])
    } else {
        (&config.engines[a], &config.engines[b])
    };
    let seed = config.seed.wrapping_add(pair as u64);
    let (white_engine, black_engine) = engines;
    white_engine.new_game(Color::White, seed);
    black_engine.new_game(Color::Black, seed);
    let mut pos = start_position();
    let mut game = Trajectory {
        rules: pos.rules,
        index: pair * 2 + usize::from(swapped),
        pair: Some(pair),
        white: white.clone(),
        black: black.clone(),
        initial_fen: pos.to_fen(),
        final_fen: String::new(),
        opening_key: key.to_owned(),
        split: if config.purpose == "arena" {
            "evaluation"
        } else {
            split(key)
        }
        .to_owned(),
        termination: "ply_limit".to_owned(),
        white_value: None,
        plies: Vec::with_capacity(config.max_plies.min(512)),
    };
    let mut legal = Vec::new();
    loop {
        generate_legal_moves(&mut pos, &mut legal);
        let outcome = adjudicate(&pos, &legal);
        if outcome != Outcome::Playing {
            let (reason, value) = terminal(&outcome);
            game.termination = reason.to_owned();
            game.white_value = value;
            break;
        }
        if stop.load(Ordering::Relaxed) {
            game.termination = "interrupted".to_owned();
            break;
        }
        let ply = game.plies.len();
        if ply >= config.max_plies {
            break;
        }
        let side = pos.side_to_move;
        let budget = SearchBudget {
            max_time_us: (config.time_ms * 1000) as i64,
            max_nodes: config.nodes as i64,
            seed: seed ^ ply as u64,
            ..SearchBudget::default()
        };
        let before = (
            pos.board,
            pos.side_to_move,
            pos.king_sq,
            pos.castling_rights,
            pos.ep_square,
            pos.halfmove_clock,
            pos.fullmove_number,
            pos.zobrist,
            pos.undo_stack.len(),
        );
        let start = Instant::now();
        let (chosen, stats) = if let Some(id) = opening.get(ply) {
            (
                *legal
                    .iter()
                    .find(|m| m.id() == *id)
                    .ok_or("paired opening no longer legal")?,
                SearchStats::default(),
            )
        } else {
            if side == Color::White {
                white_engine.choose_move(&mut pos, &budget)
            } else {
                black_engine.choose_move(&mut pos, &budget)
            }
        };
        let wall_us = i64::try_from(start.elapsed().as_micros())?;
        if before
            != (
                pos.board,
                pos.side_to_move,
                pos.king_sq,
                pos.castling_rights,
                pos.ep_square,
                pos.halfmove_clock,
                pos.fullmove_number,
                pos.zobrist,
                pos.undo_stack.len(),
            )
        {
            return Err(format!(
                "engine mutated caller position at game {} ply {ply}",
                game.index
            )
            .into());
        }
        if !legal.contains(&chosen) {
            return Err(format!("illegal engine move at game {} ply {ply}", game.index).into());
        }
        let is_opening = ply < opening.len();
        let proof = stats.diagnostics.mate_proof_plies.is_some_and(|n| n > 0);
        game.plies.push(Ply {
            action: chosen.id(),
            origin: if is_opening { "opening" } else { "search" },
            side: side as i32,
            score: (!is_opening).then_some(stats.eval_cp),
            score_perspective: "stm",
            search_complete: !is_opening && (stats.depth_reached > 0 || proof),
            nodes: stats.nodes,
            depth: stats.depth_reached,
            seldepth: stats.seldepth,
            wall_us: if is_opening { 0 } else { wall_us },
            reported_us: stats.time_used_us,
            pv: stats.pv.iter().map(|m| m.id()).collect(),
            diagnostics: stats.diagnostics,
            pieces: pos.board.iter().filter(|p| !p.is_empty()).count() as u32,
            halfmove_clock: pos.halfmove_clock,
        });
        pos.make_move(&chosen);
    }
    game.final_fen = pos.to_fen();
    Ok(game)
}

/// One worker-owned engine cache per CPU, one bounded queue and one writer.
/// Catch panics inside each worker so its peers stop immediately, not at join.
pub fn generate(corpus: &mut Corpus, config: &RunConfig, stop: &AtomicBool) -> Result<u64> {
    let roster = Roster::builtins(&config.engines)?;
    generate_controlled(corpus, config, &roster, stop, None)
}
pub fn generate_controlled(
    corpus: &mut Corpus,
    config: &RunConfig,
    roster: &Roster,
    stop: &AtomicBool,
    control: Option<&super::control::Control>,
) -> Result<u64> {
    config.validate()?;
    if corpus.disk_bytes()? >= config.max_bytes {
        return Err("corpus disk cap already reached".into());
    }
    let run = corpus.start_resolved(config, roster)?;
    let started = Instant::now();
    let next = AtomicUsize::new(0);
    let workers = config.workers.min(config.total_pairs());
    eprintln!(
        "run {run}: {} engines, {} matchups, {} games, {workers} workers",
        config.engines.len(),
        config.matchups().len(),
        config.total_pairs() * 2
    );
    let result = std::thread::scope(|scope| -> Result<()> {
        let (sender, receiver) =
            sync_channel::<std::result::Result<Trajectory, String>>(workers * 2);
        let mut handles = Vec::new();
        let mut error = None;
        for worker in 0..workers {
            let sender = sender.clone();
            let next = &next;
            let spawn = std::thread::Builder::new()
                .name(format!("lab-{worker}"))
                .stack_size(8 * 1024 * 1024)
                .spawn_scoped(scope, move || {
                    let result =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
                            // Bounded by workers * entrants * 32 MiB; no table allocation
                            // per move/game. Lazy creation avoids idle entrants' pages.
                            let mut engines: Vec<Option<Box<dyn Engine>>> =
                                (0..config.engines.len()).map(|_| None).collect();
                            while !stop.load(Ordering::Relaxed) {
                                let pair = next.fetch_add(1, Ordering::Relaxed);
                                if pair >= config.total_pairs() {
                                    break;
                                }
                                let (a, b) = config.matchup(pair);
                                for i in [a, b] {
                                    if engines[i].is_none() {
                                        let mut e = roster.create(i);
                                        let mut warm = start_position();
                                        e.choose_move(
                                            &mut warm,
                                            &SearchBudget {
                                                max_nodes: 512,
                                                ..SearchBudget::default()
                                            },
                                        );
                                        engines[i] = Some(e);
                                    }
                                }
                                let (opening, key) = opening(config, pair)?;
                                let (low, high) = engines.split_at_mut(b);
                                let a = low[a].as_mut().unwrap();
                                let b = high[0].as_mut().unwrap();
                                for swapped in [false, true] {
                                    if stop.load(Ordering::Relaxed) {
                                        break;
                                    }
                                    let players: (&mut dyn Engine, &mut dyn Engine) = if swapped {
                                        (&mut **b, &mut **a)
                                    } else {
                                        (&mut **a, &mut **b)
                                    };
                                    let game =
                                        play(config, pair, swapped, players, &opening, &key, stop)?;
                                    sender.send(Ok(game)).map_err(|_| "corpus writer stopped")?;
                                }
                            }
                            Ok(())
                        }));
                    let error = match result {
                        Ok(Ok(())) => None,
                        Ok(Err(e)) => Some(e.to_string()),
                        Err(p) => Some(format!(
                            "worker {worker} panicked: {}",
                            p.downcast_ref::<String>()
                                .map(String::as_str)
                                .or_else(|| p.downcast_ref::<&str>().copied())
                                .unwrap_or("unknown panic")
                        )),
                    };
                    if let Some(e) = error {
                        stop.store(true, Ordering::Relaxed);
                        let _ = sender.send(Err(e));
                    }
                });
            match spawn {
                Ok(handle) => handles.push(handle),
                Err(e) => {
                    stop.store(true, Ordering::Relaxed);
                    error = Some(e.to_string());
                    break;
                }
            }
        }
        drop(sender);
        let (mut saved, mut positions, mut analyses, mut terminals) = (0u64, 0u64, 0u64, 0u64);
        let mut last = Instant::now();
        loop {
            if let Some(control) = control {
                control.serve(corpus);
            }
            if config.max_seconds > 0 && started.elapsed().as_secs() >= config.max_seconds {
                stop.store(true, Ordering::Relaxed);
            }
            match receiver.recv_timeout(std::time::Duration::from_secs(1)) {
                Ok(Ok(game)) if error.is_none() => match corpus.save(run, &game) {
                    Ok(()) => {
                        saved += 1;
                        positions += game.plies.len() as u64;
                        analyses += game.plies.iter().filter(|p| p.score.is_some()).count() as u64;
                        terminals += u64::from(game.white_value.is_some());
                        if saved % 16 == 0 {
                            match corpus.disk_bytes() {
                                Ok(bytes) if bytes >= config.max_bytes => {
                                    eprintln!("run {run}: corpus disk cap reached");
                                    stop.store(true, Ordering::Relaxed);
                                }
                                Err(e) => {
                                    stop.store(true, Ordering::Relaxed);
                                    error = Some(e.to_string());
                                }
                                _ => {}
                            }
                        }
                    }
                    Err(e) => {
                        stop.store(true, Ordering::Relaxed);
                        error = Some(e.to_string());
                    }
                },
                Ok(Err(e)) => {
                    stop.store(true, Ordering::Relaxed);
                    error.get_or_insert(e);
                }
                Ok(_) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
            if last.elapsed().as_secs() >= 15 {
                eprintln!(
                    "run {run}: {saved}/{} games, {positions} plies, {analyses} analyses, {terminals} terminal, {:.1} plies/s, {:.0}s",
                    config.total_pairs() * 2,
                    positions as f64 / started.elapsed().as_secs_f64(),
                    started.elapsed().as_secs_f64()
                );
                last = Instant::now();
            }
        }
        for handle in handles {
            if handle.join().is_err() {
                error.get_or_insert("worker join failed".into());
            }
        }
        eprintln!(
            "run {run}: saved {saved} games / {positions} plies / {analyses} analyses / {terminals} terminal in {:.2}s",
            started.elapsed().as_secs_f64()
        );
        if let Some(error) = error {
            Err(error.into())
        } else {
            Ok(())
        }
    });
    match result {
        Ok(()) => {
            corpus.finish(
                run,
                if stop.load(Ordering::Relaxed) {
                    "interrupted"
                } else {
                    "finished"
                },
                None,
            )?;
            Ok(run)
        }
        Err(e) => {
            corpus.finish(run, "failed", Some(&e.to_string()))?;
            Err(e)
        }
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    pub fn config() -> RunConfig {
        RunConfig {
            engines: vec!["cataclysm".into(), "kinetic".into()],
            pairs: 1,
            workers: 1,
            nodes: 128,
            time_ms: 0,
            max_plies: 8,
            opening_plies: 4,
            seed: 7,
            purpose: "corpus".into(),
            max_seconds: 60,
            max_bytes: 1 << 30,
        }
    }
    pub fn fixture() -> Trajectory {
        let c = config();
        let (moves, key) = opening(&c, 0).unwrap();
        let mut a = (find_engine("cataclysm").unwrap().create)();
        let mut b = (find_engine("kinetic").unwrap().create)();
        play(
            &c,
            0,
            false,
            (&mut *a, &mut *b),
            &moves,
            &key,
            &AtomicBool::new(false),
        )
        .unwrap()
    }
    #[test]
    fn exact_history_replays_and_caps_remain_unknown() {
        let game = fixture();
        game.validate().unwrap();
        assert_eq!(game.white_value, None);
        assert_eq!(game.plies.len(), 8);
        assert_eq!(game.plies.iter().filter(|p| p.score.is_some()).count(), 4);
    }
    #[test]
    fn arena_openings_use_held_out_families() {
        let mut c = config();
        c.purpose = "arena".into();
        let (_, key) = opening(&c, 0).unwrap();
        assert_eq!(split(&key), "test");
    }
    #[test]
    fn ambiguous_budget_is_rejected() {
        let mut c = config();
        c.time_ms = 1;
        assert!(c.validate().is_err());
    }
    #[test]
    fn round_robin_is_interleaved() {
        let mut c = config();
        c.engines.push("astra".into());
        c.pairs = 2;
        assert_eq!(c.total_pairs(), 6);
        assert_eq!(
            (0..6).map(|p| c.matchup(p)).collect::<Vec<_>>(),
            vec![(0, 1), (0, 2), (1, 2), (0, 1), (0, 2), (1, 2)]
        );
    }
}
