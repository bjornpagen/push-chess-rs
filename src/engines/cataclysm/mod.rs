//! Cataclysm: transaction-based search and coordinated push logistics.
#![forbid(unsafe_code)]
mod board;
mod eval;
pub mod experiments;
pub mod learning;
mod network;
mod siege;

use crate::core::movegen::{generate_legal_moves, generate_pseudo_legal_moves};
use crate::core::position::Position;
use crate::core::types::*;
use crate::core::zobrist::zobrist_tables;
use crate::engine::Engine;
use crate::engine::shared::{Buckets, SearchLimits, pick_best};
use board::{Action, Board, Handwritten, Neural, Residual, pack};
use eval::{VALUE, evaluate, piece_score};
pub use network::Model;
use network::Network;
use std::borrow::Cow;

const MATE: i32 = 30_000;
const WIN: i32 = 29_800;
const INF: i32 = 31_000;
const MAX: usize = 128;
const EXACT: u8 = 1;
const LOWER: u8 = 2;
const UPPER: u8 = 3;

#[derive(Clone, Copy, Debug)]
pub enum HashSize {
    MiB4,
    MiB8,
    MiB16,
    MiB32,
}

impl TryFrom<u32> for HashSize {
    type Error = &'static str;
    fn try_from(mib: u32) -> Result<Self, Self::Error> {
        match mib {
            4 => Ok(Self::MiB4),
            8 => Ok(Self::MiB8),
            16 => Ok(Self::MiB16),
            32 => Ok(Self::MiB32),
            _ => Err("hashMiB must be 4, 8, 16, or 32"),
        }
    }
}

impl HashSize {
    fn bytes(self) -> usize {
        (match self {
            Self::MiB4 => 4,
            Self::MiB8 => 8,
            Self::MiB16 => 16,
            Self::MiB32 => 32,
        }) * 1024
            * 1024
    }
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Entry {
    key: u64,
    mv: u32,
    value: i16,
    depth: i8,
    flags: u8,
}

fn encode(v: i32, ply: usize) -> i16 {
    (if v >= WIN {
        v + ply as i32
    } else if v <= -WIN {
        v - ply as i32
    } else {
        v
    }) as i16
}
fn decode(v: i16, ply: usize) -> i32 {
    let v = i32::from(v);
    if v >= WIN {
        v - ply as i32
    } else if v <= -WIN {
        v + ply as i32
    } else {
        v
    }
}

pub type Cataclysm = Search<0>;

pub struct Search<const V: usize> {
    name: Cow<'static, str>,
    model: Option<Model>,
    table: Buckets<Entry, 4>,
    age: u8,
    history: Vec<i32>,
    killers: [[u32; 2]; MAX],
    counters: Vec<u32>,
    previous: [usize; MAX],
    statics: [i32; MAX],
    buffers: Vec<Vec<Action>>,
    quiet_buffers: Vec<Vec<usize>>,
    path: Vec<u64>,
    barrier: usize,
    root_depth: i32,
    root_best: Move,
    limits: SearchLimits,
    qnodes: u64,
    hits: u64,
}

impl<const V: usize> Search<V> {
    fn new() -> Self {
        Self::with_hash_size(HashSize::MiB32)
    }

    /// Fixed choices bound worker memory; one table is retained for the game.
    pub fn with_hash_size(size: HashSize) -> Self {
        let profile = experiments::PROFILES[V];
        Self::with_optional_model(
            profile.name,
            size,
            profile.neural_accumulator.then(Model::embedded),
        )
    }

    /// Model ownership is fixed for this search's lifetime. Its TT and history
    /// can never contain results from a previous candidate's evaluation.
    pub fn with_model(name: impl Into<Cow<'static, str>>, size: HashSize, model: Model) -> Self {
        Self::with_optional_model(name, size, Some(model))
    }

    fn with_optional_model(
        name: impl Into<Cow<'static, str>>,
        size: HashSize,
        model: Option<Model>,
    ) -> Self {
        assert!(V < experiments::PROFILES.len(), "unknown Cataclysm profile");
        assert_eq!(
            model.is_some(),
            experiments::PROFILES[V].neural_accumulator,
            "model ownership must match the profile; no evaluator fallback"
        );
        Self {
            name: name.into(),
            model,
            table: Buckets::new(size.bytes()),
            age: 0,
            history: vec![0; 2 * 3 * 64 * 64],
            killers: [[0; 2]; MAX],
            counters: vec![0; 2 * 7 * 64],
            previous: [0; MAX],
            statics: [0; MAX],
            buffers: (0..MAX).map(|_| Vec::new()).collect(),
            quiet_buffers: (0..MAX).map(|_| Vec::new()).collect(),
            path: Vec::new(),
            barrier: 0,
            root_depth: 0,
            root_best: Move::default(),
            limits: SearchLimits::default(),
            qnodes: 0,
            hits: 0,
        }
    }

    pub fn model_fingerprint(&self) -> Option<u64> {
        self.model.as_ref().map(Model::fingerprint)
    }

    fn tick(&mut self, ply: usize) -> bool {
        self.limits.tick(ply)
    }

    fn probe(&self, key: u64) -> Option<Entry> {
        self.table[key as usize & (self.table.len() - 1)]
            .iter()
            .find(|e| e.flags & 3 != 0 && e.key == key)
            .copied()
    }

    fn save(&mut self, key: u64, mv: u32, depth: i32, v: i32, bound: u8, ply: usize) {
        let index = key as usize & (self.table.len() - 1);
        let bucket = &mut self.table[index];
        let at = bucket
            .iter()
            .position(|e| e.flags & 3 == 0 || e.key == key)
            .unwrap_or_else(|| {
                (0..4)
                    .min_by_key(|&i| {
                        i32::from(bucket[i].depth)
                            - if bucket[i].flags & 252 != self.age {
                                16
                            } else {
                                0
                            }
                    })
                    .unwrap()
            });
        let old = bucket[at];
        if old.key == key && i32::from(old.depth) > depth + 3 && bound != EXACT {
            return;
        }
        bucket[at] = Entry {
            key,
            mv: if mv == 0 && old.key == key {
                old.mv
            } else {
                mv
            },
            value: encode(v, ply),
            depth: depth.min(126) as i8,
            flags: self.age | bound,
        };
    }

    fn draw(&self, b: &Board<impl Residual>, ply: usize) -> bool {
        if b.pos.halfmove_clock >= 100 {
            return true;
        }
        let needed = if ply == 0 { 2 } else { 1 };
        self.path[self.barrier..]
            .iter()
            .rev()
            .take(b.pos.halfmove_clock as usize)
            .filter(|&&key| key == b.pos.zobrist)
            .take(needed)
            .count()
            == needed
    }

    fn terminal_draw(
        &mut self,
        b: &mut Board<impl Residual>,
        ply: usize,
        check: bool,
    ) -> Option<i32> {
        if !self.draw(b, ply) {
            return None;
        }
        if !check {
            return Some(0);
        }
        let mut actions = std::mem::take(&mut self.buffers[ply]);
        b.generate(&mut actions);
        let us = b.pos.side_to_move;
        let mut legal = false;
        for a in &actions {
            let undo = b.make(a);
            legal = !b.checked(us);
            b.unmake(undo);
            if legal {
                break;
            }
        }
        self.buffers[ply] = actions;
        Some(if legal { 0 } else { -MATE + ply as i32 })
    }

    fn hi(b: &Board<impl Residual>, m: Move) -> usize {
        ((b.pos.side_to_move as usize * 3 + m.path_kind as usize) * 64 + m.from as usize) * 64
            + m.to as usize
    }

    fn action_context(b: &Board<impl Residual>, mv: Move) -> usize {
        (b.pos.side_to_move as usize * 7 + b.pos.board[mv.from as usize].piece_type as usize) * 64
            + mv.to as usize
    }

    fn parent_context(&self, ply: usize) -> Option<usize> {
        if ply == 0 {
            return None;
        }
        let index = self.previous[ply - 1];
        // No real move has piece type None, so index zero denotes no action.
        // Preserve historical lookup behavior in the unchanged controls.
        (!experiments::PROFILES[V].exact_context || index != 0).then_some(index)
    }

    fn prepare(&mut self, b: &Board<impl Residual>, ply: usize, tt: u32) -> Vec<Action> {
        let mut actions = std::mem::take(&mut self.buffers[ply]);
        b.generate(&mut actions);
        let counter = self.parent_context(ply).map(|parent| self.counters[parent]);
        for a in &mut actions {
            let id = pack(a.mv);
            if id == tt && tt != 0 {
                a.order = 10_000_000;
                continue;
            }
            let mover = b.pos.board[a.mv.from as usize];
            let mut score = self.history[Self::hi(b, a.mv)];
            if a.capture != PieceType::None {
                score +=
                    1_000_000 + VALUE[a.capture as usize] * 24 - VALUE[mover.piece_type as usize];
            }
            if a.mv.special == SpecialMove::Promotion {
                score += 900_000 + VALUE[a.mv.promo_piece as usize] * 12;
            }
            if id == self.killers[ply][0] {
                score += 90_000;
            } else if id == self.killers[ply][1] {
                score += 80_000;
            }
            if counter == Some(id) {
                score += 70_000;
            }
            if experiments::PROFILES[V].ordering {
                score += experiments::transaction_gain(b, a);
            } else if let Some(p) = &a.plan {
                for &(f, t) in p.displacements() {
                    let piece = b.pos.board[f as usize];
                    let delta = piece_score(piece, t).0 - piece_score(piece, f).0;
                    score += delta * 12;
                    if piece.piece_type == PieceType::Pawn {
                        let rank = if mover.color == Color::White {
                            rank_of(t)
                        } else {
                            7 - rank_of(t)
                        };
                        score += (rank - 3).max(0) * 100;
                    }
                }
            }
            if a.push {
                score += 300;
            }
            a.order = score;
        }
        actions
    }

    fn pick(actions: &mut [Action], i: usize) {
        pick_best::<_, true>(actions, i, |a| a.order);
    }

    fn reward(&mut self, index: usize, bonus: i32) {
        let h = &mut self.history[index];
        *h += bonus - *h * bonus.abs() / 16_384;
    }

    fn search(
        &mut self,
        b: &mut Board<impl Residual>,
        mut depth: i32,
        mut alpha: i32,
        mut beta: i32,
        ply: usize,
        null_ok: bool,
    ) -> i32 {
        if depth <= 0 {
            return self.qsearch(b, alpha, beta, ply, 0);
        }
        if self.tick(ply) {
            return 0;
        }
        let check = b.checked(b.pos.side_to_move);
        if let Some(value) = self.terminal_draw(b, ply, check) {
            return value;
        }
        if ply >= MAX - 2 {
            return evaluate::<V>(b);
        }
        alpha = alpha.max(-MATE + ply as i32);
        beta = beta.min(MATE - ply as i32 - 1);
        if alpha >= beta {
            return alpha;
        }
        let start_alpha = alpha;
        let pv = beta - alpha > 1;
        let key = b.pos.zobrist;
        let entry = self.probe(key);
        if let Some(e) = entry
            && ply > 0
            && !pv
            && i32::from(e.depth) >= depth
            && b.pos.halfmove_clock < 80
        {
            let v = decode(e.value, ply);
            if e.flags & 3 == EXACT
                || (e.flags & 3 == LOWER && v >= beta)
                || (e.flags & 3 == UPPER && v <= alpha)
            {
                self.hits += 1;
                return v;
            }
        }
        if check && (ply as i32) < self.root_depth * 2 {
            depth += 1;
        }
        let stand = evaluate::<V>(b);
        self.statics[ply] = stand;
        let improving = ply >= 2 && stand > self.statics[ply - 2];
        if !check && !pv && ply > 0 && beta.abs() < WIN {
            if depth <= 5 && stand - 120 * depth - 40 >= beta {
                // Static pruning must not turn stalemate into a material score.
                return if b.has_legal_move() { stand } else { 0 };
            }
            if null_ok
                && depth >= 3
                && stand >= beta
                && b.phase > 4
                && (b.occupied[b.pos.side_to_move as usize]
                    & !b.men[b.pos.side_to_move as usize][1]
                    & !b.men[b.pos.side_to_move as usize][6])
                    != 0
            {
                let old = (
                    b.pos.side_to_move,
                    b.pos.ep_square,
                    b.pos.zobrist,
                    self.barrier,
                );
                b.pos.side_to_move = opponent(b.pos.side_to_move);
                b.pos.zobrist ^= zobrist_tables().side_key;
                if b.pos.ep_square < 64 {
                    b.pos.zobrist ^= zobrist_tables().ep_keys[file_of(b.pos.ep_square) as usize];
                }
                b.pos.ep_square = 64;
                self.barrier = self.path.len();
                let old_context = if experiments::PROFILES[V].exact_context {
                    std::mem::replace(&mut self.previous[ply], 0)
                } else {
                    0
                };
                let reduction = (3 + depth / 5).min(depth - 1);
                let v = -self.search(b, depth - 1 - reduction, -beta, 1 - beta, ply + 1, false);
                if experiments::PROFILES[V].exact_context {
                    self.previous[ply] = old_context;
                }
                b.pos.side_to_move = old.0;
                b.pos.ep_square = old.1;
                b.pos.zobrist = old.2;
                self.barrier = old.3;
                if self.limits.stopped {
                    return 0;
                }
                if v >= beta {
                    return if b.has_legal_move() {
                        v.min(WIN - 1)
                    } else {
                        0
                    };
                }
            }
        }
        let mut actions = self.prepare(b, ply, entry.map_or(0, |e| e.mv));
        let us = b.pos.side_to_move;
        let mut legal = 0;
        let mut searched = 0;
        let mut best = -INF;
        let mut best_id = 0;
        let mut quiets = std::mem::take(&mut self.quiet_buffers[ply]);
        quiets.clear();
        for i in 0..actions.len() {
            Self::pick(&mut actions, i);
            let a = &actions[i];
            let hi = Self::hi(b, a.mv);
            let previous = Self::action_context(b, a.mv);
            let volatile = experiments::PROFILES[V].volatility && experiments::volatile_push(b, a);
            let undo = b.make(a);
            if b.checked(us) {
                b.unmake(undo);
                continue;
            }
            legal += 1;
            let gives_check = b.checked(opponent(us));
            let forcing = a.tactical() || gives_check;
            let critical = a.king_push || a.mv.special == SpecialMove::Castle || volatile;
            if searched > 0
                && !pv
                && !check
                && !forcing
                && !critical
                && !a.push
                && depth <= 3
                && legal
                    > if improving {
                        8 + depth * depth * 4
                    } else {
                        5 + depth * depth * 3
                    }
            {
                b.unmake(undo);
                continue;
            }
            let mut reduction = 0;
            if searched >= 3 && depth >= 3 && !check && !forcing && !volatile {
                reduction = ((depth as f64).ln() * (legal as f64).ln() / 2.0) as i32;
                reduction -= i32::from(pv) + i32::from(a.push || critical) + i32::from(improving);
                reduction -= self.history[hi] / 5000;
                reduction = reduction.clamp(0, depth - 2);
            }
            self.previous[ply] = previous;
            self.path.push(key);
            let mut value;
            if searched == 0 {
                value = -self.search(b, depth - 1, -beta, -alpha, ply + 1, true);
            } else {
                value = -self.search(b, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, true);
                if value > alpha && reduction > 0 && !self.limits.stopped {
                    value = -self.search(b, depth - 1, -alpha - 1, -alpha, ply + 1, true);
                }
                if value > alpha && value < beta && !self.limits.stopped {
                    value = -self.search(b, depth - 1, -beta, -alpha, ply + 1, true);
                }
            }
            self.path.pop();
            b.unmake(undo);
            if self.limits.stopped {
                break;
            }
            searched += 1;
            if !a.tactical() {
                quiets.push(hi);
            }
            if value > best {
                best = value;
                best_id = pack(a.mv);
            }
            if value > alpha {
                alpha = value;
                if ply == 0 {
                    self.root_best = a.mv;
                }
            }
            if alpha >= beta {
                if !a.tactical() {
                    let bonus = (40 * depth * depth).min(1800);
                    for &index in &quiets {
                        if index != hi {
                            self.reward(index, -bonus / 2);
                        }
                    }
                    self.reward(hi, bonus);
                    if self.killers[ply][0] != best_id {
                        self.killers[ply][1] = self.killers[ply][0];
                        self.killers[ply][0] = best_id;
                    }
                    if let Some(parent) = self.parent_context(ply) {
                        self.counters[parent] = best_id;
                    }
                }
                break;
            }
        }
        self.buffers[ply] = actions;
        self.quiet_buffers[ply] = quiets;
        if self.limits.stopped {
            return 0;
        }
        if legal == 0 {
            return if check { -MATE + ply as i32 } else { 0 };
        }
        self.save(
            key,
            best_id,
            depth,
            best,
            if best >= beta {
                LOWER
            } else if best <= start_alpha {
                UPPER
            } else {
                EXACT
            },
            ply,
        );
        best
    }

    fn qsearch(
        &mut self,
        b: &mut Board<impl Residual>,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        qply: usize,
    ) -> i32 {
        if self.tick(ply) {
            return 0;
        }
        self.qnodes += 1;
        let check = b.checked(b.pos.side_to_move);
        if let Some(value) = self.terminal_draw(b, ply, check) {
            return value;
        }
        if ply >= MAX - 2 {
            return evaluate::<V>(b);
        }
        let key = b.pos.zobrist;
        let entry = self.probe(key);
        if let Some(e) = entry
            && beta - alpha == 1
            && i32::from(e.depth) >= if qply < 2 { 0 } else { -1 }
            && b.pos.halfmove_clock < 80
        {
            let v = decode(e.value, ply);
            if e.flags & 3 == EXACT
                || (e.flags & 3 == LOWER && v >= beta)
                || (e.flags & 3 == UPPER && v <= alpha)
            {
                self.hits += 1;
                return v;
            }
        }
        let original = alpha;
        let stand = evaluate::<V>(b);
        let mut best = if check { -INF } else { stand };
        if !check {
            if stand >= beta {
                return if b.has_legal_move() { stand } else { 0 };
            }
            alpha = alpha.max(stand);
            if qply >= 12 {
                return if b.has_legal_move() { stand } else { 0 };
            }
        }
        let mut actions = self.prepare(b, ply, entry.map_or(0, |e| e.mv));
        let us = b.pos.side_to_move;
        let mut legal = 0;
        let mut best_id = 0;
        for i in 0..actions.len() {
            Self::pick(&mut actions, i);
            let a = &actions[i];
            // The first two tactical layers include checking pushes and quiet
            // checks, not just captures. Check evasions are always searched.
            let volatile =
                experiments::PROFILES[V].volatility && qply < 3 && experiments::volatile_push(b, a);
            if !check && !a.tactical() && !volatile && qply >= 2 {
                continue;
            }
            let context = if experiments::PROFILES[V].exact_context {
                Self::action_context(b, a.mv)
            } else {
                0
            };
            let undo = b.make(a);
            if b.checked(us) {
                b.unmake(undo);
                continue;
            }
            legal += 1;
            let gives_check = b.checked(opponent(us));
            if !check && !a.tactical() && !gives_check && !volatile {
                b.unmake(undo);
                continue;
            }
            if !check
                && !gives_check
                && !volatile
                && a.mv.special != SpecialMove::Promotion
                && stand + VALUE[a.capture as usize] + 220 < alpha
            {
                b.unmake(undo);
                continue;
            }
            self.path.push(key);
            if experiments::PROFILES[V].exact_context {
                self.previous[ply] = context;
            }
            let value = -self.qsearch(b, -beta, -alpha, ply + 1, qply + 1);
            self.path.pop();
            b.unmake(undo);
            if self.limits.stopped {
                break;
            }
            if value > best {
                best = value;
                best_id = pack(a.mv);
            }
            alpha = alpha.max(value);
            if alpha >= beta {
                break;
            }
        }
        self.buffers[ply] = actions;
        if self.limits.stopped {
            return 0;
        }
        if check && legal == 0 {
            return -MATE + ply as i32;
        }
        if !check && legal == 0 && !b.has_legal_move() {
            return 0;
        }
        self.save(
            key,
            best_id,
            if qply < 2 { 0 } else { -1 },
            best,
            if best >= beta {
                LOWER
            } else if best <= original {
                UPPER
            } else {
                EXACT
            },
            ply,
        );
        best
    }
}

impl<const V: usize> Engine for Search<V> {
    fn name(&self) -> &str {
        &self.name
    }
    fn new_game(&mut self, _: Color, _: u64) {
        self.table.fill([Entry::default(); 4]);
        self.history.fill(0);
        self.killers.fill([0; 2]);
        self.counters.fill(0);
        // Counter-move lookup consults per-ply action context in tactical
        // search too. An old game's context must not survive its cleared table.
        self.previous.fill(0);
        self.path.clear();
    }
    fn choose_move(&mut self, pos: &mut Position, budget: &SearchBudget) -> (Move, SearchStats) {
        self.limits.begin(budget, 94);
        // Const specialization instantiates only the profile's board layout.
        // Clone one shared owner per neural root, never per recursive node.
        // A missing required network is an invariant failure, not a fallback.
        if experiments::PROFILES[V].neural_accumulator {
            let model = self.model.as_ref().expect("required neural model").clone();
            self.choose_with_residual(pos, budget, Neural::new(model.network()))
        } else {
            self.choose_with_residual(pos, budget, Handwritten)
        }
    }
}

impl<const V: usize> Search<V> {
    fn choose_with_residual(
        &mut self,
        pos: &mut Position,
        budget: &SearchBudget,
        residual: impl Residual,
    ) -> (Move, SearchStats) {
        self.qnodes = 0;
        self.hits = 0;
        self.age = self.age.wrapping_add(4) & 252;
        self.path.clear();
        self.path.extend(pos.undo_stack.iter().map(|u| u.zobrist));
        self.barrier = 0;
        for h in &mut self.history {
            *h /= 2;
        }
        let mut legal = Vec::new();
        generate_legal_moves(pos, &mut legal);
        let Some(mut best) = legal.first().copied() else {
            return (
                Move::default(),
                SearchStats {
                    eval_cp: if pos.in_check() { -MATE } else { 0 },
                    ..SearchStats::default()
                },
            );
        };
        let mut board = Board::with_residual(pos, residual);
        let proof_cap = if budget.max_depth > 0 {
            budget.max_depth
        } else {
            11
        };
        if let Some(line) = self.siege(&mut board, proof_cap) {
            return (
                line[0],
                SearchStats {
                    nodes: self.limits.nodes,
                    depth_reached: 0,
                    seldepth: self.limits.seldepth as u32,
                    eval_cp: MATE - line.len() as i32,
                    time_used_us: self.limits.started.elapsed().as_micros() as i64,
                    diagnostics: SearchDiagnostics {
                        proof_nodes: Some(self.limits.nodes),
                        mate_proof_plies: Some(line.len() as u32),
                        ..SearchDiagnostics::default()
                    },
                    pv: line,
                },
            );
        }
        let proof_nodes = self.limits.nodes;
        let mut score = evaluate::<V>(&board);
        let mut completed = 0;
        let cap = if budget.max_depth > 0 {
            budget.max_depth.min(100)
        } else {
            64
        };
        for depth in 1..=cap {
            self.root_depth = depth;
            self.root_best = best;
            let mut width = if depth >= 4 { 35 } else { INF };
            let (mut lo, mut hi) = ((score - width).max(-INF), (score + width).min(INF));
            loop {
                let value = self.search(&mut board, depth, lo, hi, 0, true);
                if self.limits.stopped {
                    break;
                }
                if value <= lo {
                    lo = (value - width).max(-INF);
                    width = (width * 2).min(INF);
                } else if value >= hi {
                    hi = (value + width).min(INF);
                    width = (width * 2).min(INF);
                } else {
                    best = self.root_best;
                    score = value;
                    completed = depth;
                    break;
                }
            }
            if self.limits.stopped || score.abs() >= WIN {
                break;
            }
            if self.limits.micros > 0
                && self.limits.started.elapsed().as_micros() * 4 >= self.limits.micros * 3
            {
                break;
            }
        }
        let mut pv = vec![best];
        let mut view = Board::with_residual(pos, residual);
        let mut actions = Vec::new();
        let mut id = pack(best);
        let mut seen = Vec::new();
        for _ in 0..24 {
            if seen.contains(&view.pos.zobrist) {
                break;
            }
            seen.push(view.pos.zobrist);
            view.generate(&mut actions);
            let Some(a) = actions.iter().find(|a| pack(a.mv) == id) else {
                break;
            };
            let us = view.pos.side_to_move;
            view.make(a);
            if view.checked(us) {
                break;
            }
            if pv.len() == 1 && id == pack(best) {
            } else {
                pv.push(a.mv);
            }
            let Some(e) = self.probe(view.pos.zobrist) else {
                break;
            };
            id = e.mv;
        }
        (
            best,
            SearchStats {
                nodes: self.limits.nodes,
                depth_reached: completed as u32,
                seldepth: self.limits.seldepth as u32,
                eval_cp: score,
                time_used_us: self.limits.started.elapsed().as_micros() as i64,
                pv,
                diagnostics: SearchDiagnostics {
                    qnodes: self.qnodes,
                    tt_hits: self.hits,
                    proof_nodes: Some(proof_nodes),
                    mate_proof_plies: None,
                },
            },
        )
    }
}

pub fn network_fingerprint() -> u64 {
    Network::embedded().fingerprint
}

pub fn create() -> Box<dyn Engine> {
    Box::new(Cataclysm::new())
}

pub fn create_experiment<const V: usize>() -> Box<dyn Engine> {
    assert!(V < experiments::PROFILES.len());
    Box::new(Search::<V>::new())
}

/// Differential oracle used by the all-history study and regression tests.
/// Checks every pseudo-legal move, exact FEN, hash, check state, and rollback.
pub fn verify_rules(pos: &Position) -> Result<usize, String> {
    verify_rules_with_model(pos, &Model::embedded())
}

pub fn verify_rules_with_model(pos: &Position, model: &Model) -> Result<usize, String> {
    verify_board(pos, Neural::new(model.network()))
}

fn verify_board<R: Residual>(pos: &Position, residual: R) -> Result<usize, String>
where
    R::Undo: PartialEq,
{
    let mut board = Board::with_residual(pos, residual);
    let mut actions = Vec::new();
    board.generate(&mut actions);
    let mut core = Vec::new();
    generate_pseudo_legal_moves(pos, &mut core);
    let mut ours: Vec<_> = actions.iter().map(|a| pack(a.mv)).collect();
    ours.sort_unstable();
    let mut theirs: Vec<_> = core.iter().copied().map(pack).collect();
    theirs.sort_unstable();
    if ours != theirs {
        return Err(format!("move set mismatch: {}", pos.to_fen()));
    }
    for action in &actions {
        let mut reference = pos.clone();
        reference.make_move(&action.mv);
        let old = board.make(action);
        if board.pos.to_fen() != reference.to_fen()
            || board.pos.zobrist != reference.zobrist
            || board.pos.king_sq != reference.king_sq
        {
            return Err(format!(
                "transaction mismatch {:?}: {}",
                action.mv,
                pos.to_fen()
            ));
        }
        for c in [Color::White, Color::Black] {
            if board.checked(c) != reference.in_check_color(c) {
                return Err(format!("check mismatch {:?}: {}", action.mv, pos.to_fen()));
            }
        }
        let rebuilt = Board::with_residual(&reference, residual);
        if board.men != rebuilt.men
            || board.occupied != rebuilt.occupied
            || board.mg != rebuilt.mg
            || board.eg != rebuilt.eg
            || board.phase != rebuilt.phase
            || board.net.snapshot() != rebuilt.net.snapshot()
            || board.material != rebuilt.material
        {
            return Err("incremental feature mismatch".into());
        }
        board.unmake(old);
        if board.pos.to_fen() != pos.to_fen() || board.pos.zobrist != pos.zobrist {
            return Err("undo mismatch".into());
        }
    }
    Ok(actions.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "opt-in release ordering measurement, run serially on an idle machine"]
    fn ordering_probe() {
        let mut engine = Cataclysm::with_hash_size(HashSize::MiB4);
        let samples: Vec<_> = crate::engine::ordering_probe::positions()
            .iter()
            .map(|p| engine.prepare(&Board::new(p), 0, 0))
            .collect();
        crate::engine::ordering_probe::compare::<_, true, _>("cataclysm", &samples, |a| a.order);
    }
    use crate::core::position::start_position;

    #[test]
    fn transactions_match_the_rules_through_random_games() {
        for seed in 1u64..=4 {
            let mut pos = start_position();
            let mut random = seed;
            for _ in 0..80 {
                verify_rules(&pos).unwrap();
                verify_board(&pos, Handwritten).unwrap();
                let mut legal = Vec::new();
                generate_legal_moves(&mut pos, &mut legal);
                if legal.is_empty() {
                    break;
                }
                random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
                pos.make_move(&legal[random as usize % legal.len()]);
            }
        }
    }

    #[test]
    fn transactions_cover_special_moves_and_cascades() {
        for fen in [
            "7k/8/4RB2/8/4N3/8/8/K7 w - - 0 1",
            "7k/8/5BR1/8/4N3/8/8/K7 w - - 0 1",
            "7k/P7/R7/8/8/8/8/K7 w - - 0 1",
            "r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1",
        ] {
            let mut pos = Position::empty();
            pos.set_from_fen(fen);
            verify_rules(&pos).unwrap();
            verify_board(&pos, Handwritten).unwrap();
        }
    }

    #[test]
    fn granite_has_no_neural_state_or_weight_owner() {
        use board::Snapshot;
        use network::Accumulator;
        assert_eq!(std::mem::size_of::<Handwritten>(), 0);
        assert_eq!(std::mem::size_of::<<Handwritten as Residual>::Undo>(), 0);
        assert_eq!(std::mem::size_of::<Accumulator>(), 256);
        assert_eq!(
            std::mem::size_of::<Snapshot<Accumulator>>(),
            std::mem::size_of::<Snapshot<()>>() + 256
        );
        let granite = Search::<9>::with_hash_size(HashSize::MiB4);
        assert!(granite.model.is_none());
        assert_eq!(granite.model_fingerprint(), None);
        let info = crate::engines::info(granite.name()).unwrap();
        assert!(!info.neural_accumulator && !info.neural_evaluation);
        let abacus = Search::<1>::with_hash_size(HashSize::MiB4);
        assert!(abacus.model.is_some());
        assert!(
            crate::engines::info(abacus.name())
                .unwrap()
                .neural_accumulator
        );
        assert!(
            std::panic::catch_unwind(|| {
                Search::<9>::with_model("invalid-neural-granite", HashSize::MiB4, Model::embedded())
            })
            .is_err()
        );
        eprintln!(
            "GRANITE_LAYOUT neural_board={} handwritten_board={} neural_undo={} handwritten_undo={}",
            std::mem::size_of::<Board>(),
            std::mem::size_of::<Board<Handwritten>>(),
            std::mem::size_of::<Snapshot<Accumulator>>(),
            std::mem::size_of::<Snapshot<()>>()
        );
    }

    #[test]
    fn granite_matches_abacus_at_tiny_and_full_budgets_with_retained_history() {
        let mut positions: Vec<_> = [
            "8/7k/4RB2/8/4N3/8/8/K7 w - - 0 1",
            "7k/P7/R7/8/8/8/8/K7 w - - 0 1",
            "r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1",
            "7k/8/8/8/3p4/8/4P3/K7 b - - 0 1",
        ]
        .map(|fen| Position::try_from_fen(fen).unwrap())
        .into();
        let mut state = crate::selfplay::State::default();
        for ply in 0..48 {
            if state.white_value().is_some() {
                state = crate::selfplay::State::default();
            }
            if ply % 8 == 0 {
                positions.push(state.position().clone());
            }
            let legal = state.legal_moves();
            state
                .play(legal[(ply * 29 + 5) % legal.len()].id())
                .unwrap();
        }
        let mut abacus = Search::<1>::with_hash_size(HashSize::MiB4);
        let mut granite = Search::<9>::with_hash_size(HashSize::MiB4);
        for (root, pos) in positions.iter().enumerate() {
            for nodes in [1, 257, 8192] {
                abacus.new_game(pos.side_to_move, root as u64);
                granite.new_game(pos.side_to_move, root as u64);
                // The second call retains both transposition and history state.
                for _ in 0..2 {
                    let observe = |engine: &mut dyn Engine| {
                        let mut copy = pos.clone();
                        let (mv, stats) = engine.choose_move(
                            &mut copy,
                            &SearchBudget {
                                max_nodes: nodes,
                                ..SearchBudget::default()
                            },
                        );
                        assert_eq!(copy.to_fen(), pos.to_fen());
                        assert_eq!(copy.zobrist, pos.zobrist);
                        assert_eq!(copy.undo_stack.len(), pos.undo_stack.len());
                        assert!(stats.nodes <= nodes as u64);
                        (
                            mv,
                            stats.nodes,
                            stats.depth_reached,
                            stats.seldepth,
                            stats.eval_cp,
                            stats.diagnostics,
                            stats.pv,
                        )
                    };
                    assert_eq!(
                        observe(&mut abacus),
                        observe(&mut granite),
                        "root={root} nodes={nodes}"
                    );
                }
            }
        }
    }

    #[test]
    fn respects_nodes_and_restores_callers_position() {
        let mut engine = Cataclysm::new();
        let mut pos = start_position();
        let before = pos.to_fen();
        let (mv, stats) = engine.choose_move(
            &mut pos,
            &SearchBudget {
                max_nodes: 2000,
                ..SearchBudget::default()
            },
        );
        let mut legal = Vec::new();
        generate_legal_moves(&mut pos, &mut legal);
        assert!(legal.contains(&mv));
        assert!(stats.nodes <= 2000);
        assert_eq!(pos.to_fen(), before);
        assert!(pos.undo_stack.is_empty());
    }

    #[test]
    fn new_game_discards_prior_action_context() {
        let mut engine = Search::<7>::new();
        let observe = |engine: &mut Search<7>| {
            let mut pos =
                Position::try_from_fen("r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1").unwrap();
            let (mv, stats) = engine.choose_move(
                &mut pos,
                &SearchBudget {
                    max_nodes: 8192,
                    ..SearchBudget::default()
                },
            );
            (
                mv,
                stats.nodes,
                stats.depth_reached,
                stats.seldepth,
                stats.eval_cp,
                stats.diagnostics,
                stats.pv,
            )
        };
        engine.new_game(Color::White, 0);
        let expected = observe(&mut engine);
        // Valid counter-table indices that could be left by other games.
        // Deliberately poison every ply, not just those this fixture reaches.
        for previous in [1, 127, 895] {
            engine.previous.fill(previous);
            engine.new_game(Color::White, 0);
            assert!(engine.previous.iter().all(|&index| index == 0));
            assert_eq!(observe(&mut engine), expected);
        }
    }

    #[test]
    fn waypoint_counter_order_requires_a_real_parent() {
        let mut engine = Search::<8>::with_hash_size(HashSize::MiB4);
        let pos = start_position();
        let board = Board::new(&pos);
        let orders = |engine: &mut Search<8>| {
            engine
                .prepare(&board, 1, 0)
                .into_iter()
                .map(|a| (pack(a.mv), a.order))
                .collect::<Vec<_>>()
        };
        let baseline = orders(&mut engine);
        let chosen = baseline[0].0;
        engine.counters[0] = chosen;
        assert_eq!(engine.parent_context(0), None);
        assert_eq!(engine.parent_context(1), None);
        assert_eq!(orders(&mut engine), baseline);
        engine.previous[0] = 80;
        engine.counters[80] = chosen;
        assert_eq!(engine.parent_context(1), Some(80));
        for ((id, score), (before_id, before_score)) in orders(&mut engine).iter().zip(&baseline) {
            assert_eq!(id, before_id);
            assert_eq!(
                *score,
                before_score + if *id == chosen { 70_000 } else { 0 }
            );
        }
    }

    #[test]
    fn waypoint_records_tactical_and_proof_edges_before_piece_changes() {
        for (fen, expected) in [
            ("7k/8/8/8/8/1p6/P7/K7 w - - 0 1", 81),
            ("7k/P7/8/8/8/8/8/K7 w - - 0 1", 120),
        ] {
            let mut engine = Search::<8>::with_hash_size(HashSize::MiB4);
            let pos = Position::try_from_fen(fen).unwrap();
            let mut board = Board::new(&pos);
            engine.previous.fill(895);
            engine.limits.begin(
                &SearchBudget {
                    max_nodes: 2,
                    ..SearchBudget::default()
                },
                94,
            );
            engine.qsearch(&mut board, -INF, INF, 0, 0);
            assert_eq!(engine.previous[0], expected, "{fen}");
            assert_eq!(board.pos.to_fen(), pos.to_fen());
        }
        let pos = Position::try_from_fen("7k/5K2/6Q1/8/8/8/8/8 w - - 0 1").unwrap();
        let mut board = Board::new(&pos);
        let mut engine = Search::<8>::with_hash_size(HashSize::MiB4);
        engine.previous.fill(895);
        engine.limits.begin(
            &SearchBudget {
                max_nodes: 4096,
                ..SearchBudget::default()
            },
            94,
        );
        let line = engine.siege(&mut board, 1).expect("mate in one");
        assert_eq!(line.len(), 1);
        assert_eq!(
            engine.previous[0],
            Search::<8>::action_context(&board, line[0])
        );
        assert_eq!(board.pos.to_fen(), pos.to_fen());
    }

    #[test]
    fn waypoint_restores_parent_context_after_interrupted_null_search() {
        let pos = start_position();
        let mut board = Board::new(&pos);
        let mut engine = Search::<8>::with_hash_size(HashSize::MiB4);
        engine.root_depth = 6;
        engine.previous[1] = 89;
        engine.limits.begin(
            &SearchBudget {
                max_nodes: 2,
                ..SearchBudget::default()
            },
            94,
        );
        let beta = evaluate::<8>(&board) - 1;
        engine.search(&mut board, 6, beta - 1, beta, 1, true);
        assert_eq!(engine.limits.nodes, 2);
        assert!(engine.limits.stopped);
        assert_eq!(engine.previous[1], 89);
        assert_eq!(board.pos.to_fen(), pos.to_fen());
        assert_eq!(board.pos.zobrist, pos.zobrist);
    }

    #[test]
    fn candidate_network_is_fixed_shared_and_used_by_real_search() {
        let model = Model::decode(Model::CONTROL_BYTES).unwrap();
        let other = model.clone();
        assert!(std::ptr::eq(model.network(), other.network()));
        let mut control = Cataclysm::with_hash_size(HashSize::MiB4);
        let mut candidate = Cataclysm::with_model("control-copy", HashSize::MiB4, model);
        let budget = SearchBudget {
            max_nodes: 2048,
            ..SearchBudget::default()
        };
        let mut state = crate::selfplay::State::default();
        for ply in 0..12 {
            let mut pos = state.position().clone();
            control.new_game(Color::White, 0);
            candidate.new_game(Color::White, 0);
            let (a, sa) = control.choose_move(&mut pos, &budget);
            let (b, sb) = candidate.choose_move(&mut pos, &budget);
            assert_eq!(
                (a, sa.nodes, sa.eval_cp, sa.depth_reached, sa.pv),
                (b, sb.nodes, sb.eval_cp, sb.depth_reached, sb.pv)
            );
            assert_eq!(pos.to_fen(), state.position().to_fen());
            let legal = state.legal_moves();
            if legal.is_empty() {
                break;
            }
            state
                .play(legal[(ply * 19 + 7) % legal.len()].id())
                .unwrap();
        }
        let zero = vec![0; Model::BYTES];
        let zero_model = Model::decode(&zero).unwrap();
        let expected = Model::decode(&zero).unwrap();
        let mut different =
            Cataclysm::with_model("zero-fixture", HashSize::MiB4, zero_model.clone());
        assert_ne!(different.model_fingerprint(), control.model_fingerprint());
        let mut pos = state.position().clone();
        // A one-node interruption reports the initial static score. This pins
        // that the actual search board uses this candidate, not the control.
        let (_, stats) = different.choose_move(
            &mut pos,
            &SearchBudget {
                max_nodes: 1,
                ..SearchBudget::default()
            },
        );
        assert_eq!(stats.eval_cp, expected.evaluate(&pos));
        assert_eq!(stats.depth_reached, 0);
        assert_eq!(
            Some(Network::embedded().fingerprint),
            control.model_fingerprint()
        );
        for fen in [
            "8/7k/4RB2/8/4N3/8/8/K7 w - - 0 1",
            "7k/P7/R7/8/8/8/8/K7 w - - 0 1",
            "r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1",
        ] {
            verify_rules_with_model(&Position::try_from_fen(fen).unwrap(), &zero_model).unwrap();
            verify_rules_with_model(&Position::try_from_fen(fen).unwrap(), &other).unwrap();
        }
    }

    #[test]
    fn mate_encoding_is_root_independent() {
        for ply in 0..100 {
            for value in [-29_999, -100, 0, 100, 29_999] {
                assert_eq!(decode(encode(value, ply), ply), value);
            }
        }
    }

    #[test]
    fn embedded_model_is_shared_and_search_is_deterministic() {
        assert!(Network::decode(&[0; 10]).is_err());
        let mut embedded = Cataclysm::new();
        let mut repeated = Cataclysm::new();
        assert!(std::ptr::eq(Network::embedded(), Network::embedded()));
        let budget = SearchBudget {
            max_nodes: 5000,
            ..SearchBudget::default()
        };
        let mut pos = start_position();
        let (a, sa) = embedded.choose_move(&mut pos, &budget);
        let (b, sb) = repeated.choose_move(&mut pos, &budget);
        assert_eq!(a, b);
        assert_eq!(sa.eval_cp, sb.eval_cp);
        assert_eq!(sa.nodes, sb.nodes);
    }

    #[test]
    fn siege_proves_mate_and_search_handles_quiet_evasions_and_stalemate() {
        let mut engine = Cataclysm::new();
        let mut pos = Position::empty();
        pos.set_from_fen("7k/5K2/6Q1/8/8/8/8/8 w - - 0 1");
        let (mv, stats) = engine.choose_move(
            &mut pos,
            &SearchBudget {
                max_depth: 2,
                ..SearchBudget::default()
            },
        );
        assert!(stats.eval_cp >= WIN);
        pos.make_move(&mv);
        let mut legal = Vec::new();
        generate_legal_moves(&mut pos, &mut legal);
        assert!(legal.is_empty() && pos.in_check());
        engine.limits.stopped = false;
        engine.limits.limit = 0;
        engine.limits.micros = 0;
        engine.path.clear();
        pos.set_from_fen("k3r3/8/8/8/8/8/8/4K3 w - - 0 1");
        let mut board = Board::new(&pos);
        assert!(engine.qsearch(&mut board, -INF, INF, 0, 0) > -WIN);
        pos.set_from_fen("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1");
        let mut board = Board::new(&pos);
        assert_eq!(engine.qsearch(&mut board, -INF, INF, 0, 0), 0);
        assert_eq!(engine.qsearch(&mut board, -2000, -1999, 0, 0), 0);
    }

    #[test]
    fn mate_precedes_fifty_move_draw_but_checked_legal_positions_draw() {
        let mut engine = Cataclysm::new();
        let mut pos = Position::empty();
        pos.set_from_fen("7k/6Q1/5K2/8/8/8/8/8 b - - 100 1");
        assert_eq!(
            engine.qsearch(&mut Board::new(&pos), -INF, INF, 0, 0),
            -MATE
        );
        pos.set_from_fen("k3r3/8/8/8/8/8/8/4K3 w - - 100 1");
        assert_eq!(engine.qsearch(&mut Board::new(&pos), -INF, INF, 0, 0), 0);
    }

    #[test]
    fn main_search_static_pruning_preserves_stalemate() {
        let mut engine = Cataclysm::new();
        let mut pos = Position::empty();
        pos.set_from_fen("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1");
        for depth in 1..=5 {
            assert_eq!(
                engine.search(&mut Board::new(&pos), depth, -2501, -2500, 1, true),
                0
            );
        }
    }
}
