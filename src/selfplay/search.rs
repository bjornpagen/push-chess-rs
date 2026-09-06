//! Gumbel policy improvement (Danihelka et al., ICLR 2022).
//! Sequential-halving schedule adapted from DeepMind mctx (Apache-2.0).
//! Copyright 2021 DeepMind Technologies Limited. See training/THIRD_PARTY.md.
use super::{
    Features, State,
    cursor::Cursor,
    encoding::{BOARD_FLOATS, write_board},
    state::white_value,
};
use crate::core::prepared::{MoveScratch, PreparedMove, generate_prepared};
use crate::core::types::Color;
use crate::game::adjudicate_with_repetitions;
use std::{
    num::NonZeroU32,
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

static NEXT_REQUEST: AtomicU64 = AtomicU64::new(1);

pub(super) fn next_request_id() -> Result<u64, &'static str> {
    NEXT_REQUEST
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map_err(|_| "request ID overflow")
}

#[derive(Clone, Copy)]
enum Kind {
    Leaf,
    Branch(f32),
    Terminal(f32),
}

struct Node {
    start: usize,
    end: usize,
    kind: Kind,
}

/// Selection scans only these columns. Plans/links are cold during reductions.
#[derive(Default)]
struct Arena {
    nodes: Vec<Node>,
    moves: Vec<PreparedMove>,
    children: Vec<Option<NonZeroU32>>,
    visits: Vec<u32>,
    sums: Vec<f32>,
    logits: Vec<f32>,
    priors: Vec<f32>,
}

impl Arena {
    fn range(&self, node: usize) -> Range<usize> {
        self.nodes[node].start..self.nodes[node].end
    }

    fn leaf(&mut self, moves: impl IntoIterator<Item = PreparedMove>) -> usize {
        let start = self.moves.len();
        self.moves.extend(moves);
        let end = self.moves.len();
        self.children.resize(end, None);
        self.visits.resize(end, 0);
        self.sums.resize(end, 0.0);
        self.logits.resize(end, 0.0);
        self.priors.resize(end, 0.0);
        let index = self.nodes.len();
        self.nodes.push(Node {
            start,
            end,
            kind: Kind::Leaf,
        });
        index
    }

    fn transformed_q(&self, node: usize, q: &mut Vec<f32>) {
        let Kind::Branch(raw) = self.nodes[node].kind else {
            unreachable!("expanded node")
        };
        let range = self.range(node);
        let visits = &self.visits[range.clone()];
        let sums = &self.sums[range.clone()];
        let priors = &self.priors[range];
        let total: u32 = visits.iter().sum();
        let mass: f32 = visits
            .iter()
            .zip(priors)
            .filter(|(v, _)| **v > 0)
            .map(|(_, p)| p.max(1e-30))
            .sum();
        let weighted: f32 = visits
            .iter()
            .zip(priors)
            .zip(sums)
            .filter(|((v, _), _)| **v > 0)
            .map(|((&n, &p), &sum)| p.max(1e-30) / mass * sum / n as f32)
            .sum();
        let mixed = (raw + total as f32 * weighted) / (1 + total) as f32;
        q.clear();
        q.extend(
            visits
                .iter()
                .zip(sums)
                .map(|(&n, &sum)| if n == 0 { mixed } else { sum / n as f32 }),
        );
        let lo = q.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = q.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = (50 + visits.iter().copied().max().unwrap_or(0)) as f32 * 0.1;
        for v in q {
            *v = (*v - lo) / (hi - lo).max(1e-8) * scale;
        }
    }

    fn improved(&self, node: usize, scratch: &mut Vec<f32>) {
        self.transformed_q(node, scratch);
        for (p, &logit) in scratch.iter_mut().zip(&self.logits[self.range(node)]) {
            *p += logit;
        }
        softmax(scratch);
    }

    fn select(&self, node: usize, scratch: &mut Vec<f32>) -> usize {
        self.improved(node, scratch);
        let range = self.range(node);
        let visits = &self.visits[range.clone()];
        let total = 1 + visits.iter().sum::<u32>();
        range.start
            + argmax(
                scratch
                    .iter()
                    .zip(visits)
                    .map(|(p, n)| *p - *n as f32 / total as f32),
            )
    }

    fn root_action(&self, noise: &[f32], count: u32, scratch: &mut Vec<f32>) -> usize {
        self.transformed_q(0, scratch);
        let range = self.range(0);
        range.start
            + argmax(
                self.logits[range.clone()]
                    .iter()
                    .zip(noise)
                    .zip(scratch.iter())
                    .zip(&self.visits[range])
                    .map(|(((logit, noise), q), &visits)| {
                        if visits == count {
                            logit + noise + q
                        } else {
                            f32::NEG_INFINITY
                        }
                    }),
            )
    }

    fn bytes(&self) -> usize {
        self.nodes.capacity() * std::mem::size_of::<Node>()
            + self.moves.capacity() * std::mem::size_of::<PreparedMove>()
            + self.children.capacity() * std::mem::size_of::<Option<NonZeroU32>>()
            + 4 * (self.visits.capacity()
                + self.sums.capacity()
                + self.logits.capacity()
                + self.priors.capacity())
    }

    fn clear(&mut self) {
        self.nodes.clear();
        self.moves.clear();
        self.children.clear();
        self.visits.clear();
        self.sums.clear();
        self.logits.clear();
        self.priors.clear();
    }
}

fn argmax(xs: impl Iterator<Item = f32>) -> usize {
    xs.enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (i, x)| {
            if x > best.1 { (i, x) } else { best }
        })
        .0
}

fn softmax(xs: &mut [f32]) {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for x in xs.iter_mut() {
        *x = (*x - max).exp();
    }
    let sum: f32 = xs.iter().sum();
    for x in xs {
        *x /= sum;
    }
}

pub fn considered_visits(candidates: usize, simulations: usize) -> Vec<u32> {
    assert!(candidates > 0 && simulations > 0);
    if candidates == 1 {
        return (0..simulations as u32).collect();
    }
    let rounds = candidates.next_power_of_two().ilog2() as usize;
    let mut sequence = Vec::with_capacity(simulations);
    let mut visits = vec![0; candidates];
    let mut count = candidates;
    while sequence.len() < simulations {
        for _ in 0..(simulations / (rounds * count)).max(1) {
            sequence.extend_from_slice(&visits[..count]);
            for v in &mut visits[..count] {
                *v += 1;
            }
        }
        count = (count / 2).max(2);
    }
    sequence.truncate(simulations);
    sequence
}

/// Detached root input: no clone of the original game's large undo frames.
pub struct SearchRoot {
    cursor: Cursor,
    moves: Vec<PreparedMove>,
}

impl SearchRoot {
    pub fn from_state(state: &State) -> Self {
        Self {
            cursor: Cursor::from_state(state),
            moves: state.prepared.clone(),
        }
    }
    pub fn action_count(&self) -> usize {
        self.moves.len()
    }
}

#[derive(Clone, Copy)]
pub struct SearchOptions {
    pub effects: bool,
    /// Failing this explicit capacity guard is an error, never a fabricated draw.
    pub max_nodes_per_tree: usize,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            effects: false,
            max_nodes_per_tree: 16_384,
        }
    }
}

struct Tree {
    cursor: Cursor,
    arena: Arena,
    path: Vec<usize>,
    pending: Option<usize>,
    noise: Vec<f32>,
    schedule: Vec<u32>,
    selection: Vec<f32>,
    generation: MoveScratch,
    generated: Vec<PreparedMove>,
    max_nodes: usize,
}

impl Tree {
    fn new(
        root: SearchRoot,
        noise: Vec<f32>,
        simulations: usize,
        candidates: usize,
        max_nodes: usize,
    ) -> Self {
        let count = root.moves.len();
        let mut arena = Arena::default();
        arena
            .nodes
            .reserve((simulations + 1).min(max_nodes).min(4096));
        arena.leaf(root.moves);
        Self {
            cursor: root.cursor,
            arena,
            path: Vec::with_capacity(64),
            pending: None,
            noise,
            schedule: considered_visits(candidates.min(simulations).min(count), simulations),
            selection: Vec::with_capacity(count),
            generation: MoveScratch::default(),
            generated: Vec::with_capacity(count),
            max_nodes,
        }
    }

    fn restart(
        &mut self,
        root: SearchRoot,
        noise: Vec<f32>,
        simulations: usize,
        candidates: usize,
        max_nodes: usize,
    ) {
        let count = root.moves.len();
        self.cursor = root.cursor;
        self.arena.clear();
        self.arena.leaf(root.moves);
        self.path.clear();
        self.pending = None;
        self.noise = noise;
        self.schedule = considered_visits(candidates.min(simulations).min(count), simulations);
        self.max_nodes = max_nodes;
    }

    fn backup(&mut self, mut value: f32) {
        for &edge in self.path.iter().rev() {
            value = -value;
            self.arena.visits[edge] += 1;
            self.arena.sums[edge] += value;
        }
    }

    fn rewind(&mut self) {
        for _ in &self.path {
            self.cursor.pos.unmake_move();
        }
    }

    fn new_node(&mut self) -> usize {
        self.generated.clear();
        generate_prepared(
            &mut self.cursor.pos,
            &mut self.generated,
            &mut self.generation,
        );
        if let Some(v) = white_value(&adjudicate_with_repetitions(
            &self.cursor.pos,
            self.generated.is_empty(),
            self.cursor.repetitions(),
        )) {
            let value = if self.cursor.pos.side_to_move == Color::White {
                v
            } else {
                -v
            };
            let index = self.arena.nodes.len();
            self.arena.nodes.push(Node {
                start: 0,
                end: 0,
                kind: Kind::Terminal(value),
            });
            index
        } else {
            self.generated.sort_unstable_by_key(|m| m.mv().id());
            if self.selection.capacity() < self.generated.len() {
                self.selection
                    .reserve(self.generated.len() - self.selection.len());
            }
            self.arena.leaf(self.generated.drain(..))
        }
    }

    fn request(&mut self, step: Option<usize>) -> Result<bool, &'static str> {
        debug_assert!(self.pending.is_none());
        self.path.clear();
        let mut index = 0;
        loop {
            match self.arena.nodes[index].kind {
                Kind::Terminal(v) => {
                    self.backup(v);
                    self.rewind();
                    return Ok(false);
                }
                Kind::Leaf => {
                    self.pending = Some(index);
                    // Remain at the leaf until it has been encoded directly.
                    return Ok(true);
                }
                Kind::Branch(_) => {
                    let edge = if index == 0 {
                        self.arena.root_action(
                            &self.noise,
                            self.schedule[step.expect("root evaluated")],
                            &mut self.selection,
                        )
                    } else {
                        self.arena.select(index, &mut self.selection)
                    };
                    let child = self.arena.children[edge];
                    if child.is_none() && self.arena.nodes.len() >= self.max_nodes {
                        self.rewind();
                        return Err("search arena node limit reached");
                    }
                    self.arena.moves[edge].apply(&mut self.cursor.pos);
                    self.path.push(edge);
                    index = if let Some(child) = child {
                        child.get() as usize
                    } else {
                        let child = self.new_node();
                        self.arena.children[edge] = NonZeroU32::new(child as u32);
                        child
                    };
                }
            }
        }
    }

    fn pending_range(&self) -> Range<usize> {
        self.arena.range(self.pending.expect("awaiting evaluation"))
    }

    fn submit(&mut self, logits: &[f32], value: f32) {
        let index = self.pending.take().expect("validated reply");
        let range = self.arena.range(index);
        self.arena.logits[range.clone()].copy_from_slice(logits);
        self.arena.priors[range.clone()].copy_from_slice(logits);
        softmax(&mut self.arena.priors[range]);
        self.arena.nodes[index].kind = Kind::Branch(value);
        self.backup(value);
    }

    fn result(&self) -> SearchResult {
        let range = self.arena.range(0);
        let visits = self.arena.visits[range.clone()].to_vec();
        let mut scratch = Vec::with_capacity(range.len());
        let chosen =
            self.arena
                .root_action(&self.noise, *visits.iter().max().unwrap(), &mut scratch);
        self.arena.improved(0, &mut scratch);
        let value = if let Kind::Branch(raw) = self.arena.nodes[0].kind {
            raw
        } else {
            unreachable!()
        };
        SearchResult {
            mv: self.arena.moves[chosen].mv().id(),
            policy: scratch,
            visits,
            nodes: self.arena.nodes.len(),
            root_value: value,
            selected_value: if self.arena.visits[chosen] == 0 {
                value
            } else {
                self.arena.sums[chosen] / self.arena.visits[chosen] as f32
            },
        }
    }
}

#[derive(Clone, Copy)]
enum Phase {
    Ready(Option<usize>),
    Awaiting(Option<usize>),
    Finished,
    Failed,
}

#[derive(Clone)]
pub struct SearchResult {
    pub mv: u32,
    pub policy: Vec<f32>,
    pub visits: Vec<u32>,
    pub nodes: usize,
    pub root_value: f32,
    pub selected_value: f32,
}

#[derive(Default, serde::Serialize)]
pub struct SearchMetrics {
    pub nodes: usize,
    pub edges: usize,
    pub arena_bytes: usize,
    pub neural_rounds: u64,
}

/// Each request transfers fresh final-buffer ownership. Tree scratch, edges and
/// pending mappings are reused. Replies validate completely before any backup.
pub struct BatchSearch {
    trees: Vec<Tree>,
    pending: Vec<usize>,
    simulations: usize,
    phase: Phase,
    options: SearchOptions,
    request_id: u64,
    rounds: u64,
    stopped: bool,
    cancellation: Option<Arc<AtomicBool>>,
    effect_tokens: Vec<[i32; 4]>,
    effect_offsets: Vec<usize>,
}

impl BatchSearch {
    pub fn new(
        states: Vec<State>,
        noises: Vec<Vec<f32>>,
        simulations: usize,
        candidates: usize,
    ) -> Result<Self, &'static str> {
        let roots = states.iter().map(SearchRoot::from_state).collect();
        Self::with_options(
            roots,
            noises,
            simulations,
            candidates,
            SearchOptions::default(),
        )
    }

    pub fn with_options(
        roots: Vec<SearchRoot>,
        noises: Vec<Vec<f32>>,
        simulations: usize,
        candidates: usize,
        options: SearchOptions,
    ) -> Result<Self, &'static str> {
        if simulations == 0
            || simulations > 1_000_000
            || candidates == 0
            || roots.is_empty()
            || roots.len() != noises.len()
            || options.max_nodes_per_tree < 1
            || options.max_nodes_per_tree > u32::MAX as usize
        {
            return Err("invalid search batch, capacity or budget");
        }
        if roots.iter().zip(&noises).any(|(s, n)| {
            s.moves.is_empty() || s.moves.len() != n.len() || n.iter().any(|x| !x.is_finite())
        }) {
            return Err("terminal root or invalid Gumbel noise");
        }
        let count = roots.len();
        let trees = roots
            .into_iter()
            .zip(noises)
            .map(|(root, noise)| {
                Tree::new(
                    root,
                    noise,
                    simulations,
                    candidates,
                    options.max_nodes_per_tree,
                )
            })
            .collect();
        Ok(Self {
            trees,
            pending: Vec::with_capacity(count),
            simulations,
            phase: Phase::Ready(None),
            options,
            request_id: 0,
            rounds: 0,
            stopped: false,
            cancellation: None,
            effect_tokens: Vec::new(),
            effect_offsets: Vec::with_capacity(count + 1),
        })
    }

    pub fn stop(&mut self) {
        self.stopped = true;
    }

    pub fn set_cancellation(&mut self, flag: Arc<AtomicBool>) {
        self.cancellation = Some(flag);
    }

    /// A worker retains its arenas/scratch between moves and between games.
    pub fn restart(
        &mut self,
        roots: Vec<SearchRoot>,
        noises: Vec<Vec<f32>>,
        simulations: usize,
        candidates: usize,
        options: SearchOptions,
    ) -> Result<(), &'static str> {
        if !matches!(self.phase, Phase::Finished) {
            return Err("finish the previous search before reuse");
        }
        if roots.is_empty()
            || roots.len() != noises.len()
            || simulations == 0
            || simulations > 1_000_000
            || candidates == 0
            || options.max_nodes_per_tree == 0
            || options.max_nodes_per_tree > u32::MAX as usize
            || roots.iter().zip(&noises).any(|(r, n)| {
                r.moves.is_empty() || r.moves.len() != n.len() || n.iter().any(|x| !x.is_finite())
            })
        {
            return Err("invalid search restart");
        }
        self.trees.truncate(roots.len());
        for (i, (root, noise)) in roots.into_iter().zip(noises).enumerate() {
            if let Some(tree) = self.trees.get_mut(i) {
                tree.restart(
                    root,
                    noise,
                    simulations,
                    candidates,
                    options.max_nodes_per_tree,
                );
            } else {
                self.trees.push(Tree::new(
                    root,
                    noise,
                    simulations,
                    candidates,
                    options.max_nodes_per_tree,
                ));
            }
        }
        self.pending.clear();
        self.simulations = simulations;
        self.options = options;
        self.phase = Phase::Ready(None);
        self.stopped = false;
        self.rounds = 0;
        Ok(())
    }

    pub fn request_id(&self) -> Option<u64> {
        matches!(self.phase, Phase::Awaiting(_)).then_some(self.request_id)
    }

    fn advance_step(&mut self, step: Option<usize>) {
        let next = step.map_or(0, |s| s + 1);
        self.phase = if next == self.simulations || self.stopped {
            Phase::Finished
        } else {
            Phase::Ready(Some(next))
        };
    }

    pub fn request(&mut self) -> Result<Option<Features>, &'static str> {
        loop {
            if self
                .cancellation
                .as_ref()
                .is_some_and(|flag| flag.load(Ordering::Relaxed))
            {
                self.stopped = true;
            }
            let step = match self.phase {
                Phase::Ready(step) => step,
                Phase::Awaiting(_) => return Err("submit the pending batch first"),
                Phase::Finished => return Ok(None),
                Phase::Failed => return Err("search failed; discard it"),
            };
            if self.stopped && step.is_some() {
                self.phase = Phase::Finished;
                return Ok(None);
            }
            self.pending.clear();
            for (i, tree) in self.trees.iter_mut().enumerate() {
                match tree.request(step) {
                    Ok(true) => self.pending.push(i),
                    Ok(false) => {}
                    Err(error) => {
                        self.phase = Phase::Failed;
                        return Err(error);
                    }
                }
            }
            if self.pending.is_empty() {
                self.advance_step(step);
                continue;
            }
            let width = self
                .pending
                .iter()
                .map(|&i| self.trees[i].pending_range().len())
                .max()
                .unwrap()
                .max(16)
                .next_power_of_two();
            let mut batch = Features::empty(self.pending.len(), width);
            self.effect_tokens.clear();
            self.effect_offsets.clear();
            self.effect_offsets.push(0);
            for (row, &i) in self.pending.iter().enumerate() {
                let tree = &mut self.trees[i];
                let previous = tree.cursor.previous_board();
                write_board(
                    &tree.cursor.pos,
                    previous.as_ref(),
                    tree.cursor.repetitions(),
                    &mut batch.boards[row * BOARD_FLOATS..(row + 1) * BOARD_FLOATS],
                );
                let moves = &tree.arena.moves[tree.pending_range()];
                batch.write_actions(row, &tree.cursor.pos, moves);
                if self.options.effects {
                    for (a, mv) in moves.iter().enumerate() {
                        mv.effects(&tree.cursor.pos, a, &mut self.effect_tokens);
                    }
                    self.effect_offsets.push(self.effect_tokens.len());
                }
                tree.rewind();
            }
            if self.options.effects {
                batch.pack_effects(&self.effect_tokens, &self.effect_offsets);
            }
            self.request_id = next_request_id()?;
            self.rounds += 1;
            self.phase = Phase::Awaiting(step);
            return Ok(Some(batch));
        }
    }

    pub fn submit_for(
        &mut self,
        id: u64,
        logits: &[f32],
        values: &[f32],
        width: usize,
    ) -> Result<(), &'static str> {
        if self.request_id() != Some(id) {
            return Err("stale or duplicate neural reply");
        }
        self.submit(logits, values, width)
    }

    pub fn submit(
        &mut self,
        logits: &[f32],
        values: &[f32],
        width: usize,
    ) -> Result<(), &'static str> {
        let Phase::Awaiting(step) = self.phase else {
            return Err("no pending neural batch");
        };
        if values.len() != self.pending.len()
            || self.pending.len().checked_mul(width) != Some(logits.len())
            || values.iter().any(|v| !v.is_finite() || v.abs() > 1.00001)
            || self.pending.iter().enumerate().any(|(i, &t)| {
                let n = self.trees[t].pending_range().len();
                n > width
                    || logits[i * width..i * width + n]
                        .iter()
                        .any(|x| !x.is_finite())
            })
        {
            return Err("invalid neural batch shapes or values");
        }
        for (i, &t) in self.pending.iter().enumerate() {
            let n = self.trees[t].pending_range().len();
            self.trees[t].submit(&logits[i * width..i * width + n], values[i]);
        }
        self.advance_step(step);
        Ok(())
    }

    pub fn results(&self) -> Result<Vec<SearchResult>, &'static str> {
        if !matches!(self.phase, Phase::Finished) {
            return Err("search is not finished");
        }
        Ok(self.trees.iter().map(Tree::result).collect())
    }

    pub fn metrics(&self) -> SearchMetrics {
        SearchMetrics {
            nodes: self.trees.iter().map(|t| t.arena.nodes.len()).sum(),
            edges: self.trees.iter().map(|t| t.arena.moves.len()).sum(),
            arena_bytes: self.trees.iter().map(|t| t.arena.bytes()).sum(),
            neural_rounds: self.rounds,
        }
    }
}
