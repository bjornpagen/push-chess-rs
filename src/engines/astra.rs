//! Astra 001: fast legal search, check-aware tactical leaves, and push geometry.
//! No opponent-specific behavior, opening book, or access to tournament results.
#![forbid(unsafe_code)]

use crate::engine::shared::{Buckets, SearchLimits};
use std::sync::LazyLock;

use super::support::ScoredMoves;
use crate::core::movegen::{generate_legal_moves, generate_pseudo_legal_moves};
use crate::core::position::Position;
use crate::core::types::*;
use crate::core::zobrist::zobrist_tables;
use crate::engine::Engine;

const MATE: i32 = 30_000;
const MATE_BOUND: i32 = MATE - 256;
const INF: i32 = 31_000;
const MAX_PLY: usize = 128;
const BUCKETS: usize = 1 << 20;
const VALUES: [i32; 7] = [0, 100, 310, 345, 525, 980, 0];

/// Entirely handwritten evaluators: no network load, inference or neural
/// accumulator. Keep the historical search as profile zero for comparison.
#[derive(Clone, Copy)]
pub struct Profile {
    pub name: &'static str,
    pub hypothesis: &'static str,
    pressure: bool,
    races: bool,
    tactical: bool,
    conservative: bool,
}
const BASE: Profile = Profile {
    name: "astra",
    hypothesis: "Historical neural-free Astra control",
    pressure: false,
    races: false,
    tactical: false,
    conservative: false,
};
pub const PROFILES: [Profile; 7] = [
    BASE,
    Profile {
        name: "sentinel",
        hypothesis: "Terminal witnesses protect mate/draw precedence and static stalemate cutoffs",
        ..BASE
    },
    Profile {
        name: "bastion",
        hypothesis: "Sentinel plus obstruction-aware king lines and king-ring pressure",
        pressure: true,
        ..BASE
    },
    Profile {
        name: "outrider",
        hypothesis: "Sentinel plus promotion races, king interception distance and rear push support",
        races: true,
        ..BASE
    },
    Profile {
        name: "tactician",
        hypothesis: "Sentinel plus wider checking horizon and unreduced pushes",
        tactical: true,
        ..BASE
    },
    Profile {
        name: "bedrock",
        hypothesis: "Sentinel without speculative null/static/late-move/delta pruning or LMR",
        conservative: true,
        ..BASE
    },
    Profile {
        name: "meridian",
        hypothesis: "Sentinel plus pressure, races and tactical search",
        pressure: true,
        races: true,
        tactical: true,
        ..BASE
    },
];

fn king_pressure(pos: &Position, color: Color) -> i32 {
    let king = pos.king_sq[color as usize];
    let mut attacked = 0;
    let mut space = 0;
    for dr in -1..=1 {
        for df in -1..=1 {
            if dr == 0 && df == 0 {
                continue;
            }
            let (r, f) = (rank_of(king) + dr, file_of(king) + df);
            if !valid_rf(r, f) {
                continue;
            }
            let sq = make_square(r, f);
            if pos.is_attacked_by(sq, opponent(color)) {
                attacked += 1;
            } else if !pos.board[sq as usize].is_color(color) {
                space += 1;
            }
        }
    }
    // Influence in the CURRENT occupancy, not a proof that a king move is
    // legal after its source square is vacated or friendly pieces are pushed.
    space * 8 - attacked * attacked * 6
}

fn clear_line_pressure(pos: &Position, from: Square, king: Square) -> i32 {
    let (dr, df) = (
        (rank_of(king) - rank_of(from)).signum(),
        (file_of(king) - file_of(from)).signum(),
    );
    let color = pos.board[from as usize].color;
    let (mut r, mut f) = (rank_of(from) + dr, file_of(from) + df);
    let mut friends = 0;
    while valid_rf(r, f) && make_square(r, f) != king {
        let p = pos.board[make_square(r, f) as usize];
        if !p.is_empty() {
            if p.color != color {
                return 0;
            }
            friends += 1;
        }
        r += dr;
        f += df;
    }
    28 / (1 + friends)
}

fn promotion_race(pos: &Position, color: Color, sq: Square, phase: i32) -> i32 {
    let dr = if color == Color::White { 1 } else { -1 };
    let rank = if color == Color::White {
        rank_of(sq)
    } else {
        7 - rank_of(sq)
    };
    let promotion = make_square(if color == Color::White { 7 } else { 0 }, file_of(sq));
    let king = pos.king_sq[opponent(color) as usize];
    let interception = (rank_of(king) - rank_of(promotion))
        .abs()
        .max((file_of(king) - file_of(promotion)).abs());
    let mut value = (interception - (7 - rank)).clamp(-3, 3) * rank * 3 * (24 - phase) / 24;
    let ahead = rank_of(sq) + dr;
    if valid_rf(ahead, file_of(sq)) {
        let front = pos.board[make_square(ahead, file_of(sq)) as usize];
        if front.is_empty() {
            value += rank * rank * 2;
            let behind = rank_of(sq) - dr;
            if valid_rf(behind, file_of(sq))
                && pos.board[make_square(behind, file_of(sq)) as usize].is_color(color)
            {
                value += rank * 10;
            }
        } else if front.color != color {
            value -= rank * 12;
        }
    }
    value
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
#[repr(u8)]
enum Bound {
    #[default]
    Empty,
    Exact,
    Lower,
    Upper,
}

/// Two full-key entries per bucket, 32 MiB total. Mate values fit in i16 and
/// are normalized to the stored position rather than the current root.
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Entry {
    key: u64,
    mv: u32,
    score: i16,
    depth: u8,
    bound: Bound,
}

fn pack(m: Move) -> u32 {
    u32::from(m.from)
        | (u32::from(m.to) << 6)
        | (u32::from(m.path_kind) << 12)
        | (u32::from(m.stop_index) << 14)
        | ((m.special as u32) << 18)
        | ((m.promo_piece as u32) << 20)
}

fn to_table(score: i32, ply: usize) -> i16 {
    let normalized = if score >= MATE_BOUND {
        score + ply as i32
    } else if score <= -MATE_BOUND {
        score - ply as i32
    } else {
        score
    };
    i16::try_from(normalized).expect("search scores fit the table representation")
}

fn from_table(score: i16, ply: usize) -> i32 {
    let score = i32::from(score);
    if score >= MATE_BOUND {
        score - ply as i32
    } else if score <= -MATE_BOUND {
        score + ply as i32
    } else {
        score
    }
}

static REDUCTIONS: LazyLock<[[i32; 256]; 64]> = LazyLock::new(|| {
    std::array::from_fn(|depth| {
        std::array::from_fn(|moves| {
            if depth < 3 || moves < 3 {
                0
            } else {
                (0.5 + (depth as f64).ln() * (moves as f64).ln() / 2.3) as i32
            }
        })
    })
});

const PASSED_MASKS: [[u64; 64]; 2] = {
    let mut masks = [[0; 64]; 2];
    let mut color = 0;
    while color < 2 {
        let mut sq = 0;
        while sq < 64 {
            let mut target = 0;
            while target < 64 {
                let df = (sq % 8) as i32 - (target % 8) as i32;
                let ahead = if color == 0 {
                    target / 8 > sq / 8
                } else {
                    target / 8 < sq / 8
                };
                if df >= -1 && df <= 1 && ahead {
                    masks[color][sq] |= 1 << target;
                }
                target += 1;
            }
            sq += 1;
        }
        color += 1;
    }
    masks
};

fn placement(piece: PieceType, color: Color, sq: Square, endgame: bool) -> i32 {
    let rank = if color == Color::White {
        rank_of(sq)
    } else {
        7 - rank_of(sq)
    };
    let file = file_of(sq);
    let center = (file - 3).abs().min((file - 4).abs()) + (rank - 3).abs().min((rank - 4).abs());
    match piece {
        PieceType::Pawn => rank * 5 + (rank - 3).max(0).pow(2) * 4 - (file - 3).abs() * 2,
        PieceType::Knight => 32 - center * 10,
        PieceType::Bishop => 20 - center * 5,
        PieceType::Rook => {
            if rank == 6 {
                24
            } else {
                rank * 2
            }
        }
        PieceType::Queen => 12 - center * 3,
        PieceType::King if endgame => 36 - center * 12,
        PieceType::King => {
            -rank * 12
                + if rank == 0 && (file <= 2 || file >= 6) {
                    24
                } else {
                    0
                }
        }
        PieceType::None => 0,
    }
}

fn evaluate<const V: usize>(pos: &Position) -> i32 {
    let mut score = [0; 2];
    let mut phase = 0;
    let mut pawns = [0u64; 2];
    let mut pawn_files = [[0u8; 8]; 2];
    let mut bishops = [0; 2];
    for (sq, piece) in pos.board.iter().enumerate() {
        if piece.is_empty() {
            continue;
        }
        let ci = piece.color as usize;
        let pt = piece.piece_type;
        score[ci] += VALUES[pt as usize]
            + if pt == PieceType::King {
                0
            } else {
                placement(pt, piece.color, sq as u8, false)
            };
        phase += match pt {
            PieceType::Knight | PieceType::Bishop => 1,
            PieceType::Rook => 2,
            PieceType::Queen => 4,
            _ => 0,
        };
        if pt == PieceType::Pawn {
            pawns[ci] |= 1 << sq;
            pawn_files[ci][sq % 8] += 1;
        }
        if pt == PieceType::Bishop {
            bishops[ci] += 1;
        }
        // Push chess rewards latent lines to the enemy king even through friendly pieces.
        let king = pos.king_sq[1 - ci];
        let dr = (rank_of(sq as u8) - rank_of(king)).abs();
        let df = (file_of(sq as u8) - file_of(king)).abs();
        let aligned = (matches!(pt, PieceType::Rook | PieceType::Queen) && (dr == 0 || df == 0))
            || (matches!(pt, PieceType::Bishop | PieceType::Queen) && dr == df);
        if aligned {
            score[ci] += if PROFILES[V].pressure {
                clear_line_pressure(pos, sq as u8, king)
            } else if piece.color == pos.side_to_move {
                18
            } else {
                8
            };
        }
    }
    phase = phase.min(24);
    for ci in 0..2 {
        let color = if ci == 0 { Color::White } else { Color::Black };
        let king = pos.king_sq[ci];
        if king < 64 {
            score[ci] += (placement(PieceType::King, color, king, false) * phase
                + placement(PieceType::King, color, king, true) * (24 - phase))
                / 24;
            let mut shelter = 0;
            let mut own_pawns = pawns[ci];
            while own_pawns != 0 {
                let sq = own_pawns.trailing_zeros() as u8;
                own_pawns &= own_pawns - 1;
                let forward = (rank_of(sq) - rank_of(king)) * if ci == 0 { 1 } else { -1 };
                if (file_of(sq) - file_of(king)).abs() <= 1 && (1..=2).contains(&forward) {
                    shelter += 14;
                }
            }
            if pawn_files[ci][file_of(king) as usize] == 0 {
                shelter -= 24;
            }
            score[ci] += shelter * phase / 24;
        }
        if bishops[ci] >= 2 {
            score[ci] += 25;
        }
        if PROFILES[V].pressure {
            score[ci] += king_pressure(pos, color);
        }
        let mut remaining = pawns[ci];
        while remaining != 0 {
            let sq = remaining.trailing_zeros() as usize;
            remaining &= remaining - 1;
            let file = sq % 8;
            if PROFILES[V].races {
                score[ci] += promotion_race(pos, color, sq as u8, phase);
            }
            if pawn_files[ci][file] > 1 {
                score[ci] -= 9;
            }
            if (file == 0 || pawn_files[ci][file - 1] == 0)
                && (file == 7 || pawn_files[ci][file + 1] == 0)
            {
                score[ci] -= 8;
            }
            if pawns[1 - ci] & PASSED_MASKS[ci][sq] == 0 {
                let rank = if ci == 0 { sq / 8 } else { 7 - sq / 8 };
                score[ci] += [0, 0, 5, 12, 22, 38, 65, 0][rank];
            }
        }
    }
    let us = pos.side_to_move as usize;
    (score[us] - score[1 - us] + 10).clamp(-MATE_BOUND + 1, MATE_BOUND - 1)
}

/// Detect friendly contact anywhere along the chosen path, including pushes
/// whose destination is empty. This is an ordering heuristic, not a rules engine.
fn is_push(pos: &Position, mv: Move) -> bool {
    let piece = pos.board[mv.from as usize];
    let dr = rank_of(mv.to) - rank_of(mv.from);
    let df = file_of(mv.to) - file_of(mv.from);
    if piece.piece_type == PieceType::Knight {
        let (long, short) = if dr.abs() == 2 {
            ((dr.signum(), 0, 2), (0, df.signum(), 1))
        } else {
            ((0, df.signum(), 2), (dr.signum(), 0, 1))
        };
        let legs = if mv.path_kind == 1 {
            [long, short]
        } else {
            [short, long]
        };
        let (mut r, mut f) = (rank_of(mv.from), file_of(mv.from));
        for (rd, fd, distance) in legs {
            for _ in 0..distance {
                r += rd;
                f += fd;
                if pos.board[make_square(r, f) as usize].is_color(piece.color) {
                    return true;
                }
            }
        }
        false
    } else if mv.special == SpecialMove::Castle {
        false
    } else {
        let (mut r, mut f) = (rank_of(mv.from), file_of(mv.from));
        for _ in 0..dr.abs().max(df.abs()) {
            r += dr.signum();
            f += df.signum();
            if pos.board[make_square(r, f) as usize].is_color(piece.color) {
                return true;
            }
        }
        false
    }
}

struct Astra<const V: usize> {
    table: Buckets<Entry, 2>,
    history: Vec<i32>,
    killers: [[u32; 2]; MAX_PLY],
    buffer: Vec<Move>,
    null_barriers: Vec<usize>,
    limits: SearchLimits,
    qnodes: u64,
    tt_hits: u64,
    root_depth: i32,
    root_best: Move,
}

impl<const V: usize> Astra<V> {
    fn new() -> Self {
        assert!(V < PROFILES.len());
        let _ = &*REDUCTIONS;
        Self {
            table: Buckets::new(BUCKETS * size_of::<[Entry; 2]>()),
            history: vec![0; 2 * 3 * 64 * 64],
            killers: [[0; 2]; MAX_PLY],
            buffer: Vec::with_capacity(256),
            null_barriers: Vec::new(),
            limits: SearchLimits::default(),
            qnodes: 0,
            tt_hits: 0,
            root_depth: 0,
            root_best: Move::default(),
        }
    }

    fn tick(&mut self, ply: usize) -> bool {
        self.limits.tick(ply)
    }

    fn has_legal_move(&mut self, pos: &mut Position) -> bool {
        let mut buffer = std::mem::take(&mut self.buffer);
        buffer.clear();
        generate_pseudo_legal_moves(pos, &mut buffer);
        let side = pos.side_to_move;
        let mut found = false;
        for mv in &buffer {
            pos.make_move(mv);
            let legal = !pos.in_check_color(side);
            pos.unmake_move();
            if legal {
                found = true;
                break;
            }
        }
        self.buffer = buffer;
        found
    }

    fn draw_score(&mut self, pos: &mut Position, ply: usize) -> Option<i32> {
        if !self.draw(pos, ply) {
            return None;
        }
        if V != 0 && pos.in_check() && !self.has_legal_move(pos) {
            Some(-MATE + ply as i32)
        } else {
            Some(0)
        }
    }

    fn static_cutoff(&mut self, pos: &mut Position, score: i32) -> i32 {
        if V != 0 && !self.has_legal_move(pos) {
            0
        } else {
            score
        }
    }

    fn draw(&self, pos: &Position, ply: usize) -> bool {
        if pos.halfmove_clock >= 100 {
            return true;
        }
        let since_null = pos.undo_stack.len() - self.null_barriers.last().copied().unwrap_or(0);
        let available = usize::from(pos.halfmove_clock).min(since_null);
        let needed = if ply == 0 { 2 } else { 1 };
        pos.undo_stack
            .iter()
            .rev()
            .take(available)
            .filter(|undo| undo.zobrist == pos.zobrist)
            .take(needed)
            .count()
            == needed
    }

    fn probe(&self, key: u64) -> Option<Entry> {
        self.table[key as usize & (BUCKETS - 1)]
            .iter()
            .find(|entry| entry.bound != Bound::Empty && entry.key == key)
            .copied()
    }

    fn store(&mut self, key: u64, depth: i32, score: i32, bound: Bound, mv: Move, ply: usize) {
        let bucket = &mut self.table[key as usize & (BUCKETS - 1)];
        let slot = bucket
            .iter()
            .position(|e| e.bound == Bound::Empty || e.key == key)
            .unwrap_or_else(|| usize::from(bucket[1].depth < bucket[0].depth));
        bucket[slot] = Entry {
            key,
            mv: pack(mv),
            score: to_table(score, ply),
            depth: depth.clamp(0, 127) as u8,
            bound,
        };
    }

    fn history_index(pos: &Position, mv: Move) -> usize {
        ((pos.side_to_move as usize * 3 + usize::from(mv.path_kind.min(2))) * 64
            + usize::from(mv.from))
            * 64
            + usize::from(mv.to)
    }

    fn moves(&mut self, pos: &Position, ply: usize, table_move: u32) -> ScoredMoves {
        let mut buffer = std::mem::take(&mut self.buffer);
        buffer.clear();
        generate_pseudo_legal_moves(pos, &mut buffer);
        let mut moves = ScoredMoves::new();
        for &mv in &buffer {
            moves.push(mv);
            let index = moves.len() - 1;
            let target = pos.board[mv.to as usize];
            let capture = (!target.is_empty() && target.color != pos.side_to_move)
                || mv.special == SpecialMove::EnPassant;
            let mover = pos.board[mv.from as usize].piece_type;
            let mut score = self.history[Self::history_index(pos, mv)]
                + placement(mover, pos.side_to_move, mv.to, false)
                - placement(mover, pos.side_to_move, mv.from, false);
            if is_push(pos, mv) {
                score += 80_000;
            }
            if pack(mv) == self.killers[ply][1] {
                score += 200_000;
            }
            if pack(mv) == self.killers[ply][0] {
                score += 250_000;
            }
            if capture {
                score +=
                    2_000_000 + VALUES[target.piece_type as usize] * 16 - VALUES[mover as usize];
            }
            if mv.special == SpecialMove::Promotion {
                score += 1_500_000 + VALUES[mv.promo_piece as usize];
            }
            if pack(mv) == table_move && table_move != 0 {
                score = i32::MAX;
            }
            moves[index].score = score;
        }
        self.buffer = buffer;
        moves
    }

    fn search(
        &mut self,
        pos: &mut Position,
        depth: i32,
        mut alpha: i32,
        mut beta: i32,
        ply: usize,
        allow_null: bool,
    ) -> i32 {
        if depth <= 0 {
            return self.quiescence(pos, alpha, beta, ply, 0);
        }
        if self.tick(ply) {
            return 0;
        }
        if let Some(score) = self.draw_score(pos, ply) {
            return score;
        }
        if ply >= MAX_PLY - 1 {
            return evaluate::<V>(pos);
        }
        alpha = alpha.max(-MATE + ply as i32);
        beta = beta.min(MATE - ply as i32 - 1);
        if alpha >= beta {
            return alpha;
        }
        let original_alpha = alpha;
        let pv = beta - alpha > 1;
        let check = pos.in_check();
        let depth = depth + i32::from(check && (ply as i32) < 2 * self.root_depth);
        let cached = self.probe(pos.zobrist);
        if let Some(entry) = cached
            && i32::from(entry.depth) >= depth
            && !pv
            && ply > 0
        {
            let score = from_table(entry.score, ply);
            if entry.bound == Bound::Exact
                || (entry.bound == Bound::Lower && score >= beta)
                || (entry.bound == Bound::Upper && score <= alpha)
            {
                self.tt_hits += 1;
                return score;
            }
        }
        let static_eval = evaluate::<V>(pos);
        if !PROFILES[V].conservative && !check && !pv && ply > 0 && beta.abs() < MATE_BOUND {
            if depth <= 5 && static_eval - 110 * depth >= beta {
                return self.static_cutoff(pos, static_eval);
            }
            if allow_null
                && depth >= 3
                && static_eval >= beta
                && pos.board.iter().any(|p| {
                    p.is_color(pos.side_to_move)
                        && matches!(
                            p.piece_type,
                            PieceType::Knight
                                | PieceType::Bishop
                                | PieceType::Rook
                                | PieceType::Queen
                        )
                })
            {
                let (side, ep, key) = (pos.side_to_move, pos.ep_square, pos.zobrist);
                pos.side_to_move = opponent(side);
                pos.zobrist ^= zobrist_tables().side_key;
                if ep < 64 {
                    pos.zobrist ^= zobrist_tables().ep_keys[file_of(ep) as usize];
                }
                pos.ep_square = 64;
                self.null_barriers.push(pos.undo_stack.len());
                let reduction = (3 + depth / 5).min(depth - 1);
                let score =
                    -self.search(pos, depth - 1 - reduction, -beta, 1 - beta, ply + 1, false);
                self.null_barriers.pop();
                pos.side_to_move = side;
                pos.ep_square = ep;
                pos.zobrist = key;
                if self.limits.stopped {
                    return 0;
                }
                if score >= beta {
                    return self.static_cutoff(pos, if score >= MATE_BOUND { beta } else { score });
                }
            }
        }

        let mut moves = self.moves(pos, ply, cached.map_or(0, |entry| entry.mv));
        let us = pos.side_to_move;
        let key = pos.zobrist;
        let mut legal = 0;
        let mut best_score = -INF;
        let mut best = Move::default();
        for index in 0..moves.len() {
            moves.pick_best(index);
            let mv = moves[index].mv;
            let target = pos.board[mv.to as usize];
            let capture =
                (!target.is_empty() && target.color != us) || mv.special == SpecialMove::EnPassant;
            let promotion = mv.special == SpecialMove::Promotion;
            let push = is_push(pos, mv);
            let hi = Self::history_index(pos, mv);
            pos.make_move(&mv);
            if pos.in_check_color(us) {
                pos.unmake_move();
                continue;
            }
            legal += 1;
            let gives_check = pos.in_check();
            // Never prune the first legal move, captures, promotions, or checks.
            if legal > 1
                && !PROFILES[V].conservative
                && !pv
                && !check
                && !gives_check
                && !capture
                && !promotion
                && !push
                && depth <= 2
                && legal > 8 + depth * 5
            {
                pos.unmake_move();
                continue;
            }
            let child_depth = depth - 1;
            let mut reduction = 0;
            if !PROFILES[V].conservative
                && !(PROFILES[V].tactical && push)
                && legal >= 4
                && depth >= 3
                && !check
                && !gives_check
                && !capture
                && !promotion
            {
                reduction = REDUCTIONS[depth.min(63) as usize][(legal as usize).min(255)];
                if pv || push {
                    reduction -= 1;
                }
                if self.history[hi] > 3_000 {
                    reduction -= 1;
                }
                reduction = reduction.clamp(0, child_depth - 1);
            }
            let mut score;
            if legal == 1 {
                score = -self.search(pos, child_depth, -beta, -alpha, ply + 1, true);
            } else {
                score = -self.search(
                    pos,
                    child_depth - reduction,
                    -alpha - 1,
                    -alpha,
                    ply + 1,
                    true,
                );
                if score > alpha && reduction > 0 && !self.limits.stopped {
                    score = -self.search(pos, child_depth, -alpha - 1, -alpha, ply + 1, true);
                }
                if score > alpha && score < beta && !self.limits.stopped {
                    score = -self.search(pos, child_depth, -beta, -alpha, ply + 1, true);
                }
            }
            pos.unmake_move();
            if self.limits.stopped {
                return 0;
            }
            if score > best_score {
                best_score = score;
                best = mv;
            }
            if score > alpha {
                alpha = score;
                if ply == 0 {
                    self.root_best = mv;
                }
            }
            if alpha >= beta {
                if !capture && !promotion {
                    let bonus = (depth * depth * 32).min(2_000);
                    let h = &mut self.history[hi];
                    *h += bonus - *h * bonus / 16_384;
                    if self.killers[ply][0] != pack(mv) {
                        self.killers[ply][1] = self.killers[ply][0];
                        self.killers[ply][0] = pack(mv);
                    }
                }
                break;
            }
        }
        if legal == 0 {
            return if check { -MATE + ply as i32 } else { 0 };
        }
        let bound = if best_score >= beta {
            Bound::Lower
        } else if best_score <= original_alpha {
            Bound::Upper
        } else {
            Bound::Exact
        };
        self.store(key, depth, best_score, bound, best, ply);
        best_score
    }

    fn quiescence(
        &mut self,
        pos: &mut Position,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        qply: usize,
    ) -> i32 {
        if self.tick(ply) {
            return 0;
        }
        self.qnodes += 1;
        if let Some(score) = self.draw_score(pos, ply) {
            return score;
        }
        if ply >= MAX_PLY - 1 {
            return evaluate::<V>(pos);
        }
        let check = pos.in_check();
        let stand_pat = evaluate::<V>(pos);
        if !check {
            if stand_pat >= beta {
                return self.static_cutoff(pos, stand_pat);
            }
            alpha = alpha.max(stand_pat);
            if qply >= 12 {
                return self.static_cutoff(pos, alpha);
            }
        }
        let mut moves = self.moves(pos, ply, 0);
        let us = pos.side_to_move;
        let mut legal = 0;
        for index in 0..moves.len() {
            moves.pick_best(index);
            let mv = moves[index].mv;
            let target = pos.board[mv.to as usize];
            let capture =
                (!target.is_empty() && target.color != us) || mv.special == SpecialMove::EnPassant;
            let promotion = mv.special == SpecialMove::Promotion;
            if !check && !capture && !promotion && qply >= if PROFILES[V].tactical { 3 } else { 1 }
            {
                continue;
            }
            pos.make_move(&mv);
            if pos.in_check_color(us) {
                pos.unmake_move();
                continue;
            }
            legal += 1;
            let gives_check = pos.in_check();
            if !check && !capture && !promotion && !gives_check {
                pos.unmake_move();
                continue;
            }
            // A generous capture margin; never apply it to check evasions or checking moves.
            if !check
                && !PROFILES[V].conservative
                && !gives_check
                && !promotion
                && capture
                && stand_pat + VALUES[target.piece_type as usize] + 250 < alpha
            {
                pos.unmake_move();
                continue;
            }
            let score = -self.quiescence(pos, -beta, -alpha, ply + 1, qply + 1);
            pos.unmake_move();
            if self.limits.stopped {
                return 0;
            }
            if score >= beta {
                return score;
            }
            alpha = alpha.max(score);
        }
        if !check && legal == 0 && V != 0 && !self.has_legal_move(pos) {
            return 0;
        }
        if check && legal == 0 {
            -MATE + ply as i32
        } else {
            alpha
        }
    }

    fn pv(&self, pos: &mut Position, best: Move) -> Vec<Move> {
        let mut line = Vec::new();
        let mut seen = Vec::new();
        let mut next = pack(best);
        for _ in 0..32 {
            if seen.contains(&pos.zobrist) {
                break;
            }
            seen.push(pos.zobrist);
            let mut legal = Vec::new();
            generate_legal_moves(pos, &mut legal);
            let Some(mv) = legal.into_iter().find(|&mv| pack(mv) == next) else {
                break;
            };
            line.push(mv);
            pos.make_move(&mv);
            let Some(entry) = self.probe(pos.zobrist) else {
                break;
            };
            next = entry.mv;
        }
        for _ in 0..line.len() {
            pos.unmake_move();
        }
        line
    }
}

impl<const V: usize> Engine for Astra<V> {
    fn name(&self) -> &str {
        PROFILES[V].name
    }

    fn new_game(&mut self, _color: Color, _seed: u64) {
        self.table.fill([Entry::default(); 2]);
        self.history.fill(0);
        self.killers.fill([0; 2]);
        self.null_barriers.clear();
    }

    fn choose_move(&mut self, pos: &mut Position, budget: &SearchBudget) -> (Move, SearchStats) {
        self.limits.begin(budget, 92);
        self.qnodes = 0;
        self.tt_hits = 0;
        for h in &mut self.history {
            *h /= 2;
        }
        let mut legal = Vec::new();
        generate_legal_moves(pos, &mut legal);
        let Some(mut best) = legal.first().copied() else {
            return (Move::default(), SearchStats::default());
        };
        let mut score = evaluate::<V>(pos);
        let mut completed = 0;
        let max_depth = if budget.max_depth > 0 {
            budget.max_depth.min(64)
        } else {
            64
        };
        for depth in 1..=max_depth {
            self.root_depth = depth;
            let mut margin = if depth >= 4 && score.abs() < MATE_BOUND {
                35
            } else {
                INF * 2
            };
            loop {
                self.root_best = best;
                let alpha = (score - margin).max(-INF);
                let beta = (score + margin).min(INF);
                let result = self.search(pos, depth, alpha, beta, 0, true);
                if self.limits.stopped {
                    break;
                }
                if result <= alpha || result >= beta {
                    margin = (margin * 2).min(INF * 2);
                    continue;
                }
                score = result;
                best = self.root_best;
                completed = depth as u32;
                break;
            }
            if self.limits.stopped
                || score.abs() >= MATE_BOUND
                || (self.limits.micros > 0
                    && self.limits.started.elapsed().as_micros() > self.limits.micros * 2 / 3)
            {
                break;
            }
        }
        let pv = self.pv(pos, best);
        let stats = SearchStats {
            nodes: self.limits.nodes,
            depth_reached: completed,
            seldepth: self.limits.seldepth as u32,
            eval_cp: score,
            time_used_us: self.limits.started.elapsed().as_micros() as i64,
            pv,
            diagnostics: SearchDiagnostics {
                qnodes: self.qnodes,
                tt_hits: self.tt_hits,
                ..SearchDiagnostics::default()
            },
        };
        (best, stats)
    }
}

pub fn create() -> Box<dyn Engine> {
    Box::new(Astra::<0>::new())
}

pub fn create_experiment<const V: usize>() -> Box<dyn Engine> {
    Box::new(Astra::<V>::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "opt-in release ordering measurement, run serially on an idle machine"]
    fn ordering_probe() {
        let mut engine = Astra::<0>::new();
        let samples: Vec<_> = crate::engine::ordering_probe::positions()
            .iter()
            .map(|p| engine.moves(p, 0, 0).to_vec())
            .collect();
        crate::engine::ordering_probe::compare::<_, false, _>("astra", &samples, |m| m.score);
    }

    fn position(fen: &str) -> Position {
        let mut p = Position::default();
        p.set_from_fen(fen);
        p
    }

    #[test]
    fn mate_scores_round_trip_at_different_root_distances() {
        assert_eq!(std::mem::size_of::<Entry>(), 16);
        assert_eq!(from_table(to_table(MATE - 8, 3), 5), MATE - 10);
        assert_eq!(from_table(to_table(-MATE + 8, 3), 5), -MATE + 10);
        assert_eq!(from_table(to_table(125, 3), 7), 125);
    }

    #[test]
    fn recognizes_pushes_through_empty_destinations_and_knight_paths() {
        let p = position("7k/8/8/8/8/4P3/8/K3R3 w - - 0 1");
        assert!(is_push(
            &p,
            Move {
                from: 4,
                to: 36,
                ..Move::default()
            }
        ));
        let p = position("7k/8/4P3/8/4N3/8/8/K7 w - - 0 1");
        assert!(is_push(
            &p,
            Move {
                from: 28,
                to: 45,
                path_kind: 1,
                ..Move::default()
            }
        ));
    }

    #[test]
    fn quiescence_searches_quiet_check_evasions_and_detects_mate() {
        let mut engine = Astra::<0>::new();
        let mut p = position("k3r3/8/8/8/8/8/8/4K3 w - - 0 1");
        let fen = p.to_fen();
        assert!(engine.quiescence(&mut p, -INF, INF, 0, 0) > -MATE_BOUND);
        assert_eq!(p.to_fen(), fen);
        let mut p = position("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1");
        assert_eq!(engine.quiescence(&mut p, -INF, INF, 4, 0), -MATE + 4);
    }

    #[test]
    fn finds_mate_in_one_and_obeys_node_budget() {
        let mut engine = Astra::<0>::new();
        let mut p = position("7k/5K2/6Q1/8/8/8/8/8 w - - 0 1");
        let budget = SearchBudget {
            max_depth: 2,
            ..SearchBudget::default()
        };
        let (mv, stats) = engine.choose_move(&mut p, &budget);
        assert!(stats.eval_cp >= MATE_BOUND);
        p.make_move(&mv);
        assert!(p.in_check());
        let mut replies = Vec::new();
        generate_legal_moves(&mut p, &mut replies);
        assert!(replies.is_empty());
        let mut p = crate::core::position::start_position();
        let before = p.to_fen();
        let (_, stats) = engine.choose_move(
            &mut p,
            &SearchBudget {
                max_nodes: 250,
                ..SearchBudget::default()
            },
        );
        assert!(stats.nodes <= 250);
        assert_eq!(p.to_fen(), before);
        assert!(p.undo_stack.is_empty());
    }
}
