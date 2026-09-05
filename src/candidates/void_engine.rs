#![deny(unsafe_code)]
use super::support::ScoredMoves;
type MoveList = ScoredMoves<256>;
// void_002: Simple material/placement evaluation with selective search.
//
// Thesis: material + PST only eval. Chronos search architecture.
// Search and storage choices:
// - Precomputed eval table (material+PST in one lookup, no branching)
// - Packed 12-byte TT entries (33% more cache density)
// - Packed u32 moves in TT
// - Lazy selection sort (sort one move at a time in search loop)
// - IIR replacing IID (1 line vs full depth-2 search)
// - Extended reverse futility to depth 6
// - Wider qsearch (depth 10)
// - Time check every 512 nodes
// - Quiescence retains a cheap per-move material margin filter

use crate::core::movegen::generate_legal_moves;
use crate::core::position::Position;
use crate::core::types::*;
use crate::core::zobrist::zobrist_tables;
use crate::engine::Engine;
use std::sync::LazyLock;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Precomputed eval table: EVAL_TABLE[piece_type][sq] = material + PST (white perspective)
// For black: index with sq ^ 56. Eliminates pval() + pst_value() + match branching.
// ---------------------------------------------------------------------------

const PAWN_V: [i32; 64] = [
    100, 100, 100, 100, 100, 100, 100, 100, 105, 110, 110, 80, 80, 110, 110, 105, 105, 95, 90, 100,
    100, 90, 95, 105, 100, 100, 100, 120, 120, 100, 100, 100, 105, 105, 110, 125, 125, 110, 105,
    105, 110, 110, 120, 130, 130, 120, 110, 110, 150, 150, 150, 150, 150, 150, 150, 150, 100, 100,
    100, 100, 100, 100, 100, 100,
];
const KNIGHT_V: [i32; 64] = [
    270, 280, 290, 290, 290, 290, 280, 270, 280, 300, 320, 325, 325, 320, 300, 280, 290, 325, 330,
    335, 335, 330, 325, 290, 290, 320, 335, 340, 340, 335, 320, 290, 290, 325, 335, 340, 340, 335,
    325, 290, 290, 320, 330, 335, 335, 330, 320, 290, 280, 300, 320, 320, 320, 320, 300, 280, 270,
    280, 290, 290, 290, 290, 280, 270,
];
const BISHOP_V: [i32; 64] = [
    310, 320, 320, 320, 320, 320, 320, 310, 320, 335, 330, 330, 330, 330, 335, 320, 320, 340, 340,
    340, 340, 340, 340, 320, 320, 330, 340, 340, 340, 340, 330, 320, 320, 335, 335, 340, 340, 335,
    335, 320, 320, 330, 335, 340, 340, 335, 330, 320, 320, 330, 330, 330, 330, 330, 330, 320, 310,
    320, 320, 320, 320, 320, 320, 310,
];
const ROOK_V: [i32; 64] = [
    500, 500, 500, 505, 505, 500, 500, 500, 495, 500, 500, 500, 500, 500, 500, 495, 495, 500, 500,
    500, 500, 500, 500, 495, 495, 500, 500, 500, 500, 500, 500, 495, 495, 500, 500, 500, 500, 500,
    500, 495, 495, 500, 500, 500, 500, 500, 500, 495, 505, 510, 510, 510, 510, 510, 510, 505, 500,
    500, 500, 500, 500, 500, 500, 500,
];
const QUEEN_V: [i32; 64] = [
    880, 890, 890, 895, 895, 890, 890, 880, 890, 900, 900, 900, 900, 900, 900, 890, 890, 900, 905,
    905, 905, 905, 900, 890, 895, 900, 905, 905, 905, 905, 900, 895, 900, 900, 905, 905, 905, 905,
    900, 895, 890, 905, 905, 905, 905, 905, 900, 890, 890, 900, 905, 900, 900, 900, 900, 890, 880,
    890, 890, 895, 895, 890, 890, 880,
];
const KING_V: [i32; 64] = [
    20, 30, 10, 0, 0, 10, 30, 20, 20, 20, 0, 0, 0, 0, 20, 20, -10, -20, -20, -20, -20, -20, -20,
    -10, -20, -30, -30, -40, -40, -30, -30, -20, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40,
    -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50,
    -40, -40, -30,
];

// The dimensions encode piece type and square directly, with no runtime setup.
static EVAL_LUT: [[i32; 64]; 7] = [[0; 64], PAWN_V, KNIGHT_V, BISHOP_V, ROOK_V, QUEEN_V, KING_V];

// Single branchless eval lookup
#[inline(always)]
fn eval_piece(pt: PieceType, c: Color, sq: u8) -> i32 {
    let idx = if c == Color::White {
        sq as usize
    } else {
        (sq ^ 56) as usize
    };
    EVAL_LUT[pt as usize][idx]
}

// ---------------------------------------------------------------------------
// Packed TT: 12 bytes per entry (was 14-16)
// ---------------------------------------------------------------------------

#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
struct TTEntry {
    key32: u32,
    packed_move: u32, // from:6|to:6|path:2|stop:4|special:2|promo:3
    score: i16,
    depth: u8,
    flag: u8,
}

#[inline(always)]
fn pack_move(m: &Move) -> u32 {
    (m.from as u32)
        | ((m.to as u32) << 6)
        | ((m.path_kind as u32) << 12)
        | ((m.stop_index as u32) << 14)
        | ((m.special as u32) << 18)
        | ((m.promo_piece as u32) << 20)
}

#[inline(always)]
fn unpack_move(p: u32) -> Move {
    Move {
        from: (p & 0x3F) as u8,
        to: ((p >> 6) & 0x3F) as u8,
        path_kind: ((p >> 12) & 0x3) as u8,
        stop_index: ((p >> 14) & 0xF) as u8,
        special: match (p >> 18) & 0x3 {
            1 => SpecialMove::Castle,
            2 => SpecialMove::EnPassant,
            3 => SpecialMove::Promotion,
            _ => SpecialMove::None,
        },
        promo_piece: match (p >> 20) & 0x7 {
            1 => PieceType::Pawn,
            2 => PieceType::Knight,
            3 => PieceType::Bishop,
            4 => PieceType::Rook,
            5 => PieceType::Queen,
            6 => PieceType::King,
            _ => PieceType::None,
        },
    }
}

const TT_BITS: usize = 22;
const TT_SIZE: usize = 1 << TT_BITS;
const TT_MASK: usize = TT_SIZE - 1;

// ---------------------------------------------------------------------------
// History: flat Vec<i16>, manually indexed
// ---------------------------------------------------------------------------

const HISTORY_SIZE: usize = 2 * 64 * 64;
#[inline(always)]
fn hist_idx(c: usize, f: usize, t: usize) -> usize {
    (c << 12) | (f << 6) | t
}

// ---------------------------------------------------------------------------
// Move list with lazy sort
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// LMR table
// ---------------------------------------------------------------------------

struct LmrTable {
    table: [[i32; 256]; 32],
}
static LMR: LazyLock<LmrTable> = LazyLock::new(|| {
    let mut t = LmrTable {
        table: [[0i32; 256]; 32],
    };
    for d in 0..32 {
        for m in 0..256 {
            t.table[d][m] = if d < 2 || m < 3 {
                0
            } else {
                (0.5 + (d as f64).ln() * (m as f64).ln() / 1.8) as i32
            };
        }
    }
    t
});

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

struct VoidEngine {
    color: Color,
    tt: Vec<TTEntry>,
    history: Vec<i16>,
    push_history: Vec<i16>,
    killers: [[Move; 2]; 64],
    countermove: [[Move; 64]; 64],
    prev_move: Move,
    move_buf: Vec<Move>,
    nodes: u64,
    qnodes: u64,
    beta_cuts: u64,
    first_cuts: u64,
    tt_hits: u64,
    null_cuts: u64,
    lmr_tries: u64,
    lmr_re: u64,
    max_ply: u32,
    stopped: bool,
    budget_max_time_us: i64,
    t0: Instant,
    cut_hist: [u64; 6],
    cut_types: [u64; 3],
    max_qdep: i32,
}

impl VoidEngine {
    fn new() -> Self {
        let _ = &*LMR;
        Self {
            color: Color::White,
            tt: vec![TTEntry::default(); TT_SIZE],
            history: vec![0i16; HISTORY_SIZE],
            push_history: vec![0i16; HISTORY_SIZE],
            killers: [[Move::default(); 2]; 64],
            countermove: [[Move::default(); 64]; 64],
            prev_move: Move::default(),
            move_buf: Vec::with_capacity(256),
            nodes: 0,
            qnodes: 0,
            beta_cuts: 0,
            first_cuts: 0,
            tt_hits: 0,
            null_cuts: 0,
            lmr_tries: 0,
            lmr_re: 0,
            max_ply: 0,
            stopped: false,
            budget_max_time_us: 0,
            t0: Instant::now(),
            cut_hist: [0; 6],
            cut_types: [0; 3],
            max_qdep: 0,
        }
    }

    #[inline(always)]
    fn elapsed_us(&self) -> i64 {
        self.t0.elapsed().as_micros() as i64
    }

    #[inline(always)]
    fn check_time(&mut self) -> bool {
        if self.stopped {
            return true;
        }
        // Check every 512 nodes (was 256) — fewer syscalls
        if self.budget_max_time_us > 0
            && (self.nodes & 511) == 0
            && self.elapsed_us() >= self.budget_max_time_us * 9 / 10
        {
            self.stopped = true;
        }
        self.stopped
    }

    #[inline(always)]
    fn tt_store(&mut self, key: u64, depth: i32, score: i32, flag: u8, m: &Move) {
        let idx = (key as usize) & TT_MASK;
        let k32 = (key >> 32) as u32;
        let e = &mut self.tt[idx];
        if e.key32 != k32 || depth >= e.depth as i32 {
            e.key32 = k32;
            e.packed_move = pack_move(m);
            e.score = score.clamp(-32000, 32000) as i16;
            e.depth = depth.min(255) as u8;
            e.flag = flag;
        }
    }

    // VOID: branchless eval — one lookup per piece, no heuristics
    #[inline(always)]
    fn evaluate(&self, pos: &Position) -> i32 {
        let stm = pos.side_to_move;
        let mut score = 0i32;
        let board = &pos.board;
        for sq in 0u8..64 {
            let p = board[sq as usize];
            if p.piece_type != PieceType::None {
                let v = eval_piece(p.piece_type, p.color, sq);
                if p.color == stm {
                    score += v;
                } else {
                    score -= v;
                }
            }
        }
        score
    }

    fn score_moves(&self, pos: &Position, ml: &mut MoveList, ply: usize, ttm: &Move) {
        let cm = if self.prev_move.from != 0 || self.prev_move.to != 0 {
            self.countermove[self.prev_move.from as usize][self.prev_move.to as usize]
        } else {
            Move::default()
        };
        let stm = pos.side_to_move;

        for i in 0..ml.len() {
            let m = &ml[i].mv;
            if *m == *ttm {
                ml[i].score = i32::MAX;
                continue;
            }
            let mpt = pos.board[m.from as usize].piece_type;
            let tp = pos.board[m.to as usize];
            let sv;
            if !tp.is_empty() {
                if tp.color == stm {
                    // Push bracket (1M) + push history
                    sv = 1_000_000
                        + pval(tp.piece_type) * 10
                        + self.push_history[hist_idx(stm as usize, m.from as usize, m.to as usize)]
                            as i32;
                } else {
                    // Capture bracket (2M) + MVV-LVA
                    sv = 2_000_000 + pval(tp.piece_type) * 100 - pval(mpt);
                }
            } else {
                if ply < 64 && *m == self.killers[ply][0] {
                    sv = 500_000;
                } else if ply < 64 && *m == self.killers[ply][1] {
                    sv = 490_000;
                } else if *m == cm {
                    sv = 400_000;
                } else {
                    sv = self.history[hist_idx(stm as usize, m.from as usize, m.to as usize)]
                        as i32
                        + eval_piece(mpt, pos.board[m.from as usize].color, m.to)
                        - eval_piece(mpt, pos.board[m.from as usize].color, m.from);
                }
            }
            ml[i].score = if m.special == SpecialMove::Promotion {
                sv + 1_500_000 + pval(m.promo_piece)
            } else {
                sv
            };
        }
    }

    fn search(
        &mut self,
        pos: &mut Position,
        depth: i32,
        mut alpha: i32,
        mut beta: i32,
        ply: i32,
        in_check: bool,
        is_pv: bool,
    ) -> i32 {
        if self.check_time() {
            return 0;
        }
        self.nodes += 1;
        if ply as u32 > self.max_ply {
            self.max_ply = ply as u32;
        }
        if ply >= 128 {
            return self.evaluate(pos);
        }

        alpha = alpha.max(-99000 + ply);
        beta = beta.min(99000 - ply - 1);
        if alpha >= beta {
            return alpha;
        }

        let key = pos.zobrist;
        let idx = (key as usize) & TT_MASK;
        let k32 = (key >> 32) as u32;
        let mut ttm = Move::default();
        {
            let e = &self.tt[idx];
            if e.key32 == k32 {
                ttm = unpack_move(e.packed_move);
                if e.depth as i32 >= depth && !is_pv {
                    self.tt_hits += 1;
                    let s = e.score as i32;
                    if e.flag == 0 {
                        return s;
                    }
                    if e.flag == 2 && s >= beta {
                        return s;
                    }
                    if e.flag == 1 && s <= alpha {
                        return s;
                    }
                }
            }
        }

        if depth <= 0 {
            return self.qsearch(pos, alpha, beta, 0);
        }

        // Null move pruning
        if !is_pv && !in_check && ply > 0 && depth >= 3 {
            let z = zobrist_tables();
            let se = pos.ep_square;
            let ss = pos.side_to_move;
            let sz = pos.zobrist;
            pos.side_to_move = opponent(pos.side_to_move);
            pos.zobrist ^= z.side_key;
            if pos.ep_square < 64 {
                pos.zobrist ^= z.ep_keys[file_of(pos.ep_square) as usize];
                pos.ep_square = 64;
            }
            let r = if depth >= 6 { 3 } else { 2 };
            let ns = -self.search(
                pos,
                depth - 1 - r,
                -beta,
                -(beta - 1),
                ply + 1,
                false,
                false,
            );
            pos.side_to_move = ss;
            pos.ep_square = se;
            pos.zobrist = sz;
            if ns >= beta && !self.stopped {
                self.null_cuts += 1;
                return beta;
            }
        }

        // Extended reverse futility (depth <= 6, eval is ~free)
        if !in_check && !is_pv && depth <= 6 && ply > 0 {
            let eval = self.evaluate(pos);
            if eval - depth * 100 >= beta {
                self.null_cuts += 1;
                return beta;
            }
        }

        // IIR: just reduce depth by 1 when no TT move (replaces expensive IID)
        let mut search_depth = depth;
        if ttm.from == 0 && ttm.to == 0 && depth >= 4 {
            search_depth -= 1;
        }

        let mut ml = MoveList::new();
        {
            let mut buf = std::mem::take(&mut self.move_buf);
            buf.clear();
            generate_legal_moves(pos, &mut buf);
            for m in &buf {
                ml.push(*m);
            }
            self.move_buf = buf;
        }
        if ml.is_empty() {
            return if in_check { -99000 + ply } else { 0 };
        }

        // Score all moves once, then lazy-sort in the loop
        self.score_moves(pos, &mut ml, ply as usize, &ttm);

        let saved_prev = self.prev_move;
        let mut best_move = ml[0].mv;
        let mut best_score = -100000i32;
        let mut flag: u8 = 1;
        let stm = pos.side_to_move;

        for i in 0..ml.len() {
            if self.stopped {
                break;
            }

            // Lazy selection: pick best remaining move
            ml.pick_best(i);
            let m = ml[i].mv;

            let tp = pos.board[m.to as usize];
            let is_capture = !tp.is_empty() && tp.color != stm;
            let is_push = !tp.is_empty() && tp.color == stm;
            let is_promo = m.special == SpecialMove::Promotion;
            let push_hist = if is_push {
                self.push_history[hist_idx(stm as usize, m.from as usize, m.to as usize)]
            } else {
                0
            };
            let is_tactical = is_capture
                || is_promo
                || m.special == SpecialMove::EnPassant
                || (is_push && push_hist >= 0);

            if !is_tactical && !in_check && search_depth <= 2 && i as i32 >= 8 + search_depth * 4 {
                continue;
            }

            self.prev_move = m;
            pos.make_move(&m);
            let gives_check = pos.in_check();

            let score;
            let d = search_depth.min(31) as usize;
            let mi = i.min(255);

            if i >= 3 && search_depth >= 2 && !is_tactical && !gives_check && !in_check {
                self.lmr_tries += 1;
                let mut r = LMR.table[d][mi].clamp(1, search_depth - 1);
                let ci = opponent(pos.side_to_move) as usize;
                let hs = self.history[hist_idx(ci, m.from as usize, m.to as usize)] as i32;
                if hs < -500 {
                    r += 2;
                } else if hs < -100 {
                    r += 1;
                }
                r = r.clamp(1, search_depth - 1);
                let s0 = -self.search(
                    pos,
                    search_depth - 1 - r,
                    -(alpha + 1),
                    -alpha,
                    ply + 1,
                    gives_check,
                    false,
                );
                if s0 > alpha && !self.stopped {
                    self.lmr_re += 1;
                    score = -self.search(
                        pos,
                        search_depth - 1,
                        -beta,
                        -alpha,
                        ply + 1,
                        gives_check,
                        is_pv,
                    );
                } else {
                    score = s0;
                }
            } else {
                let ext = if gives_check && search_depth <= 4 {
                    1
                } else {
                    0
                };
                if i > 0 && !self.stopped {
                    let s1 = -self.search(
                        pos,
                        search_depth - 1 + ext,
                        -(alpha + 1),
                        -alpha,
                        ply + 1,
                        gives_check,
                        false,
                    );
                    if s1 > alpha && s1 < beta && !self.stopped {
                        score = -self.search(
                            pos,
                            search_depth - 1 + ext,
                            -beta,
                            -alpha,
                            ply + 1,
                            gives_check,
                            is_pv,
                        );
                    } else {
                        score = s1;
                    }
                } else {
                    score = -self.search(
                        pos,
                        search_depth - 1 + ext,
                        -beta,
                        -alpha,
                        ply + 1,
                        gives_check,
                        is_pv,
                    );
                }
            }
            pos.unmake_move();

            if score > best_score {
                best_score = score;
                best_move = m;
            }
            if score > alpha {
                alpha = score;
                flag = 0;
            }
            if alpha >= beta {
                flag = 2;
                self.beta_cuts += 1;
                if i == 0 {
                    self.first_cuts += 1;
                }
                let ct = if is_capture {
                    0
                } else if is_push {
                    1
                } else {
                    2
                };
                self.cut_types[ct] += 1;
                let hi = if i < 5 { i } else { 5 };
                self.cut_hist[hi] += 1;
                let ci = pos.side_to_move as usize;
                if is_push {
                    let h = &mut self.push_history[hist_idx(ci, m.from as usize, m.to as usize)];
                    *h = (*h + (depth * depth) as i16).min(16000);
                    for j in 0..i {
                        let pm = ml[j].mv;
                        let pt2 = pos.board[pm.to as usize];
                        if !pt2.is_empty() && pt2.color == stm {
                            let hh = &mut self.push_history
                                [hist_idx(ci, pm.from as usize, pm.to as usize)];
                            *hh = (*hh - depth as i16).max(-16000);
                        }
                    }
                } else if !is_capture && !is_promo && m.special != SpecialMove::EnPassant {
                    if ply < 64 {
                        self.killers[ply as usize][1] = self.killers[ply as usize][0];
                        self.killers[ply as usize][0] = m;
                    }
                    let h = &mut self.history[hist_idx(ci, m.from as usize, m.to as usize)];
                    *h = (*h + (depth * depth) as i16).min(16000);
                    for j in 0..i {
                        let pm = ml[j].mv;
                        let pt2 = pos.board[pm.to as usize];
                        let pc = !pt2.is_empty() && pt2.color != stm;
                        let pp = !pt2.is_empty() && pt2.color == stm;
                        if !pc
                            && !pp
                            && pm.special != SpecialMove::Promotion
                            && pm.special != SpecialMove::EnPassant
                        {
                            let hh =
                                &mut self.history[hist_idx(ci, pm.from as usize, pm.to as usize)];
                            *hh = (*hh - depth as i16).max(-16000);
                        }
                    }
                    if saved_prev.from != 0 || saved_prev.to != 0 {
                        self.countermove[saved_prev.from as usize][saved_prev.to as usize] = m;
                    }
                }
                break;
            }
        }
        self.prev_move = saved_prev;
        if !self.stopped {
            self.tt_store(key, depth, best_score, flag, &best_move);
        }
        best_score
    }

    fn qsearch(&mut self, pos: &mut Position, mut alpha: i32, beta: i32, qdepth: i32) -> i32 {
        if self.check_time() {
            return 0;
        }
        self.nodes += 1;
        self.qnodes += 1;
        if qdepth > self.max_qdep {
            self.max_qdep = qdepth;
        }
        if qdepth >= 10 {
            return self.evaluate(pos);
        }

        let sp = self.evaluate(pos);
        if sp >= beta {
            return sp;
        }
        if sp > alpha {
            alpha = sp;
        }
        // No delta pruning — eval is free, search everything

        let mut buf = std::mem::take(&mut self.move_buf);
        buf.clear();
        generate_legal_moves(pos, &mut buf);

        let mut tac: [(Move, i32); 64] = [(Move::default(), 0i32); 64];
        let mut nt: usize = 0;
        let stm = pos.side_to_move;

        for m in &buf {
            let ic = !pos.board[m.to as usize].is_empty() && pos.board[m.to as usize].color != stm;
            let t =
                ic || m.special == SpecialMove::Promotion || m.special == SpecialMove::EnPassant;
            if !t {
                continue;
            }
            let mut see = pval(pos.board[m.to as usize].piece_type);
            if m.special == SpecialMove::Promotion {
                see += 800;
            }
            if sp + see + 200 < alpha {
                continue;
            } // SEE delta pruning still kept (cheap)
            if nt < 64 {
                tac[nt] = (*m, see);
                nt += 1;
            }
        }
        self.move_buf = buf;

        // Inline selection sort for qsearch tactical moves
        for i in 0..nt {
            let mut bi = i;
            let mut bv = tac[i].1;
            for j in (i + 1)..nt {
                if tac[j].1 > bv {
                    bv = tac[j].1;
                    bi = j;
                }
            }
            if bi != i {
                tac.swap(i, bi);
            }
        }

        for i in 0..nt {
            pos.make_move(&tac[i].0);
            let s = -self.qsearch(pos, -beta, -alpha, qdepth + 1);
            pos.unmake_move();
            if self.stopped {
                return 0;
            }
            if s >= beta {
                return s;
            }
            if s > alpha {
                alpha = s;
            }
        }
        alpha
    }

    fn dump_diag(&mut self, pos: &mut Position, root: &MoveList, stats: &mut SearchStats) {
        let mut ranked: Vec<(String, i32)> = Vec::new();
        for i in 0..root.len() {
            let m = &root[i].mv;
            pos.make_move(m);
            let key = pos.zobrist;
            let e = &self.tt[(key as usize) & TT_MASK];
            let sc = if e.key32 == (key >> 32) as u32 {
                -(e.score as i32)
            } else {
                -self.evaluate(pos)
            };
            pos.unmake_move();
            let mut uci = String::new();
            uci.push((b'a' + (m.from & 7)) as char);
            uci.push((b'1' + (m.from >> 3)) as char);
            uci.push((b'a' + (m.to & 7)) as char);
            uci.push((b'1' + (m.to >> 3)) as char);
            if m.special == SpecialMove::Promotion {
                match m.promo_piece {
                    PieceType::Knight => uci.push('n'),
                    PieceType::Bishop => uci.push('b'),
                    PieceType::Rook => uci.push('r'),
                    PieceType::Queen => uci.push('q'),
                    _ => {}
                }
            }
            ranked.push((uci, sc));
        }
        ranked.sort_by_key(|entry| std::cmp::Reverse(entry.1));
        let cap = ranked.len().min(32);
        let hf = self.tt.iter().filter(|e| e.key32 != 0).count() * 1000 / TT_SIZE;
        let mut diag = format!(
            r#"{{"engine":"void_002","qn":{},"tt":{},"bcut":{},"fcut":{},"nmp":{},"lmr":[{},{}],"max_qdep":{},"hashfull":{},"cut_hist":[{},{},{},{},{},{}],"cut_types":{{"cap":{},"push":{},"quiet":{}}},"top_moves":["#,
            self.qnodes,
            self.tt_hits,
            self.beta_cuts,
            self.first_cuts,
            self.null_cuts,
            self.lmr_tries,
            self.lmr_re,
            self.max_qdep,
            hf,
            self.cut_hist[0],
            self.cut_hist[1],
            self.cut_hist[2],
            self.cut_hist[3],
            self.cut_hist[4],
            self.cut_hist[5],
            self.cut_types[0],
            self.cut_types[1],
            self.cut_types[2]
        );
        for i in 0..cap {
            if i > 0 {
                diag.push(',');
            }
            diag.push_str(&format!(r#"["{}",{}]"#, ranked[i].0, ranked[i].1));
        }
        diag.push_str("]}");
        stats.diag_json = diag;
    }
}

impl Engine for VoidEngine {
    fn name(&self) -> &str {
        "void"
    }

    fn new_game(&mut self, my_color: Color, _game_seed: u64) {
        self.color = my_color;
        // Fast clear: fill is memset-optimized
        self.tt.fill(TTEntry::default());
        self.history.fill(0);
        self.push_history.fill(0);
        self.killers = [[Move::default(); 2]; 64];
        self.countermove = [[Move::default(); 64]; 64];
        self.prev_move = Move::default();
    }

    fn choose_move(&mut self, pos: &mut Position, budget: &SearchBudget) -> (Move, SearchStats) {
        self.t0 = Instant::now();
        self.budget_max_time_us = budget.max_time_us;
        self.nodes = 0;
        self.qnodes = 0;
        self.max_ply = 0;
        self.stopped = false;
        self.beta_cuts = 0;
        self.first_cuts = 0;
        self.tt_hits = 0;
        self.null_cuts = 0;
        self.lmr_tries = 0;
        self.lmr_re = 0;
        self.prev_move = Move::default();
        self.cut_hist = [0; 6];
        self.cut_types = [0; 3];
        self.max_qdep = 0;

        let mut root = MoveList::new();
        {
            let mut buf = std::mem::take(&mut self.move_buf);
            buf.clear();
            generate_legal_moves(pos, &mut buf);
            for m in &buf {
                root.push(*m);
            }
            self.move_buf = buf;
        }

        let mut stats = SearchStats::default();
        if root.is_empty() {
            return (Move::default(), stats);
        }
        if root.len() == 1 {
            stats.nodes = 1;
            stats.depth_reached = 0;
            return (root[0].mv, stats);
        }

        let mut best = root[0].mv;
        let mut best_score = -100000i32;

        for depth in 1..=30 {
            if self.stopped {
                break;
            }
            let (mut alpha, mut beta) = if depth <= 3 || best_score.abs() > 5000 {
                (-100000i32, 100000i32)
            } else {
                (best_score - 50, best_score + 50)
            };

            let mut iter_best: Move;
            let mut iter_best_score: i32;
            let mut af = false;

            loop {
                iter_best = root[0].mv;
                iter_best_score = -100000;
                // Put previous best first
                for i in 0..root.len() {
                    if root[i].mv == best {
                        if i != 0 {
                            root.swap(0, i);
                        }
                        break;
                    }
                }

                for i in 0..root.len() {
                    if self.stopped {
                        break;
                    }
                    let m = root[i].mv;
                    self.prev_move = m;
                    pos.make_move(&m);
                    let gc = pos.in_check();
                    let score;
                    if i > 0 && !self.stopped {
                        let s1 = -self.search(pos, depth - 1, -(alpha + 1), -alpha, 1, gc, false);
                        if s1 > alpha && s1 < beta && !self.stopped {
                            score = -self.search(pos, depth - 1, -beta, -alpha, 1, gc, true);
                        } else {
                            score = s1;
                        }
                    } else {
                        score = -self.search(pos, depth - 1, -beta, -alpha, 1, gc, true);
                    }
                    pos.unmake_move();
                    if self.stopped {
                        break;
                    }
                    if score > iter_best_score {
                        iter_best_score = score;
                        iter_best = m;
                    }
                    if score > alpha {
                        alpha = score;
                    }
                    if alpha >= beta {
                        break;
                    }
                }
                if self.stopped {
                    break;
                }
                if iter_best_score <= (best_score - 50) && !af {
                    alpha = (iter_best_score - 200).max(-100000);
                    beta = 100000;
                    af = true;
                    continue;
                }
                if iter_best_score >= (best_score + 50)
                    && depth > 3
                    && best_score.abs() <= 5000
                    && !af
                {
                    alpha = -100000;
                    beta = (iter_best_score + 200).min(100000);
                    af = true;
                    continue;
                }
                break;
            }
            if !self.stopped {
                best_score = iter_best_score;
                best = iter_best;
            }
            stats.depth_reached = depth as u32;
            stats.eval_cp = best_score;
        }

        if best_score <= -90000 {
            let mut bs = -100000i32;
            for i in 0..root.len() {
                pos.make_move(&root[i].mv);
                let sc = -self.evaluate(pos);
                pos.unmake_move();
                if sc > bs {
                    bs = sc;
                    best = root[i].mv;
                }
            }
        }

        // Move validation
        let mut fresh = Vec::new();
        generate_legal_moves(pos, &mut fresh);
        if !fresh.contains(&best) {
            best = fresh.first().copied().unwrap_or_default();
        }

        stats.nodes = self.nodes;
        stats.seldepth = self.max_ply;
        stats.time_used_us = self.elapsed_us();

        // PV extraction
        stats.pv.clear();
        let mut seen = [0u64; 32];
        let mut sn = 0;
        let mut dt = 0;
        loop {
            if dt >= 32 {
                break;
            }
            let key = pos.zobrist;
            if seen[..sn].contains(&key) {
                break;
            }
            if sn < 32 {
                seen[sn] = key;
                sn += 1;
            }
            let e = &self.tt[(key as usize) & TT_MASK];
            if e.key32 != (key >> 32) as u32 {
                break;
            }
            let m = unpack_move(e.packed_move);
            if m.from >= 64 || m.to >= 64 || pos.board[m.from as usize].is_empty() {
                break;
            }
            let mut buf = std::mem::take(&mut self.move_buf);
            buf.clear();
            generate_legal_moves(pos, &mut buf);
            let found = buf.contains(&m);
            self.move_buf = buf;
            if !found {
                break;
            }
            stats.pv.push(m);
            pos.make_move(&m);
            dt += 1;
        }
        for _ in 0..dt {
            pos.unmake_move();
        }

        self.dump_diag(pos, &root, &mut stats);
        (best, stats)
    }
}

pub fn create() -> Box<dyn Engine> {
    Box::new(VoidEngine::new())
}
