// oblivion_001: The Kinetic Battery & Tactical Push Engine
//
// Paradigm Shift: Recognizes that in Push Chess, moving a piece into a friendly
// piece is NOT a "Quiet Move", it is a highly violent Tactical Move.
//
// 1. Move Ordering: Friendly pushes scored +90,000, right after captures.
// 2. Pruning Immunity: Friendly pushes are tactical — immune to LMR/LMP.
// 3. X-Ray Eval: Sliders get bonus for sharing file/rank/diagonal with enemy king.
// 4. Conservative LMR 1.8, 2M TT, stack-allocated MoveList.
//
// Engine by Gemini.

use std::sync::LazyLock;
use std::time::Instant;

use crate::core::types::*;
use crate::core::position::Position;
use crate::core::movegen::generate_legal_moves;
use crate::core::zobrist::zobrist_tables;
use crate::engine::Engine;

// ---------------------------------------------------------------------------
// PeSTO-style Piece-Square Tables
// ---------------------------------------------------------------------------

const PAWN_PST: [i32; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10,-20,-20, 10, 10,  5,
     5, -5,-10,  0,  0,-10, -5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
     0,  0,  0,  0,  0,  0,  0,  0,
];
const KNIGHT_PST: [i32; 64] = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
];
const BISHOP_PST: [i32; 64] = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
];
const ROOK_PST: [i32; 64] = [
      0,  0,  0,  5,  5,  0,  0,  0,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
      5, 10, 10, 10, 10, 10, 10,  5,
      0,  0,  0,  0,  0,  0,  0,  0,
];
const QUEEN_PST: [i32; 64] = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
];
const KING_PST: [i32; 64] = [
     20, 30, 10,  0,  0, 10, 30, 20,
     20, 20,  0,  0,  0,  0, 20, 20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
];

#[inline(always)]
fn pst_value(pt: PieceType, c: Color, sq: u8) -> i32 {
    let idx = if c == Color::White { sq as usize } else { (sq ^ 56) as usize };
    match pt {
        PieceType::Pawn => PAWN_PST[idx], PieceType::Knight => KNIGHT_PST[idx],
        PieceType::Bishop => BISHOP_PST[idx], PieceType::Rook => ROOK_PST[idx],
        PieceType::Queen => QUEEN_PST[idx], PieceType::King => KING_PST[idx],
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Structures
// ---------------------------------------------------------------------------

const MAX_MOVES: usize = 256;

struct MoveList {
    moves: [Move; MAX_MOVES],
    scores: [i32; MAX_MOVES],
    len: usize,
}

impl MoveList {
    fn new() -> Self {
        Self { moves: [Move::default(); MAX_MOVES], scores: [0i32; MAX_MOVES], len: 0 }
    }
    #[inline(always)]
    fn push(&mut self, m: Move) { self.moves[self.len] = m; self.len += 1; }
    fn selection_sort(&mut self) {
        for i in 0..self.len {
            let mut best_idx = i;
            let mut best_val = self.scores[i];
            for j in (i + 1)..self.len {
                if self.scores[j] > best_val { best_val = self.scores[j]; best_idx = j; }
            }
            if best_idx != i { self.moves.swap(i, best_idx); self.scores.swap(i, best_idx); }
        }
    }
}

const TT_BITS: usize = 21;
const TT_SIZE: usize = 1 << TT_BITS;
const TT_MASK: usize = TT_SIZE - 1;

#[derive(Clone, Copy, Default)]
struct TTEntry { key32: u32, score: i16, depth: u8, flag: u8, from: u8, to: u8, path_kind: u8, stop_idx: u8, special: u8, promo: u8 }

fn tt_to_move(e: &TTEntry) -> Move {
    Move { from: e.from, to: e.to, path_kind: e.path_kind, stop_index: e.stop_idx,
        special: match e.special { 1 => SpecialMove::Castle, 2 => SpecialMove::EnPassant, 3 => SpecialMove::Promotion, _ => SpecialMove::None },
        promo_piece: match e.promo { 1 => PieceType::Pawn, 2 => PieceType::Knight, 3 => PieceType::Bishop, 4 => PieceType::Rook, 5 => PieceType::Queen, 6 => PieceType::King, _ => PieceType::None },
    }
}

const HISTORY_SIZE: usize = 2 * 64 * 64;
#[inline(always)]
fn history_idx(color: usize, from: usize, to: usize) -> usize { color * 64 * 64 + from * 64 + to }

struct LmrTable { table: [[i32; 256]; 32] }
static LMR: LazyLock<LmrTable> = LazyLock::new(|| {
    let mut t = LmrTable { table: [[0i32; 256]; 32] };
    for d in 0..32 { for m in 0..256 {
        t.table[d][m] = if d < 2 || m < 3 { 0 } else { (0.5 + (d as f64).ln() * (m as f64).ln() / 1.8) as i32 };
    }}
    t
});

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

struct OblivionEngine {
    color: Color, tt: Vec<TTEntry>, history: Vec<i16>,
    killers: [[Move; 2]; 64], countermove: [[Move; 64]; 64], prev_move: Move,
    move_buf: Vec<Move>,
    nodes: u64, qnodes: u64, beta_cuts: u64, first_cuts: u64, tt_hits: u64,
    null_cuts: u64, lmr_tries: u64, lmr_re: u64, lazy_cuts: u64, max_ply: u32,
    stopped: bool, budget_max_time_us: i64, t0: Instant,
}

impl OblivionEngine {
    fn new() -> Self {
        let _ = &*LMR;
        Self { color: Color::White, tt: vec![TTEntry::default(); TT_SIZE],
            history: vec![0i16; HISTORY_SIZE], killers: [[Move::default(); 2]; 64],
            countermove: [[Move::default(); 64]; 64], prev_move: Move::default(),
            move_buf: Vec::new(),
            nodes: 0, qnodes: 0, beta_cuts: 0, first_cuts: 0, tt_hits: 0,
            null_cuts: 0, lmr_tries: 0, lmr_re: 0, lazy_cuts: 0, max_ply: 0,
            stopped: false, budget_max_time_us: 0, t0: Instant::now() }
    }

    fn elapsed_us(&self) -> i64 { self.t0.elapsed().as_micros() as i64 }

    fn check_time(&mut self) -> bool {
        if self.stopped { return true; }
        if self.budget_max_time_us > 0 && (self.nodes & 255) == 0 {
            if self.elapsed_us() >= self.budget_max_time_us * 9 / 10 { self.stopped = true; }
        }
        self.stopped
    }

    fn tt_store(&mut self, key: u64, depth: i32, score: i32, flag: u8, m: &Move) {
        let idx = (key as usize) & TT_MASK;
        let k32 = (key >> 32) as u32;
        let e = &mut self.tt[idx];
        if e.key32 != k32 || depth >= e.depth as i32 {
            e.key32 = k32; e.score = score.clamp(-32000, 32000) as i16;
            e.depth = depth.min(255) as u8; e.flag = flag; e.from = m.from; e.to = m.to;
            e.path_kind = m.path_kind; e.stop_idx = m.stop_index;
            e.special = m.special as u8; e.promo = m.promo_piece as u8;
        }
    }

    fn evaluate(&self, pos: &Position) -> i32 {
        let stm = pos.side_to_move;
        let mut score = 0;
        let ek_sq = pos.king_sq[opponent(stm) as usize];
        let ek_r = (ek_sq >> 3) as i32;
        let ek_f = (ek_sq & 7) as i32;
        let mut pawn_n = [0usize; 2];
        let mut pawn_sq = [[0u8; 8]; 2];
        let mut bishop_count = [0i32; 2];

        for sq in 0u8..64 {
            let p = pos.board[sq as usize];
            if p.is_empty() { continue; }
            let c = p.color;
            let ci = c as usize;
            let sign = if c == stm { 1 } else { -1 };
            let pt = p.piece_type;
            let r = (sq >> 3) as i32;
            let f = (sq & 7) as i32;

            score += sign * (pval(pt) + pst_value(pt, c, sq));

            if pt == PieceType::Pawn {
                if pawn_n[ci] < 8 { pawn_sq[ci][pawn_n[ci]] = sq; pawn_n[ci] += 1; }
            } else if pt == PieceType::Bishop { bishop_count[ci] += 1; }

            // X-Ray kinetic battery: slider pointing at enemy king
            if c == stm && (pt == PieceType::Rook || pt == PieceType::Queen) {
                if f == ek_f { score += 20; }
                if r == ek_r { score += 20; }
            }
            if c == stm && (pt == PieceType::Bishop || pt == PieceType::Queen) {
                if (r - ek_r).abs() == (f - ek_f).abs() { score += 20; }
            }
        }

        // King safety
        for ci in 0..2usize {
            let c = if ci == 0 { Color::White } else { Color::Black };
            let sign = if c == stm { 1 } else { -1 };
            let kr = (pos.king_sq[ci] >> 3) as i32;
            let kf = (pos.king_sq[ci] & 7) as i32;
            let shield_dir: i32 = if c == Color::White { 1 } else { -1 };
            for pi in 0..pawn_n[ci] {
                let psq = pawn_sq[ci][pi] as i32;
                let pr = psq >> 3; let pf = psq & 7;
                let file_dist = (pf - kf).abs();
                let rank_ahead = (pr - kr) * shield_dir;
                if file_dist <= 1 && rank_ahead >= 1 && rank_ahead <= 2 { score += sign * 15; }
            }
            let mut pawns_near_king = 0;
            for pi in 0..pawn_n[ci] {
                let psq = pawn_sq[ci][pi] as i32;
                let dist = ((psq >> 3) - kr).abs().max(((psq & 7) - kf).abs());
                if dist <= 2 { pawns_near_king += 1; }
            }
            if pawns_near_king < 2 { score += sign * (-40); }
            let mut has_pawn_on_file = false;
            for pi in 0..pawn_n[ci] { if (pawn_sq[ci][pi] & 7) as i32 == kf { has_pawn_on_file = true; break; } }
            if !has_pawn_on_file { score += sign * (-25); }
            if bishop_count[ci] >= 2 { score += sign * 30; }
        }
        score
    }

    fn order_moves(&self, pos: &Position, ml: &mut MoveList, ply: usize, ttm: &Move) {
        let cm = if self.prev_move.from != 0 || self.prev_move.to != 0 {
            self.countermove[self.prev_move.from as usize][self.prev_move.to as usize]
        } else { Move::default() };
        let stm = pos.side_to_move;

        for i in 0..ml.len {
            let m = &ml.moves[i];
            if *m == *ttm { ml.scores[i] = 10_000_000; continue; }
            let mut sv: i32 = 0;
            let mpt = pos.board[m.from as usize].piece_type;
            let mc = pos.board[m.from as usize].color;

            if !pos.board[m.to as usize].is_empty() {
                if pos.board[m.to as usize].color == stm {
                    // FRIENDLY PUSH — tactical priority
                    let pushed_pt = pos.board[m.to as usize].piece_type;
                    sv += 90_000 + if pushed_pt == PieceType::Pawn { 5000 } else { pval(pushed_pt) };
                } else {
                    // CAPTURE
                    sv += 100_000 + pval(pos.board[m.to as usize].piece_type) * 10 - pval(mpt);
                }
            }
            if m.special == SpecialMove::Promotion { sv += 95_000 + pval(m.promo_piece); }
            sv += pst_value(mpt, mc, m.to) - pst_value(mpt, mc, m.from);
            if ply < 64 {
                if *m == self.killers[ply][0] { sv += 80_000; }
                else if *m == self.killers[ply][1] { sv += 79_000; }
            }
            if *m == cm { sv += 60_000; }
            sv += self.history[history_idx(stm as usize, m.from as usize, m.to as usize)] as i32;
            ml.scores[i] = sv;
        }
        ml.selection_sort();
    }

    fn search(&mut self, pos: &mut Position, depth: i32, mut alpha: i32, mut beta: i32, ply: i32, in_check: bool, is_pv: bool) -> i32 {
        if self.check_time() { return 0; }
        self.nodes += 1;
        if ply as u32 > self.max_ply { self.max_ply = ply as u32; }
        if ply >= 128 { return self.evaluate(pos); }

        alpha = alpha.max(-99000 + ply);
        beta = beta.min(99000 - ply - 1);
        if alpha >= beta { return alpha; }

        let key = pos.zobrist;
        let idx = (key as usize) & TT_MASK;
        let k32 = (key >> 32) as u32;
        let mut ttm = Move::default();
        {
            let e = &self.tt[idx];
            if e.key32 == k32 {
                ttm = tt_to_move(e);
                if e.depth as i32 >= depth && !is_pv {
                    self.tt_hits += 1;
                    if e.flag == 0 { return e.score as i32; }
                    if e.flag == 2 && e.score as i32 >= beta { return e.score as i32; }
                    if e.flag == 1 && e.score as i32 <= alpha { return e.score as i32; }
                }
            }
        }

        if depth <= 0 { return self.qsearch(pos, alpha, beta, 0); }

        // Null move pruning
        if !is_pv && !in_check && ply > 0 && depth >= 3 {
            let z = zobrist_tables();
            let saved_ep = pos.ep_square;
            let saved_stm = pos.side_to_move;
            let saved_zobrist = pos.zobrist;
            pos.side_to_move = opponent(pos.side_to_move);
            pos.zobrist ^= z.side_key;
            if pos.ep_square < 64 { pos.zobrist ^= z.ep_keys[file_of(pos.ep_square) as usize]; pos.ep_square = 64; }
            let r = if depth >= 6 { 3 } else { 2 };
            let null_score = -self.search(pos, depth - 1 - r, -beta, -(beta - 1), ply + 1, false, false);
            pos.side_to_move = saved_stm; pos.ep_square = saved_ep; pos.zobrist = saved_zobrist;
            if null_score >= beta && !self.stopped { self.null_cuts += 1; return beta; }
        }

        // Reverse futility
        if !in_check && !is_pv && depth <= 4 && ply > 0 {
            let eval = self.evaluate(pos);
            if eval - depth * 100 >= beta { self.null_cuts += 1; return beta; }
            if depth <= 2 && eval + 300 < alpha {
                let qs = self.qsearch(pos, alpha, beta, 0);
                if qs < alpha { return qs; }
            }
        }

        // IID
        if ttm.from == 0 && ttm.to == 0 && depth >= 4 && !self.stopped {
            self.search(pos, depth - 2, alpha, beta, ply, in_check, is_pv);
            let e = &self.tt[idx];
            if e.key32 == k32 { ttm = tt_to_move(e); }
        }

        let mut ml = MoveList::new();
        { let mut buf = std::mem::take(&mut self.move_buf); buf.clear();
          generate_legal_moves(pos, &mut buf);
          for m in &buf { ml.push(*m); } self.move_buf = buf; }

        if ml.len == 0 { return if in_check { -99000 + ply } else { 0 }; }
        self.order_moves(pos, &mut ml, ply as usize, &ttm);

        let saved_prev = self.prev_move;
        let mut best_move = ml.moves[0];
        let mut best_score = -100000i32;
        let mut flag: u8 = 1;
        let stm = pos.side_to_move;

        for i in 0..ml.len {
            if self.stopped { break; }
            let m = ml.moves[i];

            let is_capture = !pos.board[m.to as usize].is_empty() && pos.board[m.to as usize].color != stm;
            let is_push = !pos.board[m.to as usize].is_empty() && pos.board[m.to as usize].color == stm;
            let is_tactical = is_capture || is_push || m.special == SpecialMove::Promotion || m.special == SpecialMove::EnPassant;

            // LMP: only prune non-tactical (pushes are immune)
            if !is_tactical && !in_check && depth <= 2 && i as i32 >= 8 + depth * 4 { continue; }

            self.prev_move = m;
            pos.make_move(&m);
            let gives_check = pos.in_check();

            let score;
            let d = depth.min(31) as usize;
            let mi = i.min(255);

            // LMR: pushes skip reduction (is_tactical = true)
            if i >= 3 && depth >= 2 && !is_tactical && !gives_check && !in_check {
                self.lmr_tries += 1;
                let mut r = LMR.table[d][mi].clamp(1, depth - 1);
                let ci = opponent(pos.side_to_move) as usize;
                let hscore = self.history[history_idx(ci, m.from as usize, m.to as usize)] as i32;
                if hscore < -500 { r += 2; } else if hscore < -100 { r += 1; }
                r = r.clamp(1, depth - 1);
                let s0 = -self.search(pos, depth - 1 - r, -(alpha + 1), -alpha, ply + 1, gives_check, false);
                if s0 > alpha && !self.stopped {
                    self.lmr_re += 1;
                    score = -self.search(pos, depth - 1, -beta, -alpha, ply + 1, gives_check, is_pv);
                } else { score = s0; }
            } else {
                let ext = if gives_check && depth <= 4 { 1 } else { 0 };
                if i > 0 && !self.stopped {
                    let s1 = -self.search(pos, depth - 1 + ext, -(alpha + 1), -alpha, ply + 1, gives_check, false);
                    if s1 > alpha && s1 < beta && !self.stopped {
                        score = -self.search(pos, depth - 1 + ext, -beta, -alpha, ply + 1, gives_check, is_pv);
                    } else { score = s1; }
                } else {
                    score = -self.search(pos, depth - 1 + ext, -beta, -alpha, ply + 1, gives_check, is_pv);
                }
            }
            pos.unmake_move();

            if score > best_score { best_score = score; best_move = m; }
            if score > alpha { alpha = score; flag = 0; }
            if alpha >= beta {
                flag = 2; self.beta_cuts += 1;
                if i == 0 { self.first_cuts += 1; }
                if !is_tactical && (ply as usize) < 64 {
                    self.killers[ply as usize][1] = self.killers[ply as usize][0];
                    self.killers[ply as usize][0] = m;
                    let ci = pos.side_to_move as usize;
                    let h = &mut self.history[history_idx(ci, m.from as usize, m.to as usize)];
                    *h = (*h + (depth * depth) as i16).min(16000);
                    for j in 0..i {
                        if pos.board[ml.moves[j].to as usize].is_empty() {
                            let hh = &mut self.history[history_idx(ci, ml.moves[j].from as usize, ml.moves[j].to as usize)];
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
        if !self.stopped { self.tt_store(key, depth, best_score, flag, &best_move); }
        best_score
    }

    fn qsearch(&mut self, pos: &mut Position, mut alpha: i32, beta: i32, qdepth: i32) -> i32 {
        if self.check_time() { return 0; }
        self.nodes += 1; self.qnodes += 1;
        if qdepth >= 4 { return self.evaluate(pos); }

        let stand_pat = self.evaluate(pos);
        if stand_pat >= beta { return stand_pat; }
        if stand_pat > alpha { alpha = stand_pat; }
        if stand_pat + 1000 < alpha { return alpha; }

        let mut buf = std::mem::take(&mut self.move_buf);
        buf.clear();
        generate_legal_moves(pos, &mut buf);

        let mut tactical: [(Move, i32); 64] = [(Move::default(), 0i32); 64];
        let mut nt: usize = 0;
        let stm = pos.side_to_move;

        for m in &buf {
            // qsearch: only captures + promotions (not pushes — those are handled in main search)
            let is_capture = !pos.board[m.to as usize].is_empty() && pos.board[m.to as usize].color != stm;
            let tac = is_capture || m.special == SpecialMove::Promotion || m.special == SpecialMove::EnPassant;
            if !tac { continue; }
            let mut see = pval(pos.board[m.to as usize].piece_type);
            if m.special == SpecialMove::Promotion { see += 800; }
            if stand_pat + see + 200 < alpha { continue; }
            if nt < 64 { tactical[nt] = (*m, see); nt += 1; }
        }
        self.move_buf = buf;

        for i in 0..nt {
            let mut best_idx = i; let mut best_val = tactical[i].1;
            for j in (i + 1)..nt { if tactical[j].1 > best_val { best_val = tactical[j].1; best_idx = j; } }
            if best_idx != i { tactical.swap(i, best_idx); }
        }

        for i in 0..nt {
            pos.make_move(&tactical[i].0);
            let score = -self.qsearch(pos, -beta, -alpha, qdepth + 1);
            pos.unmake_move();
            if self.stopped { return 0; }
            if score >= beta { return score; }
            if score > alpha { alpha = score; }
        }
        alpha
    }

    fn dump_diag(&mut self, pos: &mut Position, root: &MoveList, stats: &mut SearchStats) {
        let mut ranked: Vec<(String, i32)> = Vec::new();
        for i in 0..root.len {
            let m = &root.moves[i];
            pos.make_move(m);
            let key = pos.zobrist;
            let e = &self.tt[(key as usize) & TT_MASK];
            let sc = if e.key32 == (key >> 32) as u32 { -(e.score as i32) } else { -self.evaluate(pos) };
            pos.unmake_move();
            let mut uci = String::new();
            uci.push((b'a' + (m.from & 7)) as char); uci.push((b'1' + (m.from >> 3)) as char);
            uci.push((b'a' + (m.to & 7)) as char); uci.push((b'1' + (m.to >> 3)) as char);
            if m.special == SpecialMove::Promotion {
                match m.promo_piece { PieceType::Knight => uci.push('n'), PieceType::Bishop => uci.push('b'),
                    PieceType::Rook => uci.push('r'), PieceType::Queen => uci.push('q'), _ => {} }
            }
            ranked.push((uci, sc));
        }
        ranked.sort_by(|a, b| b.1.cmp(&a.1));
        let cap = ranked.len().min(32);
        let mut diag = format!(
            r#"{{"engine":"oblivion_001","qn":{},"tt":{},"bcut":{},"fcut":{},"nmp":{},"lmr":[{},{}],"top_moves":["#,
            self.qnodes, self.tt_hits, self.beta_cuts, self.first_cuts, self.null_cuts, self.lmr_tries, self.lmr_re);
        for i in 0..cap { if i > 0 { diag.push(','); } diag.push_str(&format!(r#"["{}",{}]"#, ranked[i].0, ranked[i].1)); }
        diag.push_str("]}"); stats.diag_json = diag;
    }
}

impl Engine for OblivionEngine {
    fn name(&self) -> &str { "oblivion" }
    fn new_game(&mut self, my_color: Color, _game_seed: u64) {
        self.color = my_color; self.tt.fill(TTEntry::default()); self.history.fill(0);
        self.killers = [[Move::default(); 2]; 64]; self.countermove = [[Move::default(); 64]; 64];
        self.prev_move = Move::default();
    }

    fn choose_move(&mut self, pos: &mut Position, budget: &SearchBudget) -> (Move, SearchStats) {
        self.t0 = Instant::now(); self.budget_max_time_us = budget.max_time_us;
        self.nodes = 0; self.qnodes = 0; self.max_ply = 0; self.stopped = false;
        self.beta_cuts = 0; self.first_cuts = 0; self.tt_hits = 0;
        self.null_cuts = 0; self.lmr_tries = 0; self.lmr_re = 0; self.lazy_cuts = 0;
        self.prev_move = Move::default();

        let mut root = MoveList::new();
        { let mut buf = std::mem::take(&mut self.move_buf); buf.clear();
          generate_legal_moves(pos, &mut buf);
          for m in &buf { root.push(*m); } self.move_buf = buf; }

        let mut stats = SearchStats::default();
        if root.len == 0 { return (Move::default(), stats); }
        if root.len == 1 { stats.nodes = 1; stats.depth_reached = 0; return (root.moves[0], stats); }

        let mut best = root.moves[0];
        let mut best_score = -100000i32;

        for depth in 1..=30 {
            if self.stopped { break; }
            let (mut alpha, mut beta) = if depth <= 3 || best_score.abs() > 5000 {
                (-100000i32, 100000i32)
            } else { (best_score - 50, best_score + 50) };

            let mut iter_best = root.moves[0];
            let mut iter_best_score = -100000i32;
            let mut aspiration_fail = false;

            loop {
                iter_best = root.moves[0]; iter_best_score = -100000;
                for i in 0..root.len { if root.moves[i] == best { if i != 0 { root.moves.swap(0, i); } break; } }

                for i in 0..root.len {
                    if self.stopped { break; }
                    let m = root.moves[i];
                    self.prev_move = m;
                    pos.make_move(&m);
                    let gives_check = pos.in_check();
                    let score;
                    if i > 0 && !self.stopped {
                        let s1 = -self.search(pos, depth - 1, -(alpha + 1), -alpha, 1, gives_check, false);
                        if s1 > alpha && s1 < beta && !self.stopped {
                            score = -self.search(pos, depth - 1, -beta, -alpha, 1, gives_check, true);
                        } else { score = s1; }
                    } else { score = -self.search(pos, depth - 1, -beta, -alpha, 1, gives_check, true); }
                    pos.unmake_move();
                    if self.stopped { break; }
                    if score > iter_best_score { iter_best_score = score; iter_best = m; }
                    if score > alpha { alpha = score; }
                    if alpha >= beta { break; }
                }
                if self.stopped { break; }
                if iter_best_score <= (best_score - 50) && !aspiration_fail {
                    alpha = (iter_best_score - 200).max(-100000); beta = 100000; aspiration_fail = true; continue;
                }
                if iter_best_score >= (best_score + 50) && depth > 3 && best_score.abs() <= 5000 && !aspiration_fail {
                    alpha = -100000; beta = (iter_best_score + 200).min(100000); aspiration_fail = true; continue;
                }
                break;
            }
            if !self.stopped { best_score = iter_best_score; best = iter_best; }
            stats.depth_reached = depth as u32; stats.eval_cp = best_score;
        }

        if best_score <= -90000 {
            let mut bs = -100000i32;
            for i in 0..root.len {
                pos.make_move(&root.moves[i]);
                let sc = -self.evaluate(pos); pos.unmake_move();
                if sc > bs { bs = sc; best = root.moves[i]; }
            }
        }

        // Move validation
        let mut fresh = Vec::new();
        generate_legal_moves(pos, &mut fresh);
        if !fresh.iter().any(|m| *m == best) { best = fresh.first().copied().unwrap_or_default(); }

        stats.nodes = self.nodes; stats.seldepth = self.max_ply; stats.time_used_us = self.elapsed_us();

        // PV extraction
        stats.pv.clear();
        let mut seen = [0u64; 32]; let mut sn = 0; let mut depth_t = 0;
        loop {
            if depth_t >= 32 { break; }
            let key = pos.zobrist;
            if seen[..sn].iter().any(|&k| k == key) { break; }
            if sn < 32 { seen[sn] = key; sn += 1; }
            let e = &self.tt[(key as usize) & TT_MASK];
            if e.key32 != (key >> 32) as u32 { break; }
            let m = tt_to_move(e);
            if m.from >= 64 || m.to >= 64 || pos.board[m.from as usize].is_empty() { break; }
            let mut buf = std::mem::take(&mut self.move_buf); buf.clear();
            generate_legal_moves(pos, &mut buf);
            let found = buf.iter().any(|lm| *lm == m); self.move_buf = buf;
            if !found { break; }
            stats.pv.push(m); pos.make_move(&m); depth_t += 1;
        }
        for _ in 0..depth_t { pos.unmake_move(); }

        self.dump_diag(pos, &root, &mut stats);
        (best, stats)
    }
}

pub fn create() -> Box<dyn Engine> { Box::new(OblivionEngine::new()) }
