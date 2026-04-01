// chimera_013: Fusion's winning formula + bug fixes + aegis eval
//
// Based on fusion_009 (76.9% win rate, conservative pruning, countermove).
// Critical fixes:
//   - is_tactical computed BEFORE make_move (was dead -- always true after)
//   - extract_pv validates every move (bounds + legality)
// Eval additions from aegis_010:
//   - Pawn shield bonus, exposed king penalty, open file penalty, bishop pair
// From nova_008:
//   - King tropism (penalize king near enemy advanced pawns)
// Search:
//   - LMR now actually fires (fixed bug means +1-2 depth for free)
//   - Conservative pruning preserved: LMP depth <= 2, no deep futility
//   - TT: 2^19 (512K)

use std::sync::LazyLock;
use std::time::Instant;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::core::movegen::generate_legal_moves;
use crate::core::position::Position;
use crate::core::types::*;
use crate::engine::Engine;

/// NEON-accelerated find-max-index for move ordering selection sort.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_find_max_index(scores: &[f32], start: usize, len: usize) -> usize {
    unsafe {
        let ptr = scores.as_ptr().add(start);
        let count = len - start;
        let chunks = count / 4;

        let mut vmax = vld1q_f32(ptr);
        for i in 1..chunks {
            let v = vld1q_f32(ptr.add(i * 4));
            vmax = vmaxq_f32(vmax, v);
        }
        let mut max_val = vmaxvq_f32(vmax);
        for i in (chunks * 4)..count {
            let v = *ptr.add(i);
            if v > max_val { max_val = v; }
        }
        let target = vdupq_n_f32(max_val);
        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * 4));
            let cmp = vceqq_f32(v, target);
            let mask: [u32; 4] = std::mem::transmute(cmp);
            for lane in 0..4 {
                if mask[lane] != 0 { return start + i * 4 + lane; }
            }
        }
        for i in (chunks * 4)..count {
            if *ptr.add(i) == max_val { return start + i; }
        }
        start
    }
}

#[inline]
fn find_max_index(scores: &[f32], start: usize, len: usize) -> usize {
    if len <= start { return start; }
    #[cfg(target_arch = "aarch64")]
    {
        if len - start >= 4 {
            return unsafe { neon_find_max_index(scores, start, len) };
        }
    }
    let mut best_idx = start;
    let mut best_val = scores[start];
    for j in (start + 1)..len {
        if scores[j] > best_val { best_val = scores[j]; best_idx = j; }
    }
    best_idx
}

// ---------------------------------------------------------------------------
// TT
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct TTEntry {
    key32: u32,
    score: i16,
    depth: u8,
    flag: u8,
    from: u8,
    to: u8,
    path_kind: u8,
    stop_idx: u8,
    special: u8,
    promo: u8,
}

const TT_BITS: usize = 19;
const TT_SIZE: usize = 1 << TT_BITS;
const TT_MASK: usize = TT_SIZE - 1;

// ---------------------------------------------------------------------------
// LMR table
// ---------------------------------------------------------------------------

fn init_lmr() -> [[i32; 256]; 32] {
    let mut table = [[0i32; 256]; 32];
    let mut d = 0;
    while d < 32 {
        let mut m = 0;
        while m < 256 {
            table[d][m] = if d < 2 || m < 3 {
                0
            } else {
                (0.5 + (d as f64).ln() * (m as f64).ln() / 1.8) as i32
            };
            m += 1;
        }
        d += 1;
    }
    table
}

static LMR: LazyLock<[[i32; 256]; 32]> = LazyLock::new(init_lmr);

// ---------------------------------------------------------------------------
// MoveList
// ---------------------------------------------------------------------------

struct MoveList {
    moves: Vec<Move>,
}

impl MoveList {
    fn new() -> Self {
        Self {
            moves: Vec::with_capacity(256),
        }
    }

    fn push(&mut self, m: Move) {
        self.moves.push(m);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const PV_TABLE: [i32; 7] = [0, 100, 320, 330, 500, 900, 0];

#[inline]
fn pv_val(pt: PieceType) -> i32 {
    PV_TABLE[pt as usize]
}

fn move_eq(a: &Move, b: &Move) -> bool {
    a.from == b.from
        && a.to == b.to
        && a.path_kind == b.path_kind
        && a.stop_index == b.stop_index
        && a.special as u8 == b.special as u8
        && a.promo_piece as u8 == b.promo_piece as u8
}

fn move_is_null(m: &Move) -> bool {
    m.from == 0 && m.to == 0
}

fn tt_to_move(e: &TTEntry) -> Move {
    Move {
        from: e.from,
        to: e.to,
        path_kind: e.path_kind,
        stop_index: e.stop_idx,
        special: unsafe { std::mem::transmute::<u8, SpecialMove>(e.special) },
        promo_piece: unsafe { std::mem::transmute::<u8, PieceType>(e.promo) },
    }
}

fn move_to_uci(m: &Move) -> String {
    let mut s = String::with_capacity(6);
    s.push((b'a' + (m.from & 7)) as char);
    s.push((b'1' + (m.from >> 3)) as char);
    s.push((b'a' + (m.to & 7)) as char);
    s.push((b'1' + (m.to >> 3)) as char);
    if m.special as u8 == SpecialMove::Promotion as u8 {
        match m.promo_piece {
            PieceType::Knight => s.push('n'),
            PieceType::Bishop => s.push('b'),
            PieceType::Rook => s.push('r'),
            PieceType::Queen => s.push('q'),
            _ => {}
        }
    }
    s
}

// ---------------------------------------------------------------------------
// ChimeraEngine
// ---------------------------------------------------------------------------

struct ChimeraEngine {
    color: Color,
    tt: Vec<TTEntry>,
    history: [[[i16; 64]; 64]; 2],
    killers: [[Move; 2]; 64],
    countermove: [[Move; 64]; 64],
    prev_move: Move,
    nodes: u64,
    qnodes: u64,
    max_ply: u32,
    stopped: bool,
    beta_cuts: u64,
    first_cuts: u64,
    tt_hits: u64,
    null_cuts: u64,
    lmr_tries: u64,
    lmr_re: u64,
    budget: SearchBudget,
    t0: Instant,
    move_buf: Vec<Move>,
}

impl ChimeraEngine {
    fn new() -> Self {
        // Force LMR init
        let _ = &*LMR;
        Self {
            color: Color::White,
            tt: vec![TTEntry::default(); TT_SIZE],
            history: [[[0i16; 64]; 64]; 2],
            killers: [[Move::default(); 2]; 64],
            countermove: [[Move::default(); 64]; 64],
            prev_move: Move::default(),
            nodes: 0,
            qnodes: 0,
            max_ply: 0,
            stopped: false,
            beta_cuts: 0,
            first_cuts: 0,
            tt_hits: 0,
            null_cuts: 0,
            lmr_tries: 0,
            lmr_re: 0,
            budget: SearchBudget::default(),
            t0: Instant::now(),
            move_buf: Vec::with_capacity(256),
        }
    }

    #[inline]
    fn elapsed_us(&self) -> i64 {
        self.t0.elapsed().as_micros() as i64
    }

    #[inline]
    fn check_time(&mut self) -> bool {
        if self.stopped {
            return true;
        }
        if self.budget.max_time_us > 0 && (self.nodes & 255) == 0 {
            if self.elapsed_us() >= self.budget.max_time_us * 9 / 10 {
                self.stopped = true;
            }
        }
        self.stopped
    }

    fn tt_store(&mut self, key: u64, depth: i32, score: i32, flag: u8, m: &Move) {
        let idx = (key as usize) & TT_MASK;
        let k32 = (key >> 32) as u32;
        let e = &mut self.tt[idx];
        if e.key32 != k32 || depth >= e.depth as i32 {
            e.key32 = k32;
            e.score = score.clamp(-32000, 32000) as i16;
            e.depth = depth.min(255) as u8;
            e.flag = flag;
            e.from = m.from;
            e.to = m.to;
            e.path_kind = m.path_kind;
            e.stop_idx = m.stop_index;
            e.special = m.special as u8;
            e.promo = m.promo_piece as u8;
        }
    }

    fn evaluate(&self, pos: &Position) -> i32 {
        let stm = pos.side_to_move;
        let mut sc_mat: i32 = 0;
        let mut sc_pos: i32 = 0;
        let mut pawn_sq: [[u8; 8]; 2] = [[0; 8]; 2];
        let mut pawn_n: [usize; 2] = [0; 2];
        let mut slider_sq: [[u8; 16]; 2] = [[0; 16]; 2];
        let mut slider_n: [usize; 2] = [0; 2];
        let mut king: [u8; 2] = [0; 2];
        let mut bishop_count: [i32; 2] = [0; 2];

        for sq in 0u8..64 {
            let piece = pos.board[sq as usize];
            if piece.is_empty() {
                continue;
            }
            let pt = piece.piece_type;
            let c = piece.color;
            let ci = c as usize;
            let sign: i32 = if c == stm { 1 } else { -1 };
            let r = (sq >> 3) as i32;
            let f = (sq & 7) as i32;

            sc_mat += sign * pv_val(pt);

            let cd = (r - 3).abs().max((f - 3).abs());
            if pt == PieceType::Knight {
                sc_pos += sign * (3 - cd) * 5;
            } else if pt != PieceType::Pawn && pt != PieceType::King {
                sc_pos += sign * (3 - cd) * 2;
            }

            if pt == PieceType::Pawn {
                let adv = if c == Color::White { r } else { 7 - r };
                sc_pos += sign * (adv * adv * 3);
                if pawn_n[ci] < 8 {
                    pawn_sq[ci][pawn_n[ci]] = sq;
                    pawn_n[ci] += 1;
                }
                // Passed pawn check
                let mut passed = true;
                let dir: i32 = if c == Color::White { 1 } else { -1 };
                let mut fr = r + dir;
                'outer: while fr >= 0 && fr < 8 {
                    for df in -1i32..=1 {
                        let ff = f + df;
                        if ff < 0 || ff > 7 {
                            continue;
                        }
                        let cs = (fr * 8 + ff) as usize;
                        let cp = pos.board[cs];
                        if !cp.is_empty() && cp.piece_type == PieceType::Pawn && cp.color != c {
                            passed = false;
                            break 'outer;
                        }
                    }
                    fr += dir;
                }
                if passed {
                    sc_pos += sign * (adv * 15);
                }
            }

            if pt == PieceType::King {
                king[ci] = sq;
            }
            if pt == PieceType::Bishop {
                bishop_count[ci] += 1;
            }
            if pt == PieceType::Rook || pt == PieceType::Queen {
                if slider_n[ci] < 16 {
                    slider_sq[ci][slider_n[ci]] = sq;
                    slider_n[ci] += 1;
                }
            }
        }

        // Piston bonus
        for ci in 0..2usize {
            let c = if ci == 0 { Color::White } else { Color::Black };
            let sign: i32 = if c == stm { 1 } else { -1 };
            for pi in 0..pawn_n[ci] {
                let psq = pawn_sq[ci][pi] as i32;
                let pr = psq >> 3;
                let pf = psq & 7;
                let adv = if c == Color::White { pr } else { 7 - pr };
                if adv < 4 {
                    continue;
                }
                for si in 0..slider_n[ci] {
                    let ssq = slider_sq[ci][si] as i32;
                    if (ssq & 7) != pf {
                        continue;
                    }
                    let sr = ssq >> 3;
                    let behind = if c == Color::White { sr < pr } else { sr > pr };
                    if !behind {
                        continue;
                    }
                    let dir: i32 = if c == Color::White { 1 } else { -1 };
                    let mut clear = true;
                    let mut cr = sr + dir;
                    while cr != pr {
                        if !pos.board[(cr * 8 + pf) as usize].is_empty() {
                            clear = false;
                            break;
                        }
                        cr += dir;
                    }
                    if !clear {
                        continue;
                    }
                    sc_pos += sign * if adv >= 5 { 150 } else { 60 };
                    break;
                }
            }
        }

        // King tropism
        for ci in 0..2usize {
            let oci = 1 - ci;
            let sign: i32 = if ci == stm as usize { 1 } else { -1 };
            let kr = (king[ci] >> 3) as i32;
            let kf = (king[ci] & 7) as i32;
            for pi in 0..pawn_n[oci] {
                let psq = pawn_sq[oci][pi] as i32;
                let pr = psq >> 3;
                let pf = psq & 7;
                let oc = if oci == 0 { Color::White } else { Color::Black };
                let adv = if oc == Color::White { pr } else { 7 - pr };
                if adv < 4 {
                    continue;
                }
                let dist = (kr - pr).abs().max((kf - pf).abs());
                if dist <= 2 {
                    sc_pos -= sign * (20 + adv * 8);
                }
            }
        }

        // King safety
        for ci in 0..2usize {
            let c = if ci == 0 { Color::White } else { Color::Black };
            let sign: i32 = if c == stm { 1 } else { -1 };
            let kr = (king[ci] >> 3) as i32;
            let kf = (king[ci] & 7) as i32;
            let shield_dir: i32 = if c == Color::White { 1 } else { -1 };

            // Pawn shield
            for pi in 0..pawn_n[ci] {
                let psq = pawn_sq[ci][pi] as i32;
                let pr = psq >> 3;
                let pf = psq & 7;
                let file_dist = (pf - kf).abs();
                let rank_ahead = (pr - kr) * shield_dir;
                if file_dist <= 1 && rank_ahead >= 1 && rank_ahead <= 2 {
                    sc_pos += sign * 15;
                }
            }

            // Exposed king
            let mut pawns_near_king = 0;
            for pi in 0..pawn_n[ci] {
                let psq = pawn_sq[ci][pi] as i32;
                let pr = psq >> 3;
                let pf = psq & 7;
                let dist = (pr - kr).abs().max((pf - kf).abs());
                if dist <= 2 {
                    pawns_near_king += 1;
                }
            }
            if pawns_near_king < 2 {
                sc_pos += sign * (-40);
            }

            // Open file
            let mut has_pawn_on_file = false;
            for pi in 0..pawn_n[ci] {
                if (pawn_sq[ci][pi] & 7) as i32 == kf {
                    has_pawn_on_file = true;
                    break;
                }
            }
            if !has_pawn_on_file {
                sc_pos += sign * (-25);
            }

            // Bishop pair
            if bishop_count[ci] >= 2 {
                sc_pos += sign * 30;
            }
        }

        sc_mat + sc_pos
    }

    fn order_moves(&self, pos: &Position, ml: &mut MoveList, ply: usize, ttm: &Move) {
        let mut scores = vec![0.0f32; ml.moves.len()];
        let cm = if !move_is_null(&self.prev_move) {
            self.countermove[self.prev_move.from as usize][self.prev_move.to as usize]
        } else {
            Move::default()
        };

        for i in 0..ml.moves.len() {
            let m = &ml.moves[i];
            let s: f32;
            if move_eq(m, ttm) {
                s = 1e7;
            } else {
                let mut sc = 0.0f32;
                if !pos.board[m.to as usize].is_empty() {
                    sc += 100000.0
                        + pv_val(pos.board[m.to as usize].piece_type) as f32 * 10.0
                        - pv_val(pos.board[m.from as usize].piece_type) as f32;
                }
                if m.special as u8 == SpecialMove::Promotion as u8 {
                    sc += 95000.0 + pv_val(m.promo_piece) as f32;
                }
                // Pawn advancement heuristic
                let mpt = pos.board[m.from as usize].piece_type;
                if mpt == PieceType::Pawn {
                    let mc = pos.board[m.from as usize].color;
                    let adv = if mc == Color::White {
                        (m.to >> 3) as i32
                    } else {
                        7 - (m.to >> 3) as i32
                    };
                    if adv >= 5 {
                        sc += 50000.0 + adv as f32 * 5000.0;
                    }
                }
                if ply < 64 {
                    if move_eq(m, &self.killers[ply][0]) {
                        sc += 80000.0;
                    } else if move_eq(m, &self.killers[ply][1]) {
                        sc += 79000.0;
                    }
                }
                if move_eq(m, &cm) {
                    sc += 60000.0;
                }
                sc += self.history[pos.side_to_move as usize][m.from as usize][m.to as usize]
                    as f32;
                s = sc;
            }
            scores[i] = s;
        }

        // Selection sort with NEON-accelerated max finding
        for i in 0..ml.moves.len() {
            let best_idx = find_max_index(&scores, i, ml.moves.len());
            if best_idx != i {
                ml.moves.swap(i, best_idx);
                scores.swap(i, best_idx);
            }
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

        // Mate distance pruning
        alpha = alpha.max(-99000 + ply);
        beta = beta.min(99000 - ply - 1);
        if alpha >= beta {
            return alpha;
        }

        // TT probe
        let key = pos.zobrist;
        let tt_idx = (key as usize) & TT_MASK;
        let k32 = (key >> 32) as u32;

        let mut ttm = Move::default();
        // Read TT entry
        let tt_key32 = self.tt[tt_idx].key32;
        let tt_depth = self.tt[tt_idx].depth;
        let tt_flag = self.tt[tt_idx].flag;
        let tt_score = self.tt[tt_idx].score as i32;

        if tt_key32 == k32 {
            ttm = tt_to_move(&self.tt[tt_idx]);
            if tt_depth as i32 >= depth {
                self.tt_hits += 1;
                if tt_flag == 0 {
                    return tt_score;
                }
                if tt_flag == 2 && tt_score >= beta {
                    return tt_score;
                }
                if tt_flag == 1 && tt_score <= alpha {
                    return tt_score;
                }
            }
        }

        if depth <= 0 {
            return self.qsearch(pos, alpha, beta, 0);
        }

        // Reverse futility pruning
        if !in_check && depth <= 4 && ply > 0 {
            let eval = self.evaluate(pos);
            if eval - depth * 100 >= beta {
                self.null_cuts += 1;
                return beta;
            }
            if depth <= 2 && eval + 300 < alpha {
                let qs = self.qsearch(pos, alpha, beta, 0);
                if qs < alpha {
                    return qs;
                }
            }
        }

        // IID
        if move_is_null(&ttm) && depth >= 4 && !self.stopped {
            self.search(pos, depth - 2, alpha, beta, ply, in_check);
            let tt_key32_2 = self.tt[tt_idx].key32;
            if tt_key32_2 == k32 {
                ttm = tt_to_move(&self.tt[tt_idx]);
            }
        }

        // Generate legal moves
        self.move_buf.clear();
        generate_legal_moves(pos, &mut self.move_buf);
        let mut ml = MoveList::new();
        for i in 0..self.move_buf.len() {
            ml.push(self.move_buf[i]);
        }

        if ml.moves.len() == 0 {
            return if in_check { -99000 + ply } else { 0 };
        }

        self.order_moves(pos, &mut ml, ply as usize, &ttm);

        let saved_prev = self.prev_move;
        let mut best_move = ml.moves[0];
        let mut best_score = -100000i32;
        let mut flag: u8 = 1;

        let mut i = 0;
        while i < ml.moves.len() && !self.stopped {
            let m = ml.moves[i];

            // is_tactical BEFORE make_move
            let is_tactical = !pos.board[m.to as usize].is_empty()
                || m.special as u8 == SpecialMove::Promotion as u8
                || m.special as u8 == SpecialMove::EnPassant as u8;

            // Late move pruning
            if !is_tactical && !in_check && depth <= 2 && i as i32 >= 8 + depth * 4 {
                i += 1;
                continue;
            }

            self.prev_move = m;
            pos.make_move(&m);
            let gives_check = pos.in_check();

            let score: i32;
            let d = depth.min(31) as usize;
            let mi = i.min(255);

            if i >= 3 && depth >= 2 && !is_tactical && !gives_check && !in_check {
                self.lmr_tries += 1;
                let mut r = LMR[d][mi].clamp(1, depth - 1);
                // History-aware LMR
                let ci = opponent(pos.side_to_move) as usize;
                let hscore = self.history[ci][m.from as usize][m.to as usize] as i32;
                if hscore < -500 {
                    r += 2;
                } else if hscore < -100 {
                    r += 1;
                }
                r = r.clamp(1, depth - 1);

                let mut s = -self.search(pos, depth - 1 - r, -(alpha + 1), -alpha, ply + 1, gives_check);
                if s > alpha && !self.stopped {
                    self.lmr_re += 1;
                    s = -self.search(pos, depth - 1, -beta, -alpha, ply + 1, gives_check);
                }
                score = s;
            } else {
                let ext = if gives_check && depth <= 4 { 1 } else { 0 };
                if i > 0 && !self.stopped {
                    let mut s = -self.search(
                        pos,
                        depth - 1 + ext,
                        -(alpha + 1),
                        -alpha,
                        ply + 1,
                        gives_check,
                    );
                    if s > alpha && s < beta && !self.stopped {
                        s = -self.search(pos, depth - 1 + ext, -beta, -alpha, ply + 1, gives_check);
                    }
                    score = s;
                } else {
                    score = -self.search(pos, depth - 1 + ext, -beta, -alpha, ply + 1, gives_check);
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
                if !is_tactical && (ply as usize) < 64 {
                    let ply_u = ply as usize;
                    self.killers[ply_u][1] = self.killers[ply_u][0];
                    self.killers[ply_u][0] = m;
                    let ci = pos.side_to_move as usize;
                    let h = &mut self.history[ci][m.from as usize][m.to as usize];
                    *h = (*h as i32 + depth * depth).min(16000) as i16;
                    for j in 0..i {
                        if pos.board[ml.moves[j].to as usize].is_empty() {
                            let hh =
                                &mut self.history[ci][ml.moves[j].from as usize][ml.moves[j].to as usize];
                            *hh = (*hh as i32 - depth).max(-16000) as i16;
                        }
                    }
                    if !move_is_null(&saved_prev) {
                        self.countermove[saved_prev.from as usize][saved_prev.to as usize] = m;
                    }
                }
                break;
            }
            i += 1;
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
        if qdepth >= 4 {
            return self.evaluate(pos);
        }

        let stand_pat = self.evaluate(pos);
        if stand_pat >= beta {
            return stand_pat;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }
        // Delta pruning
        if stand_pat + 1000 < alpha {
            return alpha;
        }

        self.move_buf.clear();
        generate_legal_moves(pos, &mut self.move_buf);

        // Collect tactical moves
        let mut tactical: Vec<(Move, i32)> = Vec::with_capacity(64);
        for idx in 0..self.move_buf.len() {
            let m = self.move_buf[idx];
            let tac = !pos.board[m.to as usize].is_empty()
                || m.special as u8 == SpecialMove::Promotion as u8
                || m.special as u8 == SpecialMove::EnPassant as u8;
            if !tac {
                continue;
            }
            let mut see = pv_val(pos.board[m.to as usize].piece_type);
            if m.special as u8 == SpecialMove::Promotion as u8 {
                see += 800;
            }
            if stand_pat + see + 200 < alpha {
                continue;
            }
            tactical.push((m, see));
        }

        // Sort tactical moves by SEE (selection sort)
        for i in 0..tactical.len() {
            for j in (i + 1)..tactical.len() {
                if tactical[j].1 > tactical[i].1 {
                    tactical.swap(i, j);
                }
            }
        }

        for i in 0..tactical.len() {
            let m = tactical[i].0;
            pos.make_move(&m);
            let score = -self.qsearch(pos, -beta, -alpha, qdepth + 1);
            pos.unmake_move();
            if self.stopped {
                return 0;
            }
            if score >= beta {
                return score;
            }
            if score > alpha {
                alpha = score;
            }
        }
        alpha
    }

    fn extract_pv(&mut self, pos: &mut Position, pv_out: &mut Vec<Move>) {
        pv_out.clear();
        let mut seen = [0u64; 32];
        let mut sn = 0usize;
        let max_pv = 32;

        for _i in 0..max_pv {
            let key = pos.zobrist;
            // Check for cycle
            for j in 0..sn {
                if seen[j] == key {
                    // Unmake all PV moves
                    for _ in 0..pv_out.len() {
                        pos.unmake_move();
                    }
                    return;
                }
            }
            if sn < 32 {
                seen[sn] = key;
                sn += 1;
            }

            let tt_idx = (key as usize) & TT_MASK;
            let k32 = (key >> 32) as u32;
            if self.tt[tt_idx].key32 != k32 {
                break;
            }
            let m = tt_to_move(&self.tt[tt_idx]);
            if m.from >= 64 || m.to >= 64 {
                break;
            }
            if pos.board[m.from as usize].is_empty() {
                break;
            }
            // Validate against legal moves
            self.move_buf.clear();
            generate_legal_moves(pos, &mut self.move_buf);
            let mut found = false;
            for lm in &self.move_buf {
                if move_eq(lm, &m) {
                    found = true;
                    break;
                }
            }
            if !found {
                break;
            }
            pv_out.push(m);
            pos.make_move(&m);
        }

        // Unmake all PV moves
        for _ in 0..pv_out.len() {
            pos.unmake_move();
        }
    }

    fn dump_diag(&mut self, pos: &mut Position, root: &MoveList) -> String {
        let mut ranked: Vec<(String, i32)> = Vec::new();
        for i in 0..root.moves.len() {
            let m = root.moves[i];
            pos.make_move(&m);
            let key = pos.zobrist;
            let tt_idx = (key as usize) & TT_MASK;
            let k32 = (key >> 32) as u32;
            let sc = if self.tt[tt_idx].key32 == k32 {
                -(self.tt[tt_idx].score as i32)
            } else {
                -self.evaluate(pos)
            };
            pos.unmake_move();
            ranked.push((move_to_uci(&m), sc));
        }

        ranked.sort_by(|a, b| b.1.cmp(&a.1));
        let cap = ranked.len().min(32);

        let mut s = format!(
            r#"{{"engine":"chimera_013","qn":{},"tt":{},"bcut":{},"fcut":{},"nmp":{},"lmr":[{},{}],"top_moves":["#,
            self.qnodes, self.tt_hits, self.beta_cuts, self.first_cuts,
            self.null_cuts, self.lmr_tries, self.lmr_re
        );
        for i in 0..cap {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!(r#"["{}",{}]"#, ranked[i].0, ranked[i].1));
        }
        s.push_str("]}");
        s
    }
}

impl Engine for ChimeraEngine {
    fn name(&self) -> &str {
        "chimera_013"
    }

    fn new_game(&mut self, my_color: Color, _game_seed: u64) {
        self.color = my_color;
        self.tt.iter_mut().for_each(|e| *e = TTEntry::default());
        self.history = [[[0i16; 64]; 64]; 2];
        self.killers = [[Move::default(); 2]; 64];
        self.countermove = [[Move::default(); 64]; 64];
        self.prev_move = Move::default();
        // Ensure LMR is initialized
        let _ = &*LMR;
    }

    fn choose_move(&mut self, pos: &mut Position, budget: &SearchBudget) -> (Move, SearchStats) {
        self.t0 = Instant::now();
        self.budget = budget.clone();
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

        self.move_buf.clear();
        generate_legal_moves(pos, &mut self.move_buf);
        let mut root = MoveList::new();
        for i in 0..self.move_buf.len() {
            root.push(self.move_buf[i]);
        }

        let mut stats = SearchStats::default();

        if root.moves.len() == 0 {
            return (Move::default(), stats);
        }
        if root.moves.len() == 1 {
            stats.nodes = 1;
            stats.depth_reached = 0;
            return (root.moves[0], stats);
        }

        let mut best = root.moves[0];
        let mut best_score: i32 = -100000;

        for depth in 1..=30 {
            if self.stopped {
                break;
            }

            let (mut alpha, mut beta) = if depth <= 3 || best_score.abs() > 5000 {
                (-100000i32, 100000i32)
            } else {
                (best_score - 50, best_score + 50)
            };

            let score;
            loop {
                let s = self.search(pos, depth, alpha, beta, 0, false);
                if self.stopped {
                    break;
                }
                if s <= alpha {
                    alpha = (alpha - 200).max(-100000);
                } else if s >= beta {
                    beta = (beta + 200).min(100000);
                } else {
                    score = s;
                    // Look up TT for root position
                    let key = pos.zobrist;
                    let tt_idx = (key as usize) & TT_MASK;
                    let k32 = (key >> 32) as u32;
                    if self.tt[tt_idx].key32 == k32 {
                        let ttm = tt_to_move(&self.tt[tt_idx]);
                        for i in 0..root.moves.len() {
                            if move_eq(&root.moves[i], &ttm) {
                                best = ttm;
                                break;
                            }
                        }
                    }
                    best_score = score;
                    stats.depth_reached = depth as u32;
                    stats.eval_cp = best_score;
                    break;
                }
            }
        }

        // Safety fallback
        if best_score <= -90000 {
            let mut bs = -100000i32;
            for i in 0..root.moves.len() {
                pos.make_move(&root.moves[i]);
                let sc = -self.evaluate(pos);
                pos.unmake_move();
                if sc > bs {
                    bs = sc;
                    best = root.moves[i];
                }
            }
        }

        // Validate returned move
        let mut valid = false;
        for i in 0..root.moves.len() {
            if move_eq(&root.moves[i], &best) {
                valid = true;
                break;
            }
        }
        if !valid {
            best = root.moves[0];
        }

        stats.nodes = self.nodes;
        stats.seldepth = self.max_ply;
        stats.time_used_us = self.elapsed_us();

        let mut pv = Vec::new();
        self.extract_pv(pos, &mut pv);
        stats.pv = pv;

        stats.diag_json = self.dump_diag(pos, &root);

        (best, stats)
    }
}

pub fn create() -> Box<dyn Engine> {
    Box::new(ChimeraEngine::new())
}
