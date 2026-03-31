// colossus_017: Maximum depth via aggressive LMR + proven eval
//
// Based on chimera's proven eval + razor's aggressive LMR divisor (1.5)
// + wider LMP (prune at move 6 not 8). Get depth of razor with accuracy of chimera.
//
// Key params:
//   - LMR divisor 1.5 (most aggressive)
//   - LMP: depth <= 3, prune at 6 + depth*3 (wider)
//   - Deeper futility: depth <= 5 with 120cp/ply
//   - Check extension up to depth 6 (not just 4)
//   - Aspiration window 40cp
//   - TT: 512K

use std::sync::LazyLock;
use std::time::Instant;
use crate::core::types::*;
use crate::core::position::Position;
use crate::core::movegen::generate_legal_moves;
use crate::engine::Engine;

const TT_BITS: usize = 19;
const TT_SIZE: usize = 1 << TT_BITS;
const TT_MASK: usize = TT_SIZE - 1;

const PV_VALS: [i32; 7] = [0, 100, 320, 330, 500, 900, 0];
fn pv(pt: PieceType) -> i32 { PV_VALS[pt as usize] }

#[derive(Clone, Copy, Default)]
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

struct MoveList {
    moves: [Move; 256],
    count: usize,
}

impl MoveList {
    fn new() -> Self {
        Self { moves: [Move::default(); 256], count: 0 }
    }
    fn push(&mut self, m: Move) {
        if self.count < 256 { self.moves[self.count] = m; self.count += 1; }
    }
}

// LMR table: divisor 1.5
static LMR_TABLE: LazyLock<[[i32; 256]; 32]> = LazyLock::new(|| {
    let mut t = [[0i32; 256]; 32];
    for d in 0..32 {
        for m in 0..256 {
            t[d][m] = if d < 2 || m < 3 { 0 }
            else { (0.5 + (d as f64).ln() * (m as f64).ln() / 1.5) as i32 };
        }
    }
    t
});

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
    if m.special == SpecialMove::Promotion {
        let c = match m.promo_piece {
            PieceType::Knight => 'n', PieceType::Bishop => 'b',
            PieceType::Rook => 'r', PieceType::Queen => 'q', _ => ' ',
        };
        if c != ' ' { s.push(c); }
    }
    s
}

struct Colossus {
    color: Color,
    tt: Vec<TTEntry>,
    history: [[[i16; 64]; 64]; 2],
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
    budget_time_us: i64,
    t0: Instant,
}

impl Colossus {
    fn new() -> Self {
        // Force LMR table init
        let _ = &*LMR_TABLE;
        Self {
            color: Color::White,
            tt: vec![TTEntry::default(); TT_SIZE],
            history: [[[0i16; 64]; 64]; 2],
            killers: [[Move::default(); 2]; 64],
            countermove: [[Move::default(); 64]; 64],
            prev_move: Move::default(),
            move_buf: Vec::new(),
            nodes: 0, qnodes: 0,
            beta_cuts: 0, first_cuts: 0, tt_hits: 0,
            null_cuts: 0, lmr_tries: 0, lmr_re: 0,
            max_ply: 0, stopped: false,
            budget_time_us: 0,
            t0: Instant::now(),
        }
    }

    fn elapsed_us(&self) -> i64 {
        self.t0.elapsed().as_micros() as i64
    }

    fn check_time(&mut self) -> bool {
        if self.stopped { return true; }
        if self.budget_time_us > 0 && (self.nodes & 255) == 0 {
            if self.elapsed_us() >= self.budget_time_us * 9 / 10 {
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
            e.depth = (depth.min(255)) as u8;
            e.flag = flag;
            e.from = m.from; e.to = m.to;
            e.path_kind = m.path_kind; e.stop_idx = m.stop_index;
            e.special = m.special as u8;
            e.promo = m.promo_piece as u8;
        }
    }

    fn evaluate(&self, pos: &Position) -> i32 {
        let stm = pos.side_to_move;
        let mut sc_mat: i32 = 0;
        let mut sc_pos: i32 = 0;
        let mut pawn_sq = [[0u8; 8]; 2];
        let mut pawn_n = [0usize; 2];
        let mut slider_sq = [[0u8; 16]; 2];
        let mut slider_n = [0usize; 2];
        let mut king = [0u8; 2];
        let mut bishop_count = [0i32; 2];

        for sq in 0..64u8 {
            let p = pos.board[sq as usize];
            if p.is_empty() { continue; }
            let pt = p.piece_type;
            let c = p.color;
            let ci = c as usize;
            let sign = if c == stm { 1 } else { -1 };
            let r = (sq >> 3) as i32;
            let f = (sq & 7) as i32;

            sc_mat += sign * pv(pt);

            let cd = (r - 3).abs().max((f - 3).abs());
            if pt == PieceType::Knight { sc_pos += sign * (3 - cd) * 5; }
            else if pt != PieceType::Pawn && pt != PieceType::King { sc_pos += sign * (3 - cd) * 2; }

            if pt == PieceType::Pawn {
                let adv = if c == Color::White { r } else { 7 - r };
                sc_pos += sign * (adv * adv * 3);
                if pawn_n[ci] < 8 { pawn_sq[ci][pawn_n[ci]] = sq; pawn_n[ci] += 1; }
                let mut passed = true;
                let dir: i32 = if c == Color::White { 1 } else { -1 };
                let mut fr = r + dir;
                'outer: while fr >= 0 && fr < 8 {
                    for df in -1..=1i32 {
                        let ff = f + df;
                        if ff < 0 || ff > 7 { continue; }
                        let cs = (fr * 8 + ff) as u8;
                        let cp = pos.board[cs as usize];
                        if !cp.is_empty() && cp.piece_type == PieceType::Pawn && cp.color != c {
                            passed = false; break 'outer;
                        }
                    }
                    fr += dir;
                }
                if passed { sc_pos += sign * (adv * 15); }
            }

            if pt == PieceType::King { king[ci] = sq; }
            if pt == PieceType::Bishop { bishop_count[ci] += 1; }
            if pt == PieceType::Rook || pt == PieceType::Queen {
                if slider_n[ci] < 16 { slider_sq[ci][slider_n[ci]] = sq; slider_n[ci] += 1; }
            }
        }

        // Piston bonus
        for ci in 0..2usize {
            let c = if ci == 0 { Color::White } else { Color::Black };
            let sign = if c == stm { 1 } else { -1 };
            for pi in 0..pawn_n[ci] {
                let psq = pawn_sq[ci][pi] as i32;
                let pr = psq >> 3; let pf = psq & 7;
                let adv = if c == Color::White { pr } else { 7 - pr };
                if adv < 4 { continue; }
                for si in 0..slider_n[ci] {
                    let ssq = slider_sq[ci][si] as i32;
                    if (ssq & 7) != pf { continue; }
                    let sr = ssq >> 3;
                    let behind = if c == Color::White { sr < pr } else { sr > pr };
                    if !behind { continue; }
                    let dir: i32 = if c == Color::White { 1 } else { -1 };
                    let mut clear = true;
                    let mut cr = sr + dir;
                    while cr != pr {
                        if !pos.board[(cr * 8 + pf) as usize].is_empty() { clear = false; break; }
                        cr += dir;
                    }
                    if !clear { continue; }
                    sc_pos += sign * (if adv >= 5 { 150 } else { 60 });
                    break;
                }
            }
        }

        // King tropism
        for ci in 0..2usize {
            let oci = 1 - ci;
            let sign = if ci == stm as usize { 1 } else { -1 };
            let kr = (king[ci] >> 3) as i32; let kf = (king[ci] & 7) as i32;
            for pi in 0..pawn_n[oci] {
                let psq = pawn_sq[oci][pi] as i32;
                let pr = psq >> 3; let pf = psq & 7;
                let adv = if oci == 0 { pr } else { 7 - pr };
                if adv < 4 { continue; }
                let dist = (kr - pr).abs().max((kf - pf).abs());
                if dist <= 2 { sc_pos -= sign * (20 + adv * 8); }
            }
        }

        // King safety
        for ci in 0..2usize {
            let c = if ci == 0 { Color::White } else { Color::Black };
            let sign = if c == stm { 1 } else { -1 };
            let kr = (king[ci] >> 3) as i32; let kf = (king[ci] & 7) as i32;
            let shield_dir: i32 = if c == Color::White { 1 } else { -1 };

            for pi in 0..pawn_n[ci] {
                let psq = pawn_sq[ci][pi] as i32;
                let pr = psq >> 3; let pf = psq & 7;
                let file_dist = (pf - kf).abs();
                let rank_ahead = (pr - kr) * shield_dir;
                if file_dist <= 1 && rank_ahead >= 1 && rank_ahead <= 2 {
                    sc_pos += sign * 15;
                }
            }

            let mut pawns_near_king = 0;
            for pi in 0..pawn_n[ci] {
                let psq = pawn_sq[ci][pi] as i32;
                let pr = psq >> 3; let pf = psq & 7;
                let dist = (pr - kr).abs().max((pf - kf).abs());
                if dist <= 2 { pawns_near_king += 1; }
            }
            if pawns_near_king < 2 { sc_pos += sign * (-40); }

            let mut has_pawn_on_file = false;
            for pi in 0..pawn_n[ci] {
                if (pawn_sq[ci][pi] & 7) as i32 == kf { has_pawn_on_file = true; break; }
            }
            if !has_pawn_on_file { sc_pos += sign * (-25); }

            if bishop_count[ci] >= 2 { sc_pos += sign * 30; }
        }

        sc_mat + sc_pos
    }

    fn order_moves(&self, pos: &Position, ml: &mut MoveList, ply: usize, ttm: &Move) {
        let mut scores = [0.0f32; 256];
        let cm = if self.prev_move.from != 0 || self.prev_move.to != 0 {
            self.countermove[self.prev_move.from as usize][self.prev_move.to as usize]
        } else { Move::default() };

        for i in 0..ml.count {
            let m = &ml.moves[i];
            let s;
            if *m == *ttm { s = 1e7; }
            else {
                let mut v = 0.0f32;
                if !pos.board[m.to as usize].is_empty() {
                    v += 100000.0 + pv(pos.board[m.to as usize].piece_type) as f32 * 10.0
                        - pv(pos.board[m.from as usize].piece_type) as f32;
                }
                if m.special == SpecialMove::Promotion {
                    v += 95000.0 + pv(m.promo_piece) as f32;
                }
                let mpt = pos.board[m.from as usize].piece_type;
                if mpt == PieceType::Pawn {
                    let mc = pos.board[m.from as usize].color;
                    let adv = if mc == Color::White { m.to >> 3 } else { 7 - (m.to >> 3) } as i32;
                    if adv >= 5 { v += 50000.0 + adv as f32 * 5000.0; }
                }
                if ply < 64 {
                    if *m == self.killers[ply][0] { v += 80000.0; }
                    else if *m == self.killers[ply][1] { v += 79000.0; }
                }
                if *m == cm { v += 60000.0; }
                v += self.history[pos.side_to_move as usize][m.from as usize][m.to as usize] as f32;
                s = v as f64;
            }
            scores[i] = s as f32;
        }
        // Selection sort
        for i in 0..ml.count {
            let mut best_idx = i;
            let mut best_val = scores[i];
            for j in (i+1)..ml.count {
                if scores[j] > best_val { best_val = scores[j]; best_idx = j; }
            }
            if best_idx != i {
                ml.moves.swap(i, best_idx);
                scores.swap(i, best_idx);
            }
        }
    }

    fn search(&mut self, pos: &mut Position, depth: i32, mut alpha: i32, mut beta: i32, ply: i32, in_check: bool) -> i32 {
        if self.check_time() { return 0; }
        self.nodes += 1;
        if ply as u32 > self.max_ply { self.max_ply = ply as u32; }

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
                if e.depth as i32 >= depth {
                    self.tt_hits += 1;
                    if e.flag == 0 { return e.score as i32; }
                    if e.flag == 2 && e.score as i32 >= beta { return e.score as i32; }
                    if e.flag == 1 && e.score as i32 <= alpha { return e.score as i32; }
                }
            }
        }

        if depth <= 0 { return self.qsearch(pos, alpha, beta, 0); }

        // Deeper reverse futility: depth <= 5, 120cp/ply
        if !in_check && depth <= 5 && ply > 0 {
            let eval = self.evaluate(pos);
            if eval - depth * 120 >= beta { self.null_cuts += 1; return beta; }
            if depth <= 2 && eval + 300 < alpha {
                let qs = self.qsearch(pos, alpha, beta, 0);
                if qs < alpha { return qs; }
            }
        }

        // IID
        if ttm.from == 0 && ttm.to == 0 && depth >= 4 && !self.stopped {
            self.search(pos, depth - 2, alpha, beta, ply, in_check);
            let e = &self.tt[idx];
            if e.key32 == k32 { ttm = tt_to_move(e); }
        }

        let mut ml = MoveList::new();
        {
            self.move_buf.clear();
            generate_legal_moves(pos, &mut self.move_buf);
            for i in 0..self.move_buf.len() { ml.push(self.move_buf[i]); }
        }

        if ml.count == 0 { return if in_check { -99000 + ply } else { 0 }; }
        self.order_moves(pos, &mut ml, ply as usize, &ttm);

        let saved_prev = self.prev_move;
        let mut best_move = ml.moves[0];
        let mut best_score = -100000i32;
        let mut flag: u8 = 1;

        let mut i = 0;
        while i < ml.count && !self.stopped {
            let m = ml.moves[i];

            let is_tactical = !pos.board[m.to as usize].is_empty()
                || m.special == SpecialMove::Promotion
                || m.special == SpecialMove::EnPassant;

            // Wider LMP: depth <= 3, prune at move 6 + depth*3
            if !is_tactical && !in_check && depth <= 3 && i as i32 >= 6 + depth * 3 {
                i += 1; continue;
            }

            self.prev_move = m;
            pos.make_move(&m);
            let gives_check = pos.in_check();

            let score;
            let d = depth.min(31) as usize;
            let mi = i.min(255);

            if i >= 3 && depth >= 2 && !is_tactical && !gives_check && !in_check {
                self.lmr_tries += 1;
                let mut r = LMR_TABLE[d][mi].clamp(1, depth - 1);
                // History-aware LMR
                let ci = if pos.side_to_move == Color::White { Color::Black } else { Color::White } as usize;
                let hscore = self.history[ci][m.from as usize][m.to as usize] as i32;
                if hscore < -500 { r += 2; }
                else if hscore < -100 { r += 1; }
                r = r.clamp(1, depth - 1);

                score = -self.search(pos, depth - 1 - r, -(alpha+1), -alpha, ply + 1, gives_check);
                if score > alpha && !self.stopped {
                    self.lmr_re += 1;
                    let s2 = -self.search(pos, depth - 1, -beta, -alpha, ply + 1, gives_check);
                    // Use s2 below
                    pos.unmake_move();
                    if s2 > best_score { best_score = s2; best_move = m; }
                    if s2 > alpha { alpha = s2; flag = 0; }
                    if alpha >= beta {
                        flag = 2; self.beta_cuts += 1;
                        if i == 0 { self.first_cuts += 1; }
                        if !is_tactical && (ply as usize) < 64 {
                            self.killers[ply as usize][1] = self.killers[ply as usize][0];
                            self.killers[ply as usize][0] = m;
                            let ci2 = pos.side_to_move as usize;
                            let h = &mut self.history[ci2][m.from as usize][m.to as usize];
                            *h = (*h + (depth * depth) as i16).min(16000);
                            for j in 0..i {
                                if pos.board[ml.moves[j].to as usize].is_empty() {
                                    let hh = &mut self.history[ci2][ml.moves[j].from as usize][ml.moves[j].to as usize];
                                    *hh = (*hh - depth as i16).max(-16000);
                                }
                            }
                            if saved_prev.from != 0 || saved_prev.to != 0 {
                                self.countermove[saved_prev.from as usize][saved_prev.to as usize] = m;
                            }
                        }
                        self.prev_move = saved_prev;
                        if !self.stopped { self.tt_store(key, depth, best_score, flag, &best_move); }
                        return best_score;
                    }
                    i += 1; continue;
                }
            } else {
                // Check extension up to depth 6
                let ext = if gives_check && depth <= 6 { 1 } else { 0 };
                if i > 0 && !self.stopped {
                    score = -self.search(pos, depth - 1 + ext, -(alpha+1), -alpha, ply + 1, gives_check);
                    if score > alpha && score < beta && !self.stopped {
                        let s2 = -self.search(pos, depth - 1 + ext, -beta, -alpha, ply + 1, gives_check);
                        pos.unmake_move();
                        if s2 > best_score { best_score = s2; best_move = m; }
                        if s2 > alpha { alpha = s2; flag = 0; }
                        if alpha >= beta {
                            flag = 2; self.beta_cuts += 1;
                            if i == 0 { self.first_cuts += 1; }
                            if !is_tactical && (ply as usize) < 64 {
                                self.killers[ply as usize][1] = self.killers[ply as usize][0];
                                self.killers[ply as usize][0] = m;
                                let ci2 = pos.side_to_move as usize;
                                let h = &mut self.history[ci2][m.from as usize][m.to as usize];
                                *h = (*h + (depth * depth) as i16).min(16000);
                                for j in 0..i {
                                    if pos.board[ml.moves[j].to as usize].is_empty() {
                                        let hh = &mut self.history[ci2][ml.moves[j].from as usize][ml.moves[j].to as usize];
                                        *hh = (*hh - depth as i16).max(-16000);
                                    }
                                }
                                if saved_prev.from != 0 || saved_prev.to != 0 {
                                    self.countermove[saved_prev.from as usize][saved_prev.to as usize] = m;
                                }
                            }
                            self.prev_move = saved_prev;
                            if !self.stopped { self.tt_store(key, depth, best_score, flag, &best_move); }
                            return best_score;
                        }
                        i += 1; continue;
                    }
                } else {
                    score = -self.search(pos, depth - 1 + ext, -beta, -alpha, ply + 1, gives_check);
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
                    let h = &mut self.history[ci][m.from as usize][m.to as usize];
                    *h = (*h + (depth * depth) as i16).min(16000);
                    for j in 0..i {
                        if pos.board[ml.moves[j].to as usize].is_empty() {
                            let hh = &mut self.history[ci][ml.moves[j].from as usize][ml.moves[j].to as usize];
                            *hh = (*hh - depth as i16).max(-16000);
                        }
                    }
                    if saved_prev.from != 0 || saved_prev.to != 0 {
                        self.countermove[saved_prev.from as usize][saved_prev.to as usize] = m;
                    }
                }
                break;
            }
            i += 1;
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

        self.move_buf.clear();
        generate_legal_moves(pos, &mut self.move_buf);

        let mut tactical: [(Move, i32); 64] = [(Move::default(), 0); 64];
        let mut nt = 0usize;
        for idx in 0..self.move_buf.len() {
            let m = self.move_buf[idx];
            let tac = !pos.board[m.to as usize].is_empty()
                || m.special == SpecialMove::Promotion
                || m.special == SpecialMove::EnPassant;
            if !tac { continue; }
            let see = pv(pos.board[m.to as usize].piece_type)
                + if m.special == SpecialMove::Promotion { 800 } else { 0 };
            if stand_pat + see + 200 < alpha { continue; }
            if nt < 64 { tactical[nt] = (m, see); nt += 1; }
        }
        // Sort by SEE descending
        for i in 0..nt {
            for j in (i+1)..nt {
                if tactical[j].1 > tactical[i].1 { tactical.swap(i, j); }
            }
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

    fn extract_pv(&mut self, pos: &mut Position, pv_out: &mut Vec<Move>) {
        pv_out.clear();
        let mut seen = [0u64; 32];
        let mut sn = 0usize;
        let mut depth = 0;
        loop {
            if depth >= 32 { break; }
            let key = pos.zobrist;
            for j in 0..sn { if seen[j] == key { self.unwind_pv(pos, pv_out); return; } }
            if sn < 32 { seen[sn] = key; sn += 1; }
            let idx = (key as usize) & TT_MASK;
            let k32 = (key >> 32) as u32;
            let e = &self.tt[idx];
            if e.key32 != k32 { break; }
            let m = tt_to_move(e);
            if m.from >= 64 || m.to >= 64 { break; }
            if pos.board[m.from as usize].is_empty() { break; }
            self.move_buf.clear();
            generate_legal_moves(pos, &mut self.move_buf);
            let mut found = false;
            for k in 0..self.move_buf.len() { if self.move_buf[k] == m { found = true; break; } }
            if !found { break; }
            pv_out.push(m);
            pos.make_move(&m);
            depth += 1;
        }
        self.unwind_pv(pos, pv_out);
    }

    fn unwind_pv(&self, pos: &mut Position, pv: &[Move]) {
        for _ in 0..pv.len() { pos.unmake_move(); }
    }

    fn dump_diag(&mut self, pos: &mut Position, root: &MoveList) -> String {
        let mut ranked: Vec<(String, i32)> = Vec::new();
        for i in 0..root.count {
            let m = root.moves[i];
            pos.make_move(&m);
            let key = pos.zobrist;
            let idx = (key as usize) & TT_MASK;
            let k32 = (key >> 32) as u32;
            let sc = if self.tt[idx].key32 == k32 {
                -(self.tt[idx].score as i32)
            } else {
                -self.evaluate(pos)
            };
            pos.unmake_move();
            ranked.push((move_to_uci(&m), sc));
        }
        ranked.sort_by(|a, b| b.1.cmp(&a.1));
        let cap = ranked.len().min(32);

        let mut s = format!(
            r#"{{"engine":"colossus_017","qn":{},"tt":{},"bcut":{},"fcut":{},"nmp":{},"lmr":[{},{}],"top_moves":["#,
            self.qnodes, self.tt_hits, self.beta_cuts, self.first_cuts,
            self.null_cuts, self.lmr_tries, self.lmr_re
        );
        for i in 0..cap {
            if i > 0 { s.push(','); }
            s.push_str(&format!(r#"["{}",{}]"#, ranked[i].0, ranked[i].1));
        }
        s.push_str("]}");
        s
    }
}

impl Engine for Colossus {
    fn name(&self) -> &str { "colossus_017" }

    fn new_game(&mut self, my_color: Color, _game_seed: u64) {
        self.color = my_color;
        self.tt.fill(TTEntry::default());
        self.history = [[[0i16; 64]; 64]; 2];
        self.killers = [[Move::default(); 2]; 64];
        self.countermove = [[Move::default(); 64]; 64];
        self.prev_move = Move::default();
    }

    fn choose_move(&mut self, pos: &mut Position, budget: &SearchBudget) -> (Move, SearchStats) {
        self.t0 = Instant::now();
        self.budget_time_us = budget.max_time_us;
        self.nodes = 0; self.qnodes = 0; self.max_ply = 0;
        self.stopped = false;
        self.beta_cuts = 0; self.first_cuts = 0; self.tt_hits = 0;
        self.null_cuts = 0; self.lmr_tries = 0; self.lmr_re = 0;
        self.prev_move = Move::default();

        let mut root = MoveList::new();
        {
            self.move_buf.clear();
            generate_legal_moves(pos, &mut self.move_buf);
            for i in 0..self.move_buf.len() { root.push(self.move_buf[i]); }
        }

        let mut stats = SearchStats::default();

        if root.count == 0 { return (Move::default(), stats); }
        if root.count == 1 {
            stats.nodes = 1; stats.depth_reached = 0;
            return (root.moves[0], stats);
        }

        let mut best = root.moves[0];
        let mut best_score = -100000i32;

        for depth in 1..=30 {
            if self.stopped { break; }
            let (mut alpha, mut beta);
            if depth <= 3 || best_score.abs() > 5000 {
                alpha = -100000; beta = 100000;
            } else {
                alpha = best_score - 40;  // aspiration window 40cp
                beta = best_score + 40;
            }

            let score;
            loop {
                let s = self.search(pos, depth, alpha, beta, 0, false);
                if self.stopped { score = s; break; }
                if s <= alpha { alpha = (alpha - 200).max(-100000); }
                else if s >= beta { beta = (beta + 200).min(100000); }
                else { score = s; break; }
            }

            if !self.stopped {
                best_score = score;
                let key = pos.zobrist;
                let idx = (key as usize) & TT_MASK;
                let k32 = (key >> 32) as u32;
                if self.tt[idx].key32 == k32 {
                    let ttm = tt_to_move(&self.tt[idx]);
                    for i in 0..root.count {
                        if root.moves[i] == ttm { best = ttm; break; }
                    }
                }
            }
            stats.depth_reached = depth as u32;
            stats.eval_cp = best_score;
        }

        // Safety fallback
        if best_score <= -90000 {
            let mut bs = -100000;
            for i in 0..root.count {
                pos.make_move(&root.moves[i]);
                let sc = -self.evaluate(pos);
                pos.unmake_move();
                if sc > bs { bs = sc; best = root.moves[i]; }
            }
        }

        // Validate
        let mut valid = false;
        for i in 0..root.count {
            if root.moves[i] == best { valid = true; break; }
        }
        if !valid { best = root.moves[0]; }

        stats.nodes = self.nodes;
        stats.seldepth = self.max_ply;
        stats.time_used_us = self.elapsed_us();

        let mut pv_vec = Vec::new();
        self.extract_pv(pos, &mut pv_vec);
        stats.pv = pv_vec;

        stats.diag_json = self.dump_diag(pos, &root);

        (best, stats)
    }
}

pub fn create() -> Box<dyn Engine> {
    Box::new(Colossus::new())
}
