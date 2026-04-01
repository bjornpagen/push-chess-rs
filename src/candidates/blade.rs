// blade_001: Lean search with null move R=2 + PV restriction + age TT + move validation
//
// Based on specter_020 (tournament champion) with:
//   1. PV-node awareness (from wraith/tempest): is_pv parameter in search().
//      - No TT cutoffs in PV nodes
//      - No reverse futility pruning or null move pruning in PV nodes
//   2. Null move pruning (done right):
//      - Only at depth >= 4, only R=2
//      - Not in check, not at root, not in PV
//      - Skip if total_pieces <= 5 (zugzwang avoidance)
//   3. Age-based TT replacement: prefer replacing older entries
//   4. Robust move validation: re-generate legal moves at end of choose_move
//
// Everything else from specter preserved:
//   - King push awareness eval (shield + escape)
//   - Direct best-move tracking at root
//   - NEON move ordering
//   - Conservative LMR (1.8 divisor)
//   - LMP, futility, check extension, aspiration
//   - 512K TT (TT_BITS=19)
//   - Ply >= 128 guard

use std::sync::LazyLock;
use std::time::Instant;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::core::types::*;
use crate::core::position::Position;
use crate::core::movegen::generate_legal_moves;
use crate::core::zobrist::zobrist_tables;
use crate::engine::Engine;

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_find_max_index(scores: &[f32], start: usize, len: usize) -> usize {
    unsafe {
        let ptr = scores.as_ptr().add(start);
        let count = len - start;
        let chunks = count / 4;
        let mut vmax = vld1q_f32(ptr);
        for i in 1..chunks { vmax = vmaxq_f32(vmax, vld1q_f32(ptr.add(i * 4))); }
        let mut max_val = vmaxvq_f32(vmax);
        for i in (chunks * 4)..count { let v = *ptr.add(i); if v > max_val { max_val = v; } }
        let target = vdupq_n_f32(max_val);
        for i in 0..chunks {
            let cmp = vceqq_f32(vld1q_f32(ptr.add(i * 4)), target);
            let mask: [u32; 4] = std::mem::transmute(cmp);
            for lane in 0..4 { if mask[lane] != 0 { return start + i * 4 + lane; } }
        }
        for i in (chunks * 4)..count { if *ptr.add(i) == max_val { return start + i; } }
        start
    }
}

#[inline]
fn find_max_index(scores: &[f32], start: usize, len: usize) -> usize {
    if len <= start { return start; }
    #[cfg(target_arch = "aarch64")]
    { if len - start >= 4 { return unsafe { neon_find_max_index(scores, start, len) }; } }
    let (mut best_idx, mut best_val) = (start, scores[start]);
    for j in (start + 1)..len { if scores[j] > best_val { best_val = scores[j]; best_idx = j; } }
    best_idx
}

// ---------------------------------------------------------------------------
// TT (512K entries with age replacement)
// ---------------------------------------------------------------------------

const TT_BITS: usize = 19;
const TT_SIZE: usize = 1 << TT_BITS;
const TT_MASK: usize = TT_SIZE - 1;

#[derive(Clone, Copy, Default)]
struct TTEntry {
    key32: u32,
    score: i16,
    depth: u8,
    flag: u8, // 0=exact, 1=upper, 2=lower
    from: u8,
    to: u8,
    path_kind: u8,
    stop_idx: u8,
    special: u8,
    promo: u8,
    age: u8,
}

fn tt_to_move(e: &TTEntry) -> Move {
    Move {
        from: e.from,
        to: e.to,
        path_kind: e.path_kind,
        stop_index: e.stop_idx,
        special: match e.special {
            1 => SpecialMove::Castle,
            2 => SpecialMove::EnPassant,
            3 => SpecialMove::Promotion,
            _ => SpecialMove::None,
        },
        promo_piece: match e.promo {
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

// ---------------------------------------------------------------------------
// LMR table (divisor 1.8 -- conservative, chimera-like)
// ---------------------------------------------------------------------------

struct LmrTable {
    table: [[i32; 256]; 32],
}

static LMR: LazyLock<LmrTable> = LazyLock::new(|| {
    let mut t = LmrTable { table: [[0i32; 256]; 32] };
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
// Blade engine
// ---------------------------------------------------------------------------

struct Blade {
    color: Color,
    tt: Vec<TTEntry>,
    tt_age: u8,
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
    budget_max_time_us: i64,
    t0: Instant,
}

impl Blade {
    fn new() -> Self {
        // Force LMR table init
        let _ = &*LMR;
        Self {
            color: Color::White,
            tt: vec![TTEntry::default(); TT_SIZE],
            tt_age: 0,
            history: [[[0i16; 64]; 64]; 2],
            killers: [[Move::default(); 2]; 64],
            countermove: [[Move::default(); 64]; 64],
            prev_move: Move::default(),
            move_buf: Vec::new(),
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
        }
    }

    fn elapsed_us(&self) -> i64 {
        self.t0.elapsed().as_micros() as i64
    }

    fn check_time(&mut self) -> bool {
        if self.stopped {
            return true;
        }
        if self.budget_max_time_us > 0 && (self.nodes & 255) == 0 {
            if self.elapsed_us() >= self.budget_max_time_us * 9 / 10 {
                self.stopped = true;
            }
        }
        self.stopped
    }

    fn tt_store(&mut self, key: u64, depth: i32, score: i32, flag: u8, m: &Move) {
        let idx = (key as usize) & TT_MASK;
        let k32 = (key >> 32) as u32;
        let e = &self.tt[idx];
        // Replace if: empty, same position, deeper, or older age
        if e.key32 == 0 || e.key32 == k32 || depth as u8 >= e.depth || e.age != self.tt_age {
            let e = &mut self.tt[idx];
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
            e.age = self.tt_age;
        }
    }

    /// Count total pieces on the board (excluding kings)
    fn total_pieces(pos: &Position) -> i32 {
        let mut count = 0i32;
        for sq in 0..64 {
            let p = pos.board[sq];
            if !p.is_empty() && p.piece_type != PieceType::King {
                count += 1;
            }
        }
        count
    }

    // Chimera eval + king-push-awareness (identical to specter_020)
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

        // Track non-king pieces for king-push-awareness
        let mut piece_sq = [[0u8; 16]; 2];
        let mut piece_n = [0usize; 2];

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

            sc_mat += sign * pval(pt);

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
                let mut passed = true;
                let dir: i32 = if c == Color::White { 1 } else { -1 };
                let mut fr = r + dir;
                'passed_check: while fr >= 0 && fr < 8 && passed {
                    for df in -1..=1 {
                        let ff = f + df;
                        if ff < 0 || ff > 7 {
                            continue;
                        }
                        let cs = (fr * 8 + ff) as u8;
                        let cp = pos.board[cs as usize];
                        if cp.piece_type == PieceType::Pawn && cp.color != c {
                            passed = false;
                            break 'passed_check;
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

            // Track non-king pieces for push awareness
            if pt != PieceType::King && piece_n[ci] < 16 {
                piece_sq[ci][piece_n[ci]] = sq;
                piece_n[ci] += 1;
            }
        }

        // Piston bonus
        for ci in 0..2 {
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
        for ci in 0..2 {
            let oci = 1 - ci;
            let sign: i32 = if ci == (stm as usize) { 1 } else { -1 };
            let kr = (king[ci] >> 3) as i32;
            let kf = (king[ci] & 7) as i32;
            for pi in 0..pawn_n[oci] {
                let psq = pawn_sq[oci][pi] as i32;
                let pr = psq >> 3;
                let pf = psq & 7;
                let adv = if oci == 0 { pr } else { 7 - pr };
                if adv < 4 {
                    continue;
                }
                let dist = (kr - pr).abs().max((kf - pf).abs());
                if dist <= 2 {
                    sc_pos -= sign * (20 + adv * 8);
                }
            }
        }

        // King safety (chimera base)
        for ci in 0..2 {
            let c = if ci == 0 { Color::White } else { Color::Black };
            let sign: i32 = if c == stm { 1 } else { -1 };
            let kr = (king[ci] >> 3) as i32;
            let kf = (king[ci] & 7) as i32;
            let shield_dir: i32 = if c == Color::White { 1 } else { -1 };

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

            if bishop_count[ci] >= 2 {
                sc_pos += sign * 30;
            }

            // === KING PUSH AWARENESS ===
            // Bonus for friendly pieces adjacent to own king (push shields)
            let mut adj_friendly = 0i32;
            let mut push_escapes = 0i32;
            for dr in -1..=1i32 {
                for df in -1..=1i32 {
                    if dr == 0 && df == 0 {
                        continue;
                    }
                    let nr = kr + dr;
                    let nf = kf + df;
                    if nr < 0 || nr > 7 || nf < 0 || nf > 7 {
                        continue;
                    }
                    let ns = (nr * 8 + nf) as u8;
                    let np = pos.board[ns as usize];
                    if np.is_empty() {
                        continue;
                    }
                    if np.color == c {
                        adj_friendly += 1;
                        // Check if this piece has an empty square behind it
                        // (forming a push escape route: king pushes friendly, friendly goes to empty)
                        let er = nr + dr;
                        let ef = nf + df;
                        if er >= 0 && er <= 7 && ef >= 0 && ef <= 7 {
                            let es = (er * 8 + ef) as u8;
                            if pos.board[es as usize].is_empty() {
                                push_escapes += 1;
                            }
                        }
                    }
                }
            }
            sc_pos += sign * (adj_friendly * 10);
            sc_pos += sign * (push_escapes * 15);
        }

        // Suppress unused variable warnings
        let _ = piece_sq;
        let _ = piece_n;

        sc_mat + sc_pos
    }

    fn order_moves(&self, pos: &Position, ml: &mut MoveList, ply: usize, ttm: &Move) {
        let mut scores = vec![0.0f32; ml.moves.len()];
        let cm = if self.prev_move.from != 0 || self.prev_move.to != 0 {
            self.countermove[self.prev_move.from as usize][self.prev_move.to as usize]
        } else {
            Move::default()
        };

        for i in 0..ml.moves.len() {
            let m = &ml.moves[i];
            let s: f32;
            if *m == *ttm {
                s = 1e7;
            } else {
                let mut sv: f32 = 0.0;
                if !pos.board[m.to as usize].is_empty() {
                    sv += 100000.0 + pval(pos.board[m.to as usize].piece_type) as f32 * 10.0
                        - pval(pos.board[m.from as usize].piece_type) as f32;
                }
                if m.special == SpecialMove::Promotion {
                    sv += 95000.0 + pval(m.promo_piece) as f32;
                }
                let mpt = pos.board[m.from as usize].piece_type;
                if mpt == PieceType::Pawn {
                    let mc = pos.board[m.from as usize].color;
                    let adv = if mc == Color::White {
                        (m.to >> 3) as i32
                    } else {
                        7 - (m.to >> 3) as i32
                    };
                    if adv >= 5 {
                        sv += 50000.0 + adv as f32 * 5000.0;
                    }
                }
                if ply < 64 {
                    if *m == self.killers[ply][0] {
                        sv += 80000.0;
                    } else if *m == self.killers[ply][1] {
                        sv += 79000.0;
                    }
                }
                if *m == cm {
                    sv += 60000.0;
                }
                sv += self.history[pos.side_to_move as usize][m.from as usize][m.to as usize] as f32;
                s = sv;
            }
            scores[i] = s;
        }

        // Selection sort with NEON-accelerated max finding
        for i in 0..ml.moves.len() {
            let best_idx = find_max_index(&scores, i, ml.moves.len());
            if best_idx != i { ml.moves.swap(i, best_idx); scores.swap(i, best_idx); }
        }
    }

    // search() with is_pv parameter (wraith/tempest-style PV-node awareness)
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
                ttm = tt_to_move(e);
                // Don't allow TT cutoffs in PV nodes
                if e.depth as i32 >= depth && !is_pv {
                    self.tt_hits += 1;
                    if e.flag == 0 {
                        return e.score as i32;
                    }
                    if e.flag == 2 && e.score as i32 >= beta {
                        return e.score as i32;
                    }
                    if e.flag == 1 && e.score as i32 <= alpha {
                        return e.score as i32;
                    }
                }
            }
        }

        if depth <= 0 {
            return self.qsearch(pos, alpha, beta, 0);
        }

        // === NULL MOVE PRUNING ===
        // Conservative: only depth >= 4, R=2, not in check, not root, not PV,
        // and enough pieces to avoid zugzwang
        if !in_check && !is_pv && ply > 0 && depth >= 4 && Self::total_pieces(pos) > 5 {
            let zt = zobrist_tables();
            // Save state
            let saved_ep = pos.ep_square;
            // Flip side to move
            pos.side_to_move = opponent(pos.side_to_move);
            pos.zobrist ^= zt.side_key;
            // Clear ep square from zobrist if present
            if saved_ep < 64 {
                pos.zobrist ^= zt.ep_keys[(saved_ep & 7) as usize];
                pos.ep_square = 64;
            }

            let null_score = -self.search(pos, depth - 1 - 2, -beta, -(beta - 1), ply + 1, false, false);

            // Restore state
            pos.side_to_move = opponent(pos.side_to_move);
            pos.zobrist ^= zt.side_key;
            if saved_ep < 64 {
                pos.zobrist ^= zt.ep_keys[(saved_ep & 7) as usize];
                pos.ep_square = saved_ep;
            }

            if null_score >= beta && !self.stopped {
                self.null_cuts += 1;
                return beta;
            }
        }

        // Reverse futility pruning (conservative: depth <= 4, 100cp/ply)
        // Don't apply in PV nodes
        if !in_check && !is_pv && depth <= 4 && ply > 0 {
            let eval = self.evaluate(pos);
            if eval - depth * 100 >= beta {
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
        if ttm.from == 0 && ttm.to == 0 && depth >= 4 && !self.stopped {
            self.search(pos, depth - 2, alpha, beta, ply, in_check, is_pv);
            let e = &self.tt[idx];
            if e.key32 == k32 {
                ttm = tt_to_move(e);
            }
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

        if ml.moves.is_empty() {
            return if in_check { -99000 + ply } else { 0 };
        }
        self.order_moves(pos, &mut ml, ply as usize, &ttm);

        let saved_prev = self.prev_move;
        let mut best_move = ml.moves[0];
        let mut best_score = -100000i32;
        let mut flag: u8 = 1;

        for i in 0..ml.moves.len() {
            if self.stopped {
                break;
            }
            let m = ml.moves[i];

            // is_tactical BEFORE make_move (bug fix)
            let is_tactical = !pos.board[m.to as usize].is_empty()
                || m.special == SpecialMove::Promotion
                || m.special == SpecialMove::EnPassant;

            // Conservative LMP: depth <= 2, prune at 8 + depth*4 (chimera-like)
            if !is_tactical && !in_check && depth <= 2 && i as i32 >= 8 + depth * 4 {
                continue;
            }

            self.prev_move = m;
            pos.make_move(&m);
            let gives_check = pos.in_check();

            let score;
            let d = depth.min(31) as usize;
            let mi = i.min(255);

            if i >= 3 && depth >= 2 && !is_tactical && !gives_check && !in_check {
                self.lmr_tries += 1;
                let mut r = LMR.table[d][mi].clamp(1, depth - 1);
                let ci = opponent(pos.side_to_move) as usize;
                let hscore = self.history[ci][m.from as usize][m.to as usize] as i32;
                if hscore < -500 {
                    r += 2;
                } else if hscore < -100 {
                    r += 1;
                }
                r = r.clamp(1, depth - 1);

                // LMR: zero-window, not PV
                let s0 = -self.search(pos, depth - 1 - r, -(alpha + 1), -alpha, ply + 1, gives_check, false);
                if s0 > alpha && !self.stopped {
                    self.lmr_re += 1;
                    // Re-search with full window if PV
                    score = -self.search(pos, depth - 1, -beta, -alpha, ply + 1, gives_check, is_pv);
                } else {
                    score = s0;
                }
            } else {
                // Check extension up to depth 4 (chimera conservative)
                let ext = if gives_check && depth <= 4 { 1 } else { 0 };
                if i > 0 && !self.stopped {
                    // Zero-window search for non-first moves
                    let s1 = -self.search(pos, depth - 1 + ext, -(alpha + 1), -alpha, ply + 1, gives_check, false);
                    if s1 > alpha && s1 < beta && !self.stopped {
                        // Re-search with full window, PV status passes through
                        score = -self.search(pos, depth - 1 + ext, -beta, -alpha, ply + 1, gives_check, is_pv);
                    } else {
                        score = s1;
                    }
                } else {
                    // First move at PV node: search with full window, is_pv=true
                    score = -self.search(pos, depth - 1 + ext, -beta, -alpha, ply + 1, gives_check, is_pv);
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
        if stand_pat + 1000 < alpha {
            return alpha;
        }

        let mut buf = std::mem::take(&mut self.move_buf);
        buf.clear();
        generate_legal_moves(pos, &mut buf);

        let mut tactical: Vec<(Move, i32)> = Vec::with_capacity(64);
        for m in &buf {
            let tac = !pos.board[m.to as usize].is_empty()
                || m.special == SpecialMove::Promotion
                || m.special == SpecialMove::EnPassant;
            if !tac {
                continue;
            }
            let mut see = pval(pos.board[m.to as usize].piece_type);
            if m.special == SpecialMove::Promotion {
                see += 800;
            }
            if stand_pat + see + 200 < alpha {
                continue;
            }
            tactical.push((*m, see));
        }
        self.move_buf = buf;

        // Sort tactical moves by SEE descending
        for i in 0..tactical.len() {
            for j in (i + 1)..tactical.len() {
                if tactical[j].1 > tactical[i].1 {
                    tactical.swap(i, j);
                }
            }
        }

        for i in 0..tactical.len() {
            pos.make_move(&tactical[i].0);
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

    fn extract_pv(&mut self, pos: &mut Position, stats: &mut SearchStats) {
        stats.pv.clear();
        let mut seen = [0u64; 32];
        let mut sn = 0usize;
        let mut depth = 0;
        let max_pv = 32;

        loop {
            if depth >= max_pv {
                break;
            }
            let key = pos.zobrist;
            let mut repeated = false;
            for j in 0..sn {
                if seen[j] == key {
                    repeated = true;
                    break;
                }
            }
            if repeated {
                break;
            }
            if sn < 32 {
                seen[sn] = key;
                sn += 1;
            }

            let idx = (key as usize) & TT_MASK;
            let k32 = (key >> 32) as u32;
            let e = &self.tt[idx];
            if e.key32 != k32 {
                break;
            }
            let m = tt_to_move(e);
            if m.from >= 64 || m.to >= 64 {
                break;
            }
            if pos.board[m.from as usize].is_empty() {
                break;
            }
            // Validate against legal moves
            let mut buf = std::mem::take(&mut self.move_buf);
            buf.clear();
            generate_legal_moves(pos, &mut buf);
            let found = buf.iter().any(|lm| *lm == m);
            self.move_buf = buf;
            if !found {
                break;
            }
            stats.pv.push(m);
            pos.make_move(&m);
            depth += 1;
        }
        for _ in 0..depth {
            pos.unmake_move();
        }
    }

    fn dump_diag(&mut self, pos: &mut Position, root: &MoveList, stats: &mut SearchStats) {
        let mut ranked: Vec<(String, i32)> = Vec::new();
        for i in 0..root.moves.len() {
            let m = &root.moves[i];
            pos.make_move(m);
            let key = pos.zobrist;
            let idx = (key as usize) & TT_MASK;
            let k32 = (key >> 32) as u32;
            let e = &self.tt[idx];
            let sc = if e.key32 == k32 {
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
                let p = match m.promo_piece {
                    PieceType::Knight => 'n',
                    PieceType::Bishop => 'b',
                    PieceType::Rook => 'r',
                    PieceType::Queen => 'q',
                    _ => ' ',
                };
                if p != ' ' {
                    uci.push(p);
                }
            }
            ranked.push((uci, sc));
        }
        ranked.sort_by(|a, b| b.1.cmp(&a.1));
        let cap = ranked.len().min(32);

        let mut diag = format!(
            r#"{{"engine":"blade_001","qn":{},"tt":{},"bcut":{},"fcut":{},"nmp":{},"lmr":[{},{}],"top_moves":["#,
            self.qnodes, self.tt_hits, self.beta_cuts, self.first_cuts,
            self.null_cuts, self.lmr_tries, self.lmr_re,
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

impl Engine for Blade {
    fn name(&self) -> &str {
        "blade"
    }

    fn new_game(&mut self, my_color: Color, _game_seed: u64) {
        self.color = my_color;
        self.tt.fill(TTEntry::default());
        self.tt_age = self.tt_age.wrapping_add(1);
        self.history = [[[0i16; 64]; 64]; 2];
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
        // Increment TT age each search
        self.tt_age = self.tt_age.wrapping_add(1);

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

        if root.moves.is_empty() {
            return (Move::default(), stats);
        }
        if root.moves.len() == 1 {
            stats.nodes = 1;
            stats.depth_reached = 0;
            return (root.moves[0], stats);
        }

        let mut best = root.moves[0];
        let mut best_score = -100000i32;

        for depth in 1..=30 {
            if self.stopped {
                break;
            }
            let (mut alpha, mut beta);
            if depth <= 3 || best_score.abs() > 5000 {
                alpha = -100000;
                beta = 100000;
            } else {
                alpha = best_score - 50;
                beta = best_score + 50;
            }

            // === DIRECT BEST-MOVE TRACKING ===
            let mut iter_best;
            let mut iter_best_score;
            let mut aspiration_fail = false;

            loop {
                iter_best = root.moves[0];
                iter_best_score = -100000;

                // Order root moves: put previous iteration's best first
                for i in 0..root.moves.len() {
                    if root.moves[i] == best {
                        if i != 0 {
                            root.moves.swap(0, i);
                        }
                        break;
                    }
                }

                for i in 0..root.moves.len() {
                    if self.stopped {
                        break;
                    }
                    let m = root.moves[i];
                    self.prev_move = m;
                    pos.make_move(&m);
                    let gives_check = pos.in_check();

                    let score;
                    if i > 0 && !self.stopped {
                        // Zero-window for non-first moves, not PV
                        let s1 = -self.search(pos, depth - 1, -(alpha + 1), -alpha, 1, gives_check, false);
                        if s1 > alpha && s1 < beta && !self.stopped {
                            // Re-search with full window, PV
                            score = -self.search(pos, depth - 1, -beta, -alpha, 1, gives_check, true);
                        } else {
                            score = s1;
                        }
                    } else {
                        // First move: full window, PV
                        score = -self.search(pos, depth - 1, -beta, -alpha, 1, gives_check, true);
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

                // Aspiration re-search
                if iter_best_score <= (best_score - 50) && !aspiration_fail {
                    alpha = (iter_best_score - 200).max(-100000);
                    beta = 100000;
                    aspiration_fail = true;
                    continue;
                }
                if iter_best_score >= (best_score + 50)
                    && depth > 3
                    && best_score.abs() <= 5000
                    && !aspiration_fail
                {
                    alpha = -100000;
                    beta = (iter_best_score + 200).min(100000);
                    aspiration_fail = true;
                    continue;
                }
                break;
            }

            if !self.stopped {
                best_score = iter_best_score;
                // Direct assignment -- no TT lookup needed
                best = iter_best;
            }
            stats.depth_reached = depth as u32;
            stats.eval_cp = best_score;
        }

        // Fallback for losing positions
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

        // Robust move validation: re-generate legal moves and verify best is legal
        let mut fresh_legal = Vec::new();
        generate_legal_moves(pos, &mut fresh_legal);
        if !fresh_legal.iter().any(|m| *m == best) {
            best = fresh_legal.first().copied().unwrap_or_default();
        }

        stats.nodes = self.nodes;
        stats.seldepth = self.max_ply;
        stats.time_used_us = self.elapsed_us();
        self.extract_pv(pos, &mut stats);
        self.dump_diag(pos, &root, &mut stats);

        (best, stats)
    }
}

pub fn create() -> Box<dyn Engine> {
    Box::new(Blade::new())
}
