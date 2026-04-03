// abyss: Every last cycle. No mercy.

use std::time::Instant;
use crate::core::types::*;
use crate::core::position::Position;
use crate::core::movegen::generate_legal_moves;
use crate::core::zobrist::zobrist_tables;
use crate::engine::Engine;

// --- Eval LUT: static array, zero runtime init cost ---
// EVAL[pt*64+sq] for white. For black: sq^56.
static EVAL: [i32; 448] = {
    let mut t = [0i32; 448];
    // Pawn
    let p: [i32;64] = [100,100,100,100,100,100,100,100,105,110,110,80,80,110,110,105,105,95,90,100,100,90,95,105,100,100,100,120,120,100,100,100,105,105,110,125,125,110,105,105,110,110,120,130,130,120,110,110,150,150,150,150,150,150,150,150,100,100,100,100,100,100,100,100];
    let n: [i32;64] = [270,280,290,290,290,290,280,270,280,300,320,325,325,320,300,280,290,325,330,335,335,330,325,290,290,320,335,340,340,335,320,290,290,325,335,340,340,335,325,290,290,320,330,335,335,330,320,290,280,300,320,320,320,320,300,280,270,280,290,290,290,290,280,270];
    let b: [i32;64] = [310,320,320,320,320,320,320,310,320,335,330,330,330,330,335,320,320,340,340,340,340,340,340,320,320,330,340,340,340,340,330,320,320,335,335,340,340,335,335,320,320,330,335,340,340,335,330,320,320,330,330,330,330,330,330,320,310,320,320,320,320,320,320,310];
    let r: [i32;64] = [500,500,500,505,505,500,500,500,495,500,500,500,500,500,500,495,495,500,500,500,500,500,500,495,495,500,500,500,500,500,500,495,495,500,500,500,500,500,500,495,495,500,500,500,500,500,500,495,505,510,510,510,510,510,510,505,500,500,500,500,500,500,500,500];
    let q: [i32;64] = [880,890,890,895,895,890,890,880,890,900,900,900,900,900,900,890,890,900,905,905,905,905,900,890,895,900,905,905,905,905,900,895,900,900,905,905,905,905,900,895,890,905,905,905,905,905,900,890,890,900,905,900,900,900,900,890,880,890,890,895,895,890,890,880];
    let k: [i32;64] = [20,30,10,0,0,10,30,20,20,20,0,0,0,0,20,20,-10,-20,-20,-20,-20,-20,-20,-10,-20,-30,-30,-40,-40,-30,-30,-20,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30];
    let mut sq = 0;
    while sq < 64 { t[64+sq]=p[sq]; t[128+sq]=n[sq]; t[192+sq]=b[sq]; t[256+sq]=r[sq]; t[320+sq]=q[sq]; t[384+sq]=k[sq]; sq+=1; }
    t
};

// Piece values for MVV-LVA — static array, no function call overhead
static PV: [i32; 7] = [0, 100, 320, 330, 500, 900, 0];

#[inline(always)]
fn ev(pt: usize, sq: usize) -> i32 { unsafe { *EVAL.get_unchecked(pt * 64 + sq) } }

// LMR table — also const-computable but f64::ln isn't const yet, so LazyLock
static LMR: std::sync::LazyLock<[[i32; 256]; 32]> = std::sync::LazyLock::new(|| {
    let mut t = [[0i32; 256]; 32];
    for d in 0..32 { for m in 0..256 { t[d][m] = if d < 2 || m < 3 { 0 } else { (0.75 + (d as f64).ln() * (m as f64).ln() / 2.0) as i32 }; } }
    t
});

// --- TT: 8-byte u64 slab, 8M entries = 64MB ---
const TT_BITS: usize = 23; const TT_SZ: usize = 1 << TT_BITS; const TT_MASK: usize = TT_SZ - 1;
// Pack: key16(16)|move23(23)|score16(16)|depth7(7)|flag2(2) = 64
#[inline(always)] fn ttp(key: u64, mv: u32, sc: i32, d: i32, f: u32) -> u64 {
    (key & 0xFFFF_0000_0000_0000) | (((mv & 0x7FFFFF) as u64) << 25) | (((sc as u16 as u64)) << 9) | (((d as u64) & 0x7F) << 2) | ((f as u64) & 3)
}
#[inline(always)] fn ttk(e: u64) -> u64 { e & 0xFFFF_0000_0000_0000 }
#[inline(always)] fn ttm(e: u64) -> u32 { ((e >> 25) & 0x7FFFFF) as u32 }
#[inline(always)] fn tts(e: u64) -> i32 { ((e >> 9) & 0xFFFF) as i16 as i32 }
#[inline(always)] fn ttd(e: u64) -> i32 { ((e >> 2) & 0x7F) as i32 }
#[inline(always)] fn ttf(e: u64) -> u32 { (e & 3) as u32 }

// Move pack: no function overhead, just bit ops
#[inline(always)] fn mp(m: &Move) -> u32 {
    (m.from as u32)|((m.to as u32)<<6)|((m.path_kind as u32)<<12)|((m.stop_index as u32)<<14)|((m.special as u32)<<18)|((m.promo_piece as u32)<<20)
}
static SP: [SpecialMove; 4] = [SpecialMove::None, SpecialMove::Castle, SpecialMove::EnPassant, SpecialMove::Promotion];
static PP: [PieceType; 8] = [PieceType::None, PieceType::Pawn, PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen, PieceType::King, PieceType::None];
#[inline(always)] fn mu(p: u32) -> Move {
    Move { from:(p&0x3F)as u8, to:((p>>6)&0x3F)as u8, path_kind:((p>>12)&0x3)as u8, stop_index:((p>>14)&0xF)as u8,
        special: SP[((p>>18)&3) as usize], promo_piece: PP[((p>>20)&7) as usize] }
}

// --- ML: no zeroing of arrays, just set n=0 ---
const MLC: usize = 256;
struct ML { m: [Move; MLC], s: [i32; MLC], n: usize }
impl ML {
    #[inline(always)] fn empty() -> Self { unsafe { let mut x: Self = std::mem::MaybeUninit::uninit().assume_init(); x.n = 0; x } }
    #[inline(always)] fn push(&mut self, mv: Move) { unsafe { *self.m.get_unchecked_mut(self.n) = mv; } self.n += 1; }
    #[inline(always)] fn pick(&mut self, i: usize) {
        let mut bi = i; let mut bv = unsafe { *self.s.get_unchecked(i) };
        let mut j = i + 1;
        while j < self.n {
            let v = unsafe { *self.s.get_unchecked(j) };
            // branchless max: bi = if v > bv { j } else { bi }
            let gt = ((bv - v) >> 31) as usize & 1; // 1 if v > bv
            bi = bi + gt * (j - bi); // bi = bi + gt*(j-bi) = if gt { j } else { bi }
            bv = bv + (gt as i32) * (v - bv);
            j += 1;
        }
        if bi != i { self.m.swap(i, bi); self.s.swap(i, bi); }
    }
}

pub struct AbyssEngine {
    tt: Vec<u64>,
    nodes: u64, sd: u32, stopped: bool,
    budget: i64, t0: Instant, buf: Vec<Move>,
}

impl AbyssEngine {
    fn new() -> Self {
        let _ = &*LMR;
        Self { tt: vec![0u64; TT_SZ], nodes: 0, sd: 0, stopped: false, budget: 0, t0: Instant::now(), buf: Vec::with_capacity(256) }
    }

    #[inline(always)]
    fn ck(&mut self) -> bool {
        if self.stopped { return true; }
        if self.budget > 0 && (self.nodes & 4095) == 0 && self.t0.elapsed().as_micros() as i64 >= self.budget - (self.budget >> 3) { self.stopped = true; }
        self.stopped
    }

    // Eval: branchless accumulation with compile-time known LUT
    #[inline(always)]
    fn eval(&self, pos: &Position) -> i32 {
        let stm = pos.side_to_move as u8;
        let brd = &pos.board;
        let mut s = 0i32;
        let mut i = 0usize;
        while i < 64 {
            // Read piece as 2 raw bytes to avoid struct field access overhead
            let p = unsafe { *brd.get_unchecked(i) };
            let pt = p.piece_type as usize;
            // Branchless skip empties: multiply by (pt != 0) as mask
            // If pt == 0, val = EVAL[0..63] which are all 0, so safe to accumulate garbage — it's 0
            let sq = i ^ ((p.color as usize) * (56)); // white: i, black: i^56 — branchless
            let val = unsafe { *EVAL.get_unchecked(pt * 64 + sq) };
            // Branchless sign: same color as stm → add, different → subtract
            let mask = ((p.color as u8 ^ stm) as i32).wrapping_neg(); // 0 if same, -1 if diff
            // But we also need to zero out empties. pt==0 → val==0 so this is automatically handled.
            s += (val ^ mask) - mask;
            i += 1;
        }
        s
    }

    fn score(&self, pos: &Position, ml: &mut ML, tmv: u32) {
        let stm = pos.side_to_move;
        let brd = &pos.board;
        let mut i = 0;
        while i < ml.n {
            let m = unsafe { *ml.m.get_unchecked(i) };
            let packed = mp(&m);
            if packed == tmv && tmv != 0 { unsafe { *ml.s.get_unchecked_mut(i) = 100_000_000; } i += 1; continue; }
            let tp = unsafe { *brd.get_unchecked(m.to as usize) };
            let v: i32;
            if tp.piece_type != PieceType::None {
                if tp.color != stm {
                    // Capture: MVV-LVA
                    v = 2_000_000 + unsafe { *PV.get_unchecked(tp.piece_type as usize) } * 10
                        - unsafe { *PV.get_unchecked(brd.get_unchecked(m.from as usize).piece_type as usize) };
                } else {
                    // Push: by pushed piece value
                    v = 1_000_000 + unsafe { *PV.get_unchecked(tp.piece_type as usize) } * 10;
                }
            } else if m.special == SpecialMove::Promotion {
                v = 1_500_000 + unsafe { *PV.get_unchecked(m.promo_piece as usize) };
            } else {
                // Quiet: PST delta
                let fp = unsafe { *brd.get_unchecked(m.from as usize) };
                let pt = fp.piece_type as usize;
                let csq = fp.color as usize * 56;
                v = unsafe { *EVAL.get_unchecked(pt * 64 + (m.to as usize ^ csq)) }
                  - unsafe { *EVAL.get_unchecked(pt * 64 + (m.from as usize ^ csq)) };
            }
            unsafe { *ml.s.get_unchecked_mut(i) = v; }
            i += 1;
        }
    }

    fn qs(&mut self, pos: &mut Position, mut a: i32, b: i32, qd: u32) -> i32 {
        if self.ck() { return 0; }
        self.nodes += 1;
        if qd > self.sd { self.sd = qd; }
        if qd >= 10 { return self.eval(pos); }
        let sp = self.eval(pos);
        if sp >= b { return sp; }
        if sp > a { a = sp; }

        let mut buf = std::mem::take(&mut self.buf);
        buf.clear(); generate_legal_moves(pos, &mut buf);
        let stm = pos.side_to_move;
        let mut ml = ML::empty();
        let mut j = 0;
        while j < buf.len() {
            let mv = unsafe { *buf.get_unchecked(j) };
            let tp = unsafe { *pos.board.get_unchecked(mv.to as usize) };
            if (tp.piece_type != PieceType::None && tp.color != stm) || mv.special == SpecialMove::Promotion || mv.special == SpecialMove::EnPassant {
                ml.push(mv);
            }
            j += 1;
        }
        self.buf = buf;
        self.score(pos, &mut ml, 0);
        let mut i = 0;
        while i < ml.n {
            ml.pick(i);
            pos.make_move(unsafe { ml.m.get_unchecked(i) });
            let s = -self.qs(pos, -b, -a, qd + 1);
            pos.unmake_move();
            if s >= b { return s; }
            if s > a { a = s; }
            i += 1;
        }
        a
    }

    fn ab(&mut self, pos: &mut Position, depth: i32, mut a: i32, mut b: i32, ply: i32, ic: bool, pv: bool) -> i32 {
        if self.ck() { return 0; }
        self.nodes += 1;
        if ply as u32 > self.sd { self.sd = ply as u32; }
        if ply >= 127 { return self.eval(pos); }
        if depth <= 0 { return self.qs(pos, a, b, ply as u32); }
        let d = depth + (ic && depth <= 4) as i32; // branchless check extension

        // TT probe
        let key = pos.zobrist;
        let idx = (key as usize) & TT_MASK;
        let e = unsafe { *self.tt.get_unchecked(idx) };
        let khi = key & 0xFFFF_0000_0000_0000;
        let mut tmv = 0u32;
        if ttk(e) == khi {
            tmv = ttm(e);
            if ttd(e) >= d && !pv {
                let s = tts(e);
                let ts = s - ((s > 90000) as i32 - (s < -90000) as i32) * ply; // branchless mate adjust
                let f = ttf(e);
                if f == 0 { return ts; }
                if f == 1 && ts <= a { return a; }
                if f == 2 && ts >= b { return b; }
            }
        }

        let ev = self.eval(pos);
        let sd = d - (tmv == 0 && d >= 4) as i32; // branchless IIR

        // RFP
        if !ic && !pv && ply > 0 && d <= 6 && ev - d * 100 >= b { return ev; }

        // NMP
        if !pv && !ic && ply > 0 && d >= 3 && ev >= b {
            let r = 2 + (d >> 2);
            let z = zobrist_tables();
            let (se, ss, sz) = (pos.ep_square, pos.side_to_move, pos.zobrist);
            pos.side_to_move = opponent(ss); pos.zobrist ^= z.side_key;
            if se < 64 { pos.zobrist ^= z.ep_keys[file_of(se) as usize]; pos.ep_square = 64; }
            let ns = -self.ab(pos, d-r-1, -b, -b+1, ply+1, false, false);
            pos.side_to_move = ss; pos.ep_square = se; pos.zobrist = sz;
            if ns >= b && !self.stopped { return b; }
        }

        // Movegen
        let mut buf = std::mem::take(&mut self.buf);
        buf.clear(); generate_legal_moves(pos, &mut buf);
        let mut ml = ML::empty();
        let mut j = 0;
        while j < buf.len() { ml.push(unsafe { *buf.get_unchecked(j) }); j += 1; }
        self.buf = buf;
        if ml.n == 0 { return if ic { -99000 + ply } else { 0 }; }
        self.score(pos, &mut ml, tmv);

        let mut bm = unsafe { *ml.m.get_unchecked(0) };
        let mut bs = -100000i32;
        let oa = a;
        let stm = pos.side_to_move;
        let mut i = 0;
        while i < ml.n {
            ml.pick(i);
            let m = unsafe { *ml.m.get_unchecked(i) };
            let tp = unsafe { *pos.board.get_unchecked(m.to as usize) };
            let tac = tp.piece_type != PieceType::None || m.special as u8 >= SpecialMove::EnPassant as u8;

            if !tac && !ic && d <= 2 && i as i32 >= 8 + d * 4 { i += 1; continue; }

            pos.make_move(&m);
            let gc = pos.in_check();
            let sc: i32;
            if i >= 3 && sd >= 2 && !tac && !gc && !ic {
                let r = unsafe { *LMR.get_unchecked(sd.min(31) as usize).get_unchecked(i.min(255)) };
                let s0 = -self.ab(pos, sd-1-r, -(a+1), -a, ply+1, gc, false);
                if s0 > a && r > 0 { sc = -self.ab(pos, sd-1, -b, -a, ply+1, gc, pv); }
                else { sc = s0; }
            } else {
                let ext = (gc && sd <= 4) as i32; // branchless
                if i == 0 {
                    sc = -self.ab(pos, sd-1+ext, -b, -a, ply+1, gc, pv);
                } else {
                    let s1 = -self.ab(pos, sd-1+ext, -(a+1), -a, ply+1, gc, false);
                    if s1 > a && s1 < b && !self.stopped { sc = -self.ab(pos, sd-1+ext, -b, -a, ply+1, gc, pv); }
                    else { sc = s1; }
                }
            }
            pos.unmake_move();
            if self.stopped { break; }
            if sc > bs { bs = sc; bm = m; }
            if sc > a { a = sc; }
            if a >= b { break; }
            i += 1;
        }

        if !self.stopped {
            let f = ((bs > oa) as u32) * (1 + (bs >= b) as u32); // branchless: 0=alpha, 1=exact(impossible here—corrected below), 2=beta
            // Actually: bs <= oa → 1(alpha), bs >= b → 2(beta), else → 0(exact)
            let f = if bs <= oa { 1u32 } else if bs >= b { 2 } else { 0 };
            let ts = bs + ((bs > 90000) as i32 - (bs < -90000) as i32) * ply; // branchless mate-to-tt
            unsafe { *self.tt.get_unchecked_mut(idx) = ttp(key, mp(&bm), ts, d, f); }
        }
        bs
    }
}

impl Engine for AbyssEngine {
    fn name(&self) -> &str { "abyss" }
    fn new_game(&mut self, _: Color, _: u64) {
        // Faster than fill(0) — memset via slice
        unsafe { std::ptr::write_bytes(self.tt.as_mut_ptr(), 0, TT_SZ); }
    }
    fn choose_move(&mut self, pos: &mut Position, budget: &SearchBudget) -> (Move, SearchStats) {
        self.t0 = Instant::now(); self.budget = budget.max_time_us;
        self.nodes = 0; self.sd = 0; self.stopped = false;

        let mut buf = std::mem::take(&mut self.buf);
        buf.clear(); generate_legal_moves(pos, &mut buf);
        let mut root = ML::empty();
        let mut j = 0;
        while j < buf.len() { root.push(unsafe { *buf.get_unchecked(j) }); j += 1; }
        self.buf = buf;

        let mut stats = SearchStats::default();
        if root.n == 0 { return (Move::default(), stats); }
        if root.n == 1 { stats.nodes = 1; return (root.m[0], stats); }

        let mut bm = root.m[0]; let mut bs = 0i32;

        for depth in 1..=64i32 {
            if self.stopped { break; }
            self.score(pos, &mut root, mp(&bm));
            let mut cb = root.m[0]; let mut cs = -100000i32;
            let mut i = 0;
            while i < root.n {
                if self.stopped { break; }
                root.pick(i);
                let m = root.m[i];
                pos.make_move(&m);
                let gc = pos.in_check();
                let sc;
                if i == 0 {
                    sc = -self.ab(pos, depth-1, -100000, 100000, 1, gc, true);
                } else {
                    let s = -self.ab(pos, depth-1, -(cs+1), -cs, 1, gc, false);
                    if s > cs && !self.stopped { sc = -self.ab(pos, depth-1, -100000, -cs, 1, gc, true); }
                    else { sc = s; }
                }
                pos.unmake_move();
                if sc > cs { cs = sc; cb = m; }
                i += 1;
            }
            if !self.stopped { bm = cb; bs = cs; stats.depth_reached = depth as u32; stats.eval_cp = bs; }
        }

        let mut fresh = Vec::new(); generate_legal_moves(pos, &mut fresh);
        if !fresh.contains(&bm) { bm = fresh.first().copied().unwrap_or_default(); }
        stats.nodes = self.nodes; stats.seldepth = self.sd;
        stats.time_used_us = self.t0.elapsed().as_micros() as i64;

        let mut cp = pos.clone();
        for _ in 0..32 {
            let idx = (cp.zobrist as usize) & TT_MASK;
            let e = unsafe { *self.tt.get_unchecked(idx) };
            if ttk(e) != cp.zobrist & 0xFFFF_0000_0000_0000 { break; }
            let m = mu(ttm(e));
            if m.from == 0 && m.to == 0 { break; }
            let mut leg = Vec::new(); generate_legal_moves(&mut cp, &mut leg);
            if !leg.contains(&m) { break; }
            stats.pv.push(m); cp.make_move(&m);
        }
        (bm, stats)
    }
}

pub fn create() -> Box<dyn Engine> { Box::new(AbyssEngine::new()) }
