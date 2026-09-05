//! Compile moves into reusable board transactions. The authoritative push
//! resolver is shared; search never resolves the same selected move again.
use super::eval::{PHASE, piece_score};
use super::network::{Accumulator, Network};
use crate::core::position::Position;
use crate::core::push::{PushPlan, resolve_knight_push, resolve_push};
use crate::core::types::*;
use crate::core::zobrist::zobrist_tables;
use std::sync::Arc;

pub const DIRS: [(i32, i32); 8] = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
];
pub const KNIGHTS: [(i32, i32); 8] = [
    (2, 1),
    (2, -1),
    (-2, 1),
    (-2, -1),
    (1, 2),
    (1, -2),
    (-1, 2),
    (-1, -2),
];

pub fn step(sq: u8, dr: i32, df: i32) -> Option<u8> {
    let (r, f) = (rank_of(sq) + dr, file_of(sq) + df);
    valid_rf(r, f).then(|| make_square(r, f))
}

pub fn pack(m: Move) -> u32 {
    u32::from(m.from)
        | (u32::from(m.to) << 6)
        | (u32::from(m.path_kind) << 12)
        | (u32::from(m.stop_index) << 14)
        | ((m.special as u32) << 18)
        | ((m.promo_piece as u32) << 20)
}

#[derive(Clone)]
pub struct Action {
    pub mv: Move,
    pub plan: Option<PushPlan>,
    pub capture: PieceType,
    pub push: bool,
    pub king_push: bool,
    pub order: i32,
}

impl Action {
    pub fn tactical(&self) -> bool {
        self.capture != PieceType::None || self.mv.special == SpecialMove::Promotion
    }
}

pub struct Board {
    pub model: Arc<Network>,
    pub pos: Position,
    pub men: [[u64; 7]; 2],
    pub occupied: [u64; 2],
    pub mg: [i32; 2],
    pub eg: [i32; 2],
    pub phase: i32,
    pub net: Accumulator,
    pub material: [i32; 2],
}

pub struct Snapshot {
    board: [Piece; 64],
    kings: [u8; 2],
    key: u64,
    castle: u8,
    ep: u8,
    half: u16,
    full: u16,
    men: [[u64; 7]; 2],
    occupied: [u64; 2],
    mg: [i32; 2],
    eg: [i32; 2],
    phase: i32,
    net: Accumulator,
    material: [i32; 2],
}

impl Board {
    pub fn new(pos: &Position) -> Self {
        Self::with_model(pos, Network::embedded())
    }

    pub fn with_model(pos: &Position, model: Arc<Network>) -> Self {
        let mut view = Position::empty();
        view.board = pos.board;
        view.side_to_move = pos.side_to_move;
        view.castling_rights = pos.castling_rights;
        view.ep_square = pos.ep_square;
        view.halfmove_clock = pos.halfmove_clock;
        view.fullmove_number = pos.fullmove_number;
        view.king_sq = pos.king_sq;
        view.zobrist = pos.zobrist;
        let mut b = Self {
            pos: view,
            men: [[0; 7]; 2],
            occupied: [0; 2],
            mg: [0; 2],
            eg: [0; 2],
            phase: 0,
            net: Accumulator::new(&model),
            material: [0; 2],
            model,
        };
        for sq in 0..64 {
            let p = b.pos.board[sq];
            if !p.is_empty() {
                let c = p.color as usize;
                b.men[c][p.piece_type as usize] |= 1 << sq;
                b.occupied[c] |= 1 << sq;
                let (mg, eg) = piece_score(p, sq as u8);
                b.mg[c] += mg;
                b.eg[c] += eg;
                b.phase += PHASE[p.piece_type as usize];
                b.net.update(p, sq as u8, 1, &b.model);
                b.material[c] += super::eval::VALUE[p.piece_type as usize];
            }
        }
        b
    }

    pub fn checked(&self, color: Color) -> bool {
        let sq = self.pos.king_sq[color as usize];
        let them = opponent(color);
        let ti = them as usize;
        let r = rank_of(sq) + if them == Color::White { -1 } else { 1 };
        for df in [-1, 1] {
            let f = file_of(sq) + df;
            if valid_rf(r, f) && self.men[ti][1] & (1 << make_square(r, f)) != 0 {
                return true;
            }
        }
        for (dr, df) in KNIGHTS {
            if let Some(from) = step(sq, dr, df)
                && self.men[ti][2] & (1 << from) != 0
                && (resolve_knight_push(&self.pos, from, sq, true)
                    .is_some_and(|p| p.captured().is_some())
                    || resolve_knight_push(&self.pos, from, sq, false)
                        .is_some_and(|p| p.captured().is_some()))
            {
                return true;
            }
        }
        for (i, (dr, df)) in DIRS.into_iter().enumerate() {
            let mut at = sq;
            let mut distance = 0;
            while let Some(next) = step(at, dr, df) {
                at = next;
                distance += 1;
                let p = self.pos.board[at as usize];
                if p.is_empty() {
                    continue;
                }
                if p.color == them
                    && (p.piece_type == PieceType::Queen
                        || (i < 4 && p.piece_type == PieceType::Rook)
                        || (i >= 4 && p.piece_type == PieceType::Bishop)
                        || (distance == 1 && p.piece_type == PieceType::King))
                {
                    return true;
                }
                break;
            }
        }
        false
    }

    /// Cheap stalemate witness before a static cutoff. Usually one king move
    /// suffices; if the king is boxed in, fall back to the complete move set.
    pub fn has_legal_move(&mut self) -> bool {
        let us = self.pos.side_to_move;
        let from = self.pos.king_sq[us as usize];
        for (dr, df) in DIRS {
            let Some(to) = step(from, dr, df) else {
                continue;
            };
            let Some(plan) = resolve_push(&self.pos, from, to, dr, df) else {
                continue;
            };
            let capture = plan
                .captured()
                .map_or(PieceType::None, |sq| self.pos.board[sq as usize].piece_type);
            let promotion = plan.displacements().iter().any(|&(f, t)| {
                self.pos.board[f as usize].piece_type == PieceType::Pawn
                    && rank_of(t) == if us == Color::White { 7 } else { 0 }
            });
            let mv = Move {
                from,
                to,
                special: if promotion {
                    SpecialMove::Promotion
                } else {
                    SpecialMove::None
                },
                promo_piece: if promotion {
                    PieceType::Queen
                } else {
                    PieceType::None
                },
                ..Move::default()
            };
            let a = Action {
                mv,
                plan: Some(plan),
                capture,
                push: false,
                king_push: false,
                order: 0,
            };
            let undo = self.make(&a);
            let legal = !self.checked(us);
            self.unmake(undo);
            if legal {
                return true;
            }
        }
        let mut actions = Vec::new();
        self.generate(&mut actions);
        for a in &actions {
            let undo = self.make(a);
            let legal = !self.checked(us);
            self.unmake(undo);
            if legal {
                return true;
            }
        }
        false
    }

    fn append(&self, out: &mut Vec<Action>, mut mv: Move, plan: Option<PushPlan>) {
        let us = self.pos.side_to_move;
        let capture = if mv.special == SpecialMove::EnPassant {
            PieceType::Pawn
        } else if let Some(sq) = plan.as_ref().and_then(PushPlan::captured) {
            self.pos.board[sq as usize].piece_type
        } else {
            PieceType::None
        };
        let disps = plan.as_ref().map_or(&[][..], PushPlan::displacements);
        let push = disps.len() > 1;
        let king_push = disps.iter().any(|&(f, _)| {
            f != mv.from && self.pos.board[f as usize].piece_type == PieceType::King
        });
        let promotion = disps.iter().any(|&(f, t)| {
            self.pos.board[f as usize].piece_type == PieceType::Pawn
                && rank_of(t) == if us == Color::White { 7 } else { 0 }
        });
        if promotion {
            mv.special = SpecialMove::Promotion;
            for pt in [
                PieceType::Queen,
                PieceType::Knight,
                PieceType::Rook,
                PieceType::Bishop,
            ] {
                mv.promo_piece = pt;
                out.push(Action {
                    mv,
                    plan: plan.clone(),
                    capture,
                    push,
                    king_push,
                    order: 0,
                });
            }
        } else {
            out.push(Action {
                mv,
                plan,
                capture,
                push,
                king_push,
                order: 0,
            });
        }
    }

    pub fn generate(&self, out: &mut Vec<Action>) {
        out.clear();
        let us = self.pos.side_to_move;
        let mut remaining = self.occupied[us as usize];
        while remaining != 0 {
            let from = remaining.trailing_zeros() as u8;
            remaining &= remaining - 1;
            let pt = self.pos.board[from as usize].piece_type;
            match pt {
                PieceType::Knight => {
                    for (dr, df) in KNIGHTS {
                        let Some(to) = step(from, dr, df) else {
                            continue;
                        };
                        let first = resolve_knight_push(&self.pos, from, to, true);
                        if let Some(p) = &first {
                            self.append(
                                out,
                                Move {
                                    from,
                                    to,
                                    path_kind: 1,
                                    ..Move::default()
                                },
                                Some(p.clone()),
                            );
                        }
                        if let Some(p) = resolve_knight_push(&self.pos, from, to, false)
                            && first.as_ref() != Some(&p)
                        {
                            self.append(
                                out,
                                Move {
                                    from,
                                    to,
                                    path_kind: 2,
                                    ..Move::default()
                                },
                                Some(p),
                            );
                        }
                    }
                }
                PieceType::Pawn => {
                    let dr = if us == Color::White { 1 } else { -1 };
                    for distance in 1..=2 {
                        if distance == 2 && rank_of(from) != if us == Color::White { 1 } else { 6 }
                        {
                            break;
                        }
                        let Some(to) = step(from, dr * distance, 0) else {
                            continue;
                        };
                        if let Some(p) = resolve_push(&self.pos, from, to, dr, 0)
                            .filter(|p| p.captured().is_none())
                        {
                            self.append(
                                out,
                                Move {
                                    from,
                                    to,
                                    stop_index: (distance - 1) as u8,
                                    ..Move::default()
                                },
                                Some(p),
                            );
                        }
                    }
                    for df in [-1, 1] {
                        let Some(to) = step(from, dr, df) else {
                            continue;
                        };
                        if self.pos.board[to as usize].is_color(opponent(us)) {
                            self.append(
                                out,
                                Move {
                                    from,
                                    to,
                                    ..Move::default()
                                },
                                Some(PushPlan::single(from, to, Some(to))),
                            );
                        }
                        if to == self.pos.ep_square {
                            self.append(
                                out,
                                Move {
                                    from,
                                    to,
                                    special: SpecialMove::EnPassant,
                                    ..Move::default()
                                },
                                None,
                            );
                        }
                    }
                }
                _ => {
                    for (i, (dr, df)) in DIRS.into_iter().enumerate() {
                        if (pt == PieceType::Bishop && i < 4) || (pt == PieceType::Rook && i >= 4) {
                            continue;
                        }
                        let mut at = from;
                        for stop in 0..if pt == PieceType::King { 1 } else { 7 } {
                            let Some(to) = step(at, dr, df) else {
                                break;
                            };
                            at = to;
                            let Some(p) = resolve_push(&self.pos, from, to, dr, df) else {
                                break;
                            };
                            let cap = p.captured().is_some();
                            // Match the core's king-to-empty fast path: stop is zero.
                            self.append(
                                out,
                                Move {
                                    from,
                                    to,
                                    stop_index: stop,
                                    ..Move::default()
                                },
                                Some(p),
                            );
                            if cap {
                                break;
                            }
                        }
                    }
                    if pt == PieceType::King {
                        let base = if us == Color::White { 0 } else { 56 };
                        if from != base + 4 {
                            continue;
                        }
                        for (flag, to, spaces, transit) in [
                            (1, base + 6, vec![base + 5, base + 6], base + 5),
                            (2, base + 2, vec![base + 1, base + 2, base + 3], base + 3),
                        ] {
                            let flag = flag << (us as u8 * 2);
                            if self.pos.castling_rights & flag != 0
                                && spaces
                                    .iter()
                                    .all(|&s| self.pos.board[s as usize].is_empty())
                                && !self.pos.is_attacked_by(from, opponent(us))
                                && !self.pos.is_attacked_by(transit, opponent(us))
                                && !self.pos.is_attacked_by(to, opponent(us))
                            {
                                self.append(
                                    out,
                                    Move {
                                        from,
                                        to,
                                        special: SpecialMove::Castle,
                                        ..Move::default()
                                    },
                                    None,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn make(&mut self, a: &Action) -> Snapshot {
        let p = &mut self.pos;
        let old = Snapshot {
            board: p.board,
            kings: p.king_sq,
            key: p.zobrist,
            castle: p.castling_rights,
            ep: p.ep_square,
            half: p.halfmove_clock,
            full: p.fullmove_number,
            men: self.men,
            occupied: self.occupied,
            mg: self.mg,
            eg: self.eg,
            phase: self.phase,
            net: self.net,
            material: self.material,
        };
        let m = a.mv;
        let us = p.side_to_move;
        let mut touched = (1u64 << m.from) | (1u64 << m.to);
        if let Some(plan) = &a.plan {
            plan.apply(&mut p.board);
            for &(f, t) in plan.displacements() {
                touched |= (1 << f) | (1 << t);
            }
            if let Some(sq) = plan.captured() {
                touched |= 1 << sq;
            }
            if m.special == SpecialMove::Promotion {
                for &(_, t) in plan.displacements() {
                    if p.board[t as usize].piece_type == PieceType::Pawn
                        && rank_of(t) == if us == Color::White { 7 } else { 0 }
                    {
                        p.board[t as usize].piece_type = m.promo_piece;
                        break;
                    }
                }
            }
        } else if m.special == SpecialMove::Castle {
            let base = if us == Color::White { 0 } else { 56 };
            let (rf, rt) = if file_of(m.to) == 6 {
                (base + 7, base + 5)
            } else {
                (base, base + 3)
            };
            p.board[m.to as usize] = p.board[m.from as usize];
            p.board[m.from as usize] = Piece::default();
            p.board[rt] = p.board[rf];
            p.board[rf] = Piece::default();
            touched |= (1 << rt) | (1 << rf);
        } else {
            let cap = make_square(rank_of(m.from), file_of(m.to));
            p.board[m.to as usize] = p.board[m.from as usize];
            p.board[m.from as usize] = Piece::default();
            p.board[cap as usize] = Piece::default();
            touched |= 1 << cap;
        }
        let z = zobrist_tables();
        if p.ep_square < 64 {
            p.zobrist ^= z.ep_keys[file_of(p.ep_square) as usize];
        }
        p.ep_square = 64;
        p.halfmove_clock = if old.board[m.from as usize].piece_type == PieceType::Pawn
            || a.capture != PieceType::None
        {
            0
        } else {
            p.halfmove_clock + 1
        };
        if old.board[m.from as usize].piece_type == PieceType::Pawn
            && (rank_of(m.to) - rank_of(m.from)).abs() == 2
        {
            p.ep_square = make_square((rank_of(m.from) + rank_of(m.to)) / 2, file_of(m.from));
            p.zobrist ^= z.ep_keys[file_of(p.ep_square) as usize];
        }
        for (sq, mask) in [(4, 3), (60, 12), (0, 2), (7, 1), (56, 8), (63, 4)] {
            if m.from == sq || m.to == sq {
                p.castling_rights &= !mask;
            }
        }
        p.zobrist ^= z.castling_keys[old.castle as usize]
            ^ z.castling_keys[p.castling_rights as usize]
            ^ z.side_key;
        while touched != 0 {
            let sq = touched.trailing_zeros() as usize;
            touched &= touched - 1;
            let before = old.board[sq];
            let after = p.board[sq];
            if before == after {
                continue;
            }
            if !before.is_empty() {
                let c = before.color as usize;
                let pt = before.piece_type as usize;
                p.zobrist ^= z.piece_keys[c][pt][sq];
                self.men[c][pt] &= !(1 << sq);
                self.occupied[c] &= !(1 << sq);
                let (mg, eg) = piece_score(before, sq as u8);
                self.mg[c] -= mg;
                self.eg[c] -= eg;
                self.phase -= PHASE[pt];
                self.net.update(before, sq as u8, -1, &self.model);
                self.material[c] -= super::eval::VALUE[pt];
            }
            if !after.is_empty() {
                let c = after.color as usize;
                let pt = after.piece_type as usize;
                p.zobrist ^= z.piece_keys[c][pt][sq];
                self.men[c][pt] |= 1 << sq;
                self.occupied[c] |= 1 << sq;
                let (mg, eg) = piece_score(after, sq as u8);
                self.mg[c] += mg;
                self.eg[c] += eg;
                self.phase += PHASE[pt];
                self.net.update(after, sq as u8, 1, &self.model);
                self.material[c] += super::eval::VALUE[pt];
                if after.piece_type == PieceType::King {
                    p.king_sq[c] = sq as u8;
                }
            }
        }
        p.side_to_move = opponent(us);
        if us == Color::Black {
            p.fullmove_number += 1;
        }
        old
    }

    pub fn unmake(&mut self, s: Snapshot) {
        self.pos.board = s.board;
        self.pos.king_sq = s.kings;
        self.pos.zobrist = s.key;
        self.pos.castling_rights = s.castle;
        self.pos.ep_square = s.ep;
        self.pos.halfmove_clock = s.half;
        self.pos.fullmove_number = s.full;
        self.pos.side_to_move = opponent(self.pos.side_to_move);
        self.men = s.men;
        self.occupied = s.occupied;
        self.mg = s.mg;
        self.eg = s.eg;
        self.phase = s.phase;
        self.net = s.net;
        self.material = s.material;
    }
}
