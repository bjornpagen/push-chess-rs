use super::push::{resolve_knight_push, resolve_push};
/// Position representation for Push Chess, ported from C++ `core/position.h` + `core/position.cc`.
use super::types::*;
use super::zobrist::zobrist_tables;
use arrayvec::ArrayVec;

// ---------------------------------------------------------------------------
// UndoInfo
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct UndoInfo {
    pub mv: Move,
    changed: ArrayVec<(Square, Piece), 32>,
    pub castling_rights: u8,
    pub ep_square: Square,
    pub halfmove_clock: u16,
    pub zobrist: u64,
    pub king_sq: [Square; 2],
}

impl Default for UndoInfo {
    fn default() -> Self {
        Self {
            mv: Move::default(),
            changed: ArrayVec::new(),
            castling_rights: 0,
            ep_square: 64,
            halfmove_clock: 0,
            zobrist: 0,
            king_sq: [4, 60],
        }
    }
}

impl UndoInfo {
    pub fn add_changed(&mut self, sq: Square, old_piece: Piece) {
        if !self.changed.iter().any(|&(changed_sq, _)| changed_sq == sq) {
            self.changed.push((sq, old_piece));
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn char_to_piece_type(c: char) -> PieceType {
    match c {
        'p' | 'P' => PieceType::Pawn,
        'n' | 'N' => PieceType::Knight,
        'b' | 'B' => PieceType::Bishop,
        'r' | 'R' => PieceType::Rook,
        'q' | 'Q' => PieceType::Queen,
        'k' | 'K' => PieceType::King,
        _ => PieceType::None,
    }
}

fn piece_to_char(p: Piece) -> char {
    if p.is_empty() {
        return '.';
    }
    const W: &[u8] = b".PNBRQK";
    const B: &[u8] = b".pnbrqk";
    let idx = p.piece_type as usize;
    if p.color as u8 == Color::White as u8 {
        W[idx] as char
    } else {
        B[idx] as char
    }
}

// ---------------------------------------------------------------------------
// Position
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Position {
    pub board: [Piece; 64],
    pub side_to_move: Color,
    pub castling_rights: u8,
    pub ep_square: Square, // 64 = none
    pub halfmove_clock: u16,
    pub fullmove_number: u16,
    pub king_sq: [Square; 2], // [white, black]
    pub zobrist: u64,
    pub undo_stack: Vec<UndoInfo>,
}

impl Default for Position {
    fn default() -> Self {
        Self {
            board: [Piece::default(); 64],
            side_to_move: Color::White,
            castling_rights: 0x0F,
            ep_square: 64,
            halfmove_clock: 0,
            fullmove_number: 1,
            king_sq: [4, 60],
            zobrist: 0,
            undo_stack: Vec::new(),
        }
    }
}

impl Position {
    /// Lightweight constructor — empty board, no zobrist, no undo stack.
    /// Used by resolve_knight_push for the temporary board copy.
    pub fn empty() -> Self {
        Self {
            board: [Piece::default(); 64],
            side_to_move: Color::White,
            castling_rights: 0,
            ep_square: 64,
            halfmove_clock: 0,
            fullmove_number: 1,
            king_sq: [64, 64],
            zobrist: 0,
            undo_stack: Vec::new(),
        }
    }

    // -------------------------------------------------------------------
    // FEN parsing
    // -------------------------------------------------------------------

    pub fn set_from_fen(&mut self, fen: &str) {
        self.board = [Piece::default(); 64];
        self.undo_stack.clear();

        let mut parts = fen.split_whitespace();
        let board_str = parts.next().unwrap_or("");
        let side_str = parts.next().unwrap_or("w");
        let castle_str = parts.next().unwrap_or("-");
        let ep_str = parts.next().unwrap_or("-");
        let hm_str = parts.next().unwrap_or("0");
        let fm_str = parts.next().unwrap_or("1");

        // Parse board (rank 8 = rank index 7 first in FEN)
        let mut rank: i32 = 7;
        let mut file: i32 = 0;
        for c in board_str.chars() {
            if c == '/' {
                rank -= 1;
                file = 0;
            } else if ('1'..='8').contains(&c) {
                file += (c as i32) - ('0' as i32);
            } else {
                let color = if c.is_ascii_uppercase() {
                    Color::White
                } else {
                    Color::Black
                };
                let pt = char_to_piece_type(c);
                let sq = make_square(rank, file);
                self.board[sq as usize] = Piece {
                    piece_type: pt,
                    color,
                };
                if pt == PieceType::King {
                    self.king_sq[color as usize] = sq;
                }
                file += 1;
            }
        }

        self.side_to_move = if side_str == "b" {
            Color::Black
        } else {
            Color::White
        };

        self.castling_rights = 0;
        if castle_str != "-" {
            for c in castle_str.chars() {
                match c {
                    'K' => self.castling_rights |= CASTLE_WK,
                    'Q' => self.castling_rights |= CASTLE_WQ,
                    'k' => self.castling_rights |= CASTLE_BK,
                    'q' => self.castling_rights |= CASTLE_BQ,
                    _ => {}
                }
            }
        }

        self.ep_square = 64;
        if ep_str != "-" && ep_str.len() == 2 {
            let bytes = ep_str.as_bytes();
            let ef = (bytes[0] as i32) - ('a' as i32);
            let er = (bytes[1] as i32) - ('1' as i32);
            if valid_rf(er, ef) {
                self.ep_square = make_square(er, ef);
            }
        }

        self.halfmove_clock = hm_str.parse::<u16>().unwrap_or(0);
        self.fullmove_number = fm_str.parse::<u16>().unwrap_or(1);

        self.compute_zobrist();
    }

    // -------------------------------------------------------------------
    // FEN serialization
    // -------------------------------------------------------------------

    pub fn to_fen(&self) -> String {
        let mut result = String::new();

        // Board
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                let p = self.board[make_square(rank, file) as usize];
                if p.is_empty() {
                    empty += 1;
                } else {
                    if empty > 0 {
                        result.push_str(&empty.to_string());
                        empty = 0;
                    }
                    result.push(piece_to_char(p));
                }
            }
            if empty > 0 {
                result.push_str(&empty.to_string());
            }
            if rank > 0 {
                result.push('/');
            }
        }

        // Side
        if self.side_to_move as u8 == Color::White as u8 {
            result.push_str(" w ");
        } else {
            result.push_str(" b ");
        }

        // Castling
        if self.castling_rights == 0 {
            result.push_str("- ");
        } else {
            if self.castling_rights & CASTLE_WK != 0 {
                result.push('K');
            }
            if self.castling_rights & CASTLE_WQ != 0 {
                result.push('Q');
            }
            if self.castling_rights & CASTLE_BK != 0 {
                result.push('k');
            }
            if self.castling_rights & CASTLE_BQ != 0 {
                result.push('q');
            }
            result.push(' ');
        }

        // EP
        if self.ep_square < 64 {
            result.push((b'a' + file_of(self.ep_square) as u8) as char);
            result.push((b'1' + rank_of(self.ep_square) as u8) as char);
        } else {
            result.push('-');
        }

        result.push(' ');
        result.push_str(&self.halfmove_clock.to_string());
        result.push(' ');
        result.push_str(&self.fullmove_number.to_string());

        result
    }

    // -------------------------------------------------------------------
    // Zobrist
    // -------------------------------------------------------------------

    pub fn compute_zobrist(&mut self) {
        let z = zobrist_tables();
        self.zobrist = 0;
        for sq in 0..64usize {
            if !self.board[sq].is_empty() {
                let c = self.board[sq].color as usize;
                let p = self.board[sq].piece_type as usize;
                self.zobrist ^= z.piece_keys[c][p][sq];
            }
        }
        if self.side_to_move as u8 == Color::Black as u8 {
            self.zobrist ^= z.side_key;
        }
        self.zobrist ^= z.castling_keys[self.castling_rights as usize];
        if self.ep_square < 64 {
            self.zobrist ^= z.ep_keys[file_of(self.ep_square) as usize];
        }
    }

    // -------------------------------------------------------------------
    // make_move
    // -------------------------------------------------------------------

    pub fn make_move(&mut self, m: &Move) {
        let z = zobrist_tables();

        // Save undo info
        let mut undo = UndoInfo {
            mv: *m,
            castling_rights: self.castling_rights,
            ep_square: self.ep_square,
            halfmove_clock: self.halfmove_clock,
            zobrist: self.zobrist,
            king_sq: self.king_sq,
            ..UndoInfo::default()
        };

        let mover = self.board[m.from as usize];
        let us = self.side_to_move;
        let them = opponent(us);

        // Clear old EP from zobrist
        if self.ep_square < 64 {
            self.zobrist ^= z.ep_keys[file_of(self.ep_square) as usize];
        }
        let mut new_ep: Square = 64;

        if m.special == SpecialMove::Castle {
            // Standard castling: king moves 2 squares, rook hops
            let rank = if us as u8 == Color::White as u8 {
                0i32
            } else {
                7i32
            };
            let king_from = m.from;
            let king_to = m.to;

            // Determine rook squares
            let (rook_from, rook_to) = if file_of(king_to) == 6 {
                // Kingside
                (make_square(rank, 7), make_square(rank, 5))
            } else {
                // Queenside
                (make_square(rank, 0), make_square(rank, 3))
            };

            // Record changes
            undo.add_changed(king_from, self.board[king_from as usize]);
            undo.add_changed(king_to, self.board[king_to as usize]);
            undo.add_changed(rook_from, self.board[rook_from as usize]);
            undo.add_changed(rook_to, self.board[rook_to as usize]);

            // Move king
            let kc = us as usize;
            let kp = PieceType::King as usize;
            self.zobrist ^= z.piece_keys[kc][kp][king_from as usize];
            self.zobrist ^= z.piece_keys[kc][kp][king_to as usize];
            self.board[king_to as usize] = self.board[king_from as usize];
            self.board[king_from as usize] = Piece::default();

            // Move rook
            let rp = PieceType::Rook as usize;
            self.zobrist ^= z.piece_keys[kc][rp][rook_from as usize];
            self.zobrist ^= z.piece_keys[kc][rp][rook_to as usize];
            self.board[rook_to as usize] = self.board[rook_from as usize];
            self.board[rook_from as usize] = Piece::default();

            self.king_sq[us as usize] = king_to;
            self.halfmove_clock += 1;
        } else if m.special == SpecialMove::EnPassant {
            // EP capture: pawn moves diagonally, captured pawn removed
            let cap_sq = make_square(rank_of(m.from), file_of(m.to));

            undo.add_changed(m.from, self.board[m.from as usize]);
            undo.add_changed(m.to, self.board[m.to as usize]);
            undo.add_changed(cap_sq, self.board[cap_sq as usize]);

            let uc = us as usize;
            let pp = PieceType::Pawn as usize;
            let tc = them as usize;

            self.zobrist ^= z.piece_keys[uc][pp][m.from as usize];
            self.zobrist ^= z.piece_keys[uc][pp][m.to as usize];
            self.zobrist ^= z.piece_keys[tc][pp][cap_sq as usize];

            self.board[m.to as usize] = self.board[m.from as usize];
            self.board[m.from as usize] = Piece::default();
            self.board[cap_sq as usize] = Piece::default();
            self.halfmove_clock = 0;
        } else {
            // Normal move or promotion: resolve push chain
            let is_pawn = mover.piece_type == PieceType::Pawn;
            let is_knight = mover.piece_type == PieceType::Knight;

            let push_info = if is_knight {
                let long_first = m.path_kind == 1;
                resolve_knight_push(self, m.from, m.to, long_first)
            } else {
                let rd = rank_of(m.to) - rank_of(m.from);
                let fd = file_of(m.to) - file_of(m.from);
                let dr = if rd != 0 {
                    if rd > 0 { 1 } else { -1 }
                } else {
                    0
                };
                let dc = if fd != 0 {
                    if fd > 0 { 1 } else { -1 }
                } else {
                    0
                };
                resolve_push(self, m.from, m.to, dr, dc)
            }
            .expect("make_move requires a generated move with a valid push path");

            // Record all squares that will change
            for di in 0..push_info.displacements().len() {
                let (f_sq, t_sq) = push_info.displacements()[di];
                undo.add_changed(f_sq, self.board[f_sq as usize]);
                undo.add_changed(t_sq, self.board[t_sq as usize]);
            }
            if let Some(sq) = push_info.captured() {
                undo.add_changed(sq, self.board[sq as usize]);
            }

            // XOR out all pieces at affected squares
            for ci in 0..undo.changed.len() {
                let sq = undo.changed[ci].0;
                if !self.board[sq as usize].is_empty() {
                    let c = self.board[sq as usize].color as usize;
                    let p = self.board[sq as usize].piece_type as usize;
                    self.zobrist ^= z.piece_keys[c][p][sq as usize];
                }
            }

            push_info.apply(&mut self.board);

            // Handle promotion (either mover or pushed pawn)
            if m.promo_piece != PieceType::None {
                let promo_rank = if us as u8 == Color::White as u8 {
                    7i32
                } else {
                    0i32
                };
                for &(_, to) in push_info.displacements() {
                    if self.board[to as usize].piece_type == PieceType::Pawn
                        && self.board[to as usize].color == us
                        && rank_of(to) == promo_rank
                    {
                        self.board[to as usize].piece_type = m.promo_piece;
                        break;
                    }
                }
            }

            // XOR in all pieces at affected squares
            for ci in 0..undo.changed.len() {
                let sq = undo.changed[ci].0;
                if !self.board[sq as usize].is_empty() {
                    let c = self.board[sq as usize].color as usize;
                    let p = self.board[sq as usize].piece_type as usize;
                    self.zobrist ^= z.piece_keys[c][p][sq as usize];
                }
            }

            // Update king position — check ALL displacements, not just the mover,
            // because push chains can displace kings of either side.
            for di in 0..push_info.displacements().len() {
                let (_f_sq, t_sq) = push_info.displacements()[di];
                let placed = self.board[t_sq as usize];
                if placed.piece_type == PieceType::King {
                    self.king_sq[placed.color as usize] = t_sq;
                }
            }

            // Update halfmove clock
            if is_pawn || push_info.captured().is_some() {
                self.halfmove_clock = 0;
            } else {
                self.halfmove_clock += 1;
            }

            // Set EP square for pawn double push
            if is_pawn
                && ((rank_of(m.to) - rank_of(m.from)) == 2
                    || (rank_of(m.from) - rank_of(m.to)) == 2)
            {
                new_ep = make_square((rank_of(m.from) + rank_of(m.to)) / 2, file_of(m.from));
            }
        }

        // Update castling rights
        self.zobrist ^= z.castling_keys[self.castling_rights as usize];
        let affected = undo
            .changed
            .iter()
            .fold(0, |mask, &(sq, _)| mask | (1u64 << sq));
        self.castling_rights = castling_after_move(self.castling_rights, affected);
        self.zobrist ^= z.castling_keys[self.castling_rights as usize];

        // Update EP
        self.ep_square = new_ep;
        if self.ep_square < 64 {
            self.zobrist ^= z.ep_keys[file_of(self.ep_square) as usize];
        }

        // Switch side
        self.zobrist ^= z.side_key;
        self.side_to_move = opponent(self.side_to_move);
        if self.side_to_move as u8 == Color::White as u8 {
            self.fullmove_number += 1;
        }

        self.undo_stack.push(undo);
    }

    // -------------------------------------------------------------------
    // unmake_move
    // -------------------------------------------------------------------

    pub fn unmake_move(&mut self) {
        let undo = self
            .undo_stack
            .pop()
            .expect("unmake_move on empty undo_stack");

        // Restore side
        self.side_to_move = opponent(self.side_to_move);
        if self.side_to_move as u8 == Color::Black as u8 {
            self.fullmove_number -= 1;
        }

        // Restore board squares
        for i in 0..undo.changed.len() {
            self.board[undo.changed[i].0 as usize] = undo.changed[i].1;
        }

        // Restore king positions from saved state
        self.king_sq = undo.king_sq;

        self.castling_rights = undo.castling_rights;
        self.ep_square = undo.ep_square;
        self.halfmove_clock = undo.halfmove_clock;
        self.zobrist = undo.zobrist;
    }

    // -------------------------------------------------------------------
    // is_attacked_by
    // -------------------------------------------------------------------

    pub fn is_attacked_by(&self, sq: Square, attacker: Color) -> bool {
        // Pawn attacks
        {
            // Pawns of 'attacker' attack from the other side
            let dr: i32 = if attacker as u8 == Color::White as u8 {
                -1
            } else {
                1
            };
            let r = rank_of(sq) + dr;
            for &dc in &[-1i32, 1i32] {
                let f = file_of(sq) + dc;
                if valid_rf(r, f) {
                    let s = make_square(r, f);
                    if self.board[s as usize].piece_type == PieceType::Pawn
                        && self.board[s as usize].color as u8 == attacker as u8
                    {
                        return true;
                    }
                }
            }
        }

        // Knight attacks
        {
            const KNIGHT_DR: [i32; 8] = [-2, -2, -1, -1, 1, 1, 2, 2];
            const KNIGHT_DC: [i32; 8] = [-1, 1, -2, 2, -2, 2, -1, 1];
            for i in 0..8usize {
                let r = rank_of(sq) + KNIGHT_DR[i];
                let f = file_of(sq) + KNIGHT_DC[i];
                if !valid_rf(r, f) {
                    continue;
                }
                let ksq = make_square(r, f);
                if self.board[ksq as usize].piece_type != PieceType::Knight
                    || self.board[ksq as usize].color as u8 != attacker as u8
                {
                    continue;
                }
                // Try both decompositions
                let info1 = resolve_knight_push(self, ksq, sq, true);
                if info1.is_some_and(|p| p.captured().is_some()) {
                    return true;
                }
                let info2 = resolve_knight_push(self, ksq, sq, false);
                if info2.is_some_and(|p| p.captured().is_some()) {
                    return true;
                }
            }
        }

        // Slider attacks (rook/queen on orthogonal, bishop/queen on diagonal)
        {
            const DR: [i32; 8] = [1, -1, 0, 0, 1, 1, -1, -1];
            const DC: [i32; 8] = [0, 0, 1, -1, 1, -1, 1, -1];
            for dir in 0..8usize {
                let ortho = dir < 4;
                let mut r = rank_of(sq) + DR[dir];
                let mut f = file_of(sq) + DC[dir];
                while valid_rf(r, f) {
                    let s = make_square(r, f);
                    if !self.board[s as usize].is_empty() {
                        if self.board[s as usize].color as u8 == attacker as u8 {
                            let pt = self.board[s as usize].piece_type;
                            if pt == PieceType::Queen {
                                return true;
                            }
                            if ortho && pt == PieceType::Rook {
                                return true;
                            }
                            if !ortho && pt == PieceType::Bishop {
                                return true;
                            }
                        }
                        break; // blocked
                    }
                    r += DR[dir];
                    f += DC[dir];
                }
            }
        }

        // King attacks
        {
            const DR: [i32; 8] = [1, -1, 0, 0, 1, 1, -1, -1];
            const DC: [i32; 8] = [0, 0, 1, -1, 1, -1, 1, -1];
            for dir in 0..8usize {
                let r = rank_of(sq) + DR[dir];
                let f = file_of(sq) + DC[dir];
                if valid_rf(r, f) {
                    let s = make_square(r, f);
                    if self.board[s as usize].piece_type == PieceType::King
                        && self.board[s as usize].color as u8 == attacker as u8
                    {
                        return true;
                    }
                }
            }
        }

        false
    }

    // -------------------------------------------------------------------
    // in_check
    // -------------------------------------------------------------------

    /// Is the side to move in check?
    pub fn in_check(&self) -> bool {
        self.in_check_color(self.side_to_move)
    }

    /// Is the given color in check?
    pub fn in_check_color(&self, c: Color) -> bool {
        self.is_attacked_by(self.king_sq[c as usize], opponent(c))
    }
}

// ---------------------------------------------------------------------------
// Free function
// ---------------------------------------------------------------------------

pub fn start_position() -> Position {
    let mut pos = Position::default();
    pos.set_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    pos
}
