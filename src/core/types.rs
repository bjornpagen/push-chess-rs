use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "wasm", tsify::declare)]
pub type Square = u8;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[repr(u8)]
pub enum Color {
    #[default]
    White = 0,
    Black = 1,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[repr(u8)]
pub enum PieceType {
    #[default]
    None = 0,
    Pawn = 1,
    Knight = 2,
    Bishop = 3,
    Rook = 4,
    Queen = 5,
    King = 6,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[repr(u8)]
pub enum SpecialMove {
    #[default]
    None = 0,
    Castle = 1,
    EnPassant = 2,
    Promotion = 3,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Move {
    pub from: Square,
    pub to: Square,
    pub path_kind: u8,
    pub stop_index: u8,
    pub special: SpecialMove,
    pub promo_piece: PieceType,
}

impl Move {
    /// Lossless wire identifier, resolved against the current legal move list.
    /// Never decode an untrusted ID directly into a move.
    pub fn id(self) -> u32 {
        u32::from(self.from)
            | (u32::from(self.to) << 6)
            | (u32::from(self.path_kind) << 12)
            | (u32::from(self.stop_index) << 14)
            | ((self.special as u32) << 18)
            | ((self.promo_piece as u32) << 20)
    }
}

#[derive(Clone, Debug, Default)]
pub struct SearchBudget {
    pub max_time_us: i64,
    pub max_nodes: i64,
    pub max_depth: i32,
    pub seed: u64,
}

#[derive(Clone, Debug, Default)]
pub struct SearchStats {
    pub nodes: u64,
    pub depth_reached: u32,
    pub seldepth: u32,
    pub eval_cp: i32,
    pub time_used_us: i64,
    pub pv: Vec<Move>,
    pub diag_json: String,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(C)]
pub struct Piece {
    pub piece_type: PieceType,
    pub color: Color,
}

impl Piece {
    pub fn is_empty(self) -> bool {
        self.piece_type == PieceType::None
    }

    pub fn is_color(self, c: Color) -> bool {
        !self.is_empty() && self.color == c
    }
}

pub fn rank_of(sq: Square) -> i32 {
    (sq / 8) as i32
}

pub fn file_of(sq: Square) -> i32 {
    (sq % 8) as i32
}

pub fn make_square(r: i32, f: i32) -> Square {
    (r * 8 + f) as Square
}

pub fn valid_rf(r: i32, f: i32) -> bool {
    (0..8).contains(&r) && (0..8).contains(&f)
}

pub fn opponent(c: Color) -> Color {
    if c == Color::White {
        Color::Black
    } else {
        Color::White
    }
}

pub const CASTLE_WK: u8 = 1;
pub const CASTLE_WQ: u8 = 2;
pub const CASTLE_BK: u8 = 4;
pub const CASTLE_BQ: u8 = 8;

/// Castling belongs to the original king/rooks, including when pushed. The
/// affected-square set is shared by the rules engine and prepared search board.
pub fn castling_after_move(rights: u8, affected: u64) -> u8 {
    [(4, 3), (60, 12), (0, 2), (7, 1), (56, 8), (63, 4)]
        .into_iter()
        .fold(rights, |rights, (square, mask)| {
            if affected & (1 << square) != 0 {
                rights & !mask
            } else {
                rights
            }
        })
}

pub const PIECE_VALUES: [i32; 7] = [0, 100, 320, 330, 500, 900, 0];

pub fn pval(pt: PieceType) -> i32 {
    PIECE_VALUES[pt as usize]
}
