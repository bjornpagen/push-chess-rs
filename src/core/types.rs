pub type Square = u8;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(u8)]
pub enum Color {
    #[default]
    White = 0,
    Black = 1,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
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

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
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

#[derive(Clone, Debug, Default)]
pub struct SearchBudget {
    pub max_time_us: i64,
    pub max_nodes: i64,
    pub max_depth: i32,
    pub seed: u64,
}

#[derive(Clone, Debug)]
pub struct SearchStats {
    pub nodes: u64,
    pub depth_reached: u32,
    pub seldepth: u32,
    pub eval_cp: i32,
    pub time_used_us: i64,
    pub pv: Vec<Move>,
    pub diag_json: String,
}

impl Default for SearchStats {
    fn default() -> Self {
        Self {
            nodes: 0,
            depth_reached: 0,
            seldepth: 0,
            eval_cp: 0,
            time_used_us: 0,
            pv: Vec::new(),
            diag_json: String::new(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
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
    r >= 0 && r < 8 && f >= 0 && f < 8
}

pub fn opponent(c: Color) -> Color {
    if c == Color::White { Color::Black } else { Color::White }
}

pub const CASTLE_WK: u8 = 1;
pub const CASTLE_WQ: u8 = 2;
pub const CASTLE_BK: u8 = 4;
pub const CASTLE_BQ: u8 = 8;

pub const PIECE_VALUES: [i32; 7] = [0, 100, 320, 330, 500, 900, 0];

pub fn pval(pt: PieceType) -> i32 {
    PIECE_VALUES[pt as usize]
}
