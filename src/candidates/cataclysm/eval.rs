//! Incremental piece-square baseline and outcome-trained neural evaluation.
//! No database access, opponent identity, or memorized game moves.
use super::board::Board;
use crate::core::types::*;
use std::sync::LazyLock;

pub const VALUE: [i32; 7] = [0, 100, 305, 365, 550, 1050, 0];
pub const PHASE: [i32; 7] = [0, 0, 1, 1, 2, 4, 0];
type PieceSquareTables = [[[(i32, i32); 64]; 7]; 2];
static PST: LazyLock<PieceSquareTables> = LazyLock::new(|| {
    std::array::from_fn(|c| {
        std::array::from_fn(|pt| {
            std::array::from_fn(|sq| {
                let r = if c == 0 { sq / 8 } else { 7 - sq / 8 } as i32;
                let f = (sq % 8) as i32;
                let center = (f - 3).abs().min((f - 4).abs()) + (r - 3).abs().min((r - 4).abs());
                let (mg, eg) = match pt {
                    1 => (
                        r * 3 + (r - 3).max(0).pow(2) * 4 - (f - 3).abs() * 2,
                        r * 7 + (r - 3).max(0).pow(2) * 8,
                    ),
                    2 => (36 - center * 11, 25 - center * 8),
                    3 => (26 - center * 6, 30 - center * 5),
                    4 => (r * 3, 12 - center * 3),
                    5 => (12 - center * 3, 20 - center * 4),
                    // No castling fetish: safety comes from the actual enemy attack geometry.
                    6 => (-r * 5, 40 - center * 12),
                    _ => (0, 0),
                };
                (VALUE[pt] + mg, VALUE[pt] + eg)
            })
        })
    })
});

pub fn piece_score(p: Piece, sq: u8) -> (i32, i32) {
    PST[p.color as usize][p.piece_type as usize][sq as usize]
}

pub fn evaluate(b: &Board) -> i32 {
    let phase = b.phase.min(24);
    let baseline = ((b.mg[0] - b.mg[1]) * phase + (b.eg[0] - b.eg[1]) * (24 - phase)) / 24;
    let mut white = baseline + b.net.white_residual(b.model) / 2;
    for c in 0..2 {
        let sign = if c == 0 { 1 } else { -1 };
        let mut pawns = b.men[c][1];
        while pawns != 0 {
            let sq = pawns.trailing_zeros() as usize;
            pawns &= pawns - 1;
            let rank = if c == 0 { sq / 8 } else { 7 - sq / 8 };
            if b.men[1 - c][1] & PASSED[c][sq] == 0 {
                white += sign * [0, 0, 5, 12, 24, 50, 100, 0][rank] * (32 - phase) / 16;
            }
        }
        if b.men[c][3].count_ones() >= 2 {
            white += sign * 30;
        }
    }
    (if b.pos.side_to_move == Color::White {
        white
    } else {
        -white
    } + 14)
        .clamp(-28_000, 28_000)
}

const PASSED: [[u64; 64]; 2] = {
    let mut result = [[0; 64]; 2];
    let mut c = 0;
    while c < 2 {
        let mut sq = 0;
        while sq < 64 {
            let mut to = 0;
            while to < 64 {
                let delta = (sq % 8) as i32 - (to % 8) as i32;
                if delta >= -1
                    && delta <= 1
                    && if c == 0 {
                        to / 8 > sq / 8
                    } else {
                        to / 8 < sq / 8
                    }
                {
                    result[c][sq] |= 1 << to;
                }
                to += 1;
            }
            sq += 1;
        }
        c += 1;
    }
    result
};
