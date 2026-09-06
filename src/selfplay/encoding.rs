use super::State;
use crate::core::position::Position;
use crate::core::prepared::PreparedMove;
use crate::core::types::{Move, Piece};

pub const BOARD_FLOATS: usize = 32 * 64;
pub const ACTION_FIELDS: usize = 6;
pub const EFFECT_FIELDS: usize = 4;

pub struct Encoded {
    pub board: [f32; BOARD_FLOATS],
    pub ids: Vec<u32>,
    pub actions: Vec<[i32; ACTION_FIELDS]>,
    pub effects: Vec<[i32; EFFECT_FIELDS]>,
}

impl State {
    pub fn encode(&self) -> Encoded {
        Encoded::new(&self.pos, &self.legal)
    }
    pub fn encode_effects(&self) -> Encoded {
        let mut row = self.encode();
        for (i, mv) in self.prepared.iter().enumerate() {
            mv.effects(&self.pos, i, &mut row.effects);
        }
        row
    }
}

pub(super) fn write_board(
    pos: &Position,
    previous: Option<&[Piece; 64]>,
    repeats: usize,
    board: &mut [f32],
) {
    assert_eq!(board.len(), BOARD_FLOATS);
    board.fill(0.0);
    let us = pos.side_to_move as usize;
    let flip = if us == 0 { 0 } else { 56 };
    for (base, pieces) in [(0, Some(&pos.board)), (12, previous)] {
        if let Some(pieces) = pieces {
            for (sq, p) in pieces.iter().enumerate() {
                if !p.is_empty() {
                    let channel = base + (p.color as usize ^ us) * 6 + p.piece_type as usize - 1;
                    board[channel * 64 + (sq ^ flip)] = 1.0;
                }
            }
        }
    }
    for i in 0..4 {
        let bit = if us == 0 { i } else { (i + 2) % 4 };
        board[(24 + i) * 64..(25 + i) * 64].fill(if pos.castling_rights & (1 << bit) != 0 {
            1.0
        } else {
            0.0
        });
    }
    if pos.ep_square < 64 {
        board[28 * 64 + (pos.ep_square as usize ^ flip)] = 1.0;
    }
    board[29 * 64..30 * 64].fill(f32::from(pos.halfmove_clock.min(100)) / 100.0);
    board[30 * 64..31 * 64].fill(repeats.min(2) as f32 / 2.0);
    board[31 * 64..].fill(us as f32);
}

pub(super) fn action(pos: &Position, m: Move) -> [i32; ACTION_FIELDS] {
    let flip = if pos.side_to_move as u8 == 0 { 0 } else { 56 };
    [
        i32::from(m.from ^ flip),
        i32::from(m.to ^ flip),
        i32::from(m.path_kind),
        i32::from(m.stop_index),
        m.promo_piece as i32,
        m.special as i32,
    ]
}

impl Encoded {
    pub(super) fn new(pos: &Position, moves: &[Move]) -> Self {
        let mut board = [0.0; BOARD_FLOATS];
        let previous = pos.previous_board();
        let repeats = pos
            .undo_stack
            .iter()
            .filter(|u| u.zobrist == pos.zobrist)
            .take(2)
            .count();
        write_board(pos, previous.as_ref(), repeats, &mut board);
        Self {
            board,
            ids: moves.iter().map(|m| m.id()).collect(),
            actions: moves.iter().map(|&m| action(pos, m)).collect(),
            effects: Vec::new(),
        }
    }
}

/// Owned final wire buffers. They may outlive the search or any subsequent
/// request; never refill storage once ownership has transferred to a caller.
pub struct Features {
    pub rows: usize,
    pub width: usize,
    pub boards: Vec<f32>,
    pub actions: Vec<i32>,
    pub lengths: Vec<i32>,
    pub effect_width: usize,
    pub effects: Vec<i32>,
}

impl Features {
    pub(super) fn empty(rows: usize, width: usize) -> Self {
        Self {
            rows,
            width,
            boards: vec![0.0; rows * BOARD_FLOATS],
            actions: vec![0; rows * width * ACTION_FIELDS],
            lengths: vec![0; rows],
            effect_width: 0,
            effects: Vec::new(),
        }
    }

    pub(super) fn write_actions(&mut self, row: usize, pos: &Position, moves: &[PreparedMove]) {
        self.lengths[row] = moves.len() as i32;
        for (i, mv) in moves.iter().enumerate() {
            let start = (row * self.width + i) * ACTION_FIELDS;
            self.actions[start..start + ACTION_FIELDS].copy_from_slice(&action(pos, mv.mv()));
        }
    }

    pub(super) fn pack_effects(&mut self, tokens: &[[i32; EFFECT_FIELDS]], offsets: &[usize]) {
        self.effect_width = offsets
            .windows(2)
            .map(|w| w[1] - w[0])
            .max()
            .unwrap_or(0)
            .max(16)
            .next_power_of_two();
        self.effects
            .resize(self.rows * self.effect_width * EFFECT_FIELDS, 0);
        for (i, range) in offsets.windows(2).enumerate() {
            for (j, effect) in tokens[range[0]..range[1]].iter().enumerate() {
                let start = (i * self.effect_width + j) * EFFECT_FIELDS;
                self.effects[start..start + EFFECT_FIELDS].copy_from_slice(effect);
            }
        }
    }
}
