use super::position::Position;
use super::types::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PushResult {
    OK,
    Capture,
    Illegal,
}

pub const MAX_DISPLACEMENTS: usize = 16;

#[derive(Clone, Copy, Debug)]
pub struct PushInfo {
    pub result: PushResult,
    pub displacements: [(Square, Square); MAX_DISPLACEMENTS],
    pub num_displacements: usize,
    pub captured_sq: Square,
}

impl Default for PushInfo {
    fn default() -> Self {
        Self {
            result: PushResult::Illegal,
            displacements: [(0, 0); MAX_DISPLACEMENTS],
            num_displacements: 0,
            captured_sq: 64,
        }
    }
}

impl PushInfo {
    pub fn add_displacement(&mut self, from: Square, to: Square) {
        self.displacements[self.num_displacements] = (from, to);
        self.num_displacements += 1;
    }
}

pub fn resolve_push(pos: &Position, from: Square, to: Square, dr: i32, dc: i32) -> PushInfo {
    let mover_color = pos.board[from as usize].color;
    let mut info = PushInfo::default();

    let mut chain: [Square; 8] = [0; 8];
    let mut chain_len: i32 = 0;
    let mut r: i32 = rank_of(from) + dr;
    let mut f: i32 = file_of(from) + dc;
    let dest = to;

    loop {
        if !valid_rf(r, f) {
            info.result = PushResult::Illegal;
            return info;
        }
        let sq = make_square(r, f);
        let at_dest = sq == dest;

        if pos.board[sq as usize].is_empty() {
            // continue
        } else if pos.board[sq as usize].is_color(mover_color) {
            chain[chain_len as usize] = sq;
            chain_len += 1;
        } else {
            if at_dest && chain_len == 0 {
                info.result = PushResult::Capture;
                info.add_displacement(from, to);
                info.captured_sq = to;
                return info;
            }
            info.result = PushResult::Illegal;
            return info;
        }

        if at_dest {
            break;
        }
        r += dr;
        f += dc;
    }

    // Phase 2: Cascade
    let mut cascade: [Square; 16] = [0; 16];
    let mut cascade_len: i32 = 0;
    for i in 0..chain_len {
        cascade[cascade_len as usize] = chain[i as usize];
        cascade_len += 1;
    }
    let mut slots_needed: i32 = cascade_len;
    let mut slots_found: i32 = 0;
    r = rank_of(to);
    f = file_of(to);

    while slots_found < slots_needed {
        r += dr;
        f += dc;
        if !valid_rf(r, f) {
            info.result = PushResult::Illegal;
            return info;
        }
        let sq = make_square(r, f);
        if !pos.board[sq as usize].is_empty() && !pos.board[sq as usize].is_color(mover_color) {
            info.result = PushResult::Illegal;
            return info;
        }
        if pos.board[sq as usize].is_color(mover_color) {
            cascade[cascade_len as usize] = sq;
            cascade_len += 1;
            slots_needed += 1;
        }
        slots_found += 1;
    }

    // Phase 3: Assign positions
    info.result = PushResult::OK;
    info.add_displacement(from, to);
    let mut target_r: i32 = rank_of(to);
    let mut target_f: i32 = file_of(to);
    for i in 0..cascade_len {
        target_r += dr;
        target_f += dc;
        info.add_displacement(cascade[i as usize], make_square(target_r, target_f));
    }

    info
}

pub fn resolve_knight_push(
    pos: &Position,
    from: Square,
    to: Square,
    long_first: bool,
) -> PushInfo {
    let dr: i32 = rank_of(to) - rank_of(from);
    let dc: i32 = file_of(to) - file_of(from);
    let abs_dr: i32 = dr.abs();
    let abs_dc: i32 = dc.abs();

    let long_dr: i32;
    let long_dc: i32;
    let short_dr: i32;
    let short_dc: i32;
    let long_dist: i32;
    let short_dist: i32;

    if abs_dr > abs_dc {
        long_dr = if dr > 0 { 1 } else { -1 };
        long_dc = 0;
        long_dist = 2;
        short_dr = 0;
        short_dc = if dc > 0 { 1 } else { -1 };
        short_dist = 1;
    } else {
        long_dr = 0;
        long_dc = if dc > 0 { 1 } else { -1 };
        long_dist = 2;
        short_dr = if dr > 0 { 1 } else { -1 };
        short_dc = 0;
        short_dist = 1;
    }

    let leg1_dr: i32;
    let leg1_dc: i32;
    let leg1_dist: i32;
    let leg2_dr: i32;
    let leg2_dc: i32;

    if long_first {
        leg1_dr = long_dr;
        leg1_dc = long_dc;
        leg1_dist = long_dist;
        leg2_dr = short_dr;
        leg2_dc = short_dc;
    } else {
        leg1_dr = short_dr;
        leg1_dc = short_dc;
        leg1_dist = short_dist;
        leg2_dr = long_dr;
        leg2_dc = long_dc;
    }

    let mid_r: i32 = rank_of(from) + leg1_dr * leg1_dist;
    let mid_f: i32 = file_of(from) + leg1_dc * leg1_dist;
    if !valid_rf(mid_r, mid_f) {
        return PushInfo::default();
    }
    let mid = make_square(mid_r, mid_f);

    let leg1 = resolve_push(pos, from, mid, leg1_dr, leg1_dc);
    if leg1.result == PushResult::Illegal || leg1.result == PushResult::Capture {
        return PushInfo::default();
    }

    // Apply leg1 displacements to board copy (just the [Piece; 64] array)
    let mut temp_board = pos.board;
    for i in 0..leg1.num_displacements {
        let (f_sq, t_sq) = leg1.displacements[i as usize];
        temp_board[t_sq as usize] = temp_board[f_sq as usize];
        temp_board[f_sq as usize] = Piece::default();
    }

    // Create minimal Position for leg 2
    let temp = Position {
        board: temp_board,
        side_to_move: pos.side_to_move,
        ..Default::default()
    };

    let leg2 = resolve_push(&temp, mid, to, leg2_dr, leg2_dc);
    if leg2.result == PushResult::Illegal {
        return PushInfo::default();
    }

    // Combine displacements
    let mut entries: [(Square, Square); 32] = [(0, 0); 32]; // (orig, current)
    let mut num_entries: i32 = 0;

    // Process leg 1 — add all entries directly (no tracking).
    // The cascade entries are independent pieces, not the mover chaining.
    for i in 0..leg1.num_displacements {
        let (f_sq, t_sq) = leg1.displacements[i as usize];
        entries[num_entries as usize] = (f_sq, t_sq);
        num_entries += 1;
    }

    // Process leg 2 — use tracking to update positions from leg 1
    for i in 0..leg2.num_displacements {
        let (f_sq, t_sq) = leg2.displacements[i as usize];
        let mut found = false;
        for j in 0..num_entries {
            if entries[j as usize].1 == f_sq {
                entries[j as usize].1 = t_sq;
                found = true;
                break;
            }
        }
        if !found {
            entries[num_entries as usize] = (f_sq, t_sq);
            num_entries += 1;
        }
    }

    let mut result = PushInfo::default();
    result.result = leg2.result;
    result.captured_sq = leg2.captured_sq;
    for i in 0..num_entries {
        if entries[i as usize].0 != entries[i as usize].1 {
            result.add_displacement(entries[i as usize].0, entries[i as usize].1);
        }
    }
    result
}
