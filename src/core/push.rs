use arrayvec::ArrayVec;

use super::position::Position;
use super::types::*;

/// A resolved move, never an illegal or partially built push.
///
/// Displacements are simultaneous: each source refers to the original board.
/// `None` represents an illegal path; a missing captured square a non-capture.
/// Counts cannot drift from the stored data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PushPlan {
    displacements: ArrayVec<(Square, Square), 16>,
    captured: Option<Square>,
}

impl PushPlan {
    pub(crate) fn single(from: Square, to: Square, captured: Option<Square>) -> Self {
        let mut displacements = ArrayVec::new();
        displacements.push((from, to));
        Self {
            displacements,
            captured,
        }
    }

    pub fn displacements(&self) -> &[(Square, Square)] {
        &self.displacements
    }

    pub fn captured(&self) -> Option<Square> {
        self.captured
    }

    /// Use a snapshot so displacements cannot overwrite each other's sources.
    pub(crate) fn apply(&self, board: &mut [Piece; 64]) {
        let original = *board;
        for &(from, _) in &self.displacements {
            board[from as usize] = Piece::default();
        }
        if let Some(sq) = self.captured {
            board[sq as usize] = Piece::default();
        }
        for &(from, to) in &self.displacements {
            board[to as usize] = original[from as usize];
        }
    }
}

/// Resolve a straight path. Invalid public inputs return `None`, never loop.
pub fn resolve_push(
    pos: &Position,
    from: Square,
    to: Square,
    dr: i32,
    dc: i32,
) -> Option<PushPlan> {
    if from >= 64
        || to >= 64
        || from == to
        || !(-1..=1).contains(&dr)
        || !(-1..=1).contains(&dc)
        || (dr == 0 && dc == 0)
        || pos.board[from as usize].is_empty()
    {
        return None;
    }
    let mover_color = pos.board[from as usize].color;
    let mut chain = ArrayVec::<Square, 8>::new();
    let mut r = rank_of(from) + dr;
    let mut f = file_of(from) + dc;

    loop {
        if !valid_rf(r, f) {
            return None;
        }
        let sq = make_square(r, f);
        let piece = pos.board[sq as usize];
        if piece.is_color(mover_color) {
            chain.push(sq);
        } else if !piece.is_empty() {
            return (sq == to && chain.is_empty()).then(|| PushPlan::single(from, to, Some(to)));
        }
        if sq == to {
            break;
        }
        r += dr;
        f += dc;
    }

    // Every friendly piece needs a slot beyond the destination. Discovering
    // another friendly piece there adds it to the same chain.
    let mut slots_found = 0;
    while slots_found < chain.len() {
        r += dr;
        f += dc;
        if !valid_rf(r, f) {
            return None;
        }
        let sq = make_square(r, f);
        let piece = pos.board[sq as usize];
        if piece.is_color(mover_color) {
            chain.push(sq);
        } else if !piece.is_empty() {
            return None;
        }
        slots_found += 1;
    }

    let mut plan = PushPlan::single(from, to, None);
    for (i, source) in chain.into_iter().enumerate() {
        let distance = i as i32 + 1;
        let target = make_square(rank_of(to) + dr * distance, file_of(to) + dc * distance);
        plan.displacements.push((source, target));
    }
    Some(plan)
}

/// Resolve either L-shaped knight path, preserving piece identity across legs.
/// Captures are allowed only on the final leg.
pub fn resolve_knight_push(
    pos: &Position,
    from: Square,
    to: Square,
    long_first: bool,
) -> Option<PushPlan> {
    if from >= 64 || to >= 64 {
        return None;
    }
    let dr = rank_of(to) - rank_of(from);
    let dc = file_of(to) - file_of(from);
    if !matches!((dr.abs(), dc.abs()), (1, 2) | (2, 1)) {
        return None;
    }
    let (long, short) = if dr.abs() == 2 {
        ((dr.signum(), 0, 2), (0, dc.signum(), 1))
    } else {
        ((0, dc.signum(), 2), (dr.signum(), 0, 1))
    };
    let (first, second) = if long_first {
        (long, short)
    } else {
        (short, long)
    };
    let mid = make_square(
        rank_of(from) + first.0 * first.2,
        file_of(from) + first.1 * first.2,
    );
    let leg1 = resolve_push(pos, from, mid, first.0, first.1)?;
    if leg1.captured.is_some() {
        return None;
    }

    let mut intermediate = Position::empty();
    intermediate.board = pos.board;
    leg1.apply(&mut intermediate.board);
    let leg2 = resolve_push(&intermediate, mid, to, second.0, second.1)?;

    // Compose against the unchanged first plan, not already-updated entries.
    let mut displacements = ArrayVec::new();
    for &(original, current) in &leg1.displacements {
        let final_sq = leg2
            .displacements
            .iter()
            .find_map(|&(from, to)| (from == current).then_some(to))
            .unwrap_or(current);
        if original != final_sq {
            displacements.push((original, final_sq));
        }
    }
    for &(from, to) in &leg2.displacements {
        if !leg1
            .displacements
            .iter()
            .any(|&(_, current)| current == from)
        {
            displacements.push((from, to));
        }
    }
    Some(PushPlan {
        displacements,
        captured: leg2.captured,
    })
}
