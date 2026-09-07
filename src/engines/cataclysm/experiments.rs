//! Immutable, independently testable hypotheses. Const specialization keeps
//! experimental evaluation/search branches out of the control's hot path.
use super::board::{Action, Board, DIRS, KNIGHTS, Residual, step};
use super::eval::piece_score;
use crate::core::types::*;

#[derive(Clone, Copy, Debug)]
pub struct Profile {
    pub name: &'static str,
    pub hypothesis: &'static str,
    pub neural_scale: i32,
    pub neural_accumulator: bool,
    pub geometry: bool,
    pub logistics: bool,
    pub ordering: bool,
    pub volatility: bool,
    pub exact_context: bool,
}

const CONTROL: Profile = Profile {
    name: "cataclysm",
    hypothesis: "Unchanged evaluation/search policy; shared scratch-buffer refactors only",
    neural_scale: 1,
    neural_accumulator: true,
    geometry: false,
    logistics: false,
    ordering: false,
    volatility: false,
    exact_context: false,
};

pub const PROFILES: [Profile; 10] = [
    CONTROL,
    Profile {
        name: "abacus",
        hypothesis: "Does the existing learned residual earn its place? (Accumulator work retained.)",
        neural_scale: 0,
        ..CONTROL
    },
    Profile {
        name: "resonance",
        hypothesis: "Is the existing learned residual underweighted?",
        neural_scale: 2,
        ..CONTROL
    },
    Profile {
        name: "perimeter",
        hypothesis: "Geometric mobility, king-ring pressure and exposed material improve evaluation",
        geometry: true,
        ..CONTROL
    },
    Profile {
        name: "convoy",
        hypothesis: "Promotion corridors and friendly pawn-push support improve evaluation",
        logistics: true,
        ..CONTROL
    },
    Profile {
        name: "kinetic",
        hypothesis: "Phase-aware whole-transaction gain orders useful pushes earlier",
        ordering: true,
        ..CONTROL
    },
    Profile {
        name: "flashpoint",
        hypothesis: "Protect near-promotion and king pushes from reductions; extend their tactical horizon",
        volatility: true,
        ..CONTROL
    },
    Profile {
        name: "synthesis",
        hypothesis: "Geometry, logistics, ordering and volatility interact beneficially",
        geometry: true,
        logistics: true,
        ordering: true,
        volatility: true,
        ..CONTROL
    },
    Profile {
        name: "waypoint",
        hypothesis: "Counter-move ordering uses the actual parent action in tactical/proof search and no parent after NULL",
        exact_context: true,
        ..CONTROL
    },
    Profile {
        name: "granite",
        hypothesis: "Abacus decisions with no neural weights, updates or accumulator snapshots",
        neural_scale: 0,
        neural_accumulator: false,
        ..CONTROL
    },
];

/// Geometric influence, not a legal-move proof. In particular knight routes
/// may be blocked in Push Chess; legal search remains the tactical authority.
fn influence(b: &Board<impl Residual>, color: usize) -> (u64, u64, i32) {
    let mut all = 0u64;
    let mut twice = 0u64;
    let mut mobility = 0;
    let mut men = b.occupied[color];
    while men != 0 {
        let sq = men.trailing_zeros() as u8;
        men &= men - 1;
        let pt = b.pos.board[sq as usize].piece_type;
        let mut attacks = 0u64;
        match pt {
            PieceType::Pawn => {
                for df in [-1, 1] {
                    if let Some(to) = step(sq, if color == 0 { 1 } else { -1 }, df) {
                        attacks |= 1 << to;
                    }
                }
            }
            PieceType::Knight => {
                for (dr, df) in KNIGHTS {
                    if let Some(to) = step(sq, dr, df) {
                        attacks |= 1 << to;
                    }
                }
            }
            _ => {
                for (i, (dr, df)) in DIRS.into_iter().enumerate() {
                    if (pt == PieceType::Bishop && i < 4) || (pt == PieceType::Rook && i >= 4) {
                        continue;
                    }
                    let mut at = sq;
                    while let Some(to) = step(at, dr, df) {
                        attacks |= 1 << to;
                        if pt == PieceType::King || !b.pos.board[to as usize].is_empty() {
                            break;
                        }
                        at = to;
                    }
                }
            }
        }
        twice |= all & attacks;
        all |= attacks;
        if pt != PieceType::Pawn && pt != PieceType::King {
            mobility += (attacks & !b.occupied[color]).count_ones() as i32;
        }
    }
    (all, twice, mobility)
}

pub(super) fn geometry(b: &Board<impl Residual>) -> i32 {
    let maps = [influence(b, 0), influence(b, 1)];
    let mut score = 0;
    for c in 0..2 {
        let (ours, supported, mobility) = maps[c];
        let theirs = maps[1 - c].0;
        let mut ring = 0u64;
        for (dr, df) in DIRS {
            if let Some(to) = step(b.pos.king_sq[1 - c], dr, df) {
                ring |= 1 << to;
            }
        }
        let pressure = (ring & ours).count_ones() as i32;
        let double = (ring & supported).count_ones() as i32;
        let exposed = b.occupied[c] & theirs & !ours & !b.men[c][6];
        let vulnerable = (exposed & (b.men[c][2] | b.men[c][3])).count_ones() as i32 * 18
            + (exposed & (b.men[c][4] | b.men[c][5])).count_ones() as i32 * 30;
        let value = 3 * mobility + 4 * pressure * pressure + 9 * double - vulnerable;
        score += if c == 0 { value } else { -value };
    }
    score
}

pub(super) fn logistics(b: &Board<impl Residual>) -> i32 {
    let mut score = 0;
    for c in 0..2 {
        let dr = if c == 0 { 1 } else { -1 };
        let mut pawns = b.men[c][1];
        while pawns != 0 {
            let sq = pawns.trailing_zeros() as u8;
            pawns &= pawns - 1;
            let rank = if c == 0 { rank_of(sq) } else { 7 - rank_of(sq) };
            let mut value = 0;
            if let Some(ahead) = step(sq, dr, 0) {
                let target = b.pos.board[ahead as usize];
                if target.is_empty() {
                    value += 3 * rank;
                } else if target.color as usize != c {
                    value -= 4 * rank;
                }
                // This is potential support, not an assertion that a push is legal.
                if target.is_empty()
                    && let Some(behind) = step(sq, -dr, 0)
                {
                    let p = b.pos.board[behind as usize];
                    if !p.is_empty() && p.color as usize == c {
                        value += 4 * rank * rank;
                    }
                }
            }
            for df in [-1, 1] {
                if let Some(other) = step(sq, 0, df)
                    && b.men[c][1] & (1 << other) != 0
                {
                    value += 2 * rank;
                }
            }
            score += if c == 0 { value } else { -value };
        }
    }
    score
}

pub(super) fn transaction_gain(b: &Board<impl Residual>, a: &Action) -> i32 {
    let phase = b.phase.min(24);
    a.plan.as_ref().map_or(0, |plan| {
        plan.displacements()
            .iter()
            .map(|&(from, to)| {
                let piece = b.pos.board[from as usize];
                let before = piece_score(piece, from);
                let after = piece_score(piece, to);
                let gain =
                    ((after.0 - before.0) * phase + (after.1 - before.1) * (24 - phase)) / 24;
                let rank = if piece.color == Color::White {
                    rank_of(to)
                } else {
                    7 - rank_of(to)
                };
                let promotion_approach = if piece.piece_type == PieceType::Pawn {
                    (rank - 3).max(0).pow(2) * 50
                } else {
                    0
                };
                let value = gain * 12 + promotion_approach;
                if piece.color == b.pos.side_to_move {
                    value
                } else {
                    -value
                }
            })
            .sum()
    })
}

pub(super) fn volatile_push(b: &Board<impl Residual>, a: &Action) -> bool {
    a.king_push
        || (a.push
            && a.plan.as_ref().is_some_and(|plan| {
                plan.displacements().iter().any(|&(from, to)| {
                    let p = b.pos.board[from as usize];
                    p.piece_type == PieceType::Pawn
                        && if p.color == Color::White {
                            rank_of(to) >= 6
                        } else {
                            rank_of(to) <= 1
                        }
                })
            }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::position::{Position, start_position};

    #[test]
    fn symmetric_start_has_no_white_geometry_or_logistics_bias() {
        let b = Board::new(&start_position());
        assert_eq!(geometry(&b), 0);
        assert_eq!(logistics(&b), 0);
    }

    #[test]
    fn features_reverse_under_color_rank_mirror() {
        let p = Position::try_from_fen("6k1/5pp1/8/2P5/2R1n3/4B3/6P1/6K1 w - - 0 1").unwrap();
        let mut q = Position::empty();
        for sq in 0..64 {
            let mut piece = p.board[sq];
            if !piece.is_empty() {
                piece.color = opponent(piece.color);
            }
            q.board[sq ^ 56] = piece;
            if piece.piece_type == PieceType::King {
                q.king_sq[piece.color as usize] = (sq ^ 56) as u8;
            }
        }
        let (a, b) = (Board::new(&p), Board::new(&q));
        assert_eq!(geometry(&a), -geometry(&b));
        assert_eq!(logistics(&a), -logistics(&b));
    }
}
