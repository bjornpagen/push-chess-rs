use super::children::{LendingIterator, PseudoLegalChildren};
/// Legal and pseudo-legal move generation for Push Chess,
/// ported 1:1 from C++ `core/movegen.cc`.
use super::position::Position;
use super::push::{PushPlan, resolve_knight_push, resolve_push};
use super::types::*;

const RAY_DR: [i32; 8] = [1, -1, 0, 0, 1, 1, -1, -1];
const RAY_DC: [i32; 8] = [0, 0, 1, -1, 1, -1, 1, -1];

const KNIGHT_DR: [i32; 8] = [-2, -2, -1, -1, 1, 1, 2, 2];
const KNIGHT_DC: [i32; 8] = [-1, 1, -2, 2, -2, 2, -1, 1];

/// Check if any displacement moves a pawn of color `us` to its promotion rank.
fn needs_promotion(pos: &Position, info: &PushPlan, us: Color) -> bool {
    let promo_rank = if us == Color::White { 7 } else { 0 };
    for i in 0..info.displacements().len() {
        let (f_sq, t_sq) = info.displacements()[i];
        let piece = pos.board[f_sq as usize];
        if piece.piece_type == PieceType::Pawn && piece.is_color(us) && rank_of(t_sq) == promo_rank
        {
            return true;
        }
    }
    false
}

/// Add a move (or 4 promotion moves) to the output list.
fn add_move(
    out: &mut Vec<Move>,
    pos: &Position,
    info: &PushPlan,
    from: Square,
    to: Square,
    path_kind: u8,
    stop_index: u8,
    special: SpecialMove,
    us: Color,
) {
    if needs_promotion(pos, info, us) {
        for promo in [
            PieceType::Queen,
            PieceType::Rook,
            PieceType::Bishop,
            PieceType::Knight,
        ] {
            out.push(Move {
                from,
                to,
                path_kind,
                stop_index,
                special: SpecialMove::Promotion,
                promo_piece: promo,
            });
        }
    } else {
        out.push(Move {
            from,
            to,
            path_kind,
            stop_index,
            special,
            promo_piece: PieceType::None,
        });
    }
}

/// Generate slider moves (bishop, rook, queen) along directions `dir_start..dir_end`.
fn gen_slider_moves(
    pos: &Position,
    from: Square,
    us: Color,
    dir_start: usize,
    dir_end: usize,
    out: &mut Vec<Move>,
) {
    for dir in dir_start..dir_end {
        let dr = RAY_DR[dir];
        let dc = RAY_DC[dir];
        let mut r = rank_of(from);
        let mut f = file_of(from);
        for stop in 0u8..7 {
            r += dr;
            f += dc;
            if !valid_rf(r, f) {
                break;
            }
            let to = make_square(r, f);

            let Some(info) = resolve_push(pos, from, to, dr, dc) else {
                break;
            };

            add_move(out, pos, &info, from, to, 0, stop, SpecialMove::None, us);

            if info.captured().is_some() {
                break;
            }
        }
    }
}

/// Generate pawn moves (forward 1, forward 2, diagonal captures, en passant).
fn gen_pawn_moves(pos: &Position, from: Square, us: Color, out: &mut Vec<Move>) {
    let dr: i32 = if us == Color::White { 1 } else { -1 };
    let start_rank: i32 = if us == Color::White { 1 } else { 6 };
    let r = rank_of(from);
    let f = file_of(from);

    // Forward 1
    {
        let nr = r + dr;
        if valid_rf(nr, f) {
            let to = make_square(nr, f);
            if pos.board[to as usize].is_empty() {
                let info = PushPlan::single(from, to, None);
                add_move(out, pos, &info, from, to, 0, 0, SpecialMove::None, us);
            } else if pos.board[to as usize].is_color(us) {
                // Push friendly piece forward
                if let Some(info) =
                    resolve_push(pos, from, to, dr, 0).filter(|p| p.captured().is_none())
                {
                    add_move(out, pos, &info, from, to, 0, 0, SpecialMove::None, us);
                }
            }
            // Pawn can't capture forward
        }
    }

    // Forward 2 from start rank
    if r == start_rank {
        let nr = r + 2 * dr;
        if valid_rf(nr, f) {
            let to = make_square(nr, f);
            if let Some(info) =
                resolve_push(pos, from, to, dr, 0).filter(|p| p.captured().is_none())
            {
                add_move(out, pos, &info, from, to, 0, 1, SpecialMove::None, us);
            }
            // Pawn can't capture with double push
        }
    }

    // Diagonal captures
    for dc in [-1i32, 1] {
        let nf = f + dc;
        let nr = r + dr;
        if !valid_rf(nr, nf) {
            continue;
        }
        let to = make_square(nr, nf);

        if !pos.board[to as usize].is_empty() && !pos.board[to as usize].is_color(us) {
            // Capture
            let info = PushPlan::single(from, to, Some(to));
            add_move(out, pos, &info, from, to, 0, 0, SpecialMove::None, us);
        }
    }

    // En passant
    if pos.ep_square < 64 {
        let ep_r = rank_of(pos.ep_square);
        let ep_f = file_of(pos.ep_square);
        if ep_r == r + dr && (ep_f == f - 1 || ep_f == f + 1) {
            let to = pos.ep_square;
            out.push(Move {
                from,
                to,
                path_kind: 0,
                stop_index: 0,
                special: SpecialMove::EnPassant,
                promo_piece: PieceType::None,
            });
        }
    }
}

/// Generate knight moves (two path decompositions per target square).
fn gen_knight_moves(pos: &Position, from: Square, us: Color, out: &mut Vec<Move>) {
    for i in 0..8 {
        let r = rank_of(from) + KNIGHT_DR[i];
        let f = file_of(from) + KNIGHT_DC[i];
        if !valid_rf(r, f) {
            continue;
        }
        let to = make_square(r, f);

        // Equal plans have the same displacements and capture, regardless of path.
        let info1 = resolve_knight_push(pos, from, to, true);
        if let Some(info) = &info1 {
            add_move(out, pos, info, from, to, 1, 0, SpecialMove::None, us);
        }
        if let Some(info) = resolve_knight_push(pos, from, to, false)
            && info1.as_ref() != Some(&info)
        {
            add_move(out, pos, &info, from, to, 2, 0, SpecialMove::None, us);
        }
    }
}

/// Generate king moves (normal + castling).
fn gen_king_moves(pos: &Position, from: Square, us: Color, out: &mut Vec<Move>) {
    let r = rank_of(from);
    let f = file_of(from);

    // Normal king moves (1 square in each direction)
    for dir in 0..8 {
        let nr = r + RAY_DR[dir];
        let nf = f + RAY_DC[dir];
        if !valid_rf(nr, nf) {
            continue;
        }
        let to = make_square(nr, nf);

        if pos.board[to as usize].is_empty() {
            out.push(Move {
                from,
                to,
                path_kind: 0,
                stop_index: 0,
                special: SpecialMove::None,
                promo_piece: PieceType::None,
            });
        } else if !pos.board[to as usize].is_color(us) {
            // Capture enemy piece
            out.push(Move {
                from,
                to,
                path_kind: 0,
                stop_index: 0,
                special: SpecialMove::None,
                promo_piece: PieceType::None,
            });
        } else {
            // Friendly piece — king pushes it along the movement direction
            if let Some(info) = resolve_push(pos, from, to, RAY_DR[dir], RAY_DC[dir]) {
                add_move(out, pos, &info, from, to, 0, 0, SpecialMove::None, us);
            }
        }
    }

    // Castling
    let castle_rank: i32 = if us == Color::White { 0 } else { 7 };
    if r != castle_rank || f != 4 {
        return;
    }

    let them = opponent(us);

    // Kingside
    let ks_flag: u8 = if us == Color::White {
        CASTLE_WK
    } else {
        CASTLE_BK
    };
    if pos.castling_rights & ks_flag != 0 {
        let f_sq = make_square(castle_rank, 5);
        let g_sq = make_square(castle_rank, 6);
        if pos.board[f_sq as usize].is_empty()
            && pos.board[g_sq as usize].is_empty()
            && !pos.is_attacked_by(from, them)
            && !pos.is_attacked_by(f_sq, them)
            && !pos.is_attacked_by(g_sq, them)
        {
            out.push(Move {
                from,
                to: g_sq,
                path_kind: 0,
                stop_index: 0,
                special: SpecialMove::Castle,
                promo_piece: PieceType::None,
            });
        }
    }

    // Queenside
    let qs_flag: u8 = if us == Color::White {
        CASTLE_WQ
    } else {
        CASTLE_BQ
    };
    if pos.castling_rights & qs_flag != 0 {
        let d_sq = make_square(castle_rank, 3);
        let c_sq = make_square(castle_rank, 2);
        let b_sq = make_square(castle_rank, 1);
        if pos.board[d_sq as usize].is_empty()
            && pos.board[c_sq as usize].is_empty()
            && pos.board[b_sq as usize].is_empty()
            && !pos.is_attacked_by(from, them)
            && !pos.is_attacked_by(d_sq, them)
            && !pos.is_attacked_by(c_sq, them)
        {
            out.push(Move {
                from,
                to: c_sq,
                path_kind: 0,
                stop_index: 0,
                special: SpecialMove::Castle,
                promo_piece: PieceType::None,
            });
        }
    }
}

/// Generate all pseudo-legal moves for the side to move.
pub fn generate_pseudo_legal_moves(pos: &Position, out: &mut Vec<Move>) {
    let us = pos.side_to_move;

    for sq in 0u8..64 {
        if !pos.board[sq as usize].is_color(us) {
            continue;
        }
        let from: Square = sq;

        match pos.board[sq as usize].piece_type {
            PieceType::Pawn => gen_pawn_moves(pos, from, us, out),
            PieceType::Knight => gen_knight_moves(pos, from, us, out),
            PieceType::Bishop => gen_slider_moves(pos, from, us, 4, 8, out),
            PieceType::Rook => gen_slider_moves(pos, from, us, 0, 4, out),
            PieceType::Queen => gen_slider_moves(pos, from, us, 0, 8, out),
            PieceType::King => gen_king_moves(pos, from, us, out),
            _ => {}
        }
    }
}

/// Generate all legal moves for the side to move.
/// Uses make/unmake on the mutable position to test legality.
pub fn generate_legal_moves(pos: &mut Position, out: &mut Vec<Move>) {
    let us = pos.side_to_move;
    let mut children = PseudoLegalChildren::new(pos);
    while let Some(child) = children.next() {
        if !child.in_check_color(us) {
            out.push(child.mv());
        }
    }
}
