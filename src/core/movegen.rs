/// Legal and pseudo-legal move generation for Push Chess,
/// ported 1:1 from C++ `core/movegen.cc`.

use super::position::Position;
use super::push::{resolve_knight_push, resolve_push, PushInfo, PushResult};
use super::types::*;

const RAY_DR: [i32; 8] = [1, -1, 0, 0, 1, 1, -1, -1];
const RAY_DC: [i32; 8] = [0, 0, 1, -1, 1, -1, 1, -1];

const KNIGHT_DR: [i32; 8] = [-2, -2, -1, -1, 1, 1, 2, 2];
const KNIGHT_DC: [i32; 8] = [-1, 1, -2, 2, -2, 2, -1, 1];

/// Check if any displacement moves a pawn of color `us` to its promotion rank.
fn needs_promotion(pos: &Position, info: &PushInfo, us: Color) -> bool {
    let promo_rank = if us == Color::White { 7 } else { 0 };
    for i in 0..info.num_displacements {
        let (f_sq, t_sq) = info.displacements[i];
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
    info: &PushInfo,
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
        let mut stop: u8 = 0;

        for _dist in 1..=7 {
            r += dr;
            f += dc;
            if !valid_rf(r, f) {
                break;
            }
            let to = make_square(r, f);

            let info = resolve_push(pos, from, to, dr, dc);
            if info.result == PushResult::Illegal {
                break;
            }

            add_move(out, pos, &info, from, to, 0, stop, SpecialMove::None, us);
            stop += 1;

            if info.result == PushResult::Capture {
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
                let mut info = PushInfo::default();
                info.result = PushResult::OK;
                info.add_displacement(from, to);
                add_move(out, pos, &info, from, to, 0, 0, SpecialMove::None, us);
            } else if pos.board[to as usize].is_color(us) {
                // Push friendly piece forward
                let info = resolve_push(pos, from, to, dr, 0);
                if info.result == PushResult::OK {
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
            let info = resolve_push(pos, from, to, dr, 0);
            if info.result == PushResult::OK {
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
            let mut info = PushInfo::default();
            info.result = PushResult::Capture;
            info.add_displacement(from, to);
            info.captured_sq = to;
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

        // Try both decompositions
        let mut added_path1 = false;

        let info1 = resolve_knight_push(pos, from, to, true);
        if info1.result != PushResult::Illegal {
            add_move(out, pos, &info1, from, to, 1, 0, SpecialMove::None, us);
            added_path1 = true;
        }

        let info2 = resolve_knight_push(pos, from, to, false);
        if info2.result != PushResult::Illegal {
            // Only add path 2 if it produces a different result than path 1
            // or if path 1 was illegal
            if !added_path1 {
                add_move(out, pos, &info2, from, to, 2, 0, SpecialMove::None, us);
            } else {
                // Check if the displacements differ
                let mut same = info1.num_displacements == info2.num_displacements;
                if same {
                    for j in 0..info1.num_displacements {
                        if info1.displacements[j] != info2.displacements[j] {
                            same = false;
                            break;
                        }
                    }
                }
                if !same {
                    add_move(out, pos, &info2, from, to, 2, 0, SpecialMove::None, us);
                }
            }
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
            let info = resolve_push(pos, from, to, RAY_DR[dir], RAY_DC[dir]);
            if info.result == PushResult::OK {
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
        if pos.board[f_sq as usize].is_empty() && pos.board[g_sq as usize].is_empty() {
            if !pos.is_attacked_by(from, them)
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
        {
            if !pos.is_attacked_by(from, them)
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
    let mut pseudo = Vec::new();
    generate_pseudo_legal_moves(pos, &mut pseudo);

    let us = pos.side_to_move;

    for m in pseudo {
        pos.make_move(&m);
        if !pos.in_check_color(us) {
            out.push(m);
        }
        pos.unmake_move();
    }
}
