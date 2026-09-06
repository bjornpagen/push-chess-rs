use std::fmt;

use super::position::Position;
use super::types::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FenError(pub &'static str);

impl fmt::Display for FenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for FenError {}

impl Position {
    /// Validated public boundary. The trusted parser remains available to the
    /// historical differential tests, which intentionally use synthetic boards.
    pub fn try_from_fen(fen: &str) -> Result<Self, FenError> {
        let fail = |s| Err(FenError(s));
        if fen.len() > 128 || !fen.is_ascii() {
            return fail("FEN must be ASCII and at most 128 bytes");
        }
        let fields: Vec<_> = fen.split_whitespace().collect();
        if fields.len() != 6 {
            return fail("FEN requires six fields");
        }
        let ranks: Vec<_> = fields[0].split('/').collect();
        if ranks.len() != 8 {
            return fail("FEN requires eight ranks");
        }
        let mut kings = [0; 2];
        let mut pieces = [0; 2];
        for rank in ranks {
            let mut width = 0;
            for c in rank.bytes() {
                if (b'1'..=b'8').contains(&c) {
                    width += usize::from(c - b'0');
                } else if b"pnbrqkPNBRQK".contains(&c) {
                    let color = usize::from(c.is_ascii_lowercase());
                    pieces[color] += 1;
                    kings[color] += usize::from(c.eq_ignore_ascii_case(&b'k'));
                    width += 1;
                } else {
                    return fail("FEN contains an invalid piece");
                }
            }
            if width != 8 {
                return fail("Every FEN rank must contain eight squares");
            }
        }
        if kings != [1, 1] || pieces.iter().any(|&n| n > 16) {
            return fail("Each side needs exactly one king and at most sixteen pieces");
        }
        if !matches!(fields[1], "w" | "b") {
            return fail("Invalid side to move");
        }
        let mut seen = 0u8;
        if fields[2] != "-" {
            for c in fields[2].bytes() {
                let bit = match c {
                    b'K' => 1,
                    b'Q' => 2,
                    b'k' => 4,
                    b'q' => 8,
                    _ => return fail("Invalid castling rights"),
                };
                if seen & bit != 0 {
                    return fail("Duplicate castling right");
                }
                seen |= bit;
            }
        }
        let ep = fields[3].as_bytes();
        if fields[3] != "-"
            && (ep.len() != 2
                || !(b'a'..=b'h').contains(&ep[0])
                || ep[1] != if fields[1] == "w" { b'6' } else { b'3' })
        {
            return fail("Invalid en-passant square");
        }
        // Leave headroom for reversible search plies and bounded saved games.
        for (i, min, max) in [(4, 0, 100), (5, 1, 60_000)] {
            if fields[i].is_empty()
                || !fields[i].bytes().all(|c| c.is_ascii_digit())
                || !fields[i]
                    .parse::<u32>()
                    .is_ok_and(|v| (min..=max).contains(&v))
            {
                return fail("Invalid FEN move counters");
            }
        }
        let mut pos = Position::empty();
        pos.set_from_fen(fen);
        for (flag, king, rook, color) in [
            (1, 4, 7, Color::White),
            (2, 4, 0, Color::White),
            (4, 60, 63, Color::Black),
            (8, 60, 56, Color::Black),
        ] {
            if seen & flag != 0
                && (pos.board[king]
                    != (Piece {
                        piece_type: PieceType::King,
                        color,
                    })
                    || pos.board[rook]
                        != (Piece {
                            piece_type: PieceType::Rook,
                            color,
                        }))
            {
                return fail("Castling rights require the king and rook on their home squares");
            }
        }
        if pos.ep_square < 64 {
            let captured = (i32::from(pos.ep_square)
                + if pos.side_to_move == Color::White {
                    -8
                } else {
                    8
                }) as usize;
            if !pos.board[pos.ep_square as usize].is_empty()
                || pos.board[captured]
                    != (Piece {
                        piece_type: PieceType::Pawn,
                        color: opponent(pos.side_to_move),
                    })
            {
                return fail("En-passant requires an empty target and the opposing pawn");
            }
        }
        if pos.in_check_color(opponent(pos.side_to_move)) {
            return fail("The side that just moved cannot be in check");
        }
        Ok(pos)
    }
}
