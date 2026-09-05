//! Exhaustively compare Cataclysm's prepared transactions with the shared rules
//! on every distinct saved FEN, without modifying either database.
use push_chess::candidates::cataclysm::verify_rules;
use push_chess::core::position::Position;
use push_chess::core::types::{Color, PieceType};
use rusqlite::{Connection, OpenFlags};
use std::collections::HashSet;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let paths: Vec<_> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        return Err("usage: verify_history <database> [database ...]".into());
    }
    let mut seen = HashSet::new();
    let mut positions = 0;
    let mut moves = 0;
    let mut malformed = 0;
    for path in paths {
        let db = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        let mut query =
            db.prepare("SELECT fen_before FROM moves UNION SELECT fen_after FROM moves")?;
        for fen in query.query_map([], |row| row.get::<_, String>(0))? {
            let fen = fen?;
            if !seen.insert(fen.clone()) {
                continue;
            }
            let mut pos = Position::empty();
            pos.set_from_fen(&fen);
            if [Color::White, Color::Black].into_iter().any(|c| {
                pos.board
                    .iter()
                    .filter(|p| p.piece_type == PieceType::King && p.color == c)
                    .count()
                    != 1
            }) {
                malformed += 1;
                continue;
            }
            moves += verify_rules(&pos)?;
            positions += 1;
            if positions % 5000 == 0 {
                println!("Verified {positions} positions, {moves} transactions");
            }
        }
    }
    println!(
        "PASS: {positions} distinct FENs, {moves} transactions; {malformed} malformed historical king layouts skipped."
    );
    Ok(())
}
