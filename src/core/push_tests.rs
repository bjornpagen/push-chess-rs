//! Exhaust the occupancy relevant to every board-valid knight route. The
//! reference remains the full transaction resolver, not another predicate.
use super::*;
use std::hint::black_box;
use std::time::Instant;

fn full_capture(pos: &Position, from: Square, to: Square, first: bool) -> bool {
    resolve_knight_push(pos, from, to, first).is_some_and(|p| p.captured().is_some())
}

#[test]
fn capture_predicate_matches_transactions_for_all_route_occupancies() {
    for from in 0..64 {
        for to in 0..64 {
            for first in [false, true] {
                let Some(route) = knight_route(from, to, first) else {
                    continue;
                };
                let mut cells = Vec::new();
                let mut r = rank_of(from) + route.first.0;
                let mut f = file_of(from) + route.first.1;
                while valid_rf(r, f) {
                    cells.push(make_square(r, f));
                    r += route.first.0;
                    f += route.first.1;
                }
                let next = make_square(
                    rank_of(route.mid) + route.second.0,
                    file_of(route.mid) + route.second.1,
                );
                if next != to {
                    cells.push(next);
                }
                for color in [Color::White, Color::Black] {
                    let choices = [
                        Piece::default(),
                        Piece {
                            piece_type: PieceType::Pawn,
                            color,
                        },
                        Piece {
                            piece_type: PieceType::Rook,
                            color: opponent(color),
                        },
                    ];
                    let mut pos = Position::empty();
                    pos.board[from as usize] = Piece {
                        piece_type: PieceType::Knight,
                        color,
                    };
                    pos.board[to as usize] = Piece {
                        piece_type: PieceType::King,
                        color: opponent(color),
                    };
                    for code in 0usize..3usize.pow(cells.len() as u32) {
                        let mut digits = code;
                        for &sq in &cells {
                            pos.board[sq as usize] = choices[digits % 3];
                            digits /= 3;
                        }
                        assert_eq!(
                            knight_captures(&pos, from, to, first),
                            full_capture(&pos, from, to, first),
                            "from={from} to={to} first={first} color={color:?} code={code}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn capture_predicate_handles_empty_friendly_and_invalid_targets() {
    let mut pos = crate::core::position::start_position();
    for from in 0..=64 {
        for to in 0..=64 {
            for first in [false, true] {
                assert_eq!(
                    knight_captures(&pos, from, to, first),
                    full_capture(&pos, from, to, first)
                );
            }
        }
    }
    pos.board.fill(Piece::default());
    assert!(!knight_captures(&pos, 0, 17, true));
    assert!(!knight_captures(&pos, 255, 1, false));
    assert!(!knight_captures(&pos, 0, 255, true));
}

#[test]
#[ignore = "opt-in release capture measurement; run serially, preferably idle"]
fn knight_capture_probe() {
    // Real reachable occupancy, all candidate origins near each king, both
    // routes. Keep only actual enemy knights, as attack callers do.
    let mut states = crate::engine::ordering_probe::positions();
    let mut queries = Vec::new();
    for (index, pos) in states.iter().enumerate() {
        for color in [Color::White, Color::Black] {
            let target = pos.king_sq[color as usize];
            for from in 0..64 {
                if pos.board[from as usize].piece_type == PieceType::Knight
                    && pos.board[from as usize].color == opponent(color)
                    && knight_route(from, target, true).is_some()
                {
                    for first in [true, false] {
                        queries.push((index, from, target, first));
                    }
                }
            }
        }
    }
    // Also include occupied targets in dense reachable openings;
    // every fixture is still a real board, never a fabricated timing score.
    let mut state = crate::selfplay::State::default();
    for i in 0..1024 {
        if i % 16 == 0 || state.white_value().is_some() {
            state = crate::selfplay::State::default();
        }
        let pos = state.position();
        let index = states.len();
        for from in 0..64 {
            let piece = pos.board[from as usize];
            if piece.piece_type != PieceType::Knight {
                continue;
            }
            for to in 0..64 {
                if pos.board[to as usize].is_color(opponent(piece.color))
                    && knight_route(from, to, true).is_some()
                {
                    for first in [true, false] {
                        queries.push((index, from, to, first));
                    }
                }
            }
        }
        states.push(pos.clone());
        let legal = state.legal_moves();
        state.play(legal[(i * 17 + 3) % legal.len()].id()).unwrap();
    }
    measure("reachable (small repeated stream)", &states, &queries);

    // Fresh, varied occupancies exceed the small-pattern predictor regime.
    // These are adversarial board inputs, NOT claimed to be legal game states.
    // The predicate is still compared with the authoritative full resolver.
    states.clear();
    queries.clear();
    let mut random = 0x8f186924ad70de31u64;
    let mut next = || {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        random
    };
    for index in 0..32768 {
        let (from, to) = loop {
            let bits = next();
            let pair = ((bits & 63) as u8, ((bits >> 6) & 63) as u8);
            if knight_route(pair.0, pair.1, true).is_some() {
                break pair;
            }
        };
        let first = next() & 1 == 0;
        let color = if next() & 1 == 0 {
            Color::White
        } else {
            Color::Black
        };
        let mut pos = Position::empty();
        for piece in &mut pos.board {
            *piece = match next() & 7 {
                0 | 1 => Piece {
                    piece_type: PieceType::Pawn,
                    color,
                },
                2 => Piece {
                    piece_type: PieceType::Rook,
                    color: opponent(color),
                },
                _ => Piece::default(),
            };
        }
        pos.board[from as usize] = Piece {
            piece_type: PieceType::Knight,
            color,
        };
        pos.board[to as usize] = Piece {
            piece_type: PieceType::King,
            color: opponent(color),
        };
        states.push(pos);
        queries.push((index, from, to, first));
    }
    measure("varied occupancy", &states, &queries);
}

fn measure(label: &str, states: &[Position], queries: &[(usize, Square, Square, bool)]) {
    assert!(!queries.is_empty());
    for &(index, from, to, first) in queries {
        assert_eq!(
            knight_captures(&states[index], from, to, first),
            full_capture(&states[index], from, to, first)
        );
    }
    let arms = [full_capture, knight_captures, full_capture];
    let mut times: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
    for rep in 0..35 {
        for step in 0..3 {
            let arm = if rep % 2 == 0 { step } else { 2 - step };
            let start = Instant::now();
            let mut captures = 0;
            for &(index, from, to, first) in queries {
                captures += usize::from(arms[arm](black_box(&states[index]), from, to, first));
            }
            black_box(captures);
            if rep >= 4 {
                times[arm].push(start.elapsed().as_nanos() as f64 / queries.len() as f64);
            }
        }
    }
    println!("knight capture regime={label} queries={}", queries.len());
    for (label, values) in ["full", "predicate", "full_repeat"]
        .into_iter()
        .zip(&mut times)
    {
        values.sort_by(f64::total_cmp);
        println!(
            "{label}: median={:.2} ns/query p10={:.2} p90={:.2}",
            values[15], values[3], values[27]
        );
    }
}
