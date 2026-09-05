use push_chess::candidates::ENGINE_REGISTRY;
use push_chess::core::children::{LendingIterator, PseudoLegalChildren};
use push_chess::core::movegen::generate_legal_moves;
use push_chess::core::position::{Position, start_position};
use push_chess::core::push::{resolve_knight_push, resolve_push};
use push_chess::core::types::*;

fn position(fen: &str) -> Position {
    let mut position = Position::default();
    position.set_from_fen(fen);
    position
}

#[test]
fn knight_first_leg_preserves_pieces_on_overlapping_squares() {
    // e4 knight pushes the e6 rook to e7; f6 bishop then moves to g6.
    let mut pos = position("7k/8/4RB2/8/4N3/8/8/K7 w - - 0 1");
    let original = pos.to_fen();
    let hash = pos.zobrist;
    let plan = resolve_knight_push(&pos, 28, 45, true).unwrap();
    assert_eq!(plan.displacements(), &[(28, 45), (44, 52), (45, 46)]);
    let mv = Move {
        from: 28,
        to: 45,
        path_kind: 1,
        ..Move::default()
    };
    pos.make_move(&mv);
    assert_eq!(pos.board[45].piece_type, PieceType::Knight);
    assert_eq!(pos.board[52].piece_type, PieceType::Rook);
    assert_eq!(pos.board[46].piece_type, PieceType::Bishop);
    assert_eq!(pos.board.iter().filter(|p| !p.is_empty()).count(), 5);
    let incremental = pos.zobrist;
    pos.compute_zobrist();
    assert_eq!(pos.zobrist, incremental);
    pos.unmake_move();
    assert_eq!(pos.to_fen(), original);
    assert_eq!(pos.zobrist, hash);
}

#[test]
fn knight_second_leg_keeps_each_cascade_piece_identity() {
    let pos = position("7k/8/5BR1/8/4N3/8/8/K7 w - - 0 1");
    let plan = resolve_knight_push(&pos, 28, 45, true).unwrap();
    assert_eq!(plan.displacements(), &[(28, 45), (45, 46), (46, 47)]);
}

#[test]
fn invalid_push_coordinates_and_directions_are_rejected() {
    let pos = start_position();
    for (from, to, dr, dc) in [
        (0, 0, 0, 0),
        (64, 1, 1, 0),
        (0, 64, 1, 0),
        (0, 8, 0, 0),
        (0, 8, 2, 0),
        (0, 10, 1, 0),
        (16, 24, 1, 0),
    ] {
        assert!(resolve_push(&pos, from, to, dr, dc).is_none());
    }
    assert!(resolve_knight_push(&pos, 1, 1, true).is_none());
    assert!(resolve_knight_push(&pos, 1, 64, false).is_none());
}

#[test]
fn borrowed_children_restore_position_on_early_exit_and_unwind() {
    let mut pos = start_position();
    let fen = pos.to_fen();
    let hash = pos.zobrist;
    {
        let mut children = PseudoLegalChildren::new(&mut pos);
        let child = children.next().unwrap();
        assert_eq!(child.side_to_move, Color::Black);
    }
    assert_eq!(pos.to_fen(), fen);
    assert_eq!(pos.zobrist, hash);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut children = PseudoLegalChildren::new(&mut pos);
        let _child = children.next().unwrap();
        panic!("exercise guard drop");
    }));
    assert!(result.is_err());
    assert_eq!(pos.to_fen(), fen);
    assert_eq!(pos.zobrist, hash);
    assert!(pos.undo_stack.is_empty());
}

#[test]
fn generated_moves_preserve_hash_piece_counts_and_round_trip() {
    for seed in 1u64..=8 {
        let mut pos = start_position();
        let mut random = seed;
        for _ in 0..40 {
            let fen = pos.to_fen();
            let hash = pos.zobrist;
            let undo_len = pos.undo_stack.len();
            let count = pos.board.iter().filter(|p| !p.is_empty()).count();
            let mut moves = Vec::new();
            generate_legal_moves(&mut pos, &mut moves);
            assert_eq!(pos.to_fen(), fen);
            assert_eq!(pos.zobrist, hash);
            assert_eq!(pos.undo_stack.len(), undo_len);
            if moves.is_empty() {
                break;
            }
            random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
            let mv = moves[random as usize % moves.len()];
            let captured = (!pos.board[mv.to as usize].is_empty()
                && pos.board[mv.to as usize].color != pos.side_to_move)
                || mv.special == SpecialMove::EnPassant;
            pos.make_move(&mv);
            assert_eq!(
                pos.board.iter().filter(|p| !p.is_empty()).count(),
                count - usize::from(captured)
            );
            let incremental = pos.zobrist;
            pos.compute_zobrist();
            assert_eq!(pos.zobrist, incremental);
            pos.unmake_move();
            assert_eq!(pos.to_fen(), fen);
            assert_eq!(pos.zobrist, hash);
            pos.make_move(&mv);
        }
    }
}

#[test]
fn every_selectable_engine_returns_legal_moves_without_changing_position() {
    // Serial on purpose: each engine allocates a large transposition table.
    for entry in ENGINE_REGISTRY {
        let mut engine = (entry.create)();
        engine.new_game(Color::White, 1);
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "7k/8/4RB2/8/4N3/8/8/K7 w - - 0 1",
        ] {
            let mut pos = position(fen);
            let original = pos.to_fen();
            let hash = pos.zobrist;
            let mut legal = Vec::new();
            generate_legal_moves(&mut pos, &mut legal);
            let budget = SearchBudget {
                max_time_us: 5_000,
                ..SearchBudget::default()
            };
            let (mv, _) = engine.choose_move(&mut pos, &budget);
            assert!(legal.contains(&mv), "{} returned {mv:?}", entry.name);
            assert_eq!(pos.to_fen(), original, "{} changed the board", entry.name);
            assert_eq!(pos.zobrist, hash, "{} changed the hash", entry.name);
            assert!(
                pos.undo_stack.is_empty(),
                "{} leaked undo records",
                entry.name
            );
        }
    }
}
