use push_chess::core::{movegen::generate_legal_moves, position::Position, types::*};

#[test]
fn castles_use_occupied_transit_and_preserve_both_knight_routes() {
    let cases = [
        ("k7/8/8/8/8/8/7n/4K2R w K - 0 1", false),
        ("7k/8/8/8/8/8/1n6/R3K3 w Q - 0 1", false),
        ("4k2r/7N/8/8/8/8/8/K7 b k - 0 1", false),
        ("r3k3/1N6/8/8/8/8/8/7K b q - 0 1", false),
        ("k7/8/8/8/8/8/8/4K2R w K - 0 1", true),
        ("7k/8/8/8/8/8/8/R3K3 w Q - 0 1", true),
        ("4k2r/8/8/8/8/8/8/K7 b k - 0 1", true),
        ("r3k3/8/8/8/8/8/8/7K b q - 0 1", true),
        // Both routes blocked: geometric L-shaped reach alone is not check.
        ("k7/8/8/8/8/8/6Pn/4K2R w K - 0 1", true),
        ("7k/8/8/8/8/8/4Pn2/R3KP2 w Q - 0 1", true),
        ("4k2r/6pN/8/8/8/8/8/K7 b k - 0 1", true),
        ("r3kp2/4pN2/8/8/8/8/8/7K b q - 0 1", true),
        // Vacating e1 opens d2-d1-e1-f1; e2 blocks the other route.
        ("k7/8/8/8/8/8/3nP3/4K2R w K - 0 1", false),
        // Final safety includes the relocated rook on f1, not empty g1.
        ("k7/8/8/8/8/8/4nP2/4K2R w K - 0 1", true),
    ];
    for (fen, allowed) in cases {
        let mut current = Position::try_from_fen(fen).unwrap();
        let original = current.clone();
        let mut moves = Vec::new();
        generate_legal_moves(&mut current, &mut moves);
        assert_eq!(
            moves.iter().any(|m| m.special == SpecialMove::Castle),
            allowed,
            "{fen}"
        );
        assert_eq!(current.to_fen(), original.to_fen());
        assert_eq!(current.zobrist, original.zobrist);
        // Specialized neural and core transactions must agree on legal play.
        push_chess::engines::cataclysm::verify_rules(&current).unwrap();
    }
}

#[test]
fn current_hash_survives_copy_make_unmake() {
    let fen = "r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1";
    let pos = Position::try_from_fen(fen).unwrap();
    assert_eq!(pos.zobrist, pos.without_history().zobrist);
    let mut view = pos.clone();
    let mut moves = Vec::new();
    generate_legal_moves(&mut view, &mut moves);
    for mv in moves {
        view.make_move(&mv);
        let hash = view.zobrist;
        view.compute_zobrist();
        assert_eq!(view.zobrist, hash);
        view.unmake_move();
        assert_eq!(view.zobrist, pos.zobrist);
    }
}
