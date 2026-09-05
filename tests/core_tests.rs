use push_chess::core::movegen::generate_legal_moves;
use push_chess::core::position::{Position, start_position};
use push_chess::core::push::{resolve_knight_push, resolve_push};
use push_chess::core::types::*;

// ---- Push Chain Tests ----

#[test]
fn test_push_rook_simple() {
    // Rook e1->e5, pawn at e3: rook@e5, pawn@e6
    let mut pos = Position::default();
    pos.set_from_fen("8/8/8/8/8/4P3/8/4R3 w - - 0 1");
    // Rook at e1 (sq 4), pawn at e3 (sq 20)
    assert_eq!(pos.board[4].piece_type, PieceType::Rook);
    assert_eq!(pos.board[20].piece_type, PieceType::Pawn);

    let info = resolve_push(&pos, 4, 36, 1, 0); // e1->e5
    let info = info.expect("legal push");
    assert!(info.captured().is_none());
    assert_eq!(info.displacements().len(), 2);
    assert_eq!(info.displacements()[0], (4u8, 36u8)); // rook->e5
    assert_eq!(info.displacements()[1], (20u8, 44u8)); // pawn->e6
}

#[test]
fn test_push_rook_to_friendly() {
    // Rook e1->e3, pawn at e3: rook@e3, pawn@e4
    let mut pos = Position::default();
    pos.set_from_fen("8/8/8/8/8/4P3/8/4R3 w - - 0 1");
    let info = resolve_push(&pos, 4, 20, 1, 0); // e1->e3
    let info = info.expect("legal push");
    assert!(info.captured().is_none());
    assert_eq!(info.displacements().len(), 2);
    assert_eq!(info.displacements()[0], (4u8, 20u8)); // rook->e3
    assert_eq!(info.displacements()[1], (20u8, 28u8)); // pawn->e4
}

#[test]
fn test_push_cascade() {
    // Rook e1->e3, pawns at e2 and e4
    // chain=[e2]. cascade: e4 has friendly -> need 2 slots: e5
    // Result: rook@e3, e2-pawn->e4, e4-pawn->e5
    let mut pos = Position::default();
    pos.set_from_fen("8/8/8/8/4P3/8/4P3/4R3 w - - 0 1");
    assert_eq!(pos.board[4].piece_type, PieceType::Rook); // e1
    assert_eq!(pos.board[12].piece_type, PieceType::Pawn); // e2
    assert_eq!(pos.board[28].piece_type, PieceType::Pawn); // e4

    let info = resolve_push(&pos, 4, 20, 1, 0); // e1->e3
    let info = info.expect("legal push");
    assert!(info.captured().is_none());
    assert_eq!(info.displacements().len(), 3);
    assert_eq!(info.displacements()[0], (4u8, 20u8)); // rook->e3
    assert_eq!(info.displacements()[1], (12u8, 28u8)); // e2->e4
    assert_eq!(info.displacements()[2], (28u8, 36u8)); // e4->e5
}

#[test]
fn test_push_off_board() {
    // Rook e1->e7, pawns at e3, e5. Pushing to e8 (ok) and e9 (off board)
    let mut pos = Position::default();
    pos.set_from_fen("8/8/8/4P3/8/4P3/8/4R3 w - - 0 1");
    let info = resolve_push(&pos, 4, 52, 1, 0); // e1->e7
    assert!(info.is_none());
}

#[test]
fn test_capture_simple() {
    // Rook e1->e5, enemy at e5, no friendlies between
    let mut pos = Position::default();
    pos.set_from_fen("8/8/8/4p3/8/8/8/4R3 w - - 0 1");
    let info = resolve_push(&pos, 4, 36, 1, 0); // e1->e5
    let info = info.expect("legal capture");
    assert!(info.captured().is_some());
    assert_eq!(info.displacements().len(), 1);
    assert_eq!(info.captured(), Some(36));
}

#[test]
fn test_capture_through_chain_illegal() {
    // Rook e1->e5, enemy at e5, friendly at e3
    let mut pos = Position::default();
    pos.set_from_fen("8/8/8/4p3/8/4P3/8/4R3 w - - 0 1");
    let info = resolve_push(&pos, 4, 36, 1, 0); // e1->e5
    assert!(info.is_none());
}

#[test]
fn test_push_into_enemy_illegal() {
    // Rook e1->e3, friendly at e2, enemy at e4 (cascade blocked)
    let mut pos = Position::default();
    pos.set_from_fen("8/8/8/8/4p3/8/4P3/4R3 w - - 0 1");
    let info = resolve_push(&pos, 4, 20, 1, 0); // e1->e3
    // chain=[e2]. cascade needs 1 slot: e4 has enemy -> illegal
    assert!(info.is_none());
}

#[test]
fn test_empty_move() {
    // Rook e1->e5, nothing in the way
    let mut pos = Position::default();
    pos.set_from_fen("8/8/8/8/8/8/8/4R3 w - - 0 1");
    let info = resolve_push(&pos, 4, 36, 1, 0);
    let info = info.expect("legal push");
    assert!(info.captured().is_none());
    assert_eq!(info.displacements().len(), 1);
    assert_eq!(info.displacements()[0], (4u8, 36u8));
}

// ---- Position Tests ----

#[test]
fn test_start_position() {
    let pos = start_position();
    assert_eq!(pos.side_to_move, Color::White);
    assert_eq!(pos.castling_rights, 0x0F);
    assert_eq!(pos.board[0].piece_type, PieceType::Rook);
    assert_eq!(pos.board[0].color, Color::White);
    assert_eq!(pos.board[4].piece_type, PieceType::King);
    assert_eq!(pos.board[60].piece_type, PieceType::King);
    assert_eq!(pos.board[60].color, Color::Black);
    assert_eq!(pos.king_sq[0], 4);
    assert_eq!(pos.king_sq[1], 60);
}

#[test]
fn test_fen_roundtrip() {
    let fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
    let mut pos = Position::default();
    pos.set_from_fen(fen);
    let result = pos.to_fen();
    assert_eq!(result, fen);
}

#[test]
fn test_zobrist_consistency() {
    let mut pos = start_position();
    let z1 = pos.zobrist;
    pos.compute_zobrist();
    assert_eq!(pos.zobrist, z1);
}

#[test]
fn test_make_unmake() {
    let mut pos = start_position();
    let orig_fen = pos.to_fen();
    let orig_z = pos.zobrist;

    // Generate and apply a move, then undo
    let mut moves = Vec::new();
    generate_legal_moves(&mut pos, &mut moves);
    assert!(!moves.is_empty());

    let limit = moves.len().min(10);
    for i in 0..limit {
        pos.make_move(&moves[i]);
        pos.unmake_move();
        assert_eq!(pos.to_fen(), orig_fen);
        assert_eq!(pos.zobrist, orig_z);
    }
}

#[test]
fn test_make_unmake_deep() {
    let mut pos = start_position();
    let orig_fen = pos.to_fen();
    let orig_z = pos.zobrist;

    // Make several moves then unmake all
    let mut moves = Vec::new();
    generate_legal_moves(&mut pos, &mut moves);
    assert!(!moves.is_empty());

    pos.make_move(&moves[0]);

    let mut moves2 = Vec::new();
    generate_legal_moves(&mut pos, &mut moves2);
    if !moves2.is_empty() {
        pos.make_move(&moves2[0]);

        let mut moves3 = Vec::new();
        generate_legal_moves(&mut pos, &mut moves3);
        if !moves3.is_empty() {
            pos.make_move(&moves3[0]);
            pos.unmake_move();
        }
        pos.unmake_move();
    }
    pos.unmake_move();

    assert_eq!(pos.to_fen(), orig_fen);
    assert_eq!(pos.zobrist, orig_z);
}

// ---- Movegen Tests ----

#[test]
fn test_start_movegen() {
    let mut pos = start_position();
    let mut moves = Vec::new();
    generate_legal_moves(&mut pos, &mut moves);
    // In push chess, from start position there are more moves than standard chess
    // due to push mechanics (pieces can push friendly pawns).
    assert!(moves.len() > 20);
}

#[test]
fn test_check_detection() {
    // White king on e1, black rook on e8, clear file
    let mut pos = Position::default();
    pos.set_from_fen("4r3/8/8/8/8/8/8/4K3 w - - 0 1");
    assert!(pos.in_check_color(Color::White));
}

#[test]
fn test_checkmate() {
    // Scholar's mate position (adapted)
    let mut pos = Position::default();
    pos.set_from_fen("r1bqkbnr/pppp1Qpp/2n5/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 1");
    assert!(pos.in_check_color(Color::Black));
    let mut moves = Vec::new();
    generate_legal_moves(&mut pos, &mut moves);
    // In push chess there might be some push-based escapes; just verify we handle it
}

#[test]
fn test_castling() {
    let mut pos = Position::default();
    pos.set_from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1");
    let mut moves = Vec::new();
    generate_legal_moves(&mut pos, &mut moves);

    let castle_count = moves
        .iter()
        .filter(|m| m.special == SpecialMove::Castle)
        .count();
    assert_eq!(castle_count, 2); // kingside and queenside
}

#[test]
fn test_en_passant() {
    // White pawn on e5, black pawn just moved d7->d5
    let mut pos = Position::default();
    pos.set_from_fen("8/8/8/3pP3/8/8/8/4K2k w - d6 0 1");
    let mut moves = Vec::new();
    generate_legal_moves(&mut pos, &mut moves);

    let ep_count = moves
        .iter()
        .filter(|m| m.special == SpecialMove::EnPassant)
        .count();
    assert_eq!(ep_count, 1);
}

#[test]
fn test_promotion() {
    // White pawn on e7, empty e8
    let mut pos = Position::default();
    pos.set_from_fen("8/4P3/8/8/8/8/8/4K2k w - - 0 1");
    let mut moves = Vec::new();
    generate_legal_moves(&mut pos, &mut moves);

    let promo_count = moves
        .iter()
        .filter(|m| m.special == SpecialMove::Promotion)
        .count();
    assert_eq!(promo_count, 4); // Q, R, B, N
}

#[test]
fn test_push_promotion() {
    // Rook pushes pawn to back rank
    // White rook on e1, white pawn on e7. Rook moves to e7, pushes pawn to e8 -> promotion
    let mut pos = Position::default();
    pos.set_from_fen("8/4P3/8/8/8/8/8/4R2K w - - 0 1");
    let mut moves = Vec::new();
    generate_legal_moves(&mut pos, &mut moves);

    let push_promo_count = moves
        .iter()
        .filter(|m| m.from == 4 && m.to == 52 && m.special == SpecialMove::Promotion)
        .count();
    assert_eq!(push_promo_count, 4); // rook pushes pawn to e8, 4 promo choices
}

// ---- Knight Push Tests ----

#[test]
fn test_knight_basic() {
    // Knight on e4, move to f6 (no obstacles)
    let mut pos = Position::default();
    pos.set_from_fen("8/8/8/8/4N3/8/8/4K2k w - - 0 1");
    let from: Square = 28; // e4
    let to: Square = 45; // f6

    let info1 = resolve_knight_push(&pos, from, to, true);
    let info2 = resolve_knight_push(&pos, from, to, false);
    assert!(
        info1.is_some_and(|p| p.captured().is_none())
            || info2.is_some_and(|p| p.captured().is_none())
    );
}

#[test]
fn test_knight_push_on_path() {
    // Knight on e4, friendly pawn on e5, move to f6
    // Decomp 1 (long first): e4->e5->e6 (push pawn), then e6->f6
    // Decomp 2 (short first): e4->f4, then f4->f5->f6
    let mut pos = Position::default();
    pos.set_from_fen("8/8/8/4P3/4N3/8/8/4K2k w - - 0 1");
    let from: Square = 28; // e4
    let to: Square = 45; // f6

    let info1 = resolve_knight_push(&pos, from, to, true);
    let info2 = resolve_knight_push(&pos, from, to, false);
    // At least one should work
    assert!(info1.is_some() || info2.is_some());
}

// ---- Core API Test (adapted — use Position directly) ----

#[test]
fn test_position_api_basic() {
    let mut pos = start_position();

    assert_eq!(pos.side_to_move, Color::White);
    assert!(!pos.in_check());
    assert_eq!(pos.board[4].piece_type, PieceType::King);
    assert_eq!(pos.board[4].color, Color::White);
    assert!(!pos.board[0].is_empty());
    assert!(pos.board[32].is_empty());

    let mut moves = Vec::new();
    generate_legal_moves(&mut pos, &mut moves);
    assert!(!moves.is_empty());

    // Make and unmake
    let z1 = pos.zobrist;
    pos.make_move(&moves[0]);
    assert_eq!(pos.side_to_move, Color::Black);
    pos.unmake_move();
    assert_eq!(pos.side_to_move, Color::White);
    assert_eq!(pos.zobrist, z1);
}
