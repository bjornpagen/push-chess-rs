use push_chess::core::position::Position;
use push_chess::core::types::*;
use push_chess::game::{Game, GameError, Outcome};
use push_chess::session::{AnalysisOptions, CataclysmSession};

fn choose(game: &Game, from: u8, to: u8) -> u32 {
    game.legal_moves()
        .iter()
        .find(|m| m.from == from && m.to == to)
        .unwrap()
        .id()
}

fn assert_animation(game: &mut Game, id: u32) {
    let before = game.snapshot();
    let preview = game.preview(id, game.revision()).unwrap();
    assert_eq!(before, game.snapshot(), "preview mutated position");
    let mut pieces = before.pieces;
    for phase in &preview.phases {
        if let Some(captured) = phase.captured {
            pieces.retain(|p| p.id != captured.id);
        }
        for d in &phase.displacements {
            let p = pieces.iter_mut().find(|p| p.id == d.piece_id).unwrap();
            assert_eq!(p.square, d.from);
            p.square = d.to;
        }
    }
    if let Some(promotion) = &preview.promotion {
        let p = pieces
            .iter_mut()
            .find(|p| p.id == promotion.piece_id)
            .unwrap();
        assert_eq!(p.square, promotion.square);
        p.kind = promotion.to;
    }
    let result = game.apply(id, game.revision()).unwrap();
    assert_eq!(preview, result.animation);
    pieces.sort_by_key(|p| p.square);
    assert_eq!(
        pieces, result.snapshot.pieces,
        "animation must reproduce the exact board and identities"
    );
}

#[test]
fn all_animation_phases_match_real_moves_and_preserve_identities() {
    for fen in [
        "7k/8/8/4R3/4N3/8/8/K7 w - - 0 1",
        "k7/8/4RB2/8/4N3/8/8/7K w - - 0 1",
        "7k/P7/R7/8/8/8/8/K7 w - - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "7k/8/8/3pP3/8/8/8/K7 w - d6 0 1",
    ] {
        let game = Game::from_fen(fen).unwrap();
        for mv in game.legal_moves() {
            assert_animation(&mut Game::from_fen(fen).unwrap(), mv.id());
        }
    }
}

#[test]
fn knight_choices_and_pushed_underpromotions_are_lossless() {
    let game = Game::from_fen("7k/8/8/4R3/4N3/8/8/K7 w - - 0 1").unwrap();
    let routes: Vec<_> = game
        .legal_moves()
        .iter()
        .filter(|m| m.from == 28 && m.to == 45)
        .collect();
    assert_eq!(routes.len(), 2);
    assert_ne!(routes[0].id(), routes[1].id());
    for mv in routes {
        assert_eq!(game.preview(mv.id(), 0).unwrap().phases.len(), 2);
    }
    let mut game = Game::from_fen("7k/P7/R7/8/8/8/8/K7 w - - 0 1").unwrap();
    let id = game
        .legal_moves()
        .iter()
        .find(|m| m.from == 40 && m.to == 48 && m.promo_piece == PieceType::Knight)
        .unwrap()
        .id();
    let preview = game.preview(id, 0).unwrap();
    assert_eq!(preview.promotion.unwrap().piece_id, 48);
    assert_animation(&mut game, id);
    let restored = Game::from_json(&serde_json::to_string(&game.save()).unwrap()).unwrap();
    assert_eq!(game.snapshot(), restored.snapshot());
    game.undo(1, 1).unwrap();
    assert_eq!(
        game.snapshot()
            .pieces
            .iter()
            .find(|p| p.id == 48)
            .unwrap()
            .kind,
        PieceType::Pawn
    );
}

#[test]
fn mutation_is_atomic_and_stale_inputs_are_rejected() {
    let mut game = Game::default();
    let before = game.snapshot();
    assert_eq!(game.apply(u32::MAX, 0), Err(GameError::IllegalMove));
    assert_eq!(game.apply(0, 99), Err(GameError::StaleRevision));
    assert_eq!(game.undo(0, 0), Err(GameError::NothingToUndo));
    assert_eq!(game.undo(1, 0), Err(GameError::NothingToUndo));
    assert_eq!(game.snapshot(), before);
    let id = choose(&game, 12, 28);
    game.apply(id, 0).unwrap();
    assert_eq!(game.apply(id, 0), Err(GameError::StaleRevision));
    game.undo(1, 1).unwrap();
    assert_eq!(game.snapshot().pieces, before.pieces);
    assert_eq!(game.snapshot().fen, before.fen);
    assert_eq!(game.apply(id, 0), Err(GameError::StaleRevision));
}

#[test]
fn draw_history_survives_save_and_undo() {
    let mut game = Game::from_fen("7k/8/8/8/8/8/8/K7 w - - 0 1").unwrap();
    for _ in 0..2 {
        for (from, to) in [(0, 1), (63, 62), (1, 0), (62, 63)] {
            game.apply(choose(&game, from, to), game.revision())
                .unwrap();
        }
    }
    assert_eq!(*game.outcome(), Outcome::Repetition);
    let save = serde_json::to_string(&game.save()).unwrap();
    assert_eq!(
        *Game::from_json(&save).unwrap().outcome(),
        Outcome::Repetition
    );
    game.undo(1, game.revision()).unwrap();
    assert_eq!(*game.outcome(), Outcome::Playing);
    for (fen, expected) in [
        (
            "7k/6Q1/5K2/8/8/8/8/8 b - - 100 1",
            Outcome::Checkmate {
                winner: Color::White,
            },
        ),
        ("7k/5Q2/5K2/8/8/8/8/8 b - - 0 1", Outcome::Stalemate),
        ("7k/8/8/8/8/8/8/K7 w - - 100 1", Outcome::FiftyMove),
    ] {
        assert_eq!(*Game::from_fen(fen).unwrap().outcome(), expected);
    }
}

#[test]
fn untrusted_fen_and_saves_never_reach_unchecked_board_operations() {
    for fen in [
        "",
        "💥",
        "8/8/8/8/8/8/8/8 w - - 0 1",
        "7k/8/8/8/8/8/8/K8 w - - 0 1",
        "7k/8/8/8/8/8/8/K7 x - - 0 1",
        "7k/8/8/8/8/8/8/K7 w K - 0 1",
        "7k/8/8/8/8/8/8/K7 w - d6 0 1",
        "7k/8/8/8/8/8/8/K7 w - - 65535 1",
        "7k/8/8/8/8/8/8/K7 w - - 0 65535",
        "7k/8/8/8/8/8/8/K7 w - - 0 1 junk",
        "7k/8/8/8/8/8/8/K6R w - - 0 1",
    ] {
        assert!(Position::try_from_fen(fen).is_err(), "{fen}");
    }
    let mut session = CataclysmSession::new(4).unwrap();
    let original = session.snapshot();
    for json in [
        "{}",
        "[]",
        "null",
        "{",
        "{\"version\":2,\"initialFen\":\"\",\"moves\":[]}",
    ] {
        assert!(session.restore(json).is_err());
        assert_eq!(session.snapshot(), original);
    }
    let mut save = session.save();
    save.moves.push(u32::MAX);
    assert!(
        session
            .restore(&serde_json::to_string(&save).unwrap())
            .is_err()
    );
    assert_eq!(session.snapshot(), original);
    for len in 0..512 {
        let s: String = (0..len)
            .map(|i| char::from(32 + ((i * 17 + len) % 95) as u8))
            .collect();
        let _ = Position::try_from_fen(&s);
    }
}

#[test]
fn random_play_previews_save_restore_and_undo_agree() {
    let mut seed = 17u64;
    for _ in 0..12 {
        let mut game = Game::default();
        for ply in 0..100 {
            if game.legal_moves().is_empty() {
                break;
            }
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let id = game.legal_moves()[(seed >> 32) as usize % game.legal_moves().len()].id();
            assert_animation(&mut game, id);
            if ply % 13 == 0 {
                let restored =
                    Game::from_json(&serde_json::to_string(&game.save()).unwrap()).unwrap();
                assert_eq!(game.snapshot(), restored.snapshot());
            }
        }
        if game.snapshot().ply > 0 {
            game.undo(game.snapshot().ply, game.revision()).unwrap();
            assert_eq!(game.snapshot().fen, Game::default().snapshot().fen);
            assert_eq!(game.snapshot().pieces, Game::default().snapshot().pieces);
        }
    }
}

#[test]
fn mobile_search_is_bounded_repeatable_and_never_mutates_the_game() {
    assert!(CataclysmSession::new(0).is_err());
    assert!(CataclysmSession::new(128).is_err());
    for hash in [4, 8, 16, 32] {
        let mut a = CataclysmSession::new(hash).unwrap();
        let mut b = CataclysmSession::new(hash).unwrap();
        let before = a.snapshot();
        let options = AnalysisOptions {
            time_ms: 0,
            max_nodes: 1000,
            max_depth: 4,
        };
        let x = a.analyse(options, 0).unwrap();
        let y = b.analyse(options, 0).unwrap();
        assert_eq!(x.mv, y.mv);
        assert_eq!(x.nodes, y.nodes);
        assert!(x.nodes <= 1000);
        assert_eq!(a.snapshot(), before);
        assert!(
            a.analyse(
                AnalysisOptions {
                    time_ms: 0,
                    max_nodes: 0,
                    max_depth: 32
                },
                0
            )
            .is_err()
        );
        assert!(
            a.analyse(
                AnalysisOptions {
                    time_ms: 5001,
                    ..options
                },
                0
            )
            .is_err()
        );
        assert!(
            a.analyse(
                AnalysisOptions {
                    max_nodes: u32::MAX,
                    ..options
                },
                0
            )
            .is_err()
        );
        assert!(a.analyse(options, 1).is_err());
    }
}

#[test]
fn moving_a_king_by_knight_push_revokes_castling_and_exports_valid_fen() {
    let mut game = Game::from_fen("k7/8/8/8/8/8/8/2N1K2R w K - 0 1").unwrap();
    push_chess::candidates::cataclysm::verify_rules(game.position()).unwrap();
    let mv = game
        .legal_moves()
        .iter()
        .find(|m| m.from == 2 && m.to == 12 && m.path_kind == 1)
        .copied()
        .unwrap();
    game.apply(mv.id(), game.revision()).unwrap();
    assert_eq!(game.position().king_sq[0], 5);
    assert_eq!(game.position().castling_rights, 0);
    Position::try_from_fen(&game.snapshot().fen).unwrap();
    push_chess::candidates::cataclysm::verify_rules(game.position()).unwrap();
}
