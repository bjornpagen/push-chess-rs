//! Fixed-node behavioral controls for mechanism-only refactors. These are
//! not strength tests. Change a fingerprint only for an intentional strategy
//! change, never to make a purported behavior-preserving optimization pass.
use push_chess::core::{position::Position, types::*};
use push_chess::engines::ENGINE_REGISTRY;
use push_chess::selfplay::State;

fn mix(hash: &mut u64, word: u64) {
    for byte in word.to_le_bytes() {
        *hash = (*hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
    }
}

#[test]
fn fixed_node_search_fingerprints() {
    let mut positions: Vec<_> = [
        "8/7k/4RB2/8/4N3/8/8/K7 w - - 0 1",
        "7k/P7/R7/8/8/8/8/K7 w - - 0 1",
        "r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1",
        "7k/8/8/8/3p4/8/4P3/K7 b - - 0 1",
    ]
    .map(|fen| Position::try_from_fen(fen).unwrap_or_else(|error| panic!("{fen}: {error}")))
    .into();
    let mut state = State::default();
    for ply in 0..64 {
        if state.white_value().is_some() {
            state = State::default();
        }
        if ply % 9 == 0 {
            positions.push(state.position().clone());
        }
        let legal = state.legal_moves();
        state
            .play(legal[(ply * 29 + 5) % legal.len()].id())
            .unwrap();
    }
    let mut fingerprints = Vec::new();
    for entry in ENGINE_REGISTRY {
        let mut engine = (entry.create)();
        let mut hash = 0xcbf29ce484222325;
        for (index, position) in positions.iter().enumerate() {
            engine.new_game(position.side_to_move, index as u64);
            // Also exercise a retained TT/history on the same root.
            for _ in 0..2 {
                let mut pos = position.clone();
                let (mv, stats) = engine.choose_move(
                    &mut pos,
                    &SearchBudget {
                        max_nodes: 2048,
                        ..SearchBudget::default()
                    },
                );
                for word in [
                    u64::from(mv.id()),
                    stats.nodes,
                    stats.depth_reached as u64,
                    stats.seldepth as u64,
                    stats.eval_cp as u64,
                    stats.diagnostics.qnodes,
                    stats.diagnostics.tt_hits,
                    stats.diagnostics.proof_nodes.unwrap_or(u64::MAX),
                    stats
                        .diagnostics
                        .mate_proof_plies
                        .map(u64::from)
                        .unwrap_or(u64::MAX),
                    stats.pv.len() as u64,
                ] {
                    mix(&mut hash, word);
                }
                for step in stats.pv {
                    mix(&mut hash, u64::from(step.id()));
                }
                assert_eq!(pos.to_fen(), position.to_fen());
                assert_eq!(pos.zobrist, position.zobrist);
                assert_eq!(pos.undo_stack.len(), position.undo_stack.len());
            }
        }
        fingerprints.push((entry.name, hash));
    }
    assert_eq!(
        fingerprints,
        vec![
            ("cataclysm", 6497933308806905378),
            ("abacus", 18076246204620320795),
            ("resonance", 9429648648516425643),
            ("perimeter", 16263288030244163530),
            ("convoy", 12329239644556675728),
            ("kinetic", 13552763403531440284),
            ("flashpoint", 9277680671124538036),
            ("synthesis", 11513562103666673480),
            ("astra", 15877867728975295300),
            ("sentinel", 15877867728975295300),
            ("bastion", 17631093111658239132),
            ("outrider", 10543874249971508930),
            ("tactician", 7540476717129506322),
            ("bedrock", 18143612453828631594),
            ("meridian", 10636715918183943797),
            ("waypoint", 3445330567361932315),
        ]
    );
}
