//! Opt-in whole-search evidence. Compare saved before/after test executables
//! in alternating order; do not treat a contended single process as a verdict.
use push_chess::core::rules::Rules;
use push_chess::core::{position::Position, types::SearchBudget};
use push_chess::engines::ENGINE_REGISTRY;
use push_chess::selfplay::State;
use std::hint::black_box;
use std::time::Instant;

#[test]
#[ignore = "bounded whole-search measurement; run serially, preferably idle"]
fn whole_search_cost() {
    let rules = Rules::parse(
        &std::env::var("PUSH_CHESS_PROBE_RULES").unwrap_or_else(|_| Rules::HistoryV1.name().into()),
    )
    .unwrap();
    let initial = State::default().position().to_fen();
    let name = std::env::var("PUSH_CHESS_PROBE_ENGINE").unwrap_or_else(|_| "astra".into());
    let entry = ENGINE_REGISTRY
        .iter()
        .find(|e| e.name == name)
        .expect("known engine");
    let mut positions: Vec<_> = [
        "8/7k/4RB2/8/4N3/8/8/K7 w - - 0 1",
        "7k/P7/R7/8/8/8/8/K7 w - - 0 1",
        "r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1",
        "7k/8/8/8/3p4/8/4P3/K7 b - - 0 1",
    ]
    .map(|fen| Position::try_from_fen_with_rules(fen, rules).unwrap())
    .into();
    let mut state = State::from_fen_with_rules(&initial, rules).unwrap();
    for ply in 0..128 {
        if state.white_value().is_some() {
            state = State::from_fen_with_rules(&initial, rules).unwrap();
        }
        if ply % 17 == 0 {
            positions.push(state.position().clone());
        }
        let legal = state.legal_moves();
        state
            .play(legal[(ply * 29 + 5) % legal.len()].id())
            .unwrap();
    }
    let budget = SearchBudget {
        max_nodes: 8192,
        ..SearchBudget::default()
    };
    let mut engine = (entry.create)();
    let mut samples = Vec::new();
    let mut expected = None;
    let mut expected_roots = Vec::new();
    for repetition in 0..3 {
        let (mut elapsed, mut nodes) = (0u128, 0u64);
        let mut fingerprint = 0xcbf29ce484222325u64;
        let mut mix = |word: u64| {
            for byte in word.to_le_bytes() {
                fingerprint = (fingerprint ^ u64::from(byte)).wrapping_mul(0x100000001b3);
            }
        };
        for (index, root) in positions.iter().enumerate() {
            let mut pos = root.clone();
            // Table/history reset and input copying are not timed. Search,
            // root setup, move selection and final PV extraction are timed.
            engine.new_game(pos.side_to_move, index as u64);
            let start = Instant::now();
            let (mv, stats) = engine.choose_move(black_box(&mut pos), black_box(&budget));
            elapsed += start.elapsed().as_nanos();
            nodes += stats.nodes;
            let signature = (
                mv,
                stats.nodes,
                stats.depth_reached,
                stats.seldepth,
                stats.eval_cp,
                stats.diagnostics.clone(),
                stats.pv.clone(),
            );
            if repetition == 0 {
                expected_roots.push(signature);
            } else {
                assert_eq!(
                    expected_roots[index],
                    signature,
                    "engine={name} repetition={repetition} root={index} fen={}",
                    root.to_fen()
                );
            }
            for word in [
                u64::from(mv.id()),
                stats.nodes,
                stats.eval_cp as u64,
                stats.depth_reached as u64,
                stats.seldepth as u64,
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
                mix(word);
            }
            for mv in stats.pv {
                mix(u64::from(mv.id()));
            }
            assert_eq!(pos.to_fen(), root.to_fen());
            assert_eq!(pos.zobrist, root.zobrist);
            assert_eq!(pos.undo_stack.len(), root.undo_stack.len());
        }
        if let Some(previous) = expected {
            assert_eq!(previous, (nodes, fingerprint));
        }
        expected = Some((nodes, fingerprint));
        if repetition > 0 {
            samples.push(elapsed);
        }
    }
    let (nodes, fingerprint) = expected.unwrap();
    println!(
        "SEARCH_COST engine={name} roots={} nodes={nodes} fingerprint={fingerprint} ns={}",
        positions.len(),
        samples.iter().sum::<u128>() / samples.len() as u128
    );
}
