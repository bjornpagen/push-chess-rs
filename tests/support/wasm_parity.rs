use push_chess::session::{AnalysisOptions, CataclysmSession};
use serde_json::json;

fn main() {
    let options = AnalysisOptions {
        time_ms: 0,
        max_nodes: 2000,
        max_depth: 4,
    };
    let mut fixtures = Vec::new();
    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "7k/8/8/4R3/4N3/8/8/K7 w - - 0 1",
        "7k/P7/R7/8/8/8/8/K7 w - - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "7k/8/8/3pP3/8/8/8/K7 w - d6 0 1",
    ] {
        let mut session = CataclysmSession::new(8).unwrap();
        let initial = session.reset(Some(fen)).unwrap();
        let mut turns = Vec::new();
        for _ in 0..6 {
            let before = session.snapshot();
            if before.legal_moves.is_empty() {
                break;
            }
            let analysis = session.analyse(options, before.revision).unwrap();
            let result = session.play(analysis.mv.id, before.revision).unwrap();
            turns.push(json!({ "analysis": analysis, "result": result }));
        }
        fixtures.push(json!({ "fen": fen, "initial": initial, "turns": turns }));
    }
    println!("{}", json!({ "options": options, "fixtures": fixtures }));
}
