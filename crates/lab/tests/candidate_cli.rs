//! Complete CLI path, in an isolated temporary bumbledb, never a live corpus.
use serde_json::Value;
use std::process::Command;

fn run(args: &[&str]) -> Value {
    let result = Command::new(env!("CARGO_BIN_EXE_lab"))
        .args(args)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    serde_json::from_slice(&result.stdout).unwrap()
}

#[test]
fn repeated_candidate_flags_generate_normalized_games_without_replacing_controls() {
    let temp = tempfile::tempdir().unwrap();
    let db = temp.path().join("corpus");
    let model = temp.path().join("candidate.bin");
    std::fs::write(&model, push_chess::engines::cataclysm::Model::CONTROL_BYTES).unwrap();
    let db = db.to_str().unwrap();
    let first = format!("first-copy={}", model.display());
    let second = format!("second-copy={}", model.display());
    run(&["init", "--db", db]);
    let report = run(&[
        "tournament",
        "--db",
        db,
        "--engines",
        "first-copy,second-copy",
        "--nnue",
        &first,
        "--nnue",
        &second,
        "--pairs",
        "1",
        "--workers",
        "1",
        "--nodes",
        "128",
        "--max-plies",
        "8",
        "--opening-plies",
        "4",
        "--max-seconds",
        "60",
        "--purpose",
        "arena",
    ]);
    assert_eq!(report["status"], "finished");
    assert_eq!(report["matchups"][0]["a"], "first-copy");
    assert_eq!(report["matchups"][0]["b"], "second-copy");
    assert_eq!(report["matchups"][0]["saved"], 2);
    assert_eq!(
        run(&["verify", "--db", db, "--run", "1"])["verified_games"],
        2
    );
    let shadow = format!("cataclysm={}", model.display());
    let result = Command::new(env!("CARGO_BIN_EXE_lab"))
        .args([
            "tournament",
            "--db",
            db,
            "--engines",
            "all",
            "--nnue",
            &shadow,
            "--pairs",
            "1",
            "--workers",
            "1",
            "--nodes",
            "128",
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("cannot replace"));
    assert_eq!(
        run(&["summary", "--db", db])["runs"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
}
