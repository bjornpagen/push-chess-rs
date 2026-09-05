//! Bounded, reproducible self-play -> warm-start training -> promotion loop.
//! Gate games are never fed back into training. Rejected models cannot replace
//! the champion. Everything, including failed generations, stays in the run dir.
use rusqlite::{Connection, OpenFlags};
use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const TRAIN_NODES: usize = 6_000;
const GATE_NODES: usize = 20_000;
const TIME_US: usize = 1_000_000;

fn run(mut command: Command, log: &Path) -> Result<()> {
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(log)?;
    command
        .stdout(Stdio::from(file.try_clone()?))
        .stderr(Stdio::from(file));
    let status = command.status()?;
    if !status.success() {
        return Err(format!("child failed ({status}); see {}", log.display()).into());
    }
    Ok(())
}

struct MatchConfig<'a> {
    binary: &'a Path,
    db: &'a Path,
    champion: &'a Path,
    candidate: &'a Path,
    games: usize,
    nodes: usize,
    seed: u64,
    selfplay: bool,
}

fn play(config: MatchConfig<'_>, log: &Path) -> Result<()> {
    let mut cmd = Command::new(config.binary);
    cmd.args([
        "cataclysm-reference",
        if config.selfplay {
            "cataclysm-reference"
        } else {
            "cataclysm-candidate"
        },
    ]);
    cmd.arg(config.games.to_string())
        .arg(TIME_US.to_string())
        .arg(config.db);
    cmd.env("CATACLYSM_REFERENCE_MODEL", config.champion)
        .env("CATACLYSM_CANDIDATE_MODEL", config.candidate)
        .env("SHOWDOWN_NODES", config.nodes.to_string())
        .env("SHOWDOWN_OPENING_PLIES", "6")
        .env("SHOWDOWN_OPENING_SEED", config.seed.to_string())
        .env("SHOWDOWN_JOBS", "4")
        .env_remove("SHOWDOWN_VERBOSE");
    run(cmd, log)?;
    let db = Connection::open_with_flags(config.db, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let (finished,bad):(usize,usize)=db.query_row("SELECT COUNT(*),COALESCE(SUM(result='' OR termination IN ('timeout','illegal_move','error')),0) FROM games",[],|r|Ok((r.get(0)?,r.get(1)?)))?;
    if finished != config.games || bad != 0 {
        return Err(format!(
            "invalid match: {finished} game rows, {bad} unfinished/failed games in {}",
            config.db.display()
        )
        .into());
    }
    let max_nodes: usize = db.query_row("SELECT COALESCE(MAX(nodes),0) FROM search", [], |r| {
        r.get(0)
    })?;
    if max_nodes > config.nodes {
        return Err("an engine exceeded the specified node cap".into());
    }
    Ok(())
}

#[derive(Debug)]
struct Gate {
    wins: usize,
    draws: usize,
    losses: usize,
    score: f64,
    paired_lower: f64,
    promote: bool,
}

fn gate_from_scores(scores: &[f64]) -> Result<Gate> {
    if scores.len() < 8 || !scores.len().is_multiple_of(2) {
        return Err("promotion gate needs at least four complete color pairs".into());
    }
    let pair_scores: Vec<_> = scores
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| (p[0] + p[1]) / 2.)
        .collect();
    let score = pair_scores.iter().sum::<f64>() / pair_scores.len() as f64;
    let variance = pair_scores.iter().map(|p| (p - score).powi(2)).sum::<f64>()
        / (pair_scores.len() - 1) as f64;
    // An approximate one-sided 95% normal paired interval, not an Elo claim or a
    // multiple-testing-adjusted statistical guarantee. The explicit 62.5%
    // hurdle guards against promoting a negligible observed improvement.
    let paired_lower = score - 1.645 * (variance / pair_scores.len() as f64).sqrt();
    let wins = scores.iter().filter(|&&s| s == 1.).count();
    let draws = scores.iter().filter(|&&s| s == 0.5).count();
    Ok(Gate {
        wins,
        draws,
        losses: scores.len() - wins - draws,
        score,
        paired_lower,
        promote: score >= 0.625 && paired_lower > 0.5,
    })
}

fn read_gate(path: &Path) -> Result<Gate> {
    let db = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    // Verify exact paired starting positions, independently of the seed code.
    let mut opening_query = db.prepare(
        "SELECT m.game_id,m.ply,m.fen_after FROM moves m WHERE m.ply<6 ORDER BY m.game_id,m.ply",
    )?;
    let openings = opening_query
        .query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, usize>(1)?,
                r.get::<_, String>(2)?,
            ))
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut lines = std::collections::BTreeMap::<i64, Vec<String>>::new();
    for (id, _, fen) in openings {
        lines.entry(id).or_default().push(fen);
    }
    let lines: Vec<_> = lines.into_values().collect();
    for pair in lines.as_chunks::<2>().0 {
        if pair[0] != pair[1] {
            return Err("gate openings were not paired identically".into());
        }
    }
    let mut query = db.prepare("SELECT white_id,result FROM games ORDER BY game_num")?;
    let scores = query
        .query_map([], |r| {
            let white: String = r.get(0)?;
            let result: String = r.get(1)?;
            Ok(match result.as_str() {
                "1/2-1/2" => 0.5,
                "1-0" => f64::from(white == "cataclysm-candidate"),
                "0-1" => f64::from(white != "cataclysm-candidate"),
                _ => -1.,
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if scores.iter().any(|&s| s < 0.) {
        return Err("incomplete gate games".into());
    }
    gate_from_scores(&scores)
}

fn model_id(path: &Path) -> Result<String> {
    let hash = std::fs::read(path)?
        .iter()
        .fold(0xcbf29ce484222325, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
        });
    Ok(format!("{hash:016x}"))
}

fn manifest(dir: &Path, champion: &Path, history: &str, status: &str, seconds: u64) -> Result<()> {
    let text = format!(
        "{{\"status\":{status:?},\"champion\":{:?},\"champion_model\":{:?},\"selfplay_nodes\":{TRAIN_NODES},\"gate_nodes\":{GATE_NODES},\"time_cap_us\":{TIME_US},\"opening_plies\":6,\"workers\":4,\"elapsed_seconds\":{seconds},\"generations\":[{history}]}}\n",
        champion.display().to_string(),
        model_id(champion)?
    );
    std::fs::write(dir.join("manifest.json"), text)?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        return Err("usage: evolve <new-run-directory> <initial-model.bin> [generations=3] [selfplay-games=128] [gate-games=48]".into());
    }
    let generations = args
        .get(2)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(3usize);
    let training_games = args
        .get(3)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(128usize);
    let gate_games = args
        .get(4)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(48usize);
    if generations == 0 || training_games < 16 || gate_games < 8 || !gate_games.is_multiple_of(2) {
        return Err(
            "need positive generations, >=16 training games, and an even gate size >=8".into(),
        );
    }
    let initial = std::fs::canonicalize(&args[1])?;
    let dir = PathBuf::from(&args[0]);
    if let Some(parent) = dir.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::create_dir(&dir)?; // Never overwrite a previous experiment.
    let dir = std::fs::canonicalize(dir)?;
    let binaries = std::env::current_exe()?
        .parent()
        .ok_or("missing binary directory")?
        .to_path_buf();
    let showdown = binaries.join("showdown");
    let trainer = binaries.join("train_cataclysm");
    if !showdown.is_file() || !trainer.is_file() {
        return Err("build all release binaries before evolving".into());
    }
    let mut champion = dir.join("champion-000.bin");
    std::fs::copy(initial, &champion)?;
    let mut replay = Vec::new();
    let mut history = String::new();
    let start = Instant::now();
    manifest(&dir, &champion, &history, "running", 0)?;
    for generation in 1..=generations {
        let stem = format!("generation-{generation:03}");
        let selfplay = dir.join(format!("{stem}-selfplay.db"));
        let candidate = dir.join(format!("{stem}-candidate.bin"));
        let gate_db = dir.join(format!("{stem}-gate.db"));
        println!(
            "Generation {generation}/{generations}: {training_games} varied self-play games with champion {}",
            model_id(&champion)?
        );
        play(
            MatchConfig {
                binary: &showdown,
                db: &selfplay,
                champion: &champion,
                candidate: &champion,
                games: training_games,
                nodes: TRAIN_NODES,
                seed: 10_000 + generation as u64 * 1_000_000,
                selfplay: true,
            },
            &dir.join(format!("{stem}-selfplay.log")),
        )?;
        replay.push(selfplay);
        println!(
            "Generation {generation}: training on {} self-play batches; gate games excluded",
            replay.len()
        );
        let mut train = Command::new(&trainer);
        train
            .arg(&candidate)
            .arg(dir.join(format!("{stem}-training.json")))
            .args(["24", "2147483647", "--warm-start"])
            .arg(&champion)
            .args(&replay);
        run(train, &dir.join(format!("{stem}-training.log")))?;
        let id = model_id(&candidate)?;
        let same = id == model_id(&champion)?;
        let gate = if same {
            Gate {
                wins: 0,
                draws: 0,
                losses: 0,
                score: 0.5,
                paired_lower: 0.5,
                promote: false,
            }
        } else {
            println!(
                "Generation {generation}: {gate_games} held-out, color-paired promotion games"
            );
            play(
                MatchConfig {
                    binary: &showdown,
                    db: &gate_db,
                    champion: &champion,
                    candidate: &candidate,
                    games: gate_games,
                    nodes: GATE_NODES,
                    seed: (1u64 << 48) + generation as u64 * 1_000_000,
                    selfplay: false,
                },
                &dir.join(format!("{stem}-gate.log")),
            )?;
            read_gate(&gate_db)?
        };
        if gate.promote {
            let next = dir.join(format!("champion-{generation:03}.bin"));
            std::fs::copy(&candidate, &next)?;
            champion = next;
        }
        if !history.is_empty() {
            history.push(',');
        }
        write!(
            history,
            "{{\"generation\":{generation},\"candidate_model\":{id:?},\"selfplay_games\":{training_games},\"wins\":{},\"draws\":{},\"losses\":{},\"score\":{},\"paired_lower\":{},\"promoted\":{},\"unchanged\":{same}}}",
            gate.wins, gate.draws, gate.losses, gate.score, gate.paired_lower, gate.promote
        )?;
        manifest(
            &dir,
            &champion,
            &history,
            "running",
            start.elapsed().as_secs(),
        )?;
        println!(
            "Generation {generation}: {}W/{}D/{}L, {:.1}%; {}",
            gate.wins,
            gate.draws,
            gate.losses,
            gate.score * 100.,
            if gate.promote {
                "PROMOTED"
            } else {
                "champion retained"
            }
        );
    }
    manifest(
        &dir,
        &champion,
        &history,
        "finished",
        start.elapsed().as_secs(),
    )?;
    println!("Evolution complete. Champion: {}", champion.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn equal_and_losing_models_are_never_promoted() {
        assert!(!gate_from_scores(&[0.5; 48]).unwrap().promote);
        assert!(!gate_from_scores(&[0.; 48]).unwrap().promote);
        assert!(gate_from_scores(&[1.; 48]).unwrap().promote);
        let alternating: Vec<_> = (0..48).map(|i| if i % 2 == 0 { 1. } else { 0. }).collect();
        assert!(!gate_from_scores(&alternating).unwrap().promote);
    }
    #[test]
    fn unfinished_pairs_cannot_enter_the_gate() {
        assert!(gate_from_scores(&[1.; 9]).is_err());
    }
}
