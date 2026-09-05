//! Resumable, bounded training-corpus collection. No training or model promotion.
//! Each run freezes its playing executable; raw games and rejected labels remain
//! intact. The catalog records provenance, per-shard progress, and quality counts.
use push_chess::core::movegen::generate_legal_moves;
use push_chess::core::position::start_position;
use push_chess::core::types::*;
use rusqlite::{Connection, params};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const BATCH_SIZE: usize = 64;
// Half the schedule is self-play/Astra; the rest introduces other strong styles.
const POOL: [(&str, &str); 12] = [
    ("cataclysm", "cataclysm"),
    ("cataclysm", "astra"),
    ("cataclysm", "void"),
    ("cataclysm", "cataclysm"),
    ("cataclysm", "eternity"),
    ("cataclysm", "chronos"),
    ("cataclysm", "cataclysm"),
    ("cataclysm", "astra"),
    ("cataclysm", "oblivion"),
    ("cataclysm", "cataclysm"),
    ("astra", "void"),
    ("astra", "eternity"),
];

fn hash_bytes(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
    })
}

fn hash_file(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut hash = 0xcbf29ce484222325u64;
    let mut buffer = [0; 65536];
    loop {
        let n = f.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        for b in &buffer[..n] {
            hash = (hash ^ u64::from(*b)).wrapping_mul(0x100000001b3);
        }
    }
    Ok(format!("{hash:016x}"))
}

fn opening_key(fen: &str) -> String {
    fen.split_whitespace().take(4).collect::<Vec<_>>().join(" ")
}

fn split(fen: &str) -> &'static str {
    if hash_bytes(opening_key(fen).as_bytes()).is_multiple_of(5) {
        "validation"
    } else {
        "train"
    }
}

fn parse_special(s: &str) -> Result<SpecialMove> {
    Ok(match s {
        "none" => SpecialMove::None,
        "castle" => SpecialMove::Castle,
        "en_passant" => SpecialMove::EnPassant,
        "promotion" => SpecialMove::Promotion,
        _ => return Err(format!("invalid special move: {s}").into()),
    })
}

fn parse_promotion(s: &str) -> Result<PieceType> {
    Ok(match s {
        "none" => PieceType::None,
        "knight" => PieceType::Knight,
        "bishop" => PieceType::Bishop,
        "rook" => PieceType::Rook,
        "queen" => PieceType::Queen,
        _ => return Err(format!("invalid promotion: {s}").into()),
    })
}

struct Audit {
    opening: String,
    signature: String,
    searched: usize,
}

fn audit_game(db: &Connection, id: i64, opening_plies: usize) -> Result<Audit> {
    let (result, end, plies, final_fen): (String, String, usize, String) = db.query_row(
        "SELECT result,termination,ply_count,final_fen FROM games WHERE game_id=?1",
        [id],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
    )?;
    if ![
        "checkmate",
        "stalemate",
        "threefold_repetition",
        "50_move_rule",
    ]
    .contains(&end.as_str())
    {
        return Err(format!("excluded outcome: {end}").into());
    }
    let mut q = db.prepare(
        "SELECT m.ply,fen_before,fen_after,move_from,move_to,path_kind,stop_index,
                special,promo_piece,s.nodes,s.eval_cp,s.diag_json
         FROM moves m LEFT JOIN search s USING(move_id) WHERE game_id=?1 ORDER BY ply",
    )?;
    let mut rows = q.query([id])?;
    let mut pos = start_position();
    let mut history = HashMap::from([(pos.zobrist, 1usize)]);
    let mut count = 0;
    let mut searched = 0;
    let mut opening = pos.to_fen();
    let mut signature = String::new();
    while let Some(r) = rows.next()? {
        let ply: usize = r.get(0)?;
        let before: String = r.get(1)?;
        let after: String = r.get(2)?;
        if ply != count || before != pos.to_fen() {
            return Err(format!("discontinuous game at ply {ply}").into());
        }
        if pos.halfmove_clock >= 100 || history[&pos.zobrist] >= 3 {
            return Err("moves recorded after a rule-terminal position".into());
        }
        let mv = Move {
            from: r.get(3)?,
            to: r.get(4)?,
            path_kind: r.get(5)?,
            stop_index: r.get(6)?,
            special: parse_special(&r.get::<_, String>(7)?)?,
            promo_piece: parse_promotion(&r.get::<_, String>(8)?)?,
        };
        let mut legal = Vec::new();
        generate_legal_moves(&mut pos, &mut legal);
        if !legal.contains(&mv) {
            return Err(format!("illegal saved move at ply {ply}").into());
        }
        pos.make_move(&mv);
        if after != pos.to_fen() {
            return Err(format!("wrong saved board at ply {ply}").into());
        }
        let nodes: u64 = r.get(9)?;
        let _: i32 = r.get(10)?;
        let diag: String = r.get(11)?;
        let is_opening = diag == "{\"opening\":true}";
        if is_opening != (ply < opening_plies) {
            return Err("opening telemetry mismatch".into());
        }
        if !is_opening && nodes > 0 {
            searched += 1;
        }
        if ply < opening_plies {
            opening.clone_from(&after);
        }
        signature.push_str(&format!("{mv:?}:{after}\n"));
        *history.entry(pos.zobrist).or_default() += 1;
        count += 1;
    }
    if count != plies || final_fen != pos.to_fen() || searched == 0 {
        return Err("missing moves, final board, or searched positions".into());
    }
    let mut legal = Vec::new();
    generate_legal_moves(&mut pos, &mut legal);
    let expected = match end.as_str() {
        "checkmate" if legal.is_empty() && pos.in_check() => {
            if pos.side_to_move == Color::White {
                "0-1"
            } else {
                "1-0"
            }
        }
        "stalemate" if legal.is_empty() && !pos.in_check() => "1/2-1/2",
        "50_move_rule" if !legal.is_empty() && pos.halfmove_clock >= 100 => "1/2-1/2",
        "threefold_repetition" if !legal.is_empty() && history[&pos.zobrist] >= 3 => "1/2-1/2",
        _ => return Err("terminal position does not support the saved result".into()),
    };
    if result != expected {
        return Err("incorrect outcome label".into());
    }
    Ok(Audit {
        opening: opening_key(&opening),
        signature: format!("{:016x}", hash_bytes(signature.as_bytes())),
        searched,
    })
}

fn audit_shard(path: &Path, expected_games: usize, opening_plies: usize) -> Result<(usize, usize)> {
    let mut db = Connection::open(path)?;
    let check: String = db.query_row("PRAGMA quick_check", [], |r| r.get(0))?;
    if check != "ok" {
        return Err(format!("database integrity failure: {check}").into());
    }
    let ids = db
        .prepare("SELECT game_id FROM games ORDER BY game_id")?
        .query_map([], |r| r.get::<_, i64>(0))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if ids.len() != expected_games {
        return Err("wrong game count in completed batch".into());
    }
    let tx = db.transaction()?;
    tx.execute_batch(
        "CREATE TABLE corpus_audit (
        game_id INTEGER PRIMARY KEY REFERENCES games(game_id),
        accepted INTEGER NOT NULL, split TEXT NOT NULL,
        opening_key TEXT NOT NULL, signature TEXT NOT NULL,
        searched_positions INTEGER NOT NULL, reason TEXT NOT NULL
    );",
    )?;
    let (mut accepted, mut searched) = (0, 0);
    for id in ids {
        match audit_game(&tx, id, opening_plies) {
            Ok(a) => {
                tx.execute(
                    "INSERT INTO corpus_audit VALUES (?1,1,?2,?3,?4,?5,'')",
                    params![id, split(&a.opening), a.opening, a.signature, a.searched],
                )?;
                accepted += 1;
                searched += a.searched;
            }
            Err(error) => {
                tx.execute(
                    "INSERT INTO corpus_audit VALUES (?1,0,'','','',0,?2)",
                    params![id, error.to_string()],
                )?;
            }
        }
    }
    tx.commit()?;
    db.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
    Ok((accepted, searched))
}

fn metadata(db: &Connection, key: &str) -> Result<String> {
    Ok(
        db.query_row("SELECT value FROM metadata WHERE key=?1", [key], |r| {
            r.get(0)
        })?,
    )
}

fn manifest(db: &Connection, dir: &Path, state: &str) -> Result<()> {
    let json: String = db.query_row(
        "SELECT json_object('status',?1,
          'metadata',(SELECT json_group_object(key,value) FROM metadata),
          'completed_games',COALESCE(SUM(CASE WHEN status='finished' THEN games ELSE 0 END),0),
          'eligible_games',COALESCE(SUM(accepted),0),
          'searched_positions',COALESCE(SUM(searched),0),
          'batches',(SELECT json_group_array(json_object(
            'batch',id,'white_pool',engine1,'black_pool',engine2,'games',games,
            'opening_plies',opening_plies,'seed',seed,'nodes',nodes,'time_us',time_us,
            'status',status,'database',database,'accepted',accepted,'searched',searched))
            FROM batches)) FROM batches",
        [state],
        |r| r.get(0),
    )?;
    let next = dir.join("manifest.next.json");
    std::fs::write(&next, format!("{json}\n"))?;
    std::fs::rename(next, dir.join("manifest.json"))?;
    Ok(())
}

fn initialize(dir: &Path, games: usize, seed: u64, workers: usize) -> Result<Connection> {
    if games < 2 || !games.is_multiple_of(2) || workers == 0 || workers > 12 || seed >= (1 << 62) {
        return Err("need an even game count >=2, 1..12 workers, and seed <2^62".into());
    }
    let collector = std::env::current_exe()?;
    let binary = collector
        .parent()
        .ok_or("missing binary directory")?
        .join("showdown");
    if !binary.is_file() {
        return Err("build release binaries before collection".into());
    }
    std::fs::create_dir_all(dir.parent().unwrap_or(Path::new(".")))?;
    std::fs::create_dir(dir)?;
    std::fs::copy(binary, dir.join("showdown"))?;
    std::fs::copy(collector, dir.join("collect"))?;
    let mut db = Connection::open(dir.join("catalog.db"))?;
    db.execute_batch("PRAGMA journal_mode=WAL;
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE batches (
            id INTEGER PRIMARY KEY, engine1 TEXT NOT NULL, engine2 TEXT NOT NULL,
            games INTEGER NOT NULL, opening_plies INTEGER NOT NULL, seed INTEGER NOT NULL,
            nodes INTEGER NOT NULL, time_us INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
            attempt INTEGER NOT NULL DEFAULT 0, database TEXT NOT NULL DEFAULT '',
            accepted INTEGER NOT NULL DEFAULT 0, searched INTEGER NOT NULL DEFAULT 0
        );")?;
    let tx = db.transaction()?;
    let revision = Command::new("git").args(["rev-parse", "HEAD"]).output()?;
    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()?;
    for (key, value) in [
        (
            "source_commit",
            String::from_utf8_lossy(&revision.stdout).trim().to_owned(),
        ),
        ("source_dirty", (!dirty.stdout.is_empty()).to_string()),
        ("binary_fnv1a64", hash_file(&dir.join("showdown"))?),
        (
            "model_fnv1a64",
            format!(
                "{:016x}",
                hash_bytes(include_bytes!("../candidates/cataclysm/network.bin"))
            ),
        ),
        ("target_games", games.to_string()),
        ("seed", seed.to_string()),
        ("workers", workers.to_string()),
        (
            "purpose",
            "training only; no tournament or promotion-gate games".to_owned(),
        ),
        (
            "split",
            "opening-position FNV modulo five; paired colors stay together".to_owned(),
        ),
    ] {
        tx.execute("INSERT INTO metadata VALUES (?1,?2)", params![key, value])?;
    }
    for (i, offset) in (0..games).step_by(BATCH_SIZE).enumerate() {
        let (a, b) = POOL[i % POOL.len()];
        let deep = (i / POOL.len()) % 4 == 3;
        let opening_plies = [4usize, 8, 12][(i / POOL.len()) % 3];
        let batch_seed = seed
            .checked_add((i as u64).checked_mul(1 << 32).ok_or("too many batches")?)
            .ok_or("seed overflow")?;
        tx.execute(
            "INSERT INTO batches (id,engine1,engine2,games,opening_plies,seed,nodes,time_us)
            VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
            params![
                i + 1,
                a,
                b,
                (games - offset).min(BATCH_SIZE),
                opening_plies,
                batch_seed,
                if deep { 200_000 } else { 50_000 },
                if deep { 500_000 } else { 250_000 }
            ],
        )?;
    }
    tx.commit()?;
    manifest(&db, dir, "ready")?;
    Ok(db)
}

fn collect(db: &Connection, dir: &Path) -> Result<()> {
    let workers = metadata(db, "workers")?;
    let frozen = dir.join("showdown");
    if hash_file(&frozen)? != metadata(db, "binary_fnv1a64")? {
        return Err("frozen playing executable changed; refusing mixed-version corpus".into());
    }
    let total: usize = db.query_row("SELECT COUNT(*) FROM batches", [], |r| r.get(0))?;
    for id in 1..=total {
        let (a,b,games,opening,seed,nodes,time,status,attempt): (String,String,usize,usize,u64,usize,usize,String,usize) =
            db.query_row("SELECT engine1,engine2,games,opening_plies,seed,nodes,time_us,status,attempt FROM batches WHERE id=?1",[id],
                |r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?,r.get(6)?,r.get(7)?,r.get(8)?)))?;
        if status == "finished" {
            continue;
        }
        // An interrupted shard is retained, never overwritten or trained on.
        let attempt = attempt + 1;
        let stem = format!("batch-{id:04}-attempt-{attempt:03}");
        let filename = format!("{stem}.db");
        let path = dir.join(&filename);
        if path.exists() {
            return Err("attempt path already exists; refusing overwrite".into());
        }
        db.execute(
            "UPDATE batches SET status='running',attempt=?2,database=?3 WHERE id=?1",
            params![id, attempt, filename],
        )?;
        manifest(db, dir, "running")?;
        println!(
            "Batch {id}/{total}: {games} games, {a} / {b}, {nodes} nodes / {time} us, {opening} opening plies"
        );
        std::io::stdout().flush()?;
        let log = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(dir.join(format!("{stem}.log")))?;
        let result = Command::new(&frozen)
            .args([&a, &b, &games.to_string(), &time.to_string()])
            .arg(&path)
            .env("SHOWDOWN_JOBS", &workers)
            .env("SHOWDOWN_NODES", nodes.to_string())
            .env("SHOWDOWN_OPENING_PLIES", opening.to_string())
            .env("SHOWDOWN_OPENING_SEED", seed.to_string())
            .env_remove("SHOWDOWN_VERBOSE")
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log))
            .status()?;
        if !result.success() {
            return Err(format!(
                "batch {id} failed; inspect {stem}.log and resume the run after diagnosis"
            )
            .into());
        }
        let (accepted, searched) = audit_shard(&path, games, opening)?;
        db.execute(
            "UPDATE batches SET status='finished',accepted=?2,searched=?3 WHERE id=?1",
            params![id, accepted, searched],
        )?;
        manifest(db, dir, "running")?;
        println!(
            "Batch {id}: {accepted}/{games} verified outcomes; {searched} searched positions (before deduplication)"
        );
        if accepted * 2 < games {
            return Err(format!("batch {id}: more than half the outcome labels failed quality checks; investigate before continuing").into());
        }
    }
    manifest(db, dir, "finished")?;
    println!(
        "Collection complete: {}",
        dir.join("manifest.json").display()
    );
    Ok(())
}

fn main() -> Result<()> {
    let mut args: Vec<_> = std::env::args().skip(1).collect();
    let prepare = args.iter().any(|s| s == "--prepare");
    args.retain(|s| s != "--prepare");
    if args.is_empty() || args.len() > 4 {
        return Err("usage: collect <run-directory> [games=8192] [seed=2305843009233954857] [workers=6] [--prepare]; resume an existing run with its directory only".into());
    }
    let dir = PathBuf::from(&args[0]);
    let db = if dir.exists() {
        if args.len() != 1 || !dir.join("catalog.db").is_file() {
            return Err(
                "resume requires only the existing run directory; configuration is immutable"
                    .into(),
            );
        }
        Connection::open(dir.join("catalog.db"))?
    } else {
        initialize(
            &dir,
            args.get(1).map(|s| s.parse()).transpose()?.unwrap_or(8192),
            args.get(2)
                .map(|s| s.parse())
                .transpose()?
                .unwrap_or(2305843009233954857),
            args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(6),
        )?
    };
    let dir = std::fs::canonicalize(dir)?;
    // OS-released exclusive lock: a crash does not leave an unrecoverable stale lock.
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(dir.join("collector.lock"))?;
    lock.try_lock()?;
    if prepare {
        println!("Prepared frozen collection run: {}", dir.display());
        return Ok(());
    }
    let outcome = collect(&db, &dir);
    if outcome.is_err() {
        manifest(&db, &dir, "failed")?;
    }
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opening_families_ignore_counters_but_keep_side_and_rights() {
        let a = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";
        let b = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 24 30";
        assert_eq!(opening_key(a), opening_key(b));
        assert_eq!(split(a), split(b));
        assert_ne!(opening_key(a), opening_key(&a.replace(" w ", " b ")));
        assert_ne!(opening_key(a), opening_key(&a.replace(" KQkq ", " - ")));
    }

    #[test]
    fn unknown_move_encodings_are_rejected() {
        assert!(parse_special("teleport").is_err());
        assert!(parse_promotion("king").is_err());
        assert_eq!(parse_promotion("queen").unwrap(), PieceType::Queen);
    }

    #[test]
    fn truncated_games_cannot_supply_outcome_labels() {
        let db = Connection::open_in_memory().unwrap();
        db.execute_batch(
            "CREATE TABLE games (game_id, result, termination, ply_count, final_fen);
            INSERT INTO games VALUES (1,'1/2-1/2','adjudication',300,'');
            INSERT INTO games VALUES (2,'1-0','timeout',4,'');
            INSERT INTO games VALUES (3,'','',0,'');",
        )
        .unwrap();
        for id in 1..=3 {
            assert!(audit_game(&db, id, 4).is_err());
        }
    }
}
