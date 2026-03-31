use push_chess::core::types::*;
use push_chess::core::position::{Position, start_position};
use push_chess::core::movegen::generate_legal_moves;
use push_chess::engine::Engine;
use push_chess::candidates::{find_engine, ENGINE_REGISTRY};

use rusqlite::{Connection, params};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

// ============================================================================
// Helpers — pure functions, no IO
// ============================================================================

fn color_str(c: Color) -> &'static str {
    match c {
        Color::White => "white",
        Color::Black => "black",
    }
}

fn piece_str(pt: PieceType) -> &'static str {
    match pt {
        PieceType::None => "none",
        PieceType::Pawn => "pawn",
        PieceType::Knight => "knight",
        PieceType::Bishop => "bishop",
        PieceType::Rook => "rook",
        PieceType::Queen => "queen",
        PieceType::King => "king",
    }
}

fn special_str(s: SpecialMove) -> &'static str {
    match s {
        SpecialMove::None => "none",
        SpecialMove::Castle => "castle",
        SpecialMove::EnPassant => "en_passant",
        SpecialMove::Promotion => "promotion",
    }
}

fn sq_name(sq: Square) -> String {
    let file = (b'a' + sq % 8) as char;
    let rank = (b'1' + sq / 8) as char;
    format!("{file}{rank}")
}

fn move_uci(m: &Move) -> String {
    let mut s = format!("{}{}", sq_name(m.from), sq_name(m.to));
    if m.special == SpecialMove::Promotion {
        let ch = match m.promo_piece {
            PieceType::Knight => 'n',
            PieceType::Bishop => 'b',
            PieceType::Rook => 'r',
            PieceType::Queen => 'q',
            _ => ' ',
        };
        if ch != ' ' {
            s.push(ch);
        }
    }
    s
}

fn pv_string(stats: &SearchStats) -> String {
    stats
        .pv
        .iter()
        .map(|m| move_uci(m))
        .collect::<Vec<_>>()
        .join(" ")
}

fn derive_generation(name: &str) -> i32 {
    match name {
        "chimera" => 3,
        "tempest" | "colossus" | "phantom" => 4,
        _ => 5,
    }
}

// ============================================================================
// Schema
// ============================================================================

fn create_schema(conn: &Connection) {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS candidates (
            candidate_id  TEXT PRIMARY KEY,
            source_path   TEXT NOT NULL,
            engine_path   TEXT NOT NULL,
            file_hash     TEXT NOT NULL,
            generation    INTEGER NOT NULL,
            description   TEXT NOT NULL DEFAULT '',
            created_epoch INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS tournaments (
            tournament_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            name              TEXT NOT NULL,
            budget_us         INTEGER NOT NULL,
            games_per_matchup INTEGER NOT NULL,
            started_at        TEXT NOT NULL DEFAULT (datetime('now')),
            finished_at       TEXT,
            status            TEXT NOT NULL DEFAULT 'running'
                              CHECK(status IN ('running','finished','aborted'))
        );

        CREATE TABLE IF NOT EXISTS matches (
            match_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            tournament_id INTEGER REFERENCES tournaments(tournament_id),
            budget_us     INTEGER NOT NULL,
            num_games     INTEGER NOT NULL,
            started_at    TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS match_slots (
            match_id     INTEGER NOT NULL REFERENCES matches(match_id),
            slot         INTEGER NOT NULL CHECK(slot IN (1,2)),
            candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id),
            PRIMARY KEY (match_id, slot)
        );

        CREATE TABLE IF NOT EXISTS positions (
            fen              TEXT PRIMARY KEY,
            material_balance INTEGER,
            white_pieces     INTEGER,
            black_pieces     INTEGER,
            in_check_white   INTEGER,
            in_check_black   INTEGER
        );

        CREATE TABLE IF NOT EXISTS games (
            game_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id   INTEGER NOT NULL REFERENCES matches(match_id),
            game_num   INTEGER NOT NULL,
            seed       INTEGER NOT NULL,
            white_id   TEXT NOT NULL REFERENCES candidates(candidate_id),
            black_id   TEXT NOT NULL REFERENCES candidates(candidate_id),
            result     TEXT NOT NULL DEFAULT '',
            termination TEXT NOT NULL DEFAULT '',
            ply_count  INTEGER NOT NULL DEFAULT 0,
            final_fen  TEXT NOT NULL DEFAULT '',
            wall_time_ms INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS moves (
            move_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id          INTEGER NOT NULL REFERENCES games(game_id),
            ply              INTEGER NOT NULL,
            side             TEXT NOT NULL,
            candidate_id     TEXT NOT NULL REFERENCES candidates(candidate_id),
            fen_before       TEXT NOT NULL REFERENCES positions(fen),
            fen_after        TEXT NOT NULL REFERENCES positions(fen),
            move_from        INTEGER NOT NULL,
            move_to          INTEGER NOT NULL,
            move_uci         TEXT NOT NULL,
            path_kind        INTEGER NOT NULL,
            stop_index       INTEGER NOT NULL,
            special          TEXT NOT NULL,
            promo_piece      TEXT NOT NULL,
            moving_piece     TEXT NOT NULL,
            captured_piece   TEXT NOT NULL,
            legal_move_count INTEGER NOT NULL,
            is_capture       INTEGER NOT NULL,
            is_promotion     INTEGER NOT NULL,
            is_castle        INTEGER NOT NULL,
            is_en_passant    INTEGER NOT NULL,
            is_knight_move   INTEGER NOT NULL,
            creates_promo_threat INTEGER NOT NULL,
            blocks_promo_threat  INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS search (
            move_id   INTEGER PRIMARY KEY REFERENCES moves(move_id),
            nodes     INTEGER,
            depth     INTEGER,
            seldepth  INTEGER,
            eval_cp   INTEGER,
            time_us   INTEGER,
            pv        TEXT,
            diag_json TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_moves_game ON moves(game_id, ply);
        CREATE INDEX IF NOT EXISTS idx_games_match ON games(match_id);
        CREATE INDEX IF NOT EXISTS idx_matches_tournament ON matches(tournament_id);

        DROP VIEW IF EXISTS engine_standings;
        CREATE VIEW engine_standings AS
            SELECT
                c.candidate_id,
                c.generation,
                c.description,
                COALESCE(s.total_games, 0)  AS total_games,
                COALESCE(s.wins, 0)         AS wins,
                COALESCE(s.draws, 0)        AS draws,
                COALESCE(s.losses, 0)       AS losses,
                CASE WHEN COALESCE(s.total_games, 0) > 0
                     THEN ROUND(100.0 * (COALESCE(s.wins,0) + 0.5*COALESCE(s.draws,0)) / s.total_games, 1)
                     ELSE 0.0 END AS score_pct
            FROM candidates c
            LEFT JOIN (
                SELECT candidate_id,
                       COUNT(*) AS total_games,
                       SUM(CASE WHEN won=1 THEN 1 ELSE 0 END) AS wins,
                       SUM(CASE WHEN won=-1 THEN 1 ELSE 0 END) AS losses,
                       SUM(CASE WHEN won=0 THEN 1 ELSE 0 END) AS draws
                FROM (
                    SELECT white_id AS candidate_id,
                           CASE result WHEN '1-0' THEN 1 WHEN '0-1' THEN -1 ELSE 0 END AS won
                    FROM games WHERE result != ''
                    UNION ALL
                    SELECT black_id AS candidate_id,
                           CASE result WHEN '0-1' THEN 1 WHEN '1-0' THEN -1 ELSE 0 END AS won
                    FROM games WHERE result != ''
                ) GROUP BY candidate_id
            ) s ON s.candidate_id = c.candidate_id
            ORDER BY score_pct DESC;

        DROP VIEW IF EXISTS engines_of_interest;
        CREATE VIEW engines_of_interest AS
            SELECT * FROM engine_standings
            ORDER BY score_pct DESC LIMIT 5;
        ",
    )
    .expect("failed to create schema");
}

// ============================================================================
// Candidate registration
// ============================================================================

fn register_candidate(conn: &Connection, name: &str) {
    let generation = derive_generation(name);
    conn.execute(
        "INSERT OR IGNORE INTO candidates \
         (candidate_id, source_path, engine_path, file_hash, generation, created_epoch) \
         VALUES (?1,?2,?3,?4,?5,?6)",
        params![name, name, name, "", generation, 0i64],
    )
    .expect("failed to register candidate");
}

// ============================================================================
// Position helpers
// ============================================================================

fn material_balance(pos: &Position) -> i32 {
    let mut bal = 0i32;
    for sq in 0..64usize {
        let p = pos.board[sq];
        if p.is_empty() {
            continue;
        }
        let v = pval(p.piece_type);
        if p.color == Color::White {
            bal += v;
        } else {
            bal -= v;
        }
    }
    bal
}

fn ensure_position(conn: &Connection, pos: &Position) {
    let fen = pos.to_fen();

    let exists: bool = conn
        .prepare_cached("SELECT 1 FROM positions WHERE fen=?1")
        .unwrap()
        .exists(params![fen])
        .unwrap_or(false);
    if exists {
        return;
    }

    let mat = material_balance(pos);
    let mut wp = 0i32;
    let mut bp = 0i32;
    for sq in 0..64usize {
        if pos.board[sq].is_empty() {
            continue;
        }
        if pos.board[sq].color == Color::White {
            wp += 1;
        } else {
            bp += 1;
        }
    }

    let chk_w = if pos.is_attacked_by(pos.king_sq[0], Color::Black) {
        1
    } else {
        0
    };
    let chk_b = if pos.is_attacked_by(pos.king_sq[1], Color::White) {
        1
    } else {
        0
    };

    conn.execute(
        "INSERT OR IGNORE INTO positions \
         (fen, material_balance, white_pieces, black_pieces, in_check_white, in_check_black) \
         VALUES (?1,?2,?3,?4,?5,?6)",
        params![fen, mat, wp, bp, chk_w, chk_b],
    )
    .expect("failed to insert position");
}

// ============================================================================
// Game play
// ============================================================================

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum GameResult {
    WhiteWin,
    BlackWin,
    Draw,
}

fn result_str(r: GameResult) -> &'static str {
    match r {
        GameResult::WhiteWin => "1-0",
        GameResult::BlackWin => "0-1",
        GameResult::Draw => "1/2-1/2",
    }
}

#[derive(Clone, Debug)]
struct GameOutcome {
    result: GameResult,
    termination: String,
    ply_count: i32,
    final_fen: String,
    wall_time_ms: i64,
}

impl Default for GameOutcome {
    fn default() -> Self {
        Self {
            result: GameResult::Draw,
            termination: String::new(),
            ply_count: 0,
            final_fen: String::new(),
            wall_time_ms: 0,
        }
    }
}

fn play_game(
    white_engine: &mut dyn Engine,
    white_id: &str,
    black_engine: &mut dyn Engine,
    black_id: &str,
    game_seed: u64,
    budget_us: i64,
    db_path: &str,
    game_id: i64,
    print_mu: &Mutex<()>,
) -> GameOutcome {
    let conn = Connection::open(db_path).expect("failed to open db in play_game");
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
        .ok();

    let mut pos = start_position();
    let mut out = GameOutcome::default();

    white_engine.new_game(Color::White, game_seed);
    black_engine.new_game(Color::Black, game_seed);

    let game_t0 = Instant::now();
    const MAX_PLY: i32 = 300;

    let mut zobrist_history: Vec<u64> = Vec::with_capacity(MAX_PLY as usize);
    zobrist_history.push(pos.zobrist);
    let mut halfmove_clock: i32 = 0;

    let verbose = std::env::var("SHOWDOWN_VERBOSE").is_ok();

    for ply in 0..MAX_PLY {
        let stm = pos.side_to_move;
        let cid = if stm == Color::White {
            white_id
        } else {
            black_id
        };

        let mut legal = Vec::new();
        generate_legal_moves(&mut pos, &mut legal);

        if legal.is_empty() {
            out.result = if pos.in_check() {
                if stm == Color::White {
                    GameResult::BlackWin
                } else {
                    GameResult::WhiteWin
                }
            } else {
                GameResult::Draw
            };
            out.termination = if pos.in_check() {
                "checkmate".to_string()
            } else {
                "stalemate".to_string()
            };
            out.ply_count = ply;
            break;
        }

        if halfmove_clock >= 100 {
            out.result = GameResult::Draw;
            out.termination = "50_move_rule".to_string();
            out.ply_count = ply;
            break;
        }

        // Threefold repetition check
        {
            let cur = pos.zobrist;
            let rep = zobrist_history.iter().filter(|&&z| z == cur).count();
            if rep >= 3 {
                out.result = GameResult::Draw;
                out.termination = "threefold_repetition".to_string();
                out.ply_count = ply;
                break;
            }
        }

        let fen_before = pos.to_fen();
        ensure_position(&conn, &pos);
        let move_count = legal.len() as i32;

        let mut budget = SearchBudget::default();
        budget.max_time_us = budget_us;
        budget.seed = game_seed ^ (ply as u64);

        let eng: &mut dyn Engine = if stm == Color::White {
            white_engine
        } else {
            black_engine
        };

        let t0 = Instant::now();
        let (chosen, mut stats) = eng.choose_move(&mut pos, &budget);
        let elapsed_us = t0.elapsed().as_micros() as i64;
        stats.time_used_us = elapsed_us;

        // Timeout check
        if stats.time_used_us > budget_us * 2 {
            out.result = if stm == Color::White {
                GameResult::BlackWin
            } else {
                GameResult::WhiteWin
            };
            out.termination = "timeout".to_string();
            out.ply_count = ply;
            break;
        }

        // Validate move
        let ok = legal.iter().any(|m| *m == chosen);
        if !ok {
            out.result = if stm == Color::White {
                GameResult::BlackWin
            } else {
                GameResult::WhiteWin
            };
            out.termination = "illegal_move".to_string();
            out.ply_count = ply;
            break;
        }

        let mover = pos.board[chosen.from as usize].piece_type;
        let mut captured = if pos.board[chosen.to as usize].is_empty() {
            PieceType::None
        } else {
            pos.board[chosen.to as usize].piece_type
        };
        if chosen.special == SpecialMove::EnPassant {
            captured = PieceType::Pawn;
        }

        pos.make_move(&chosen);
        zobrist_history.push(pos.zobrist);
        if captured != PieceType::None || mover == PieceType::Pawn {
            halfmove_clock = 0;
        } else {
            halfmove_clock += 1;
        }

        let fen_after = pos.to_fen();
        ensure_position(&conn, &pos);
        let check_after = pos.in_check();

        let uci = move_uci(&chosen);

        conn.execute(
            "INSERT INTO moves (game_id, ply, side, candidate_id, fen_before, fen_after, \
             move_from, move_to, move_uci, path_kind, stop_index, special, promo_piece, \
             moving_piece, captured_piece, legal_move_count, \
             is_capture, is_promotion, is_castle, is_en_passant, is_knight_move, \
             creates_promo_threat, blocks_promo_threat) \
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21,?22,?23)",
            params![
                game_id,
                ply,
                color_str(stm),
                cid,
                fen_before,
                fen_after,
                chosen.from as i32,
                chosen.to as i32,
                uci,
                chosen.path_kind as i32,
                chosen.stop_index as i32,
                special_str(chosen.special),
                piece_str(chosen.promo_piece),
                piece_str(mover),
                piece_str(captured),
                move_count,
                if captured != PieceType::None { 1 } else { 0 },
                if chosen.special == SpecialMove::Promotion { 1 } else { 0 },
                if chosen.special == SpecialMove::Castle { 1 } else { 0 },
                if chosen.special == SpecialMove::EnPassant { 1 } else { 0 },
                if chosen.path_kind > 0 { 1 } else { 0 },
                0i32, // creates_promo_threat (not implemented in Rust crate)
                0i32, // blocks_promo_threat (not implemented in Rust crate)
            ],
        )
        .expect("failed to insert move");

        let mid = conn.last_insert_rowid();

        let pvs = pv_string(&stats);
        let dj = if stats.diag_json.is_empty() {
            "{}"
        } else {
            &stats.diag_json
        };

        conn.execute(
            "INSERT INTO search (move_id, nodes, depth, seldepth, eval_cp, time_us, pv, diag_json) \
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
            params![
                mid,
                stats.nodes as i64,
                stats.depth_reached as i32,
                stats.seldepth as i32,
                stats.eval_cp,
                stats.time_used_us,
                pvs,
                dj,
            ],
        )
        .expect("failed to insert search");

        if verbose {
            let _lk = print_mu.lock().unwrap();
            eprintln!(
                "  [g{} p{:3}] {:>5} {:<6} {:<6} d={} n={}k t={}ms eval={}cp{}{}{}",
                game_id,
                ply,
                color_str(stm),
                piece_str(mover),
                uci,
                stats.depth_reached,
                stats.nodes / 1000,
                stats.time_used_us / 1000,
                stats.eval_cp,
                if captured != PieceType::None {
                    " CAP"
                } else {
                    ""
                },
                if chosen.special == SpecialMove::Promotion {
                    " PROMO"
                } else {
                    ""
                },
                if check_after { " CHK" } else { "" },
            );
        }

        out.ply_count = ply + 1;
        if ply >= MAX_PLY - 1 {
            out.result = GameResult::Draw;
            out.termination = "adjudication".to_string();
        }
    }

    if out.final_fen.is_empty() {
        out.final_fen = pos.to_fen();
    }
    out.wall_time_ms = game_t0.elapsed().as_millis() as i64;
    out
}

// ============================================================================
// Elo computation (fully derived, per generation -- no storage)
// ============================================================================

struct EloEntry {
    candidate_id: String,
    generation: i32,
    elo: f64,
    games: i32,
    wins: i32,
    draws: i32,
    losses: i32,
    score_pct: f64,
    description: String,
}

fn compute_elo(
    conn: &Connection,
    generation: i32,
    engine_set: &[String],
) -> HashMap<String, f64> {
    let mut rating: HashMap<String, f64> = HashMap::new();

    if generation >= 0 {
        let mut stmt = conn
            .prepare("SELECT candidate_id FROM candidates WHERE generation=?1")
            .unwrap();
        let rows = stmt
            .query_map(params![generation], |row| row.get::<_, String>(0))
            .unwrap();
        for id in rows.flatten() {
            rating.insert(id, 1500.0);
        }
    } else {
        for id in engine_set {
            rating.insert(id.clone(), 1500.0);
        }
    }

    if rating.len() < 2 {
        return rating;
    }

    // Build parameterized IN clause
    let ids: Vec<String> = rating.keys().cloned().collect();
    let placeholders: String = (0..ids.len())
        .map(|i| format!("?{}", i + 1))
        .collect::<Vec<_>>()
        .join(",");

    // Bind twice — once for white_id IN, once for black_id IN
    // We need to use dynamic params since the count varies
    let n = ids.len();
    let mut all_params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::with_capacity(n * 2);
    for id in &ids {
        all_params.push(Box::new(id.clone()));
    }
    for id in &ids {
        all_params.push(Box::new(id.clone()));
    }

    // Build the SQL with correct parameter numbering for the second IN clause
    let placeholders2: String = (0..ids.len())
        .map(|i| format!("?{}", n + i + 1))
        .collect::<Vec<_>>()
        .join(",");

    let sql = format!(
        "SELECT g.white_id, g.black_id, g.result FROM games g \
         WHERE g.result != '' \
         AND g.white_id IN ({}) \
         AND g.black_id IN ({}) \
         ORDER BY g.game_id",
        placeholders, placeholders2
    );

    let param_refs: Vec<&dyn rusqlite::types::ToSql> =
        all_params.iter().map(|p| p.as_ref()).collect();

    let mut stmt = conn.prepare(&sql).unwrap();
    let mut rows = stmt.query(param_refs.as_slice()).unwrap();

    while let Some(row) = rows.next().unwrap() {
        let wid: String = row.get(0).unwrap();
        let bid: String = row.get(1).unwrap();
        let res: String = row.get(2).unwrap();

        let (sw, sb) = match res.as_str() {
            "1-0" => (1.0, 0.0),
            "0-1" => (0.0, 1.0),
            _ => (0.5, 0.5),
        };

        let rw = *rating.get(&wid).unwrap_or(&1500.0);
        let rb = *rating.get(&bid).unwrap_or(&1500.0);
        let ea = 1.0 / (1.0 + 10.0_f64.powf((rb - rw) / 400.0));
        let eb = 1.0 - ea;
        *rating.get_mut(&wid).unwrap() += 32.0 * (sw - ea);
        *rating.get_mut(&bid).unwrap() += 32.0 * (sb - eb);
    }

    rating
}

// ============================================================================
// Standings
// ============================================================================

fn print_standings(conn: &Connection) {
    let mut stmt = conn
        .prepare("SELECT DISTINCT generation FROM candidates ORDER BY generation")
        .unwrap();
    let gens: Vec<i32> = stmt
        .query_map([], |row| row.get(0))
        .unwrap()
        .flatten()
        .collect();

    if gens.is_empty() {
        println!("\nNo engines registered.\n");
        return;
    }

    for generation in &gens {
        let elo_map = compute_elo(conn, *generation, &[]);

        let mut entries: Vec<EloEntry> = Vec::new();
        let mut stmt = conn
            .prepare(
                "SELECT candidate_id, generation, total_games, wins, draws, losses, score_pct, description \
                 FROM engine_standings WHERE generation=?1",
            )
            .unwrap();
        let rows = stmt
            .query_map(params![generation], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i32>(1)?,
                    row.get::<_, i32>(2)?,
                    row.get::<_, i32>(3)?,
                    row.get::<_, i32>(4)?,
                    row.get::<_, i32>(5)?,
                    row.get::<_, f64>(6)?,
                    row.get::<_, String>(7).unwrap_or_default(),
                ))
            })
            .unwrap();

        for r in rows.flatten() {
            let elo = elo_map.get(&r.0).copied().unwrap_or(1500.0);
            entries.push(EloEntry {
                candidate_id: r.0,
                generation: r.1,
                elo,
                games: r.2,
                wins: r.3,
                draws: r.4,
                losses: r.5,
                score_pct: r.6,
                description: r.7,
            });
        }

        entries.sort_by(|a, b| b.elo.partial_cmp(&a.elo).unwrap());

        println!("\n=== Generation {} ===", generation);
        println!(
            "{:<12}  {:>7}  {:>5}  {:>4}  {:>4}  {:>4}  {:>6}  {}",
            "Engine", "Elo", "Games", "W", "D", "L", "Score", "Description"
        );
        println!(
            "{:<12}  {:>7}  {:>5}  {:>4}  {:>4}  {:>4}  {:>6}  {}",
            "------------",
            "-------",
            "-----",
            "----",
            "----",
            "----",
            "------",
            "-----------"
        );

        for e in &entries {
            println!(
                "{:<12}  {:>7.1}  {:>5}  {:>4}  {:>4}  {:>4}  {:>5.1}%  {}",
                e.candidate_id,
                e.elo,
                e.games,
                e.wins,
                e.draws,
                e.losses,
                e.score_pct,
                e.description
            );
        }
    }
    println!();
}

// ============================================================================
// Tournament
// ============================================================================

#[allow(dead_code)]
struct TourneyGame {
    matchup_idx: i32,
    game_num: i32,
    match_id: i64,
    game_id: i64,
    seed: u64,
    #[allow(dead_code)]
    swap: bool,
    wid: String,
    bid: String,
}

fn run_tournament(
    name: &str,
    budget_us: i64,
    games_per: i32,
    engines: &[String],
    db_path: &str,
) -> i32 {
    let conn = Connection::open(db_path).expect("failed to open DB");
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
        .ok();
    create_schema(&conn);

    // Validate and register engines
    let mut ids: Vec<String> = Vec::new();
    for eng in engines {
        if find_engine(eng).is_none() {
            eprintln!("Failed to find engine: {}", eng);
            return 1;
        }
        ids.push(eng.clone());
        register_candidate(&conn, eng);
    }

    let n = ids.len();
    let pairings = n * (n - 1) / 2;
    let total_games = pairings * (games_per as usize);

    conn.execute(
        "INSERT INTO tournaments (name, budget_us, games_per_matchup) VALUES (?1,?2,?3)",
        params![name, budget_us, games_per],
    )
    .expect("failed to insert tournament");
    let tid = conn.last_insert_rowid();

    println!("=== TOURNAMENT \"{}\" (#{}) ===", name, tid);
    let max_concurrent = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);
    println!(
        "  {} engines, {} pairings x {} games = {} total games ({} concurrent)",
        n, pairings, games_per, total_games, max_concurrent
    );
    println!(
        "  Budget: {}ms | DB: {}\n",
        budget_us / 1000,
        db_path
    );

    // Pre-create all matches and game rows
    let mut all_games: Vec<TourneyGame> = Vec::with_capacity(total_games);

    let mut matchup = 0i32;
    for i in 0..n {
        for j in (i + 1)..n {
            matchup += 1;

            conn.execute(
                "INSERT INTO matches (tournament_id, budget_us, num_games) VALUES (?1,?2,?3)",
                params![tid, budget_us, games_per],
            )
            .expect("failed to insert match");
            let mid = conn.last_insert_rowid();

            conn.execute(
                "INSERT INTO match_slots (match_id, slot, candidate_id) VALUES (?1,1,?2)",
                params![mid, ids[i]],
            )
            .expect("failed to insert match_slot 1");
            conn.execute(
                "INSERT INTO match_slots (match_id, slot, candidate_id) VALUES (?1,2,?2)",
                params![mid, ids[j]],
            )
            .expect("failed to insert match_slot 2");

            for g in 0..games_per {
                let seed = ((matchup as u64) * 1000 + (g as u64))
                    .wrapping_mul(6364136223846793005u64)
                    .wrapping_add(1442695040888963407u64);
                let swap = g % 2 == 1;
                let wid = if swap {
                    ids[j].clone()
                } else {
                    ids[i].clone()
                };
                let bid = if swap {
                    ids[i].clone()
                } else {
                    ids[j].clone()
                };

                conn.execute(
                    "INSERT INTO games (match_id, game_num, seed, white_id, black_id) VALUES (?1,?2,?3,?4,?5)",
                    params![mid, g + 1, seed as i64, wid, bid],
                )
                .expect("failed to insert game");
                let gid = conn.last_insert_rowid();

                all_games.push(TourneyGame {
                    matchup_idx: matchup,
                    game_num: g + 1,
                    match_id: mid,
                    game_id: gid,
                    seed,
                    swap,
                    wid,
                    bid,
                });
            }
        }
    }

    // Run all games via bounded worker pool
    let print_mu = Mutex::new(());
    let completed = AtomicUsize::new(0);
    let next_idx = AtomicUsize::new(0);
    let total = all_games.len();

    // We need to share all_games across threads — use a Vec of Mutex<Option<GameOutcome>>
    // to store results, and index into the read-only game info.
    struct GameInfo {
        game_id: i64,
        seed: u64,
        wid: String,
        bid: String,
    }

    let game_infos: Vec<GameInfo> = all_games
        .iter()
        .map(|tg| GameInfo {
            game_id: tg.game_id,
            seed: tg.seed,
            wid: tg.wid.clone(),
            bid: tg.bid.clone(),
        })
        .collect();

    let results: Vec<Mutex<Option<GameOutcome>>> =
        (0..total).map(|_| Mutex::new(None)).collect();

    std::thread::scope(|s| {
        let num_workers = max_concurrent.min(total);
        for _ in 0..num_workers {
            s.spawn(|| {
                loop {
                    let idx = next_idx.fetch_add(1, Ordering::Relaxed);
                    if idx >= total {
                        break;
                    }

                    let gi = &game_infos[idx];

                    let we = find_engine(&gi.wid).expect("engine not found");
                    let be = find_engine(&gi.bid).expect("engine not found");
                    let mut white_eng = (we.create)();
                    let mut black_eng = (be.create)();

                    let outcome = play_game(
                        white_eng.as_mut(),
                        &gi.wid,
                        black_eng.as_mut(),
                        &gi.bid,
                        gi.seed,
                        budget_us,
                        db_path,
                        gi.game_id,
                        &print_mu,
                    );

                    // Update game result in DB
                    let gdb =
                        Connection::open(db_path).expect("failed to open db for result update");
                    gdb.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
                        .ok();
                    gdb.execute(
                        "UPDATE games SET result=?1, termination=?2, ply_count=?3, \
                         final_fen=?4, wall_time_ms=?5 WHERE game_id=?6",
                        params![
                            result_str(outcome.result),
                            outcome.termination,
                            outcome.ply_count,
                            outcome.final_fen,
                            outcome.wall_time_ms,
                            gi.game_id,
                        ],
                    )
                    .expect("failed to update game result");

                    *results[idx].lock().unwrap() = Some(outcome);

                    let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                    {
                        let _lk = print_mu.lock().unwrap();
                        eprint!("\r  {}/{}", done, total);
                    }
                }
            });
        }
    });

    conn.execute(
        "UPDATE tournaments SET status='finished', finished_at=datetime('now') WHERE tournament_id=?1",
        params![tid],
    )
    .expect("failed to update tournament status");

    println!("\n=== TOURNAMENT COMPLETE ===");

    // Compute Elo for tournament participants (closed universe)
    let elo_all = compute_elo(&conn, -1, &ids);
    let mut entries: Vec<EloEntry> = Vec::new();
    for id in &ids {
        let elo = elo_all.get(id).copied().unwrap_or(1500.0);

        let mut stmt = conn
            .prepare(
                "SELECT \
                 SUM(CASE WHEN (white_id=?1 AND result='1-0') OR (black_id=?1 AND result='0-1') THEN 1 ELSE 0 END), \
                 SUM(CASE WHEN result='1/2-1/2' THEN 1 ELSE 0 END), \
                 SUM(CASE WHEN (white_id=?1 AND result='0-1') OR (black_id=?1 AND result='1-0') THEN 1 ELSE 0 END), \
                 COUNT(*) \
                 FROM games WHERE (white_id=?1 OR black_id=?1) AND result<>'' \
                 AND match_id IN (SELECT match_id FROM matches WHERE tournament_id=?2)",
            )
            .unwrap();

        let row = stmt
            .query_row(params![id, tid], |row| {
                Ok((
                    row.get::<_, i32>(0).unwrap_or(0),
                    row.get::<_, i32>(1).unwrap_or(0),
                    row.get::<_, i32>(2).unwrap_or(0),
                    row.get::<_, i32>(3).unwrap_or(0),
                ))
            })
            .unwrap_or((0, 0, 0, 0));

        let score_pct = if row.3 > 0 {
            100.0 * (row.0 as f64 + 0.5 * row.1 as f64) / row.3 as f64
        } else {
            0.0
        };

        entries.push(EloEntry {
            candidate_id: id.clone(),
            generation: derive_generation(id),
            elo,
            games: row.3,
            wins: row.0,
            draws: row.1,
            losses: row.2,
            score_pct,
            description: String::new(),
        });
    }
    entries.sort_by(|a, b| b.elo.partial_cmp(&a.elo).unwrap());

    println!("\n=== TOURNAMENT ELO (all participants) ===");
    println!(
        "{:<12}  Gen  {:>7}  {:>5}  {:>4}  {:>4}  {:>4}  {:>6}",
        "Engine", "Elo", "Games", "W", "D", "L", "Score"
    );
    println!(
        "{:<12}  ---  {:>7}  {:>5}  {:>4}  {:>4}  {:>4}  {:>6}",
        "------------", "-------", "-----", "----", "----", "----", "------"
    );
    for e in &entries {
        println!(
            "{:<12}  {:>3}  {:>7.1}  {:>5}  {:>4}  {:>4}  {:>4}  {:>5.1}%",
            e.candidate_id,
            e.generation,
            e.elo,
            e.games,
            e.wins,
            e.draws,
            e.losses,
            e.score_pct
        );
    }
    println!();

    0
}

// ============================================================================
// Standalone match mode
// ============================================================================

fn run_standalone_match(
    name1: &str,
    name2: &str,
    num_games: i32,
    budget_us: i64,
    db_path: &str,
) -> i32 {
    // Validate engines exist
    if find_engine(name1).is_none() {
        eprintln!("Failed to find engine: {}", name1);
        return 1;
    }
    if find_engine(name2).is_none() {
        eprintln!("Failed to find engine: {}", name2);
        return 1;
    }

    let conn = Connection::open(db_path).expect("failed to open DB");
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
        .ok();
    create_schema(&conn);

    register_candidate(&conn, name1);
    register_candidate(&conn, name2);

    conn.execute(
        "INSERT INTO matches (tournament_id, budget_us, num_games) VALUES (NULL,?1,?2)",
        params![budget_us, num_games],
    )
    .expect("failed to insert match");
    let match_id = conn.last_insert_rowid();

    conn.execute(
        "INSERT INTO match_slots (match_id, slot, candidate_id) VALUES (?1,1,?2)",
        params![match_id, name1],
    )
    .expect("failed to insert match_slot 1");
    conn.execute(
        "INSERT INTO match_slots (match_id, slot, candidate_id) VALUES (?1,2,?2)",
        params![match_id, name2],
    )
    .expect("failed to insert match_slot 2");

    println!("=== SHOWDOWN #{} ===", match_id);
    println!(
        "  {} vs {} | {} games | {}ms budget | {}\n",
        name1,
        name2,
        num_games,
        budget_us / 1000,
        db_path
    );

    // Pre-create game rows
    #[allow(dead_code)]
    struct GameSlot {
        game_num: i32,
        seed: u64,
        swap: bool,
        wid: String,
        bid: String,
        gid: i64,
    }

    let mut slots: Vec<GameSlot> = Vec::with_capacity(num_games as usize);
    for g in 0..num_games {
        let seed = (g as u64)
            .wrapping_mul(6364136223846793005u64)
            .wrapping_add(1442695040888963407u64);
        let swap = g % 2 == 1;
        let wid = if swap {
            name2.to_string()
        } else {
            name1.to_string()
        };
        let bid = if swap {
            name1.to_string()
        } else {
            name2.to_string()
        };

        conn.execute(
            "INSERT INTO games (match_id, game_num, seed, white_id, black_id) VALUES (?1,?2,?3,?4,?5)",
            params![match_id, g + 1, seed as i64, wid, bid],
        )
        .expect("failed to insert game");
        let gid = conn.last_insert_rowid();

        slots.push(GameSlot {
            game_num: g + 1,
            seed,
            swap,
            wid,
            bid,
            gid,
        });
    }

    // Run via bounded worker pool
    let print_mu = Mutex::new(());
    let next_idx = AtomicUsize::new(0);
    let total = slots.len();
    let max_concurrent = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);

    // Collect results
    let outcomes: Vec<Mutex<Option<(GameResult, bool)>>> =
        (0..total).map(|_| Mutex::new(None)).collect();

    std::thread::scope(|s| {
        let num_workers = max_concurrent.min(total);
        for _ in 0..num_workers {
            s.spawn(|| {
                loop {
                    let idx = next_idx.fetch_add(1, Ordering::Relaxed);
                    if idx >= total {
                        break;
                    }

                    let sl = &slots[idx];

                    let we = find_engine(&sl.wid).expect("engine not found");
                    let be = find_engine(&sl.bid).expect("engine not found");
                    let mut white_eng = (we.create)();
                    let mut black_eng = (be.create)();

                    let outcome = play_game(
                        white_eng.as_mut(),
                        &sl.wid,
                        black_eng.as_mut(),
                        &sl.bid,
                        sl.seed,
                        budget_us,
                        db_path,
                        sl.gid,
                        &print_mu,
                    );

                    // Update game result in DB
                    let gdb =
                        Connection::open(db_path).expect("failed to open db for result update");
                    gdb.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
                        .ok();
                    gdb.execute(
                        "UPDATE games SET result=?1, termination=?2, ply_count=?3, \
                         final_fen=?4, wall_time_ms=?5 WHERE game_id=?6",
                        params![
                            result_str(outcome.result),
                            outcome.termination,
                            outcome.ply_count,
                            outcome.final_fen,
                            outcome.wall_time_ms,
                            sl.gid,
                        ],
                    )
                    .expect("failed to update game result");

                    *outcomes[idx].lock().unwrap() = Some((outcome.result, sl.swap));
                }
            });
        }
    });

    let mut w1 = 0;
    let mut w2 = 0;
    let mut draws = 0;
    for o in &outcomes {
        if let Some((result, swap)) = o.lock().unwrap().as_ref() {
            match result {
                GameResult::WhiteWin => {
                    if *swap {
                        w2 += 1;
                    } else {
                        w1 += 1;
                    }
                }
                GameResult::BlackWin => {
                    if *swap {
                        w1 += 1;
                    } else {
                        w2 += 1;
                    }
                }
                GameResult::Draw => {
                    draws += 1;
                }
            }
        }
    }

    println!("=== FINAL: {} {} - {} - {} {} ===", name1, w1, draws, w2, name2);

    println!("\nDB: {}", db_path);
    {
        let mut stmt = conn
            .prepare(
                "SELECT COUNT(*) FROM moves WHERE game_id IN \
                 (SELECT game_id FROM games WHERE match_id=?1)",
            )
            .unwrap();
        let count: i32 = stmt.query_row(params![match_id], |r| r.get(0)).unwrap_or(0);
        println!("  Moves: {}", count);
    }
    {
        let mut stmt = conn
            .prepare("SELECT COUNT(DISTINCT fen) FROM positions")
            .unwrap();
        let count: i32 = stmt.query_row([], |r| r.get(0)).unwrap_or(0);
        println!("  Unique positions: {}", count);
    }
    {
        let mut stmt = conn
            .prepare(
                "SELECT candidate_id, ROUND(AVG(s.nodes)), ROUND(AVG(s.depth),1), ROUND(AVG(s.time_us/1000)) \
                 FROM moves m JOIN search s ON s.move_id=m.move_id \
                 WHERE m.game_id IN (SELECT game_id FROM games WHERE match_id=?1) \
                 GROUP BY candidate_id",
            )
            .unwrap();
        let mut rows = stmt.query(params![match_id]).unwrap();
        while let Some(row) = rows.next().unwrap() {
            let cid: String = row.get(0).unwrap();
            let avg_nodes: f64 = row.get(1).unwrap_or(0.0);
            let avg_depth: f64 = row.get(2).unwrap_or(0.0);
            let avg_time: i32 = row.get(3).unwrap_or(0);
            println!(
                "  {}: avg {:.0} nodes, depth {:.1}, {}ms",
                cid, avg_nodes, avg_depth, avg_time
            );
        }
    }

    0
}

// ============================================================================
// Main -- mode dispatcher
// ============================================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!(
            "Usage:\n\
             \x20 {} <engine1> <engine2> <N> [budget_us] [db_path]                -- match\n\
             \x20 {} tournament <name> <budget_us> <games_per> e1 e2 [e3 ...]     -- tournament\n\
             \x20 {} standings [db_path]                                           -- standings\n\
             Engines: {}",
            args[0],
            args[0],
            args[0],
            ENGINE_REGISTRY
                .iter()
                .map(|e| e.name)
                .collect::<Vec<_>>()
                .join(" "),
        );
        std::process::exit(1);
    }

    let mode = &args[1];

    if mode == "standings" {
        let db_path = if args.len() > 2 {
            &args[2]
        } else {
            "pushchess.db"
        };
        let conn = Connection::open(db_path).expect("failed to open DB");
        create_schema(&conn);
        print_standings(&conn);
        std::process::exit(0);
    }

    if mode == "tournament" {
        if args.len() < 6 {
            eprintln!(
                "Usage: {} tournament <name> <budget_us> <games_per> e1 e2 [e3 ...]",
                args[0]
            );
            std::process::exit(1);
        }
        let name = &args[2];
        let budget_us: i64 = args[3].parse().expect("invalid budget_us");
        let games_per: i32 = args[4].parse().expect("invalid games_per");
        let engines: Vec<String> = args[5..].to_vec();
        if engines.len() < 2 {
            eprintln!("Need at least 2 engines");
            std::process::exit(1);
        }
        let rc = run_tournament(name, budget_us, games_per, &engines, "pushchess.db");
        std::process::exit(rc);
    }

    // Standalone match mode
    if args.len() < 4 {
        eprintln!(
            "Usage: {} <engine1> <engine2> <num_games> [budget_us] [db_path]",
            args[0]
        );
        std::process::exit(1);
    }
    let e1 = &args[1];
    let e2 = &args[2];
    let num_games: i32 = args[3].parse().expect("invalid num_games");
    let budget_us: i64 = if args.len() > 4 {
        args[4].parse().expect("invalid budget_us")
    } else {
        50000
    };
    let db_path = if args.len() > 5 {
        &args[5]
    } else {
        "pushchess.db"
    };

    let rc = run_standalone_match(e1, e2, num_games, budget_us, db_path);
    std::process::exit(rc);
}
