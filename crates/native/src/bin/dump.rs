use rusqlite::Connection;
use std::collections::HashSet;
use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 || args[1] == "list" || args[1] == "-l" {
        let db_path = if args.len() >= 2 && args[1] != "list" && args[1] != "-l" {
            &args[1]
        } else {
            "pushchess.db"
        };
        let conn = Connection::open(db_path).expect("failed to open db");
        let mut stmt = conn.prepare(
            "SELECT t.tournament_id, t.name, t.budget_us, t.games_per_matchup, t.status,
                (SELECT COUNT(*) FROM games g JOIN matches m ON g.match_id=m.match_id WHERE m.tournament_id=t.tournament_id AND g.result!='')
            FROM tournaments t ORDER BY t.tournament_id"
        ).unwrap();
        let rows = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, i32>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, i64>(2)?,
                    r.get::<_, i32>(3)?,
                    r.get::<_, String>(4)?,
                    r.get::<_, i32>(5)?,
                ))
            })
            .unwrap();
        eprintln!("Tournaments in {}:", db_path);
        eprintln!(
            "{:<4} {:<20} {:>8} {:>5} {:>6} {:>6}",
            "ID", "Name", "budget", "g/m", "games", "status"
        );
        for r in rows {
            let (id, name, budget, gpm, status, games) = r.unwrap();
            eprintln!(
                "{:<4} {:<20} {:>6}ms {:>5} {:>6} {:>6}",
                id,
                name,
                budget / 1000,
                gpm,
                games,
                status
            );
        }
        // Show standalone matches (tournament_id IS NULL)
        let mut stmt2 = conn.prepare(
            "SELECT m.match_id, m.budget_us, m.num_games,
                (SELECT GROUP_CONCAT(ms.candidate_id, ' vs ') FROM match_slots ms WHERE ms.match_id=m.match_id),
                (SELECT COUNT(*) FROM games g WHERE g.match_id=m.match_id AND g.result!='')
            FROM matches m WHERE m.tournament_id IS NULL ORDER BY m.match_id"
        ).unwrap();
        let rows2 = stmt2
            .query_map([], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, i32>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, i32>(4)?,
                ))
            })
            .unwrap();
        let mut has_matches = false;
        for r in rows2 {
            if !has_matches {
                eprintln!("\nStandalone matches:");
                eprintln!(
                    "{:<6} {:<30} {:>8} {:>6}",
                    "m:ID", "Engines", "budget", "games"
                );
                has_matches = true;
            }
            let (id, budget, _ng, engines, games) = r.unwrap();
            eprintln!(
                "m:{:<4} {:<30} {:>6}ms {:>6}",
                id,
                engines,
                budget / 1000,
                games
            );
        }
        eprintln!("\nUsage: dump <tournament_id>  OR  dump m:<match_id>  OR  dump latest");
        std::process::exit(0);
    }

    // Parse args
    let (db_path, selector) = if args.len() >= 3 {
        (args[1].as_str(), args[2].as_str())
    } else {
        ("pushchess.db", args[1].as_str())
    };

    let conn = Connection::open(db_path).expect("failed to open db");

    // Determine filter: tournament_id, match_id, or all games
    enum Filter {
        Tournament(i32),
        Match(i64),
        All,
    }

    let filter = if selector == "latest" {
        // Find the most recent match (tournament or standalone)
        let mid: i64 = conn
            .prepare("SELECT match_id FROM matches ORDER BY match_id DESC LIMIT 1")
            .unwrap()
            .query_row([], |r| r.get(0))
            .expect("no matches");
        // Check if it belongs to a tournament
        let tid: Option<i32> = conn
            .prepare("SELECT tournament_id FROM matches WHERE match_id=?1")
            .unwrap()
            .query_row([mid], |r| r.get(0))
            .ok()
            .flatten();
        if let Some(t) = tid {
            Filter::Tournament(t)
        } else {
            Filter::Match(mid)
        }
    } else if selector == "all" {
        Filter::All
    } else if let Some(id) = selector.strip_prefix("m:") {
        Filter::Match(id.parse().expect("invalid match id"))
    } else {
        Filter::Tournament(
            selector
                .parse()
                .expect("invalid tournament id — use `dump list`"),
        )
    };

    // Build the WHERE clause for games
    let (game_where, header) = match &filter {
        Filter::Tournament(tid) => (
            format!(
                "g.match_id IN (SELECT match_id FROM matches WHERE tournament_id={})",
                tid
            ),
            format!("TOURNAMENT {}", tid),
        ),
        Filter::Match(mid) => (format!("g.match_id={}", mid), format!("MATCH {}", mid)),
        Filter::All => ("1=1".to_string(), "ALL GAMES".to_string()),
    };

    // Tournament info (if applicable)
    if let Filter::Tournament(tournament_id) = &filter {
        let mut stmt = conn.prepare("SELECT name, budget_us, games_per_matchup, started_at, finished_at, status FROM tournaments WHERE tournament_id=?1").unwrap();
        if let Ok((name, budget, gpm, started, finished, status)) =
            stmt.query_row([tournament_id], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, i32>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, String>(4)?,
                    r.get::<_, String>(5)?,
                ))
            })
        {
            println!("=== {} : {} ===", header, name);
            println!(
                "budget={}us  games_per_matchup={}  status={}  started={}  finished={}",
                budget, gpm, status, started, finished
            );
        }
    } else {
        println!("=== {} ===", header);
    }

    // Collect engines
    let mut engines: HashSet<String> = HashSet::new();
    {
        let sql = format!(
            "SELECT DISTINCT candidate_id FROM (
                SELECT white_id as candidate_id FROM games g WHERE {} AND g.result!=''
                UNION SELECT black_id FROM games g WHERE {} AND g.result!=''
            )",
            game_where, game_where
        );
        let mut stmt = conn.prepare(&sql).unwrap();
        let rows = stmt.query_map([], |r| r.get::<_, String>(0)).unwrap();
        for r in rows {
            engines.insert(r.unwrap());
        }
    }

    // Standings
    println!("\n=== STANDINGS ===");
    {
        let sql = format!(
            "SELECT a.candidate_id,
                SUM(won=1) as W, SUM(won=0) as D, SUM(won=-1) as L,
                ROUND(100.0*(SUM(won=1)+0.5*SUM(won=0))/(SUM(won=1)+SUM(won=0)+SUM(won=-1)),1) as score
            FROM (
                SELECT white_id as candidate_id, CASE result WHEN '1-0' THEN 1 WHEN '0-1' THEN -1 ELSE 0 END as won
                FROM games g WHERE {} AND g.result!=''
                UNION ALL
                SELECT black_id, CASE result WHEN '0-1' THEN 1 WHEN '1-0' THEN -1 ELSE 0 END
                FROM games g WHERE {} AND g.result!=''
            ) a GROUP BY a.candidate_id ORDER BY score DESC", game_where, game_where);
        let mut stmt = conn.prepare(&sql).unwrap();
        let rows = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, i32>(1)?,
                    r.get::<_, i32>(2)?,
                    r.get::<_, i32>(3)?,
                    r.get::<_, f64>(4)?,
                ))
            })
            .unwrap();
        for r in rows {
            let (e, w, d, l, s) = r.unwrap();
            println!("{:<16} {:>2}W {:>2}D {:>2}L  {:>5.1}%", e, w, d, l, s);
        }
    }

    // All games
    println!("\n=== GAMES ===");
    {
        let sql = format!(
            "SELECT g.game_id, g.white_id, g.black_id, g.result, g.termination, g.ply_count, g.wall_time_ms
            FROM games g WHERE {} AND g.result!='' ORDER BY g.game_id", game_where);
        let mut stmt = conn.prepare(&sql).unwrap();
        let rows = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, i32>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, String>(4)?,
                    r.get::<_, i32>(5)?,
                    r.get::<_, i64>(6)?,
                ))
            })
            .unwrap();
        for r in rows {
            let (gid, w, b, res, term, ply, ms) = r.unwrap();
            println!(
                "g{:<4} {:<14} v {:<14} {:<7} {:<20} {:>3}p {:>5}ms",
                gid, w, b, res, term, ply, ms
            );
        }
    }

    // Search stats per engine
    println!("\n=== SEARCH STATS ===");
    {
        let sql = format!(
            "SELECT mv.candidate_id,
                ROUND(AVG(s.depth),1), ROUND(AVG(s.seldepth),1), ROUND(AVG(s.nodes),0),
                ROUND(AVG(s.eval_cp),0), ROUND(AVG(s.time_us),0), COUNT(*)
            FROM search s JOIN moves mv ON mv.move_id=s.move_id
            JOIN games g ON g.game_id=mv.game_id
            WHERE {}
            GROUP BY mv.candidate_id ORDER BY AVG(s.depth) DESC",
            game_where
        );
        let mut stmt = conn.prepare(&sql).unwrap();
        let rows = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, f64>(1)?,
                    r.get::<_, f64>(2)?,
                    r.get::<_, f64>(3)?,
                    r.get::<_, f64>(4)?,
                    r.get::<_, f64>(5)?,
                    r.get::<_, i32>(6)?,
                ))
            })
            .unwrap();
        println!(
            "{:<16} {:>5} {:>5} {:>8} {:>7} {:>8} {:>6}",
            "engine", "depth", "sdep", "nodes", "eval", "time_us", "moves"
        );
        for r in rows {
            let (e, d, sd, n, ev, t, mv) = r.unwrap();
            println!(
                "{:<16} {:>5.1} {:>5.1} {:>8.0} {:>7.0} {:>8.0} {:>6}",
                e, d, sd, n, ev, t, mv
            );
        }
    }

    // Full per-move data with all telemetry
    println!("\n=== FULL MOVE DATA ===");
    {
        let sql = format!(
            "SELECT mv.game_id, mv.ply, mv.side, mv.candidate_id, mv.move_uci, mv.moving_piece,
                mv.captured_piece, mv.special, mv.is_capture, mv.is_promotion, mv.is_castle,
                mv.is_en_passant, mv.is_knight_move, mv.legal_move_count,
                mv.fen_before, mv.fen_after,
                s.eval_cp, s.depth, s.seldepth, s.nodes, s.time_us, s.pv, s.diag_json
            FROM moves mv JOIN search s ON s.move_id=mv.move_id
            JOIN games g ON g.game_id=mv.game_id
            WHERE {}
            ORDER BY mv.game_id, mv.ply",
            game_where
        );
        let mut stmt = conn.prepare(&sql).unwrap();
        let rows = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, i32>(0)?,
                    r.get::<_, i32>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, String>(4)?,
                    r.get::<_, String>(5)?,
                    r.get::<_, String>(6)?,
                    r.get::<_, String>(7)?,
                    r.get::<_, i32>(8)?,
                    r.get::<_, i32>(9)?,
                    r.get::<_, i32>(10)?,
                    r.get::<_, i32>(11)?,
                    r.get::<_, i32>(12)?,
                    r.get::<_, i32>(13)?,
                    r.get::<_, String>(14)?,
                    r.get::<_, String>(15)?,
                    r.get::<_, i32>(16)?,
                    r.get::<_, i32>(17)?,
                    r.get::<_, i32>(18)?,
                    r.get::<_, i64>(19)?,
                    r.get::<_, i64>(20)?,
                    r.get::<_, String>(21)?,
                    r.get::<_, String>(22)?,
                ))
            })
            .unwrap();
        let mut last_gid = -1;
        for r in rows {
            let (
                gid,
                ply,
                side,
                eng,
                uci,
                piece,
                cap,
                special,
                _is_cap,
                _is_promo,
                _is_castle,
                _is_ep,
                _is_knight,
                legal_count,
                fen_before,
                fen_after,
                eval,
                depth,
                seldepth,
                nodes,
                time_us,
                pv,
                diag,
            ) = r.unwrap();
            if gid != last_gid {
                if last_gid != -1 {
                    println!();
                }
                println!("=== GAME {} ===", gid);
                last_gid = gid;
            }
            // Dense one-line format with all data
            print!("p{} {} {} {} {}", ply, side, eng, uci, piece);
            if cap != "none" {
                print!(" x{}", cap);
            }
            if special != "none" {
                print!(" [{}]", special);
            }
            print!(
                " | eval={} d={} sd={} n={} t={}us legal={}",
                eval, depth, seldepth, nodes, time_us, legal_count
            );
            if !pv.is_empty() {
                print!(" pv={}", pv);
            }
            println!();
            if !diag.is_empty() && diag != "{}" {
                println!("  diag: {}", diag);
            }
            // Print FENs for first and last ply of each game
            if ply == 0 {
                println!("  fen_before: {}", fen_before);
            }
            println!("  fen_after: {}", fen_after);
        }
        println!();
    }

    // Source files
    println!("\n=== ENGINE SOURCES ===");
    let mut sorted_engines: Vec<&String> = engines.iter().collect();
    sorted_engines.sort();
    for eng in sorted_engines {
        let path = format!("src/candidates/{}.rs", eng);
        if let Ok(content) = fs::read_to_string(&path) {
            println!("\n--- {} ({} lines) ---", eng, content.lines().count());
            print!("{}", content);
        }
    }
}
