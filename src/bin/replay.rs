use crossterm::{
    event::{self, Event, KeyCode, KeyEvent},
    terminal::{self, ClearType},
    execute, cursor,
};
use rusqlite::Connection;
use std::io::{Write, stdout};

// ============================================================================
// Data types
// ============================================================================

struct MoveRecord {
    ply: i32,
    side: String,
    moving_piece: String,
    move_uci: String,
    captured: String,
    special: String,
    depth: i32,
    eval_cp: i32,
    mat: i32,
    nodes: i32,
    time_us: i32,
    legal_count: i32,
    is_capture: bool,
    is_promotion: bool,
    in_check_after: bool,
    fen_before: String,
    fen_after: String,
}

struct GameInfo {
    game_id: i32,
    white_id: String,
    black_id: String,
    result: String,
    termination: String,
    ply_count: i32,
    wall_time_ms: i32,
    match_id: i32,
}

struct TournamentInfo {
    tournament_id: i32,
    name: String,
    status: String,
    games_per_matchup: i32,
    budget_us: i32,
}

// ============================================================================
// Terminal helpers
// ============================================================================

fn read_key() -> char {
    loop {
        if let Ok(Event::Key(KeyEvent { code, .. })) = event::read() {
            return match code {
                KeyCode::Char('q') | KeyCode::Char('Q') => 'q',
                KeyCode::Enter => '\n',
                KeyCode::Char(' ') => 'R',
                KeyCode::Up => 'U',
                KeyCode::Down => 'D',
                KeyCode::Right => 'R',
                KeyCode::Left => 'L',
                KeyCode::Home => 'H',
                KeyCode::End => 'F',
                KeyCode::Char(c) => c,
                _ => continue,
            };
        }
    }
}

fn clear_screen() {
    let mut out = stdout();
    let _ = execute!(out, terminal::Clear(ClearType::All), cursor::MoveTo(0, 0));
}

// Board colors
const SQ_LIGHT: &str = "\x1b[48;2;220;220;220m";
const SQ_DARK: &str = "\x1b[48;2;80;80;80m";
const ANSI_RESET: &str = "\x1b[0m";
const ANSI_BOLD: &str = "\x1b[1m";
const ANSI_DIM: &str = "\x1b[2m";
const PC_BLUE: &str = "\x1b[38;2;30;100;220m";
const PC_RED: &str = "\x1b[38;2;220;40;40m";

fn piece_glyph(c: char) -> &'static str {
    match c {
        'K' | 'k' => "\u{265A}",
        'Q' | 'q' => "\u{265B}",
        'R' | 'r' => "\u{265C}",
        'B' | 'b' => "\u{265D}",
        'N' | 'n' => "\u{265E}",
        'P' | 'p' => "\u{265F}",
        _ => " ",
    }
}

fn is_white_piece(c: char) -> bool {
    c.is_ascii_uppercase()
}

fn render_board(fen: &str) {
    let mut board = [['.'; 8]; 8];
    let mut rank: i32 = 7;
    let mut file: usize = 0;

    for c in fen.chars() {
        if c == ' ' { break; }
        if c == '/' {
            rank -= 1;
            file = 0;
        } else if c.is_ascii_digit() {
            file += (c as usize) - ('0' as usize);
        } else {
            if rank >= 0 && rank < 8 && file < 8 {
                board[rank as usize][file] = c;
            }
            file += 1;
        }
    }

    let mut out = stdout();
    let _ = write!(out, "\n");
    for r in (0..8).rev() {
        let _ = write!(out, "  {} ", r + 1);
        for f in 0..8 {
            let dark = (r + f) % 2 == 0;
            let p = board[r][f];
            let sq_bg = if dark { SQ_DARK } else { SQ_LIGHT };

            if p == '.' {
                let _ = write!(out, "{}   {}", sq_bg, ANSI_RESET);
            } else {
                let pc_fg = if is_white_piece(p) { PC_BLUE } else { PC_RED };
                let _ = write!(out, "{}{} {} {}", sq_bg, pc_fg, piece_glyph(p), ANSI_RESET);
            }
        }
        let _ = writeln!(out);
    }
    let _ = writeln!(out, "    a  b  c  d  e  f  g  h");
    let _ = out.flush();
}

// ============================================================================
// Database queries
// ============================================================================

fn load_tournaments(conn: &Connection) -> Vec<TournamentInfo> {
    let mut stmt = conn.prepare(
        "SELECT tournament_id, name, status, games_per_matchup, budget_us \
         FROM tournaments ORDER BY tournament_id DESC"
    ).unwrap();

    stmt.query_map([], |row| {
        Ok(TournamentInfo {
            tournament_id: row.get(0)?,
            name: row.get(1)?,
            status: row.get(2)?,
            games_per_matchup: row.get(3)?,
            budget_us: row.get(4)?,
        })
    }).unwrap().filter_map(|r| r.ok()).collect()
}

fn load_games(conn: &Connection, tournament_id: i32) -> Vec<GameInfo> {
    if tournament_id > 0 {
        let mut stmt = conn.prepare(
            "SELECT g.game_id, g.white_id, g.black_id, g.result, g.termination, \
             g.ply_count, g.wall_time_ms, g.match_id \
             FROM games g JOIN matches m ON g.match_id = m.match_id \
             WHERE m.tournament_id=?1 AND g.result<>'' \
             ORDER BY g.game_id"
        ).unwrap();

        stmt.query_map([tournament_id], |row| {
            Ok(GameInfo {
                game_id: row.get(0)?,
                white_id: row.get(1)?,
                black_id: row.get(2)?,
                result: row.get(3)?,
                termination: row.get(4)?,
                ply_count: row.get(5)?,
                wall_time_ms: row.get(6)?,
                match_id: row.get(7)?,
            })
        }).unwrap().filter_map(|r| r.ok()).collect()
    } else {
        let mut stmt = conn.prepare(
            "SELECT game_id, white_id, black_id, result, termination, \
             ply_count, wall_time_ms, match_id \
             FROM games WHERE result<>'' ORDER BY game_id"
        ).unwrap();

        stmt.query_map([], |row| {
            Ok(GameInfo {
                game_id: row.get(0)?,
                white_id: row.get(1)?,
                black_id: row.get(2)?,
                result: row.get(3)?,
                termination: row.get(4)?,
                ply_count: row.get(5)?,
                wall_time_ms: row.get(6)?,
                match_id: row.get(7)?,
            })
        }).unwrap().filter_map(|r| r.ok()).collect()
    }
}

fn load_moves(conn: &Connection, game_id: i32) -> (Vec<MoveRecord>, String) {
    let mut stmt = conn.prepare(
        "SELECT m.ply, m.side, m.moving_piece, m.move_uci, m.captured_piece, m.special, \
         s.depth, s.eval_cp, p.material_balance, s.nodes, s.time_us, \
         m.legal_move_count, m.is_capture, m.is_promotion, \
         CASE WHEN m.side='white' THEN p2.in_check_black ELSE p2.in_check_white END, \
         m.fen_before, m.fen_after \
         FROM moves m \
         JOIN search s ON s.move_id=m.move_id \
         JOIN positions p ON p.fen=m.fen_before \
         JOIN positions p2 ON p2.fen=m.fen_after \
         WHERE m.game_id=?1 ORDER BY m.ply"
    ).unwrap();

    let mut start_fen = String::new();
    let moves: Vec<MoveRecord> = stmt.query_map([game_id], |row| {
        Ok(MoveRecord {
            ply: row.get(0)?,
            side: row.get(1)?,
            moving_piece: row.get(2)?,
            move_uci: row.get(3)?,
            captured: row.get(4)?,
            special: row.get(5)?,
            depth: row.get(6)?,
            eval_cp: row.get(7)?,
            mat: row.get(8)?,
            nodes: row.get(9)?,
            time_us: row.get(10)?,
            legal_count: row.get(11)?,
            is_capture: row.get::<_, i32>(12)? != 0,
            is_promotion: row.get::<_, i32>(13)? != 0,
            in_check_after: row.get::<_, i32>(14)? != 0,
            fen_before: row.get(15)?,
            fen_after: row.get(16)?,
        })
    }).unwrap().filter_map(|r| r.ok()).collect();

    if let Some(first) = moves.first() {
        start_fen = first.fen_before.clone();
    }

    (moves, start_fen)
}

// ============================================================================
// TUI Screens
// ============================================================================

fn select_from_list(title: &str, items: &[String]) -> Option<usize> {
    if items.is_empty() {
        clear_screen();
        let mut out = stdout();
        let _ = writeln!(out, "  {}\n\n  (empty)\n\n  Press any key...", title);
        let _ = out.flush();
        read_key();
        return None;
    }

    let mut sel: i32 = 0;
    let mut scroll: i32 = 0;
    let max_visible: i32 = 20;
    let n = items.len() as i32;

    loop {
        clear_screen();
        let mut out = stdout();

        let _ = writeln!(out, "{}  {}{}\n", ANSI_BOLD, title, ANSI_RESET);

        if sel < scroll { scroll = sel; }
        if sel >= scroll + max_visible { scroll = sel - max_visible + 1; }

        let end = n.min(scroll + max_visible);
        for i in scroll..end {
            if i == sel {
                let _ = writeln!(out, "  \x1b[7m {} \x1b[0m", items[i as usize]);
            } else {
                let _ = writeln!(out, "   {}", items[i as usize]);
            }
        }

        if n > max_visible {
            let _ = write!(out, "\n  {}({}-{} of {}){}", ANSI_DIM,
                scroll + 1, end.min(n), n, ANSI_RESET);
        }

        let _ = writeln!(out, "\n\n  {}Up/Down: navigate  Enter: select  q: back{}", ANSI_DIM, ANSI_RESET);
        let _ = out.flush();

        let k = read_key();
        match k {
            'q' => return None,
            '\n' => return Some(sel as usize),
            'U' => { if sel > 0 { sel -= 1; } }
            'D' => { if sel < n - 1 { sel += 1; } }
            'H' => { sel = 0; }
            'F' => { sel = n - 1; }
            _ => {}
        }
    }
}

fn replay_game(conn: &Connection, gi: &GameInfo) {
    let (moves, start_fen) = load_moves(conn, gi.game_id);
    if moves.is_empty() {
        clear_screen();
        let mut out = stdout();
        let _ = writeln!(out, "  No moves for game {}\n\n  Press any key...", gi.game_id);
        let _ = out.flush();
        read_key();
        return;
    }

    let total = moves.len() as i32;
    let mut pos: i32 = -1;

    loop {
        clear_screen();
        let mut out = stdout();

        let _ = writeln!(out, "{}  Push Chess Replay{}  --  Game #{}", ANSI_BOLD, ANSI_RESET, gi.game_id);
        let _ = writeln!(out, "  W: {}  vs  B: {}", gi.white_id, gi.black_id);
        let _ = writeln!(out, "  Result: {} ({}, {} ply)", gi.result, gi.termination, gi.ply_count);
        let _ = out.flush();

        if pos < 0 {
            render_board(&start_fen);
        } else {
            render_board(&moves[pos as usize].fen_after);
        }

        let mut out = stdout();
        let _ = writeln!(out);

        if pos < 0 {
            let _ = writeln!(out, "  {}Start position{}", ANSI_DIM, ANSI_RESET);
            let _ = writeln!(out);
            let _ = writeln!(out);
            let _ = writeln!(out);
        } else {
            let m = &moves[pos as usize];
            let movenum = m.ply / 2 + 1;
            let dot = if m.side == "white" { "." } else { "..." };

            let _ = write!(out, "  {}{}{}  {}{}", ANSI_BOLD, movenum, dot, m.moving_piece, ANSI_RESET);
            let _ = write!(out, " {}", m.move_uci);
            if m.is_capture {
                let _ = write!(out, "  \x1b[31mx{}\x1b[0m", m.captured);
            }
            if m.is_promotion {
                let _ = write!(out, "  \x1b[33mPROMO\x1b[0m");
            }
            if m.in_check_after {
                let _ = write!(out, "  \x1b[31;1m+CHECK\x1b[0m");
            }
            let _ = writeln!(out);

            // Eval bar
            let eval = if m.side == "black" { -m.eval_cp } else { m.eval_cp };
            let bar_len: i32 = 30;
            let center = bar_len / 2;
            let filled = center + (eval / 50).clamp(-center, center);
            let _ = write!(out, "  eval: ");
            for i in 0..bar_len {
                if i < filled {
                    let _ = write!(out, "\x1b[48;2;255;255;255m \x1b[0m");
                } else {
                    let _ = write!(out, "\x1b[48;2;60;60;60m \x1b[0m");
                }
            }
            if eval.abs() >= 90000 {
                let _ = writeln!(out, " \x1b[1;31mM{}\x1b[0m", 99000 - eval.abs());
            } else {
                let _ = writeln!(out, " {:+.1}", eval as f64 / 100.0);
            }

            let _ = writeln!(out, "  depth: {}  nodes: {}k  time: {}ms  legal: {}  mat: {:+.1}",
                m.depth, m.nodes / 1000, m.time_us / 1000, m.legal_count, m.mat as f64 / 100.0);
        }

        // Progress bar
        let progress = pos + 1;
        let bar_w: i32 = 40;
        let fill = if total > 0 { progress * bar_w / total } else { 0 };
        let _ = write!(out, "\n  [");
        for i in 0..bar_w {
            if i < fill {
                let _ = write!(out, "=");
            } else if i == fill {
                let _ = write!(out, ">");
            } else {
                let _ = write!(out, " ");
            }
        }
        let _ = writeln!(out, "] {}/{}", progress, total);

        let _ = writeln!(out, "\n  {}Left/Right: step  Home/End: jump  q: back to list{}", ANSI_DIM, ANSI_RESET);
        let _ = out.flush();

        let k = read_key();
        match k {
            'q' => return,
            'R' => { if pos < total - 1 { pos += 1; } }
            'L' => { if pos > -1 { pos -= 1; } }
            'H' => { pos = -1; }
            'F' => { pos = total - 1; }
            _ => {}
        }
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let db_path = std::env::args().nth(1).unwrap_or_else(|| "pushchess.db".into());

    let conn = match Connection::open(&db_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Can't open {}: {}", db_path, e);
            std::process::exit(1);
        }
    };

    terminal::enable_raw_mode().expect("failed to enable raw mode");

    // Ensure we restore terminal on exit
    struct RawGuard;
    impl Drop for RawGuard {
        fn drop(&mut self) {
            let _ = terminal::disable_raw_mode();
        }
    }
    let _guard = RawGuard;

    loop {
        let tournaments = load_tournaments(&conn);

        let mut menu_items: Vec<String> = Vec::new();
        menu_items.push("[All games]".to_string());
        for t in &tournaments {
            menu_items.push(format!(
                "Tournament #{}: {}  ({}, {}ms budget, {} games/matchup)",
                t.tournament_id, t.name, t.status,
                t.budget_us / 1000, t.games_per_matchup
            ));
        }

        let choice = select_from_list("Push Chess Replay -- Select source", &menu_items);
        let Some(choice) = choice else { break };

        let tid = if choice > 0 {
            tournaments[choice - 1].tournament_id
        } else {
            0
        };

        let games = load_games(&conn, tid);

        let game_items: Vec<String> = games.iter().map(|g| {
            format!("#{:<4}  {:<10} vs {:<10}  {:<7}  {:<20}  {:>3} ply  {:>5}ms",
                g.game_id, g.white_id, g.black_id,
                g.result, g.termination, g.ply_count, g.wall_time_ms)
        }).collect();

        let title = if tid > 0 {
            format!("Games in Tournament #{}", tid)
        } else {
            "All Games".to_string()
        };

        loop {
            let gsel = select_from_list(&title, &game_items);
            let Some(gsel) = gsel else { break };
            replay_game(&conn, &games[gsel]);
        }
    }

    clear_screen();
}
