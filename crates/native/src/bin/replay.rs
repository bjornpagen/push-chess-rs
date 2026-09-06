use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, ListState, Paragraph},
};
use rusqlite::Connection;

// ============================================================================
// Data types
// ============================================================================

struct MoveRecord {
    ply: i32,
    side: String,
    moving_piece: String,
    move_uci: String,
    captured: String,
    depth: i32,
    eval_cp: i32,
    nodes: i64,
    time_us: i64,
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
    wall_time_ms: i64,
}

struct TournamentInfo {
    tournament_id: i32,
    name: String,
    status: String,
    games_per_matchup: i32,
    budget_us: i64,
}

// ============================================================================
// Database queries
// ============================================================================

fn load_tournaments(conn: &Connection) -> Vec<TournamentInfo> {
    let mut stmt = conn
        .prepare(
            "SELECT tournament_id, name, status, games_per_matchup, budget_us \
             FROM tournaments ORDER BY tournament_id DESC",
        )
        .unwrap();
    stmt.query_map([], |row| {
        Ok(TournamentInfo {
            tournament_id: row.get(0)?,
            name: row.get(1)?,
            status: row.get(2)?,
            games_per_matchup: row.get(3)?,
            budget_us: row.get(4)?,
        })
    })
    .unwrap()
    .filter_map(|r| r.ok())
    .collect()
}

fn load_games(conn: &Connection, tournament_id: i32) -> Vec<GameInfo> {
    if tournament_id > 0 {
        let mut stmt = conn
            .prepare(
                "SELECT g.game_id, g.white_id, g.black_id, g.result, g.termination, \
                 g.ply_count, g.wall_time_ms \
                 FROM games g JOIN matches m ON g.match_id = m.match_id \
                 WHERE m.tournament_id=?1 AND g.result<>'' \
                 ORDER BY g.game_id",
            )
            .unwrap();
        stmt.query_map([tournament_id], |row| {
            Ok(GameInfo {
                game_id: row.get(0)?,
                white_id: row.get(1)?,
                black_id: row.get(2)?,
                result: row.get(3)?,
                termination: row.get(4)?,
                ply_count: row.get(5)?,
                wall_time_ms: row.get(6)?,
            })
        })
        .unwrap()
        .filter_map(|r| r.ok())
        .collect()
    } else {
        let mut stmt = conn
            .prepare(
                "SELECT game_id, white_id, black_id, result, termination, \
                 ply_count, wall_time_ms \
                 FROM games WHERE result<>'' ORDER BY game_id",
            )
            .unwrap();
        stmt.query_map([], |row| {
            Ok(GameInfo {
                game_id: row.get(0)?,
                white_id: row.get(1)?,
                black_id: row.get(2)?,
                result: row.get(3)?,
                termination: row.get(4)?,
                ply_count: row.get(5)?,
                wall_time_ms: row.get(6)?,
            })
        })
        .unwrap()
        .filter_map(|r| r.ok())
        .collect()
    }
}

fn load_moves(conn: &Connection, game_id: i32) -> (Vec<MoveRecord>, String) {
    let mut stmt = conn
        .prepare(
            "SELECT m.ply, m.side, m.moving_piece, m.move_uci, m.captured_piece, \
             s.depth, s.eval_cp, s.nodes, s.time_us, \
             m.legal_move_count, m.is_capture, m.is_promotion, \
             0, \
             m.fen_before, m.fen_after \
             FROM moves m \
             JOIN search s ON s.move_id=m.move_id \
             WHERE m.game_id=?1 ORDER BY m.ply",
        )
        .unwrap();

    let mut start_fen = String::new();
    let moves: Vec<MoveRecord> = stmt
        .query_map([game_id], |row| {
            Ok(MoveRecord {
                ply: row.get(0)?,
                side: row.get(1)?,
                moving_piece: row.get(2)?,
                move_uci: row.get(3)?,
                captured: row.get(4)?,
                depth: row.get(5)?,
                eval_cp: row.get(6)?,
                nodes: row.get(7)?,
                time_us: row.get(8)?,
                legal_count: row.get(9)?,
                is_capture: row.get::<_, i32>(10)? != 0,
                is_promotion: row.get::<_, i32>(11)? != 0,
                in_check_after: row.get::<_, i32>(12)? != 0,
                fen_before: row.get(13)?,
                fen_after: row.get(14)?,
            })
        })
        .unwrap()
        .filter_map(|r| r.ok())
        .collect();

    if let Some(first) = moves.first() {
        start_fen = first.fen_before.clone();
    }
    (moves, start_fen)
}

// ============================================================================
// Board rendering
// ============================================================================

fn piece_char(c: char) -> &'static str {
    match c.to_ascii_lowercase() {
        'k' => "\u{265A}",
        'q' => "\u{265B}",
        'r' => "\u{265C}",
        'b' => "\u{265D}",
        'n' => "\u{265E}",
        'p' => "\u{265F}",
        _ => " ",
    }
}

fn parse_board(fen: &str) -> [[char; 8]; 8] {
    let mut board = [['.'; 8]; 8];
    let mut rank: i32 = 7;
    let mut file: usize = 0;
    for c in fen.chars() {
        if c == ' ' {
            break;
        }
        if c == '/' {
            rank -= 1;
            file = 0;
        } else if c.is_ascii_digit() {
            file += (c as usize) - ('0' as usize);
        } else if (0..8).contains(&rank) && file < 8 {
            board[rank as usize][file] = c;
            file += 1;
        }
    }
    board
}

fn render_board_widget(fen: &str, area: Rect, frame: &mut Frame) {
    let board = parse_board(fen);

    let light = Color::Rgb(220, 220, 220);
    let dark = Color::Rgb(80, 80, 80);
    let white_piece = Color::Rgb(30, 100, 220);
    let black_piece = Color::Rgb(220, 40, 40);

    let mut lines: Vec<Line> = Vec::new();

    for r in (0..8).rev() {
        let mut spans: Vec<Span> = Vec::new();
        spans.push(Span::styled(
            format!("  {} ", r + 1),
            Style::default().fg(Color::DarkGray),
        ));
        for f in 0..8 {
            let is_dark = (r + f) % 2 == 0;
            let bg = if is_dark { dark } else { light };
            let p = board[r][f];
            if p == '.' {
                spans.push(Span::styled("   ", Style::default().bg(bg)));
            } else {
                let fg = if p.is_ascii_uppercase() {
                    white_piece
                } else {
                    black_piece
                };
                spans.push(Span::styled(
                    format!(" {} ", piece_char(p)),
                    Style::default().fg(fg).bg(bg),
                ));
            }
        }
        lines.push(Line::from(spans));
    }
    lines.push(Line::from(Span::styled(
        "     a  b  c  d  e  f  g  h",
        Style::default().fg(Color::DarkGray),
    )));

    let para = Paragraph::new(lines);
    frame.render_widget(para, area);
}

// ============================================================================
// App state
// ============================================================================

enum Screen {
    Tournaments,
    Games,
    Replay,
}

struct App {
    screen: Screen,
    conn: Connection,
    // Tournaments
    tournaments: Vec<TournamentInfo>,
    tournament_state: ListState,
    // Games
    games: Vec<GameInfo>,
    game_state: ListState,
    selected_tournament: i32, // 0 = all
    tournament_label: String,
    // Replay
    moves: Vec<MoveRecord>,
    start_fen: String,
    replay_pos: i32, // -1 = start position
    replay_game: Option<usize>,
    // Move list scroll
    move_list_state: ListState,
}

impl App {
    fn new(conn: Connection) -> Self {
        let tournaments = load_tournaments(&conn);
        let mut ts = ListState::default();
        if !tournaments.is_empty() {
            ts.select(Some(0));
        }
        Self {
            screen: Screen::Tournaments,
            conn,
            tournaments,
            tournament_state: ts,
            games: Vec::new(),
            game_state: ListState::default(),
            selected_tournament: 0,
            tournament_label: String::new(),
            moves: Vec::new(),
            start_fen: String::new(),
            replay_pos: -1,
            replay_game: None,
            move_list_state: ListState::default(),
        }
    }

    fn enter_games(&mut self) {
        let sel = self.tournament_state.selected().unwrap_or(0);
        if sel == 0 {
            self.selected_tournament = 0;
            self.tournament_label = "All Games".to_string();
        } else {
            let t = &self.tournaments[sel - 1];
            self.selected_tournament = t.tournament_id;
            self.tournament_label = format!("Tournament #{}: {}", t.tournament_id, t.name);
        }
        self.games = load_games(&self.conn, self.selected_tournament);
        self.game_state = ListState::default();
        if !self.games.is_empty() {
            self.game_state.select(Some(0));
        }
        self.screen = Screen::Games;
    }

    fn enter_replay(&mut self) {
        let sel = self.game_state.selected().unwrap_or(0);
        if sel >= self.games.len() {
            return;
        }
        self.replay_game = Some(sel);
        let gid = self.games[sel].game_id;
        let (moves, start_fen) = load_moves(&self.conn, gid);
        self.moves = moves;
        self.start_fen = start_fen;
        self.replay_pos = -1;
        self.move_list_state = ListState::default();
        self.screen = Screen::Replay;
    }

    fn handle_key(&mut self, code: KeyCode) -> bool {
        match self.screen {
            Screen::Tournaments => match code {
                KeyCode::Char('q') | KeyCode::Esc => return true,
                KeyCode::Up | KeyCode::Char('k') => {
                    let i = self.tournament_state.selected().unwrap_or(0);
                    if i > 0 {
                        self.tournament_state.select(Some(i - 1));
                    }
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    let i = self.tournament_state.selected().unwrap_or(0);
                    let max = self.tournaments.len(); // +1 for "All"
                    if i < max {
                        self.tournament_state.select(Some(i + 1));
                    }
                }
                KeyCode::Enter => self.enter_games(),
                _ => {}
            },
            Screen::Games => match code {
                KeyCode::Char('q') | KeyCode::Esc => self.screen = Screen::Tournaments,
                KeyCode::Up | KeyCode::Char('k') => {
                    let i = self.game_state.selected().unwrap_or(0);
                    if i > 0 {
                        self.game_state.select(Some(i - 1));
                    }
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    let i = self.game_state.selected().unwrap_or(0);
                    if i + 1 < self.games.len() {
                        self.game_state.select(Some(i + 1));
                    }
                }
                KeyCode::Enter => self.enter_replay(),
                _ => {}
            },
            Screen::Replay => match code {
                KeyCode::Char('q') | KeyCode::Esc => self.screen = Screen::Games,
                KeyCode::Right | KeyCode::Char('l') | KeyCode::Char(' ') => {
                    if self.replay_pos < self.moves.len() as i32 - 1 {
                        self.replay_pos += 1;
                        self.move_list_state.select(Some(self.replay_pos as usize));
                    }
                }
                KeyCode::Left | KeyCode::Char('h') => {
                    if self.replay_pos > -1 {
                        self.replay_pos -= 1;
                        if self.replay_pos >= 0 {
                            self.move_list_state.select(Some(self.replay_pos as usize));
                        } else {
                            self.move_list_state.select(None);
                        }
                    }
                }
                KeyCode::Home => {
                    self.replay_pos = -1;
                    self.move_list_state.select(None);
                }
                KeyCode::End => {
                    if !self.moves.is_empty() {
                        self.replay_pos = self.moves.len() as i32 - 1;
                        self.move_list_state.select(Some(self.replay_pos as usize));
                    }
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    if self.replay_pos > -1 {
                        self.replay_pos -= 1;
                        if self.replay_pos >= 0 {
                            self.move_list_state.select(Some(self.replay_pos as usize));
                        } else {
                            self.move_list_state.select(None);
                        }
                    }
                }
                KeyCode::Down | KeyCode::Char('j')
                    if self.replay_pos < self.moves.len() as i32 - 1 =>
                {
                    self.replay_pos += 1;
                    self.move_list_state.select(Some(self.replay_pos as usize));
                }
                _ => {}
            },
        }
        false
    }

    fn draw(&mut self, frame: &mut Frame) {
        match self.screen {
            Screen::Tournaments => self.draw_tournaments(frame),
            Screen::Games => self.draw_games(frame),
            Screen::Replay => self.draw_replay(frame),
        }
    }

    fn draw_tournaments(&mut self, frame: &mut Frame) {
        let area = frame.area();
        let chunks = Layout::vertical([
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .split(area);

        let title = Paragraph::new(Line::from(vec![" Push Chess Replay ".bold().cyan()])).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        );
        frame.render_widget(title, chunks[0]);

        let mut items: Vec<ListItem> = Vec::new();
        items.push(ListItem::new(Line::from(" [All games]".yellow())));
        for t in &self.tournaments {
            items.push(ListItem::new(Line::from(format!(
                " Tournament #{}: {}  ({}, {}ms, {} g/m)",
                t.tournament_id,
                t.name,
                t.status,
                t.budget_us / 1000,
                t.games_per_matchup
            ))));
        }

        let list = List::new(items)
            .block(
                Block::default()
                    .title(" Select Source ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray)),
            )
            .highlight_style(
                Style::default()
                    .bg(Color::Rgb(40, 60, 80))
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("▸ ");
        frame.render_stateful_widget(list, chunks[1], &mut self.tournament_state);

        let help = Paragraph::new(Line::from(vec![
            " ↑↓".dark_gray(),
            " navigate ".dark_gray(),
            " Enter".dark_gray(),
            " select ".dark_gray(),
            " q".dark_gray(),
            " quit".dark_gray(),
        ]));
        frame.render_widget(help, chunks[2]);
    }

    fn draw_games(&mut self, frame: &mut Frame) {
        let area = frame.area();
        let chunks = Layout::vertical([
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .split(area);

        let title = Paragraph::new(Line::from(vec![
            format!(" {} ", self.tournament_label).cyan().bold(),
        ]))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        );
        frame.render_widget(title, chunks[0]);

        let items: Vec<ListItem> = self
            .games
            .iter()
            .map(|g| {
                let result_style = match g.result.as_str() {
                    "1-0" => Style::default().fg(Color::Cyan),
                    "0-1" => Style::default().fg(Color::Red),
                    _ => Style::default().fg(Color::Yellow),
                };
                let term_style = if g.termination == "illegal_move" {
                    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::DarkGray)
                };
                ListItem::new(Line::from(vec![
                    Span::styled(
                        format!(" #{:<4}", g.game_id),
                        Style::default().fg(Color::DarkGray),
                    ),
                    Span::raw(format!(" {:<10} vs {:<10}  ", g.white_id, g.black_id)),
                    Span::styled(format!("{:<7}", g.result), result_style),
                    Span::styled(format!("  {:<16}", g.termination), term_style),
                    Span::styled(
                        format!("  {:>3}p", g.ply_count),
                        Style::default().fg(Color::DarkGray),
                    ),
                    Span::styled(
                        format!("  {:>5}ms", g.wall_time_ms),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]))
            })
            .collect();

        let list = List::new(items)
            .block(
                Block::default()
                    .title(format!(" {} games ", self.games.len()))
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray)),
            )
            .highlight_style(
                Style::default()
                    .bg(Color::Rgb(40, 60, 80))
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("▸ ");
        frame.render_stateful_widget(list, chunks[1], &mut self.game_state);

        let help = Paragraph::new(Line::from(vec![
            " ↑↓".dark_gray(),
            " navigate ".dark_gray(),
            " Enter".dark_gray(),
            " select ".dark_gray(),
            " q".dark_gray(),
            " back".dark_gray(),
        ]));
        frame.render_widget(help, chunks[2]);
    }

    fn draw_replay(&mut self, frame: &mut Frame) {
        let gi_idx = self.replay_game.unwrap_or(0);
        let gi = &self.games[gi_idx];
        let area = frame.area();

        // Top: header (3) | Main: board+info | Bottom: progress + help (3)
        let outer = Layout::vertical([
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(3),
        ])
        .split(area);

        // Header
        let result_color = match gi.result.as_str() {
            "1-0" => Color::Cyan,
            "0-1" => Color::Red,
            _ => Color::Yellow,
        };
        let header = Paragraph::new(Line::from(vec![
            Span::styled(
                format!(" Game #{}", gi.game_id),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(
                &gi.white_id,
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" vs ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                &gi.black_id,
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(
                &gi.result,
                Style::default()
                    .fg(result_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  {}  {} ply", gi.termination, gi.ply_count),
                Style::default().fg(Color::DarkGray),
            ),
        ]))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        );
        frame.render_widget(header, outer[0]);

        // Main area: board (left) | info panel (right)
        let main = Layout::horizontal([
            Constraint::Length(32), // board
            Constraint::Min(30),    // move list + info
        ])
        .split(outer[1]);

        // Board
        let fen = if self.replay_pos < 0 {
            &self.start_fen
        } else {
            &self.moves[self.replay_pos as usize].fen_after
        };
        let board_block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray))
            .title(" Board ");
        let board_inner = board_block.inner(main[0]);
        frame.render_widget(board_block, main[0]);
        render_board_widget(fen, board_inner, frame);

        // Right panel: split into move info (top) and move list (bottom)
        let right = Layout::vertical([
            Constraint::Length(9), // current move info
            Constraint::Min(0),    // move list
        ])
        .split(main[1]);

        // Move info panel
        let info_lines = if self.replay_pos < 0 {
            vec![
                Line::from(""),
                Line::from(Span::styled(
                    "  Start position",
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::ITALIC),
                )),
            ]
        } else {
            let m = &self.moves[self.replay_pos as usize];
            let movenum = m.ply / 2 + 1;
            let dot = if m.side == "white" { "." } else { "..." };
            let eval = if m.side == "black" {
                -m.eval_cp
            } else {
                m.eval_cp
            };

            let mut move_spans = vec![
                Span::styled(
                    format!("  {}{} ", movenum, dot),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(
                    &m.moving_piece,
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::raw(format!(" {}", m.move_uci)),
            ];
            if m.is_capture {
                move_spans.push(Span::styled(
                    format!(" x{}", m.captured),
                    Style::default().fg(Color::Red),
                ));
            }
            if m.is_promotion {
                move_spans.push(Span::styled(
                    " PROMO",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ));
            }
            if m.in_check_after {
                move_spans.push(Span::styled(
                    " +CHECK",
                    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                ));
            }

            let eval_str = if eval.abs() >= 90000 {
                format!("M{}", 99000 - eval.abs())
            } else {
                format!("{:+.1}", eval as f64 / 100.0)
            };
            let eval_color = if eval > 100 {
                Color::Green
            } else if eval < -100 {
                Color::Red
            } else {
                Color::Yellow
            };

            vec![
                Line::from(""),
                Line::from(move_spans),
                Line::from(""),
                Line::from(vec![
                    Span::styled("  eval ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        eval_str.clone(),
                        Style::default().fg(eval_color).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!("   depth {}", m.depth),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]),
                Line::from(vec![
                    Span::styled(
                        format!("  nodes {}k", m.nodes / 1000),
                        Style::default().fg(Color::DarkGray),
                    ),
                    Span::styled(
                        format!("   time {}ms", m.time_us / 1000),
                        Style::default().fg(Color::DarkGray),
                    ),
                    Span::styled(
                        format!("   legal {}", m.legal_count),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]),
            ]
        };

        let info = Paragraph::new(info_lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(" Move Info "),
        );
        frame.render_widget(info, right[0]);

        // Move list
        let move_items: Vec<ListItem> = self
            .moves
            .iter()
            .map(|m| {
                let movenum = m.ply / 2 + 1;
                let dot = if m.side == "white" { "." } else { "..." };
                let eval = if m.side == "black" {
                    -m.eval_cp
                } else {
                    m.eval_cp
                };
                let eval_str = if eval.abs() >= 90000 {
                    format!("M{}", 99000 - eval.abs())
                } else {
                    format!("{:+.1}", eval as f64 / 100.0)
                };
                let side_color = if m.side == "white" {
                    Color::Cyan
                } else {
                    Color::Red
                };
                let eval_color = if eval > 100 {
                    Color::Green
                } else if eval < -100 {
                    Color::Red
                } else {
                    Color::Yellow
                };
                let mut spans = vec![
                    Span::styled(
                        format!("{:>3}{:<3} ", movenum, dot),
                        Style::default().fg(Color::DarkGray),
                    ),
                    Span::styled(
                        format!("{:<6}", m.moving_piece),
                        Style::default().fg(side_color),
                    ),
                    Span::raw(format!("{:<6}", m.move_uci)),
                ];
                if m.is_capture {
                    spans.push(Span::styled(
                        format!("x{:<5} ", m.captured),
                        Style::default().fg(Color::Red),
                    ));
                } else {
                    spans.push(Span::styled(
                        "       ",
                        Style::default().fg(Color::DarkGray),
                    ));
                }
                spans.push(Span::styled(
                    format!("{:>6}", eval_str),
                    Style::default().fg(eval_color),
                ));
                spans.push(Span::styled(
                    format!("  d{}", m.depth),
                    Style::default().fg(Color::DarkGray),
                ));
                ListItem::new(Line::from(spans))
            })
            .collect();

        let move_list = List::new(move_items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray))
                    .title(format!(" Moves ({}) ", self.moves.len())),
            )
            .highlight_style(
                Style::default()
                    .bg(Color::Rgb(40, 60, 80))
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("▸ ");
        frame.render_stateful_widget(move_list, right[1], &mut self.move_list_state);

        // Bottom: progress gauge + help
        let bottom = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(outer[2]);

        let progress = (self.replay_pos + 1) as f64;
        let total = self.moves.len() as f64;
        let ratio = if total > 0.0 { progress / total } else { 0.0 };
        let gauge = Gauge::default()
            .ratio(ratio)
            .label(format!("{}/{}", self.replay_pos + 1, self.moves.len()))
            .gauge_style(Style::default().fg(Color::Cyan).bg(Color::Rgb(30, 30, 30)));
        frame.render_widget(gauge, bottom[0]);

        let help = Paragraph::new(Line::from(vec![
            " ←→".dark_gray(),
            " step ".dark_gray(),
            " ↑↓".dark_gray(),
            " scroll ".dark_gray(),
            " Home/End".dark_gray(),
            " jump ".dark_gray(),
            " Space".dark_gray(),
            " fwd ".dark_gray(),
            " q".dark_gray(),
            " back".dark_gray(),
        ]));
        frame.render_widget(help, bottom[2]);
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "pushchess.db".into());

    let conn = Connection::open(&db_path)?;

    let mut terminal = ratatui::init();
    let mut app = App::new(conn);

    let result = loop {
        terminal.draw(|frame| app.draw(frame))?;
        if let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
            && app.handle_key(key.code)
        {
            break Ok(());
        }
    };

    ratatui::restore();
    result
}
