use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
};

use std::sync::mpsc;
use std::time::Duration;

use push_chess::candidates::find_engine;
use push_chess::core::movegen::generate_legal_moves;
use push_chess::core::position::{Position, start_position};
use push_chess::core::types::{
    self as chess, Move, PieceType, SearchBudget, SearchStats, SpecialMove,
};

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

fn sq_name(sq: u8) -> String {
    format!("{}{}", (b'a' + sq % 8) as char, (b'1' + sq / 8) as char)
}

fn move_uci(m: &Move) -> String {
    let mut s = format!("{}{}", sq_name(m.from), sq_name(m.to));
    if m.special == SpecialMove::Promotion {
        s.push(match m.promo_piece {
            PieceType::Queen => 'q',
            PieceType::Rook => 'r',
            PieceType::Bishop => 'b',
            PieceType::Knight => 'n',
            _ => '?',
        });
    }
    s
}

// ============================================================================
// App state
// ============================================================================

struct MoveRecord {
    uci: String,
    side: &'static str,
    eval: i32,
    depth: u32,
}

struct EngineResult {
    chosen: Move,
    stats: SearchStats,
}

struct App {
    pos: Position,
    engine_name: String,
    player_is_white: bool,
    legal_moves: Vec<Move>,
    move_history: Vec<MoveRecord>,
    cursor_sq: u8,
    selected_from: Option<u8>,
    move_list_state: ListState,
    status: String,
    game_over: bool,
    ply: i32,
    think_time_ms: u64,
    // Async engine state
    engine_thinking: bool,
    result_rx: Option<mpsc::Receiver<EngineResult>>,
    thinking_dots: u8,
}

impl App {
    fn new(engine_name: &str, play_white: bool, think_time_ms: u64) -> Self {
        let pos = start_position();
        let mut app = App {
            pos,
            engine_name: engine_name.to_string(),
            player_is_white: play_white,
            legal_moves: Vec::new(),
            move_history: Vec::new(),
            cursor_sq: if play_white { 4 } else { 60 },
            selected_from: None,
            move_list_state: ListState::default(),
            status: "Your move".to_string(),
            game_over: false,
            ply: 0,
            think_time_ms,
            engine_thinking: false,
            result_rx: None,
            thinking_dots: 0,
        };
        app.refresh_legal_moves();

        if !play_white {
            app.start_engine_think();
        }
        app
    }

    fn refresh_legal_moves(&mut self) {
        self.legal_moves.clear();
        generate_legal_moves(&mut self.pos, &mut self.legal_moves);
    }

    fn is_player_turn(&self) -> bool {
        !self.game_over
            && !self.engine_thinking
            && ((self.player_is_white && self.pos.side_to_move == chess::Color::White)
                || (!self.player_is_white && self.pos.side_to_move == chess::Color::Black))
    }

    fn start_engine_think(&mut self) {
        if self.game_over {
            return;
        }
        self.refresh_legal_moves();
        if self.legal_moves.is_empty() {
            self.game_over = true;
            self.status = if self.pos.in_check() {
                "You win! Engine is checkmated.".to_string()
            } else {
                "Draw — stalemate.".to_string()
            };
            return;
        }

        self.engine_thinking = true;
        self.thinking_dots = 0;
        self.status = "Engine thinking".to_string();

        let (tx, rx) = mpsc::channel();
        self.result_rx = Some(rx);

        let mut pos_clone = self.pos.clone();
        let ply = self.ply;
        let engine_name = self.engine_name.clone();
        let player_is_white = self.player_is_white;
        let think_us = self.think_time_ms as i64 * 1000;

        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(move || {
                let entry = find_engine(&engine_name).unwrap();
                let mut engine = (entry.create)();
                engine.new_game(
                    if player_is_white {
                        chess::Color::Black
                    } else {
                        chess::Color::White
                    },
                    42,
                );

                let budget = SearchBudget {
                    max_time_us: think_us,
                    max_nodes: 0,
                    max_depth: 0,
                    seed: ply as u64,
                };
                let (chosen, stats) = engine.choose_move(&mut pos_clone, &budget);
                let _ = tx.send(EngineResult { chosen, stats });
            })
            .expect("failed to spawn engine thread");
    }

    fn check_engine_result(&mut self) {
        if !self.engine_thinking {
            return;
        }

        let got = if let Some(ref rx) = self.result_rx {
            rx.try_recv().ok()
        } else {
            None
        };

        if let Some(result) = got {
            self.result_rx = None;
            self.engine_thinking = false;

            self.pos.make_move(&result.chosen);
            self.ply += 1;

            self.move_history.push(MoveRecord {
                uci: move_uci(&result.chosen),
                side: if self.player_is_white {
                    "black"
                } else {
                    "white"
                },
                eval: result.stats.eval_cp,
                depth: result.stats.depth_reached,
            });
            self.move_list_state
                .select(Some(self.move_history.len() - 1));

            self.refresh_legal_moves();
            if self.legal_moves.is_empty() {
                self.game_over = true;
                self.status = if self.pos.in_check() {
                    "Engine wins. You are checkmated.".to_string()
                } else {
                    "Draw — stalemate.".to_string()
                };
            } else {
                self.status = format!(
                    "Engine played {}  (d{} eval {}cp  {}ms)",
                    move_uci(&result.chosen),
                    result.stats.depth_reached,
                    result.stats.eval_cp,
                    result.stats.time_used_us / 1000,
                );
            }
        } else if self.engine_thinking {
            self.thinking_dots = (self.thinking_dots + 1) % 4;
            let dots = ".".repeat(self.thinking_dots as usize);
            self.status = format!("Engine thinking{}", dots);
        }
    }

    fn try_move(&mut self, from: u8, to: u8) -> bool {
        let matches: Vec<&Move> = self
            .legal_moves
            .iter()
            .filter(|m| m.from == from && m.to == to)
            .collect();
        if matches.is_empty() {
            return false;
        }

        let chosen = if matches.len() > 1 {
            *matches
                .iter()
                .find(|m| m.promo_piece == PieceType::Queen)
                .unwrap_or(&matches[0])
        } else {
            matches[0]
        };
        let chosen = *chosen;

        self.pos.make_move(&chosen);
        self.ply += 1;

        self.move_history.push(MoveRecord {
            uci: move_uci(&chosen),
            side: if self.player_is_white {
                "white"
            } else {
                "black"
            },
            eval: 0,
            depth: 0,
        });
        self.move_list_state
            .select(Some(self.move_history.len() - 1));

        self.selected_from = None;

        self.refresh_legal_moves();
        if self.legal_moves.is_empty() {
            self.game_over = true;
            self.status = if self.pos.in_check() {
                "You win! Engine is checkmated.".to_string()
            } else {
                "Draw — stalemate.".to_string()
            };
            return true;
        }

        self.start_engine_think();
        true
    }

    fn handle_key(&mut self, code: KeyCode) -> bool {
        match code {
            KeyCode::Char('q') | KeyCode::Esc => return true,
            KeyCode::Up | KeyCode::Char('k') => {
                if self.cursor_sq / 8 < 7 {
                    self.cursor_sq += 8;
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.cursor_sq / 8 > 0 {
                    self.cursor_sq -= 8;
                }
            }
            KeyCode::Left | KeyCode::Char('h') => {
                if !self.cursor_sq.is_multiple_of(8) {
                    self.cursor_sq -= 1;
                }
            }
            KeyCode::Right | KeyCode::Char('l') => {
                if self.cursor_sq % 8 < 7 {
                    self.cursor_sq += 1;
                }
            }
            KeyCode::Enter | KeyCode::Char(' ') => {
                if !self.is_player_turn() || self.game_over {
                    return false;
                }

                if let Some(from) = self.selected_from {
                    if self.cursor_sq == from {
                        self.selected_from = None;
                        self.status = "Deselected.".to_string();
                    } else {
                        let to = self.cursor_sq;
                        if !self.try_move(from, to) {
                            self.status = format!("Illegal move {}{}", sq_name(from), sq_name(to));
                        }
                    }
                } else {
                    let sq = self.cursor_sq;
                    let piece = self.pos.board[sq as usize];
                    if !piece.is_empty() && piece.color == self.pos.side_to_move {
                        let has_moves = self.legal_moves.iter().any(|m| m.from == sq);
                        if has_moves {
                            self.selected_from = Some(sq);
                            let count = self.legal_moves.iter().filter(|m| m.from == sq).count();
                            self.status = format!("Selected {} — {} moves", sq_name(sq), count);
                        } else {
                            self.status = "No legal moves for this piece.".to_string();
                        }
                    }
                }
            }
            _ => {}
        }
        false
    }

    fn draw(&mut self, frame: &mut Frame) {
        let outer = Layout::vertical([
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(3),
        ])
        .split(frame.area());

        let you = if self.player_is_white {
            "White"
        } else {
            "Black"
        };
        let header = Paragraph::new(Line::from(vec![
            Span::styled(" Play vs ", Style::default().fg(Color::White)),
            Span::styled(
                &self.engine_name,
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  (you are {}, {}ms/move)", you, self.think_time_ms),
                Style::default().fg(Color::DarkGray),
            ),
        ]))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        );
        frame.render_widget(header, outer[0]);

        let main =
            Layout::horizontal([Constraint::Length(32), Constraint::Min(25)]).split(outer[1]);

        let board_block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray))
            .title(" Board ");
        let board_inner = board_block.inner(main[0]);
        frame.render_widget(board_block, main[0]);

        let fen = self.pos.to_fen();
        render_board_with_highlights(
            &fen,
            self.cursor_sq,
            self.selected_from,
            &self.legal_moves,
            board_inner,
            frame,
        );

        let right = Layout::vertical([Constraint::Min(0), Constraint::Length(3)]).split(main[1]);

        let items: Vec<ListItem> = self
            .move_history
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let movenum = i / 2 + 1;
                let dot = if m.side == "white" { "." } else { "..." };
                let side_color = if m.side == "white" {
                    Color::Cyan
                } else {
                    Color::Red
                };
                let mut spans = vec![
                    Span::styled(
                        format!("{:>3}{:<3} ", movenum, dot),
                        Style::default().fg(Color::DarkGray),
                    ),
                    Span::styled(format!("{:<6}", m.uci), Style::default().fg(side_color)),
                ];
                if m.depth > 0 {
                    spans.push(Span::styled(
                        format!(" d{} {}cp", m.depth, m.eval),
                        Style::default().fg(Color::DarkGray),
                    ));
                }
                ListItem::new(Line::from(spans))
            })
            .collect();

        let move_list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray))
                    .title(format!(" Moves ({}) ", self.move_history.len())),
            )
            .highlight_style(Style::default().bg(Color::Rgb(40, 60, 80)));
        frame.render_stateful_widget(move_list, right[0], &mut self.move_list_state);

        let turn = if self.game_over {
            "GAME OVER".to_string()
        } else if self.engine_thinking {
            "Engine thinking...".to_string()
        } else if self.is_player_turn() {
            format!(
                "Your turn ({})",
                if self.pos.side_to_move == chess::Color::White {
                    "white"
                } else {
                    "black"
                }
            )
        } else {
            "Waiting...".to_string()
        };

        let status_style = if self.engine_thinking {
            Style::default().fg(Color::Magenta)
        } else {
            Style::default().fg(Color::Yellow)
        };

        let status = Paragraph::new(vec![
            Line::from(Span::styled(&self.status, status_style)),
            Line::from(Span::styled(turn, Style::default().fg(Color::DarkGray))),
        ])
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        );
        frame.render_widget(status, right[1]);

        let help = Paragraph::new(Line::from(vec![
            " ←→↑↓".dark_gray(),
            " move cursor ".dark_gray(),
            " Enter".dark_gray(),
            " select/move ".dark_gray(),
            " q".dark_gray(),
            " quit".dark_gray(),
        ]));
        frame.render_widget(help, outer[2]);
    }
}

fn render_board_with_highlights(
    fen: &str,
    cursor: u8,
    selected: Option<u8>,
    legal: &[Move],
    area: Rect,
    frame: &mut Frame,
) {
    let board = parse_board(fen);
    let light = Color::Rgb(220, 220, 220);
    let dark = Color::Rgb(80, 80, 80);
    let white_piece = Color::Rgb(30, 100, 220);
    let black_piece = Color::Rgb(220, 40, 40);
    let cursor_color = Color::Rgb(100, 160, 100);
    let selected_color = Color::Rgb(180, 180, 50);
    let target_color = Color::Rgb(140, 180, 100);

    let targets: Vec<u8> = if let Some(from) = selected {
        legal
            .iter()
            .filter(|m| m.from == from)
            .map(|m| m.to)
            .collect()
    } else {
        Vec::new()
    };

    let mut lines: Vec<Line> = Vec::new();
    for r in (0..8).rev() {
        let mut spans: Vec<Span> = Vec::new();
        spans.push(Span::styled(
            format!("  {} ", r + 1),
            Style::default().fg(Color::DarkGray),
        ));
        for f in 0..8 {
            let sq = (r * 8 + f) as u8;
            let is_dark = (r + f) % 2 == 0;
            let bg = if selected == Some(sq) {
                selected_color
            } else if sq == cursor {
                cursor_color
            } else if targets.contains(&sq) {
                target_color
            } else if is_dark {
                dark
            } else {
                light
            };
            let p = board[r][f];
            if p == '.' {
                let dot = if targets.contains(&sq) { " · " } else { "   " };
                spans.push(Span::styled(
                    dot,
                    Style::default().fg(Color::DarkGray).bg(bg),
                ));
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
    frame.render_widget(Paragraph::new(lines), area);
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let engine_name = args.get(1).map(|s| s.as_str()).unwrap_or("oblivion");
    let play_as = args.get(2).map(|s| s.as_str()).unwrap_or("white");
    let think_ms: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1000);
    let play_white = play_as != "black";

    if find_engine(engine_name).is_none() {
        eprintln!("Unknown engine: {}", engine_name);
        eprintln!("Usage: play [engine] [white|black] [think_ms]");
        eprintln!(
            "Available: {:?}",
            push_chess::candidates::ENGINE_REGISTRY
                .iter()
                .map(|e| e.name)
                .collect::<Vec<_>>()
        );
        std::process::exit(1);
    }

    let mut terminal = ratatui::init();
    let mut app = App::new(engine_name, play_white, think_ms);

    let result = loop {
        terminal.draw(|frame| app.draw(frame))?;

        // Non-blocking poll: check for input every 100ms so we can update thinking animation
        if event::poll(Duration::from_millis(100))?
            && let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
            && app.handle_key(key.code)
        {
            break Ok(());
        }

        // Check if engine finished thinking
        app.check_engine_result();
    };

    ratatui::restore();
    result
}
