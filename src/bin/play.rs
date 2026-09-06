use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph},
};

use std::io::IsTerminal;
use std::sync::mpsc;
use std::time::Duration;

use push_chess::candidates::find_engine;
use push_chess::core::movegen::generate_legal_moves;
use push_chess::core::position::{Position, start_position};
use push_chess::core::types::{
    self as chess, Move, PieceType, SearchBudget, SearchStats, SpecialMove,
};
use push_chess::engine::Engine;

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

struct SearchRequest {
    pos: Position,
    budget: SearchBudget,
}

/// One engine per game. Its transposition table and ordering history survive
/// between turns, exactly as they do in the tournament runner.
struct EngineWorker {
    requests: mpsc::Sender<SearchRequest>,
    results: mpsc::Receiver<EngineResult>,
}

impl EngineWorker {
    fn spawn(create: fn() -> Box<dyn Engine>, color: chess::Color) -> std::io::Result<Self> {
        let (requests, incoming) = mpsc::channel::<SearchRequest>();
        let (outgoing, results) = mpsc::channel();
        std::thread::Builder::new()
            .name("push-chess-engine".into())
            .stack_size(8 * 1024 * 1024)
            .spawn(move || {
                let mut engine = create();
                engine.new_game(color, 42);
                for mut request in incoming {
                    let (chosen, stats) = engine.choose_move(&mut request.pos, &request.budget);
                    if outgoing.send(EngineResult { chosen, stats }).is_err() {
                        break;
                    }
                }
            })?;
        Ok(Self { requests, results })
    }
}

enum Selection {
    None,
    Piece(u8),
    Alternatives { moves: Vec<Move>, at: usize },
}

impl Selection {
    fn origin_square(&self) -> Option<u8> {
        match self {
            Self::None => None,
            Self::Piece(sq) => Some(*sq),
            Self::Alternatives { moves, .. } => Some(moves[0].from),
        }
    }
}

fn move_label(mv: &Move) -> String {
    let route = match mv.path_kind {
        1 => " · long leg first",
        2 => " · short leg first",
        _ => "",
    };
    format!("{}{route}", move_uci(mv))
}

struct App {
    pos: Position,
    engine_name: String,
    player_is_white: bool,
    legal_moves: Vec<Move>,
    move_history: Vec<MoveRecord>,
    cursor_sq: u8,
    selection: Selection,
    move_list_state: ListState,
    status: String,
    game_over: bool,
    think_time_ms: u64,
    // Async engine state
    engine_thinking: bool,
    worker: EngineWorker,
    thinking_dots: u8,
}

impl App {
    fn new(engine_name: &str, play_white: bool, think_time_ms: u64) -> std::io::Result<Self> {
        let entry = find_engine(engine_name).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "unknown engine")
        })?;
        let worker = EngineWorker::spawn(
            entry.create,
            if play_white {
                chess::Color::Black
            } else {
                chess::Color::White
            },
        )?;
        let pos = start_position();
        let mut app = App {
            pos,
            engine_name: engine_name.to_string(),
            player_is_white: play_white,
            legal_moves: Vec::new(),
            move_history: Vec::new(),
            cursor_sq: if play_white { 4 } else { 60 },
            selection: Selection::None,
            move_list_state: ListState::default(),
            status: "Your move".to_string(),
            game_over: false,
            think_time_ms,
            engine_thinking: false,
            worker,
            thinking_dots: 0,
        };
        app.refresh_legal_moves();

        if !play_white {
            app.start_engine_think();
        }
        Ok(app)
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

    fn refresh_state(&mut self) {
        self.refresh_legal_moves();
        let ending = if self.legal_moves.is_empty() {
            Some(if self.pos.in_check() {
                if (self.pos.side_to_move == chess::Color::White) == self.player_is_white {
                    "Engine wins. You are checkmated."
                } else {
                    "You win! Engine is checkmated."
                }
            } else {
                "Draw — stalemate."
            })
        } else if self.pos.halfmove_clock >= 100 {
            Some("Draw — fifty-move rule.")
        } else if self
            .pos
            .undo_stack
            .iter()
            .filter(|u| u.zobrist == self.pos.zobrist)
            .count()
            >= 2
        {
            Some("Draw — threefold repetition.")
        } else {
            None
        };
        if let Some(message) = ending {
            self.game_over = true;
            self.status = message.into();
        }
    }

    fn start_engine_think(&mut self) {
        if self.game_over || self.engine_thinking {
            return;
        }
        self.refresh_state();
        if self.game_over {
            return;
        }
        let request = SearchRequest {
            pos: self.pos.clone(),
            budget: SearchBudget {
                max_time_us: (self.think_time_ms * 1000) as i64,
                seed: self.move_history.len() as u64,
                ..SearchBudget::default()
            },
        };
        if self.worker.requests.send(request).is_err() {
            self.game_over = true;
            self.status = "Engine stopped unexpectedly. Quit and start a new game.".into();
            return;
        }
        self.engine_thinking = true;
        self.thinking_dots = 0;
        self.status = "Engine thinking".into();
    }

    fn check_engine_result(&mut self) {
        if !self.engine_thinking {
            return;
        }
        match self.worker.results.try_recv() {
            Ok(result) => {
                self.engine_thinking = false;
                if !self.play_move(result.chosen, Some(&result.stats)) {
                    self.game_over = true;
                    self.status = "Engine returned an illegal move; board left unchanged.".into();
                }
            }
            Err(mpsc::TryRecvError::Empty) => {
                self.thinking_dots = (self.thinking_dots + 1) % 4;
                self.status = format!("Engine thinking{}", ".".repeat(self.thinking_dots as usize));
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                self.engine_thinking = false;
                self.game_over = true;
                self.status = "Engine stopped unexpectedly. Quit and start a new game.".into();
            }
        }
    }

    fn play_move(&mut self, chosen: Move, stats: Option<&SearchStats>) -> bool {
        if !self.legal_moves.contains(&chosen) {
            return false;
        }
        let side = if self.pos.side_to_move == chess::Color::White {
            "white"
        } else {
            "black"
        };
        self.pos.make_move(&chosen);
        self.move_history.push(MoveRecord {
            uci: move_label(&chosen),
            side,
            eval: stats.map_or(0, |s| s.eval_cp),
            depth: stats.map_or(0, |s| s.depth_reached),
        });
        self.move_list_state
            .select(Some(self.move_history.len() - 1));
        self.selection = Selection::None;
        self.status = if let Some(stats) = stats {
            format!(
                "Engine played {}  (d{} eval {}cp  {}ms)",
                move_label(&chosen),
                stats.depth_reached,
                stats.eval_cp,
                stats.time_used_us / 1000
            )
        } else {
            format!("You played {}", move_label(&chosen))
        };
        self.refresh_state();
        true
    }

    fn play_human_move(&mut self, chosen: Move) {
        if self.play_move(chosen, None) && !self.game_over {
            self.start_engine_think();
        }
    }

    fn try_move(&mut self, from: u8, to: u8) -> bool {
        if !self.is_player_turn() {
            return false;
        }
        let moves: Vec<_> = self
            .legal_moves
            .iter()
            .copied()
            .filter(|m| m.from == from && m.to == to)
            .collect();
        match moves.as_slice() {
            [] => false,
            [chosen] => {
                self.play_human_move(*chosen);
                true
            }
            _ => {
                self.selection = Selection::Alternatives { moves, at: 0 };
                self.status = "Choose the knight path or promotion, then press Enter.".into();
                true
            }
        }
    }

    fn handle_key(&mut self, code: KeyCode) -> bool {
        if code == KeyCode::Char('q') {
            return true;
        }
        if let Selection::Alternatives { moves, at } = &mut self.selection {
            match code {
                KeyCode::Down | KeyCode::Right | KeyCode::Tab | KeyCode::Char('j') => {
                    *at = (*at + 1) % moves.len()
                }
                KeyCode::Up | KeyCode::Left | KeyCode::BackTab | KeyCode::Char('k') => {
                    *at = (*at + moves.len() - 1) % moves.len()
                }
                KeyCode::Enter | KeyCode::Char(' ') => {
                    let chosen = moves[*at];
                    self.play_human_move(chosen);
                }
                KeyCode::Esc => {
                    self.selection = Selection::Piece(moves[0].from);
                    self.status = "Move choice cancelled.".into();
                }
                _ => {}
            }
            return false;
        }
        match code {
            KeyCode::Esc => {
                if matches!(self.selection, Selection::None) {
                    return true;
                }
                self.selection = Selection::None;
                self.status = "Deselected.".into();
            }
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

                if let Some(from) = self.selection.origin_square() {
                    if self.cursor_sq == from {
                        self.selection = Selection::None;
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
                            self.selection = Selection::Piece(sq);
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
            self.selection.origin_square(),
            &self.legal_moves,
            board_inner,
            frame,
        );

        let right = Layout::vertical([Constraint::Min(0), Constraint::Length(5)]).split(main[1]);

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
            " Tab".dark_gray(),
            " choose path/promotion ".dark_gray(),
            " q".dark_gray(),
            " quit".dark_gray(),
        ]));
        frame.render_widget(help, outer[2]);
        if let Selection::Alternatives { moves, at } = &self.selection {
            let screen = frame.area();
            let width = screen.width.min(58);
            let height = screen.height.min(moves.len() as u16 + 4);
            let popup = Rect::new(
                screen.x + (screen.width - width) / 2,
                screen.y + (screen.height - height) / 2,
                width,
                height,
            );
            let items: Vec<_> = moves
                .iter()
                .map(|mv| ListItem::new(move_label(mv)))
                .collect();
            let choices = List::new(items)
                .block(Block::bordered().title(" Choose move · ↑↓ / Tab · Enter · Esc "))
                .highlight_symbol("› ")
                .highlight_style(Style::default().fg(Color::Black).bg(Color::Cyan));
            frame.render_widget(Clear, popup);
            frame.render_stateful_widget(
                choices,
                popup,
                &mut ListState::default().with_selected(Some(*at)),
            );
        }
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

struct Config {
    engine: String,
    play_white: bool,
    think_ms: u64,
}

impl Config {
    fn parse(args: &[String]) -> Result<Option<Self>, String> {
        if args.iter().any(|s| s == "--help" || s == "-h") {
            return Ok(None);
        }
        if args.len() > 3 {
            return Err("Too many arguments.".into());
        }
        let engine = args.first().map(String::as_str).unwrap_or("cataclysm");
        if find_engine(engine).is_none() {
            return Err(format!("Unknown engine: {engine}"));
        }
        let play_white = match args.get(1).map(String::as_str).unwrap_or("white") {
            "white" => true,
            "black" => false,
            other => return Err(format!("Invalid side: {other}. Use white or black.")),
        };
        let think_ms = args
            .get(2)
            .map(|s| s.parse::<u64>())
            .transpose()
            .map_err(|_| "Think time must be a whole number of milliseconds.")?
            .unwrap_or(1000);
        if !(1..=60_000).contains(&think_ms) {
            return Err("Think time must be between 1 and 60000 milliseconds.".into());
        }
        Ok(Some(Self {
            engine: engine.into(),
            play_white,
            think_ms,
        }))
    }
}

fn help() -> String {
    format!(
        "Push Chess — play Cataclysm\n\nUsage: play [engine] [white|black] [think_ms]\n\nDefaults: cataclysm white 1000\nFrom this repository: cargo run --release\n\nArrows or hjkl: move cursor\nEnter/Space: select a piece, then its destination\nIf a move has multiple paths or promotions: arrows/Tab to choose, Enter to confirm\nEsc: cancel selection; q: quit\n\nEngines: {}\n",
        push_chess::candidates::ENGINE_REGISTRY
            .iter()
            .map(|e| e.name)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn run(terminal: &mut ratatui::DefaultTerminal, app: &mut App) -> std::io::Result<()> {
    loop {
        terminal.draw(|frame| app.draw(frame))?;
        if event::poll(Duration::from_millis(50))?
            && let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
            && app.handle_key(key.code)
        {
            return Ok(());
        }
        app.check_engine_result();
    }
}

struct RestoreTerminal;
impl Drop for RestoreTerminal {
    fn drop(&mut self) {
        ratatui::restore();
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let config = match Config::parse(&args) {
        Ok(Some(config)) => config,
        Ok(None) => {
            print!("{}", help());
            return Ok(());
        }
        Err(message) => {
            eprintln!("{message}\n\n{}", help());
            std::process::exit(2);
        }
    };
    if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        return Err("Play needs an interactive terminal. Run cargo run --release in your terminal; use --help for controls.".into());
    }
    let mut app = App::new(&config.engine, config.play_white, config.think_ms)?;
    let _restore = RestoreTerminal;
    let mut terminal = ratatui::try_init()?;
    run(&mut terminal, &mut app)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app_at(fen: &str) -> App {
        let mut app = App::new("cataclysm", true, 1).unwrap();
        app.pos.set_from_fen(fen);
        app.refresh_legal_moves();
        app
    }

    #[test]
    fn cataclysm_is_default_and_bad_arguments_are_rejected() {
        let config = Config::parse(&[]).unwrap().unwrap();
        assert_eq!(config.engine, "cataclysm");
        assert!(config.play_white);
        assert_eq!(config.think_ms, 1000);
        for args in [
            vec!["missing"],
            vec!["cataclysm", "purple"],
            vec!["cataclysm", "white", "0"],
            vec!["cataclysm", "white", "oops"],
            vec!["cataclysm", "white", "18446744073709551615"],
        ] {
            assert!(
                Config::parse(&args.into_iter().map(str::to_owned).collect::<Vec<_>>()).is_err()
            );
        }
        assert!(Config::parse(&["--help".into()]).unwrap().is_none());
    }

    #[test]
    fn engine_state_survives_multiple_turns() {
        #[derive(Default)]
        struct Stateful {
            calls: u64,
            initialized: bool,
        }
        impl Engine for Stateful {
            fn name(&self) -> &str {
                "stateful test"
            }
            fn new_game(&mut self, _: chess::Color, _: u64) {
                assert!(!self.initialized);
                self.initialized = true;
            }
            fn choose_move(&mut self, _: &mut Position, _: &SearchBudget) -> (Move, SearchStats) {
                assert!(self.initialized);
                self.calls += 1;
                (
                    Move::default(),
                    SearchStats {
                        nodes: self.calls,
                        ..SearchStats::default()
                    },
                )
            }
        }
        let worker =
            EngineWorker::spawn(|| Box::new(Stateful::default()), chess::Color::Black).unwrap();
        for expected in 1..=2 {
            worker
                .requests
                .send(SearchRequest {
                    pos: start_position(),
                    budget: SearchBudget::default(),
                })
                .unwrap();
            let result = worker.results.recv_timeout(Duration::from_secs(3)).unwrap();
            assert_eq!(result.stats.nodes, expected);
        }
    }

    #[test]
    fn knight_route_is_selected_explicitly_and_popup_renders() {
        let mut app = app_at("7k/8/8/4R3/4N3/8/8/K7 w - - 0 1");
        let before = app.pos.to_fen();
        assert!(app.try_move(28, 45));
        assert_eq!(app.pos.to_fen(), before);
        let Selection::Alternatives { moves, at } = &mut app.selection else {
            panic!("missing path chooser")
        };
        assert_eq!(moves.len(), 2);
        *at = moves.iter().position(|m| m.path_kind == 2).unwrap();
        let mut terminal =
            ratatui::Terminal::new(ratatui::backend::TestBackend::new(90, 26)).unwrap();
        terminal.draw(|frame| app.draw(frame)).unwrap();
        let screen = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect::<String>();
        assert!(screen.contains("Choose move"));
        assert!(screen.contains("short leg first"));
        assert!(!app.handle_key(KeyCode::Enter));
        assert_eq!(app.pos.undo_stack.last().unwrap().mv.path_kind, 2);
    }

    #[test]
    fn underpromotion_is_available_for_pushed_pawns() {
        let mut app = app_at("7k/P7/R7/8/8/8/8/K7 w - - 0 1");
        assert!(app.try_move(40, 48));
        let Selection::Alternatives { moves, at } = &mut app.selection else {
            panic!("missing promotion chooser")
        };
        *at = moves
            .iter()
            .position(|m| m.promo_piece == PieceType::Knight)
            .unwrap();
        app.handle_key(KeyCode::Enter);
        assert_eq!(app.pos.board[56].piece_type, PieceType::Knight);
        assert_eq!(app.pos.board[48].piece_type, PieceType::Rook);
    }

    #[test]
    fn cancelling_a_choice_does_not_move_or_quit() {
        let mut app = app_at("7k/8/8/4R3/4N3/8/8/K7 w - - 0 1");
        let before = app.pos.to_fen();
        assert!(app.try_move(28, 45));
        assert!(!app.handle_key(KeyCode::Esc));
        assert!(matches!(app.selection, Selection::Piece(28)));
        assert_eq!(app.pos.to_fen(), before);
        assert!(!app.engine_thinking);
    }

    #[test]
    fn equivalent_knight_paths_do_not_need_a_prompt() {
        let mut app = app_at("7k/8/8/8/4N3/8/8/K7 w - - 0 1");
        assert!(app.try_move(28, 45));
        assert!(matches!(app.selection, Selection::None));
        assert_eq!(app.pos.board[45].piece_type, PieceType::Knight);
    }

    #[test]
    fn fifty_move_draw_and_mate_precedence_match_the_runner() {
        let mut app = app_at("7k/8/8/8/8/8/8/K7 w - - 99 1");
        let mv = app
            .legal_moves
            .iter()
            .find(|m| m.from == 0 && m.to == 1)
            .copied()
            .unwrap();
        assert!(app.play_move(mv, None));
        assert!(app.game_over);
        assert!(app.status.contains("fifty-move"));
        let mut app = app_at("7k/6Q1/5K2/8/8/8/8/8 b - - 100 1");
        app.refresh_state();
        assert!(app.status.contains("Engine is checkmated"));
    }

    #[test]
    fn threefold_draw_uses_actual_game_history() {
        let mut app = app_at("7k/8/8/8/8/8/8/K7 w - - 0 1");
        for _ in 0..2 {
            for (from, to) in [(0, 1), (63, 62), (1, 0), (62, 63)] {
                let mv = app
                    .legal_moves
                    .iter()
                    .find(|m| m.from == from && m.to == to)
                    .copied()
                    .unwrap();
                assert!(app.play_move(mv, None));
            }
        }
        assert!(app.game_over);
        assert!(app.status.contains("threefold"));
    }

    #[test]
    fn invalid_engine_move_and_disconnected_worker_leave_board_intact() {
        for disconnect in [false, true] {
            let mut app = app_at("7k/8/8/8/8/8/8/K7 w - - 0 1");
            let before = app.pos.to_fen();
            let (outgoing, results) = mpsc::channel();
            let (requests, _incoming) = mpsc::channel();
            app.worker = EngineWorker { requests, results };
            app.engine_thinking = true;
            if !disconnect {
                outgoing
                    .send(EngineResult {
                        chosen: Move::default(),
                        stats: SearchStats::default(),
                    })
                    .unwrap();
            }
            drop(outgoing);
            app.check_engine_result();
            assert!(!app.engine_thinking);
            assert!(app.game_over);
            assert_eq!(app.pos.to_fen(), before);
        }
    }
}
