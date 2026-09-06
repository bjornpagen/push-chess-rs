//! UI-independent game sessions. Rust owns legality, history, and animation
//! transactions; callers choose a legal move ID at an explicit revision.
use serde::{Deserialize, Serialize};

use crate::core::movegen::generate_legal_moves;
use crate::core::position::{Position, start_position};
use crate::core::push::{resolve_knight_legs, resolve_push};
use crate::core::types::*;

pub const SAVE_VERSION: u8 = 1;
pub const MAX_PLIES: usize = 4096;
pub const MAX_SAVE_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GameError {
    InvalidPosition(String),
    InvalidSave,
    UnsupportedVersion,
    IllegalMove,
    StaleRevision,
    GameOver,
    HistoryLimit,
    NothingToUndo,
}

impl std::fmt::Display for GameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPosition(message) => write!(f, "Invalid position: {message}"),
            Self::InvalidSave => f.write_str("Invalid or oversized saved game"),
            Self::UnsupportedVersion => f.write_str("Unsupported saved-game version"),
            Self::IllegalMove => f.write_str("Move is not legal in this position"),
            Self::StaleRevision => f.write_str("Position changed; refresh before moving"),
            Self::GameOver => f.write_str("Game has finished"),
            Self::HistoryLimit => f.write_str("Game exceeds the 4096-ply session limit"),
            Self::NothingToUndo => f.write_str("No moves to undo"),
        }
    }
}
impl std::error::Error for GameError {}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
pub enum Outcome {
    Playing,
    Checkmate { winner: Color },
    Stalemate,
    FiftyMove,
    Repetition,
}

/// Shared with the native CLI. Mate takes precedence over either draw rule.
pub fn adjudicate(pos: &Position, legal: &[Move]) -> Outcome {
    let repeats = pos
        .undo_stack
        .iter()
        .filter(|u| u.zobrist == pos.zobrist)
        .take(2)
        .count();
    adjudicate_with_repetitions(pos, legal.is_empty(), repeats)
}

/// Search cursors supply repetitions from their immutable prefix + local path.
pub(crate) fn adjudicate_with_repetitions(
    pos: &Position,
    no_legal_moves: bool,
    repeats: usize,
) -> Outcome {
    if no_legal_moves {
        if pos.in_check() {
            Outcome::Checkmate {
                winner: opponent(pos.side_to_move),
            }
        } else {
            Outcome::Stalemate
        }
    } else if pos.halfmove_clock >= 100 {
        Outcome::FiftyMove
    } else if repeats >= 2 {
        Outcome::Repetition
    } else {
        Outcome::Playing
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
pub struct SavedGame {
    pub version: u8,
    pub initial_fen: String,
    /// Lossless move IDs include the knight route and promotion choice.
    pub moves: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
pub struct MoveOption {
    pub id: u32,
    pub from: Square,
    pub to: Square,
    pub path_kind: u8,
    pub special: SpecialMove,
    pub promotion: PieceType,
}

impl From<Move> for MoveOption {
    fn from(mv: Move) -> Self {
        Self {
            id: mv.id(),
            from: mv.from,
            to: mv.to,
            path_kind: mv.path_kind,
            special: mv.special,
            promotion: mv.promo_piece,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
pub struct BoardPiece {
    /// Stable through pushes, captures/undo, promotion, and save/reload.
    pub id: u8,
    pub square: Square,
    pub color: Color,
    pub kind: PieceType,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
pub struct Snapshot {
    pub revision: u32,
    pub fen: String,
    pub turn: Color,
    pub in_check: bool,
    pub outcome: Outcome,
    pub pieces: Vec<BoardPiece>,
    pub legal_moves: Vec<MoveOption>,
    pub ply: u32,
    pub last_move: Option<MoveOption>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
pub struct Displacement {
    pub piece_id: u8,
    pub from: Square,
    pub to: Square,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
pub struct AnimationPhase {
    /// Simultaneous within one phase; phases run sequentially for knights.
    pub displacements: Vec<Displacement>,
    pub captured: Option<BoardPiece>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
pub struct Promotion {
    pub piece_id: u8,
    pub square: Square,
    pub to: PieceType,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
pub struct MovePreview {
    pub revision: u32,
    pub mv: MoveOption,
    pub phases: Vec<AnimationPhase>,
    pub promotion: Option<Promotion>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
pub struct MoveResult {
    pub animation: MovePreview,
    pub snapshot: Snapshot,
}

pub struct Game {
    pos: Position,
    initial_fen: String,
    legal: Vec<Move>,
    outcome: Outcome,
    revision: u32,
    identities: [Option<u8>; 64],
    identity_history: Vec<[Option<u8>; 64]>,
}

/// A legal move plus its already-resolved presentation transaction. Application
/// consumes this proof instead of rechecking the same move or rebuilding IDs.
struct PreparedMove {
    mv: Move,
    animation: MovePreview,
    identities: [Option<u8>; 64],
}

impl Default for Game {
    fn default() -> Self {
        Self::from_position(start_position())
    }
}

impl Game {
    fn from_position(pos: Position) -> Self {
        let identities = std::array::from_fn(|sq| (!pos.board[sq].is_empty()).then_some(sq as u8));
        let mut game = Self {
            initial_fen: pos.to_fen(),
            pos,
            identities,
            legal: Vec::new(),
            outcome: Outcome::Playing,
            revision: 0,
            identity_history: Vec::new(),
        };
        game.refresh();
        game
    }

    pub fn from_fen(fen: &str) -> Result<Self, GameError> {
        Position::try_from_fen(fen)
            .map(Self::from_position)
            .map_err(|e| GameError::InvalidPosition(e.to_string()))
    }

    pub fn restore(save: SavedGame) -> Result<Self, GameError> {
        if save.version != SAVE_VERSION {
            return Err(GameError::UnsupportedVersion);
        }
        if save.moves.len() > MAX_PLIES {
            return Err(GameError::HistoryLimit);
        }
        let mut game = Self::from_fen(&save.initial_fen)?;
        for id in save.moves {
            game.apply(id, game.revision)?;
        }
        Ok(game)
    }

    pub fn from_json(json: &str) -> Result<Self, GameError> {
        if json.len() > MAX_SAVE_BYTES {
            return Err(GameError::InvalidSave);
        }
        Self::restore(serde_json::from_str(json).map_err(|_| GameError::InvalidSave)?)
    }

    pub fn save(&self) -> SavedGame {
        SavedGame {
            version: SAVE_VERSION,
            initial_fen: self.initial_fen.clone(),
            moves: self.pos.undo_stack.iter().map(|u| u.mv.id()).collect(),
        }
    }

    pub fn position(&self) -> &Position {
        &self.pos
    }
    pub fn revision(&self) -> u32 {
        self.revision
    }
    pub(crate) fn advance_revision_from(&mut self, previous: u32) {
        self.revision = previous.wrapping_add(1);
    }
    pub(crate) fn restore_revision(&mut self, revision: u32) {
        self.revision = revision;
    }
    pub fn outcome(&self) -> &Outcome {
        &self.outcome
    }
    pub fn legal_moves(&self) -> &[Move] {
        &self.legal
    }

    fn refresh(&mut self) {
        self.legal.clear();
        generate_legal_moves(&mut self.pos, &mut self.legal);
        self.outcome = adjudicate(&self.pos, &self.legal);
        if self.outcome != Outcome::Playing {
            self.legal.clear();
        }
    }

    fn checked_move(&self, id: u32, revision: u32) -> Result<Move, GameError> {
        if revision != self.revision {
            return Err(GameError::StaleRevision);
        }
        if self.outcome != Outcome::Playing {
            return Err(GameError::GameOver);
        }
        self.legal
            .iter()
            .find(|m| m.id() == id)
            .copied()
            .ok_or(GameError::IllegalMove)
    }

    pub fn snapshot(&self) -> Snapshot {
        let pieces = self
            .pos
            .board
            .iter()
            .enumerate()
            .filter_map(|(sq, p)| {
                self.identities[sq].map(|id| BoardPiece {
                    id,
                    square: sq as u8,
                    color: p.color,
                    kind: p.piece_type,
                })
            })
            .collect();
        Snapshot {
            revision: self.revision,
            fen: self.pos.to_fen(),
            turn: self.pos.side_to_move,
            in_check: self.pos.in_check(),
            outcome: self.outcome.clone(),
            pieces,
            legal_moves: self.legal.iter().copied().map(Into::into).collect(),
            ply: self.pos.undo_stack.len() as u32,
            last_move: self.pos.undo_stack.last().map(|u| u.mv.into()),
        }
    }

    pub fn preview(&self, id: u32, revision: u32) -> Result<MovePreview, GameError> {
        Ok(self.prepare(id, revision)?.animation)
    }

    fn prepare(&self, id: u32, revision: u32) -> Result<PreparedMove, GameError> {
        let mv = self.checked_move(id, revision)?;
        let mover = self.pos.board[mv.from as usize];
        let raw = if mv.special == SpecialMove::Castle {
            let base = if mover.color == Color::White { 0 } else { 56 };
            let (rook_from, rook_to) = if mv.to % 8 == 6 {
                (base + 7, base + 5)
            } else {
                (base, base + 3)
            };
            vec![(vec![(mv.from, mv.to), (rook_from, rook_to)], None)]
        } else if mv.special == SpecialMove::EnPassant {
            vec![(
                vec![(mv.from, mv.to)],
                Some(make_square(rank_of(mv.from), file_of(mv.to))),
            )]
        } else if mover.piece_type == PieceType::Knight {
            resolve_knight_legs(&self.pos, mv.from, mv.to, mv.path_kind == 1)
                .expect("legal knight path")
                .into_iter()
                .map(|p| (p.displacements().to_vec(), p.captured()))
                .collect()
        } else {
            let plan = resolve_push(
                &self.pos,
                mv.from,
                mv.to,
                (rank_of(mv.to) - rank_of(mv.from)).signum(),
                (file_of(mv.to) - file_of(mv.from)).signum(),
            )
            .expect("legal straight path");
            vec![(plan.displacements().to_vec(), plan.captured())]
        };
        let mut identities = self.identities;
        let mut board = self.pos.board;
        let mut phases = Vec::new();
        for (pairs, capture) in raw {
            let previous_ids = identities;
            let previous_board = board;
            let captured = capture.map(|sq| BoardPiece {
                id: previous_ids[sq as usize].expect("captured piece"),
                square: sq,
                color: board[sq as usize].color,
                kind: board[sq as usize].piece_type,
            });
            let displacements = pairs
                .iter()
                .map(|&(from, to)| Displacement {
                    piece_id: previous_ids[from as usize].expect("moving piece"),
                    from,
                    to,
                })
                .collect();
            for &(from, _) in &pairs {
                identities[from as usize] = None;
                board[from as usize] = Piece::default();
            }
            if let Some(sq) = capture {
                identities[sq as usize] = None;
                board[sq as usize] = Piece::default();
            }
            for &(from, to) in &pairs {
                identities[to as usize] = previous_ids[from as usize];
                board[to as usize] = previous_board[from as usize];
            }
            phases.push(AnimationPhase {
                displacements,
                captured,
            });
        }
        // Match core promotion order, including a pawn displaced by another man.
        let promotion = if mv.promo_piece != PieceType::None {
            let mut after = self.pos.clone();
            after.make_move(&mv);
            (0..64)
                .find(|&sq| {
                    board[sq].piece_type == PieceType::Pawn
                        && after.board[sq].piece_type == mv.promo_piece
                })
                .map(|sq| Promotion {
                    piece_id: identities[sq].expect("promoted piece"),
                    square: sq as u8,
                    to: mv.promo_piece,
                })
        } else {
            None
        };
        Ok(PreparedMove {
            mv,
            identities,
            animation: MovePreview {
                revision,
                mv: mv.into(),
                phases,
                promotion,
            },
        })
    }

    pub fn apply(&mut self, id: u32, revision: u32) -> Result<MoveResult, GameError> {
        let PreparedMove {
            mv,
            identities,
            animation,
        } = self.prepare(id, revision)?;
        if self.pos.undo_stack.len() >= MAX_PLIES {
            return Err(GameError::HistoryLimit);
        }
        self.identity_history.push(self.identities);
        self.identities = identities;
        self.pos.make_move(&mv);
        self.revision = self.revision.wrapping_add(1);
        self.refresh();
        Ok(MoveResult {
            animation,
            snapshot: self.snapshot(),
        })
    }

    pub fn undo(&mut self, plies: u32, revision: u32) -> Result<Snapshot, GameError> {
        if revision != self.revision {
            return Err(GameError::StaleRevision);
        }
        if plies == 0 || plies as usize > self.pos.undo_stack.len() {
            return Err(GameError::NothingToUndo);
        }
        for _ in 0..plies {
            self.pos.unmake_move();
            self.identities = self
                .identity_history
                .pop()
                .expect("identity history matches moves");
        }
        self.revision = self.revision.wrapping_add(1);
        self.refresh();
        Ok(self.snapshot())
    }
}
