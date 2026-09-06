use serde::{Deserialize, Serialize};

use crate::candidates::cataclysm::{Cataclysm, HashSize};
use crate::core::types::{Color, SearchBudget};
use crate::engine::Engine;
use crate::game::{Game, MoveOption, MovePreview, MoveResult, Outcome, SavedGame, Snapshot};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default, deny_unknown_fields)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
pub struct AnalysisOptions {
    pub time_ms: u32,
    pub max_nodes: u32,
    pub max_depth: u8,
}

impl Default for AnalysisOptions {
    fn default() -> Self {
        Self {
            time_ms: 250,
            max_nodes: 100_000,
            max_depth: 32,
        }
    }
}

impl AnalysisOptions {
    fn budget(self) -> Result<SearchBudget, String> {
        if self.time_ms > 5000
            || self.max_nodes > 2_000_000
            || !(1..=32).contains(&self.max_depth)
            || (self.time_ms == 0 && self.max_nodes == 0)
        {
            return Err("Search requires timeMs <= 5000, maxNodes <= 2000000, maxDepth 1..32, and a positive time or node limit".into());
        }
        Ok(SearchBudget {
            max_time_us: i64::from(self.time_ms) * 1000,
            max_nodes: i64::from(self.max_nodes),
            max_depth: i32::from(self.max_depth),
            seed: 0,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
pub struct Analysis {
    pub revision: u32,
    pub mv: MoveOption,
    pub nodes: u32,
    pub depth: u32,
    pub selective_depth: u32,
    /// Positive always means White is ahead, regardless of whose turn it is.
    pub white_eval_cp: i32,
    pub time_ms: f64,
    pub pv: Vec<MoveOption>,
}

/// One engine for the lifetime of the game; no per-move allocation of its table.
/// The mobile default is an 8 MiB table (native CLI keeps its 32 MiB default).
pub struct CataclysmSession {
    game: Game,
    engine: Cataclysm,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
pub struct Checkpoint {
    pub saved: SavedGame,
    pub revision: u32,
}

impl CataclysmSession {
    pub fn new(hash_mib: u32) -> Result<Self, String> {
        let size = HashSize::try_from(hash_mib).map_err(str::to_owned)?;
        Ok(Self {
            game: Game::default(),
            engine: Cataclysm::with_hash_size(size),
        })
    }
    pub fn snapshot(&self) -> Snapshot {
        self.game.snapshot()
    }
    pub fn save(&self) -> SavedGame {
        self.game.save()
    }

    /// Resume the same session after worker suspension. Unlike importing a
    /// game, this preserves its revision, including after undo/restart.
    pub fn recover(&mut self, checkpoint: Checkpoint) -> Result<Snapshot, String> {
        let mut game = Game::restore(checkpoint.saved).map_err(|e| e.to_string())?;
        game.restore_revision(checkpoint.revision);
        self.game = game;
        self.engine.new_game(Color::White, 0);
        Ok(self.snapshot())
    }

    /// Replacing a game is transactional: malformed input leaves it untouched.
    pub fn restore(&mut self, json: &str) -> Result<Snapshot, String> {
        let mut game = Game::from_json(json).map_err(|e| e.to_string())?;
        game.advance_revision_from(self.game.revision());
        self.game = game;
        self.engine.new_game(Color::White, 0);
        Ok(self.snapshot())
    }

    pub fn reset(&mut self, fen: Option<&str>) -> Result<Snapshot, String> {
        let mut game = match fen {
            Some(fen) => Game::from_fen(fen).map_err(|e| e.to_string())?,
            None => Game::default(),
        };
        game.advance_revision_from(self.game.revision());
        self.game = game;
        self.engine.new_game(Color::White, 0);
        Ok(self.snapshot())
    }

    pub fn preview(&self, id: u32, revision: u32) -> Result<MovePreview, String> {
        self.game.preview(id, revision).map_err(|e| e.to_string())
    }

    pub fn play(&mut self, id: u32, revision: u32) -> Result<MoveResult, String> {
        self.game.apply(id, revision).map_err(|e| e.to_string())
    }

    pub fn undo(&mut self, plies: u32, revision: u32) -> Result<Snapshot, String> {
        let snapshot = self.game.undo(plies, revision).map_err(|e| e.to_string())?;
        // Remove search entries from the abandoned continuation.
        self.engine.new_game(Color::White, 0);
        Ok(snapshot)
    }

    pub fn analyse(&mut self, options: AnalysisOptions, revision: u32) -> Result<Analysis, String> {
        let budget = options.budget()?;
        if revision != self.game.revision() {
            return Err("Position changed; refresh before searching".into());
        }
        if *self.game.outcome() != Outcome::Playing {
            return Err("Game has finished".into());
        }
        let mut pos = self.game.position().clone();
        let (mv, stats) = self.engine.choose_move(&mut pos, &budget);
        if !self.game.legal_moves().contains(&mv) {
            return Err("Engine returned an illegal move; board unchanged".into());
        }
        Ok(Analysis {
            revision,
            mv: mv.into(),
            nodes: stats.nodes as u32,
            depth: stats.depth_reached,
            selective_depth: stats.seldepth,
            white_eval_cp: stats.eval_cp
                * if self.game.position().side_to_move == Color::White {
                    1
                } else {
                    -1
                },
            time_ms: stats.time_used_us as f64 / 1000.0,
            pv: stats.pv.into_iter().map(Into::into).collect(),
        })
    }
}
