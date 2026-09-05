use crate::core::position::Position;
use crate::core::types::*;

/// Engine trait — equivalent to C++ Engine base class.
/// Each engine implements this to provide its search algorithm.
pub trait Engine: Send {
    fn name(&self) -> &str;
    fn new_game(&mut self, my_color: Color, game_seed: u64);
    fn choose_move(&mut self, pos: &mut Position, budget: &SearchBudget) -> (Move, SearchStats);
}

/// Factory function type for creating engines.
pub struct EngineEntry {
    pub name: &'static str,
    pub create: fn() -> Box<dyn Engine>,
}
