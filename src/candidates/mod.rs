pub mod chimera;
pub mod tempest;
pub mod colossus;
pub mod phantom;
pub mod titan;
pub mod specter;
pub mod avalanche;
pub mod flux;

use crate::engine::EngineEntry;

pub const ENGINE_REGISTRY: &[EngineEntry] = &[
    EngineEntry { name: "chimera", create: chimera::create },
    EngineEntry { name: "tempest", create: tempest::create },
    EngineEntry { name: "colossus", create: colossus::create },
    EngineEntry { name: "phantom", create: phantom::create },
    EngineEntry { name: "titan", create: titan::create },
    EngineEntry { name: "specter", create: specter::create },
    EngineEntry { name: "avalanche", create: avalanche::create },
    EngineEntry { name: "flux", create: flux::create },
];

pub fn find_engine(name: &str) -> Option<&'static EngineEntry> {
    ENGINE_REGISTRY.iter().find(|e| e.name == name)
}
