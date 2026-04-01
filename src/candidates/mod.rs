pub mod chimera;
pub mod tempest;
pub mod colossus;
pub mod phantom;
pub mod titan;
pub mod specter;
pub mod avalanche;
pub mod flux;
pub mod wraith;
pub mod ember;
pub mod nexus;
pub mod vortex;
pub mod razor;
pub mod surge;
pub mod blade;
pub mod pulse;
pub mod echo;
pub mod warden;
pub mod torrent;
pub mod prism;

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
    EngineEntry { name: "wraith", create: wraith::create },
    EngineEntry { name: "ember", create: ember::create },
    EngineEntry { name: "nexus", create: nexus::create },
    EngineEntry { name: "vortex", create: vortex::create },
    EngineEntry { name: "razor", create: razor::create },
    EngineEntry { name: "surge", create: surge::create },
    EngineEntry { name: "blade", create: blade::create },
    EngineEntry { name: "pulse", create: pulse::create },
    EngineEntry { name: "echo", create: echo::create },
    EngineEntry { name: "torrent", create: torrent::create },
    EngineEntry { name: "warden", create: warden::create },
    EngineEntry { name: "prism", create: prism::create },
];

pub fn find_engine(name: &str) -> Option<&'static EngineEntry> {
    ENGINE_REGISTRY.iter().find(|e| e.name == name)
}
