pub mod abyss;
pub mod apex;
pub mod apotheosis;
pub mod astra;
pub mod avalanche;
pub mod blade;
pub mod cataclysm;
pub mod catalyst;
pub mod chimera;
pub mod chronos;
pub mod colossus;
pub mod dynamo;
pub mod echo;
pub mod ember;
pub mod eternity;
pub mod flux;
pub mod fortress;
pub mod herald;
pub mod hyperion;
pub mod inferno;
pub mod juggernaut;
pub mod knight_king;
pub mod leviathan;
pub mod nexus;
pub mod oblivion;
pub mod omega;
pub mod ozymandias;
pub mod phantom;
pub mod prism;
pub mod pulse;
pub mod razor;
pub mod singularity;
pub mod specter;
pub mod surge;
pub mod tempest;
pub mod terminus;
pub mod titan;
pub mod torrent;
pub mod void_engine;
pub mod vortex;
pub mod warden;
pub mod wraith;
pub mod zenith;

use crate::engine::EngineEntry;

// Selectable engines. Other modules above retain historical experiments.
pub const ENGINE_REGISTRY: &[EngineEntry] = &[
    EngineEntry {
        name: "cataclysm",
        create: cataclysm::create,
    },
    EngineEntry {
        name: "astra",
        create: astra::create,
    },
    EngineEntry {
        name: "phantom",
        create: phantom::create,
    },
    EngineEntry {
        name: "blade",
        create: blade::create,
    },
    EngineEntry {
        name: "ember",
        create: ember::create,
    },
    EngineEntry {
        name: "razor",
        create: razor::create,
    },
    EngineEntry {
        name: "singularity",
        create: singularity::create,
    },
    EngineEntry {
        name: "surge",
        create: surge::create,
    },
    EngineEntry {
        name: "chimera",
        create: chimera::create,
    },
    EngineEntry {
        name: "inferno",
        create: inferno::create,
    },
    EngineEntry {
        name: "vortex",
        create: vortex::create,
    },
    EngineEntry {
        name: "dynamo",
        create: dynamo::create,
    },
    EngineEntry {
        name: "fortress",
        create: fortress::create,
    },
    EngineEntry {
        name: "omega",
        create: omega::create,
    },
    EngineEntry {
        name: "zenith",
        create: zenith::create,
    },
    EngineEntry {
        name: "oblivion",
        create: oblivion::create,
    },
    EngineEntry {
        name: "apotheosis",
        create: apotheosis::create,
    },
    EngineEntry {
        name: "terminus",
        create: terminus::create,
    },
    EngineEntry {
        name: "hyperion",
        create: hyperion::create,
    },
    EngineEntry {
        name: "ozymandias",
        create: ozymandias::create,
    },
    EngineEntry {
        name: "leviathan",
        create: leviathan::create,
    },
    EngineEntry {
        name: "chronos",
        create: chronos::create,
    },
    EngineEntry {
        name: "eternity",
        create: eternity::create,
    },
    EngineEntry {
        name: "void",
        create: void_engine::create,
    },
    EngineEntry {
        name: "abyss",
        create: abyss::create,
    },
];

pub fn find_engine(name: &str) -> Option<&'static EngineEntry> {
    ENGINE_REGISTRY.iter().find(|e| e.name == name)
}
mod support;
