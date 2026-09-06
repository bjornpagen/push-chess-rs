#[cfg(feature = "historical-engines")]
pub mod abyss;
#[cfg(feature = "historical-engines")]
pub mod apex;
#[cfg(feature = "historical-engines")]
pub mod apotheosis;
#[cfg(feature = "historical-engines")]
pub mod astra;
#[cfg(feature = "historical-engines")]
pub mod avalanche;
#[cfg(feature = "historical-engines")]
pub mod blade;
pub mod cataclysm;
#[cfg(feature = "historical-engines")]
pub mod catalyst;
#[cfg(feature = "historical-engines")]
pub mod chimera;
#[cfg(feature = "historical-engines")]
pub mod chronos;
#[cfg(feature = "historical-engines")]
pub mod colossus;
#[cfg(feature = "historical-engines")]
pub mod dynamo;
#[cfg(feature = "historical-engines")]
pub mod echo;
#[cfg(feature = "historical-engines")]
pub mod ember;
#[cfg(feature = "historical-engines")]
pub mod eternity;
#[cfg(feature = "historical-engines")]
pub mod flux;
#[cfg(feature = "historical-engines")]
pub mod fortress;
#[cfg(feature = "historical-engines")]
pub mod herald;
#[cfg(feature = "historical-engines")]
pub mod hyperion;
#[cfg(feature = "historical-engines")]
pub mod inferno;
#[cfg(feature = "historical-engines")]
pub mod juggernaut;
#[cfg(feature = "historical-engines")]
pub mod knight_king;
#[cfg(feature = "historical-engines")]
pub mod leviathan;
#[cfg(feature = "historical-engines")]
pub mod nexus;
#[cfg(feature = "historical-engines")]
pub mod oblivion;
#[cfg(feature = "historical-engines")]
pub mod omega;
#[cfg(feature = "historical-engines")]
pub mod ozymandias;
#[cfg(feature = "historical-engines")]
pub mod phantom;
#[cfg(feature = "historical-engines")]
pub mod prism;
#[cfg(feature = "historical-engines")]
pub mod pulse;
#[cfg(feature = "historical-engines")]
pub mod razor;
#[cfg(feature = "historical-engines")]
pub mod singularity;
#[cfg(feature = "historical-engines")]
pub mod specter;
#[cfg(feature = "historical-engines")]
pub mod surge;
#[cfg(feature = "historical-engines")]
pub mod tempest;
#[cfg(feature = "historical-engines")]
pub mod terminus;
#[cfg(feature = "historical-engines")]
pub mod titan;
#[cfg(feature = "historical-engines")]
pub mod torrent;
#[cfg(feature = "historical-engines")]
pub mod void_engine;
#[cfg(feature = "historical-engines")]
pub mod vortex;
#[cfg(feature = "historical-engines")]
pub mod warden;
#[cfg(feature = "historical-engines")]
pub mod wraith;
#[cfg(feature = "historical-engines")]
pub mod zenith;

use crate::engine::EngineEntry;

// Selectable engines. Other modules above retain historical experiments.
pub const ENGINE_REGISTRY: &[EngineEntry] = &[
    EngineEntry {
        name: "cataclysm",
        create: cataclysm::create,
    },
    #[cfg(feature = "historical-engines")]
    EngineEntry {
        name: "astra",
        create: astra::create,
    },
    #[cfg(feature = "historical-engines")]
    EngineEntry {
        name: "phantom",
        create: phantom::create,
    },
    #[cfg(feature = "historical-engines")]
    EngineEntry {
        name: "blade",
        create: blade::create,
    },
    #[cfg(feature = "historical-engines")]
    EngineEntry {
        name: "ember",
        create: ember::create,
    },
    #[cfg(feature = "historical-engines")]
    EngineEntry {
        name: "razor",
        create: razor::create,
    },
    #[cfg(feature = "historical-engines")]
    EngineEntry {
        name: "singularity",
        create: singularity::create,
    },
    #[cfg(feature = "historical-engines")]
    EngineEntry {
        name: "surge",
        create: surge::create,
    },
    #[cfg(feature = "historical-engines")]
    EngineEntry {
        name: "chimera",
        create: chimera::create,
    },
    #[cfg(feature = "historical-engines")]
    EngineEntry {
        name: "inferno",
        create: inferno::create,
    },
    #[cfg(feature = "historical-engines")]
    EngineEntry {
        name: "vortex",
        create: vortex::create,
    },
    #[cfg(feature = "historical-engines")]
    EngineEntry {
        name: "dynamo",
        create: dynamo::create,
    },
    #[cfg(feature = "historical-engines")]
    EngineEntry {
        name: "fortress",
        create: fortress::create,
    },
    #[cfg(feature = "historical-engines")]
    EngineEntry {
        name: "omega",
        create: omega::create,
    },
    #[cfg(feature = "historical-engines")]
    EngineEntry {
        name: "zenith",
        create: zenith::create,
    },
    #[cfg(feature = "historical-engines")]
    EngineEntry {
        name: "oblivion",
        create: oblivion::create,
    },
    #[cfg(feature = "historical-engines")]
    EngineEntry {
        name: "apotheosis",
        create: apotheosis::create,
    },
    #[cfg(feature = "historical-engines")]
    EngineEntry {
        name: "terminus",
        create: terminus::create,
    },
    #[cfg(feature = "historical-engines")]
    EngineEntry {
        name: "hyperion",
        create: hyperion::create,
    },
    #[cfg(feature = "historical-engines")]
    EngineEntry {
        name: "ozymandias",
        create: ozymandias::create,
    },
    #[cfg(feature = "historical-engines")]
    EngineEntry {
        name: "leviathan",
        create: leviathan::create,
    },
    #[cfg(feature = "historical-engines")]
    EngineEntry {
        name: "chronos",
        create: chronos::create,
    },
    #[cfg(feature = "historical-engines")]
    EngineEntry {
        name: "eternity",
        create: eternity::create,
    },
    #[cfg(feature = "historical-engines")]
    EngineEntry {
        name: "void",
        create: void_engine::create,
    },
    #[cfg(feature = "historical-engines")]
    EngineEntry {
        name: "abyss",
        create: abyss::create,
    },
];

pub fn find_engine(name: &str) -> Option<&'static EngineEntry> {
    ENGINE_REGISTRY.iter().find(|e| e.name == name)
}
#[cfg(feature = "historical-engines")]
mod support;
