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
pub mod catalyst;
pub mod fortress;
pub mod knight_king;
pub mod inferno;
pub mod apex;
pub mod dynamo;
pub mod juggernaut;
pub mod herald;
pub mod singularity;
pub mod omega;
pub mod zenith;
pub mod oblivion;
pub mod apotheosis;
pub mod terminus;
pub mod hyperion;
pub mod ozymandias;
pub mod leviathan;
pub mod chronos;
pub mod eternity;

use crate::engine::EngineEntry;

// Active roster: gen9_brawl top 9 + singularity + omega + zenith
pub const ENGINE_REGISTRY: &[EngineEntry] = &[
    EngineEntry { name: "phantom", create: phantom::create },
    EngineEntry { name: "blade", create: blade::create },
    EngineEntry { name: "ember", create: ember::create },
    EngineEntry { name: "razor", create: razor::create },
    EngineEntry { name: "singularity", create: singularity::create },
    EngineEntry { name: "surge", create: surge::create },
    EngineEntry { name: "chimera", create: chimera::create },
    EngineEntry { name: "inferno", create: inferno::create },
    EngineEntry { name: "vortex", create: vortex::create },
    EngineEntry { name: "dynamo", create: dynamo::create },
    EngineEntry { name: "fortress", create: fortress::create },
    EngineEntry { name: "omega", create: omega::create },
    EngineEntry { name: "zenith", create: zenith::create },
    EngineEntry { name: "oblivion", create: oblivion::create },
    EngineEntry { name: "apotheosis", create: apotheosis::create },
    EngineEntry { name: "terminus", create: terminus::create },
    EngineEntry { name: "hyperion", create: hyperion::create },
    EngineEntry { name: "ozymandias", create: ozymandias::create },
    EngineEntry { name: "leviathan", create: leviathan::create },
    EngineEntry { name: "chronos", create: chronos::create },
    EngineEntry { name: "eternity", create: eternity::create },
];

pub fn find_engine(name: &str) -> Option<&'static EngineEntry> {
    ENGINE_REGISTRY.iter().find(|e| e.name == name)
}
