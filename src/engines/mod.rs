//! Active research population only. Retired generations live in git history.
pub mod astra;
pub mod cataclysm;
mod support;

use crate::engine::EngineEntry;

pub const ENGINE_REGISTRY: &[EngineEntry] = &[
    EngineEntry {
        name: "cataclysm",
        create: cataclysm::create,
    },
    EngineEntry {
        name: "abacus",
        create: cataclysm::create_experiment::<1>,
    },
    EngineEntry {
        name: "resonance",
        create: cataclysm::create_experiment::<2>,
    },
    EngineEntry {
        name: "perimeter",
        create: cataclysm::create_experiment::<3>,
    },
    EngineEntry {
        name: "convoy",
        create: cataclysm::create_experiment::<4>,
    },
    EngineEntry {
        name: "kinetic",
        create: cataclysm::create_experiment::<5>,
    },
    EngineEntry {
        name: "flashpoint",
        create: cataclysm::create_experiment::<6>,
    },
    EngineEntry {
        name: "synthesis",
        create: cataclysm::create_experiment::<7>,
    },
    EngineEntry {
        name: "astra",
        create: astra::create,
    },
    EngineEntry {
        name: "sentinel",
        create: astra::create_experiment::<1>,
    },
    EngineEntry {
        name: "bastion",
        create: astra::create_experiment::<2>,
    },
    EngineEntry {
        name: "outrider",
        create: astra::create_experiment::<3>,
    },
    EngineEntry {
        name: "tactician",
        create: astra::create_experiment::<4>,
    },
    EngineEntry {
        name: "bedrock",
        create: astra::create_experiment::<5>,
    },
    EngineEntry {
        name: "meridian",
        create: astra::create_experiment::<6>,
    },
    EngineEntry {
        name: "waypoint",
        create: cataclysm::create_experiment::<8>,
    },
];

pub fn find_engine(name: &str) -> Option<&'static EngineEntry> {
    ENGINE_REGISTRY.iter().find(|e| e.name == name)
}

#[derive(serde::Serialize)]
pub struct EngineInfo<'a> {
    pub name: &'a str,
    pub lineage: &'static str,
    pub hypothesis: &'static str,
    pub neural_accumulator: bool,
    pub neural_evaluation: bool,
}
pub fn info(name: &str) -> Option<EngineInfo<'static>> {
    if let Some(p) = cataclysm::experiments::PROFILES
        .iter()
        .find(|p| p.name == name)
    {
        Some(EngineInfo {
            name: p.name,
            lineage: "cataclysm",
            hypothesis: p.hypothesis,
            neural_accumulator: true,
            neural_evaluation: p.neural_scale != 0,
        })
    } else {
        astra::PROFILES
            .iter()
            .find(|p| p.name == name)
            .map(|p| EngineInfo {
                name: p.name,
                lineage: "astra",
                hypothesis: p.hypothesis,
                neural_accumulator: false,
                neural_evaluation: false,
            })
    }
}
