//! Resolve immutable participants once, before opening a corpus or allocating
//! search tables. Artifacts supply weights only; bumbledb owns all game facts.
use super::Result;
use push_chess::engine::{Engine, EngineEntry};
use push_chess::engines::{
    self, EngineInfo,
    cataclysm::{Cataclysm, HashSize, Model},
};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;

pub struct Candidate {
    name: String,
    model: Model,
    digest: [u8; 32],
}
impl Candidate {
    pub fn decode(name: &str, bytes: &[u8]) -> Result<Self> {
        if name.is_empty()
            || name.len() > 64
            || !name
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_')
        {
            return Err(
                "candidate name must be 1..=64 lowercase ASCII letters, digits, - or _".into(),
            );
        }
        if engines::find_engine(name).is_some() {
            return Err("a candidate cannot replace a built-in engine name".into());
        }
        Ok(Self {
            name: name.into(),
            model: Model::decode(bytes)?,
            digest: Sha256::digest(bytes).into(),
        })
    }
    pub fn load(name: &str, path: &Path) -> Result<Self> {
        let mut bytes = Vec::with_capacity(Model::BYTES + 1);
        std::fs::File::open(path)?
            .take((Model::BYTES + 1) as u64)
            .read_to_end(&mut bytes)?;
        Self::decode(name, &bytes)
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn network_fingerprint(&self) -> u64 {
        self.model.fingerprint()
    }
    pub fn create(&self) -> Box<dyn Engine> {
        Box::new(Cataclysm::with_model(
            self.name.clone(),
            HashSize::MiB32,
            self.model.clone(),
        ))
    }
}

enum Factory {
    Builtin(&'static EngineEntry),
    Residual(Candidate),
}
pub(super) struct Entrant {
    pub name: String,
    factory: Factory,
}
impl Entrant {
    pub fn info(&self) -> EngineInfo<'_> {
        match &self.factory {
            Factory::Builtin(entry) => engines::info(entry.name).expect("registered engine info"),
            Factory::Residual(_) => {
                let mut info = engines::info("cataclysm").unwrap();
                info.name = &self.name;
                info.hypothesis = "Cataclysm control search and handwritten baseline with an isolated candidate NNUE residual";
                info
            }
        }
    }
    pub fn network_fingerprint(&self) -> Option<u64> {
        match &self.factory {
            Factory::Builtin(_) => self
                .info()
                .neural_accumulator
                .then(engines::cataclysm::network_fingerprint),
            Factory::Residual(candidate) => Some(candidate.model.fingerprint()),
        }
    }
    pub fn identity(&self, binary: [u8; 32]) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(binary);
        digest.update(self.name.as_bytes());
        if let Factory::Residual(candidate) = &self.factory {
            // Strong content binding, not just a 64-bit accumulator checksum.
            digest.update([0]);
            digest.update(candidate.digest);
        }
        digest.finalize().into()
    }
    fn create(&self) -> Box<dyn Engine> {
        match &self.factory {
            Factory::Builtin(entry) => (entry.create)(),
            Factory::Residual(candidate) => candidate.create(),
        }
    }
}

pub struct Roster {
    pub(super) entries: Vec<Entrant>,
}
impl Roster {
    pub fn builtins(names: &[String]) -> Result<Self> {
        Self::resolve(names, Vec::new())
    }
    pub fn resolve(names: &[String], candidates: Vec<Candidate>) -> Result<Self> {
        let mut supplied = std::collections::BTreeMap::new();
        for candidate in candidates {
            if supplied.insert(candidate.name.clone(), candidate).is_some() {
                return Err("duplicate candidate name".into());
            }
        }
        let mut entries = Vec::with_capacity(names.len());
        let mut seen = std::collections::BTreeSet::new();
        for name in names {
            if !seen.insert(name) {
                return Err("duplicate entrant name".into());
            }
            let factory = match (engines::find_engine(name), supplied.remove(name)) {
                (Some(entry), None) => Factory::Builtin(entry),
                (None, Some(candidate)) => Factory::Residual(candidate),
                _ => return Err(format!("unresolved entrant {name}").into()),
            };
            entries.push(Entrant {
                name: name.clone(),
                factory,
            });
        }
        if !supplied.is_empty() {
            return Err("a supplied candidate is absent from --engines".into());
        }
        Ok(Self { entries })
    }
    pub(super) fn validate(&self, names: &[String]) -> Result<()> {
        if !self.entries.iter().map(|e| &e.name).eq(names) {
            return Err("resolved roster does not match scheduled entrants".into());
        }
        Ok(())
    }
    pub(super) fn create(&self, slot: usize) -> Box<dyn Engine> {
        self.entries[slot].create()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const CONTROL: &[u8] = Model::CONTROL_BYTES;

    #[test]
    fn candidates_cannot_shadow_controls_or_be_silently_ignored() {
        assert!(Candidate::decode("cataclysm", CONTROL).is_err());
        assert!(Candidate::decode("../candidate", CONTROL).is_err());
        assert!(Candidate::decode("aurora", b"bad").is_err());
        assert!(Roster::builtins(&["unknown".into()]).is_err());
        assert!(
            Roster::resolve(
                &["cataclysm".into()],
                vec![Candidate::decode("aurora", CONTROL).unwrap()]
            )
            .is_err()
        );
        let names = ["cataclysm".into(), "aurora".into()];
        let roster =
            Roster::resolve(&names, vec![Candidate::decode("aurora", CONTROL).unwrap()]).unwrap();
        roster.validate(&names).unwrap();
        assert!(
            roster
                .validate(&["aurora".into(), "cataclysm".into()])
                .is_err()
        );
        assert_ne!(
            roster.entries[0].identity([0; 32]),
            roster.entries[1].identity([0; 32])
        );
        assert_eq!(
            roster.entries[0].network_fingerprint(),
            roster.entries[1].network_fingerprint()
        );
        assert_eq!(roster.create(1).name(), "aurora");
    }

    #[test]
    fn candidate_identity_binds_full_weights_and_binary() {
        let mut changed = CONTROL.to_vec();
        changed[0] ^= 1;
        let names = ["aurora".into()];
        let a =
            Roster::resolve(&names, vec![Candidate::decode("aurora", CONTROL).unwrap()]).unwrap();
        let b =
            Roster::resolve(&names, vec![Candidate::decode("aurora", &changed).unwrap()]).unwrap();
        assert_ne!(
            a.entries[0].identity([0; 32]),
            b.entries[0].identity([0; 32])
        );
        assert_ne!(
            a.entries[0].identity([0; 32]),
            a.entries[0].identity([1; 32])
        );
    }
}
