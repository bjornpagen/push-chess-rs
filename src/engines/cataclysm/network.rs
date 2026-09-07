//! Tiny, embedded NNUE residual, trained on outcomes of whole saved games.
//! Both color perspectives are incrementally maintained with integer arithmetic.
use crate::core::types::*;
use std::sync::{Arc, LazyLock};

pub const WIDTH: usize = 32;
pub const FEATURES: usize = 2 * 6 * 64;
type FeatureWeights = [[i16; WIDTH]; FEATURES];

pub(super) fn feature_index(piece: Piece, sq: u8, perspective: usize) -> usize {
    let color = piece.color as usize ^ perspective;
    let square = sq ^ if perspective == 0 { 0 } else { 56 };
    (color * 6 + piece.piece_type as usize - 1) * 64 + square as usize
}
pub struct Network {
    weights: FeatureWeights,
    bias: [i32; WIDTH],
    output: [i16; WIDTH],
    pub fingerprint: u64,
}
static MODEL: LazyLock<Arc<Network>> = LazyLock::new(|| {
    Arc::new(
        Network::decode(include_bytes!("network.bin"))
            .expect("embedded network has the specified shape"),
    )
});

/// Immutable weights shared by worker-owned searches. No process-global
/// candidate installation, leaked allocation, or per-node reference counting.
#[derive(Clone)]
pub struct Model(Arc<Network>);
impl Model {
    pub const BYTES: usize = (FEATURES + 2) * WIDTH * 2;
    pub fn embedded() -> Self {
        Self(Arc::clone(&MODEL))
    }
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        Network::decode(bytes)
            .map(|n| Self(Arc::new(n)))
            .map_err(|e| e.to_string())
    }
    pub fn fingerprint(&self) -> u64 {
        self.0.fingerprint
    }
    pub(super) fn network(&self) -> &Network {
        &self.0
    }
}

impl Network {
    pub fn embedded() -> &'static Self {
        &MODEL
    }
    pub fn decode(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        if bytes.len() != Model::BYTES {
            return Err("invalid Cataclysm model shape".into());
        }
        let mut words = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| i16::from_le_bytes([p[0], p[1]]));
        let weights = std::array::from_fn(|_| std::array::from_fn(|_| words.next().unwrap()));
        let bias = std::array::from_fn(|_| i32::from(words.next().unwrap()));
        let output = std::array::from_fn(|_| words.next().unwrap());
        Ok(Network {
            weights,
            bias,
            output,
            fingerprint: bytes.iter().fold(0xcbf29ce484222325, |h, b| {
                (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
            }),
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Accumulator {
    pub hidden: [[i32; WIDTH]; 2],
}

impl Accumulator {
    pub fn new(model: &Network) -> Self {
        Self {
            hidden: [model.bias; 2],
        }
    }
}

impl Accumulator {
    pub fn update(&mut self, piece: Piece, sq: u8, sign: i32, model: &Network) {
        for (perspective, hidden) in self.hidden.iter_mut().enumerate() {
            let weights = &model.weights[feature_index(piece, sq, perspective)];
            for (acc, &w) in hidden.iter_mut().zip(weights) {
                *acc += sign * i32::from(w);
            }
        }
    }
    pub fn white_residual(&self, model: &Network) -> i32 {
        let mut sum = 0i64;
        for i in 0..WIDTH {
            let difference = self.hidden[0][i].clamp(0, 256) - self.hidden[1][i].clamp(0, 256);
            sum += i64::from(difference) * i64::from(model.output[i]);
        }
        (sum / (256 * 16)) as i32
    }
}
