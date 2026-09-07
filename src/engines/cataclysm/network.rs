//! Tiny, embedded NNUE residual, trained on outcomes of whole saved games.
//! Both color perspectives are incrementally maintained with integer arithmetic.
use crate::core::types::*;
use std::sync::{Arc, LazyLock};

pub const WIDTH: usize = 32;
pub const FEATURES: usize = 2 * 6 * 64;
pub const SLOTS: usize = 64;
pub const HIDDEN_CLIP: i32 = 256;
pub const RESIDUAL_DIVISOR: i64 = 8192;
pub const TEMPO: i32 = 14;
pub const SCORE_LIMIT: i32 = 28_000;
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
        Network::decode(Model::CONTROL_BYTES).expect("embedded network has the specified shape"),
    )
});

/// Immutable weights shared by worker-owned searches. No process-global
/// candidate installation, leaked allocation, or per-node reference counting.
#[derive(Clone)]
pub struct Model(Arc<Network>);
impl Model {
    pub const BYTES: usize = (FEATURES + 2) * WIDTH * 2;
    pub const FORMAT: &str = "cataclysm-residual-v1";
    pub const CONTROL_BYTES: &'static [u8] = include_bytes!("network.bin");
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
    /// Rebuild through the deployed board/evaluator, never a learning-only
    /// copy of the scoring formula. Tree search maintains this state by delta.
    pub fn evaluate(&self, pos: &crate::core::position::Position) -> i32 {
        super::eval::evaluate::<0>(&super::board::Board::with_model(pos, self.network()))
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
        for perspective in 0..2 {
            self.update_feature(
                perspective,
                feature_index(piece, sq, perspective),
                sign,
                model,
            );
        }
    }
    pub(super) fn update_feature(
        &mut self,
        perspective: usize,
        id: usize,
        sign: i32,
        model: &Network,
    ) {
        for (acc, &w) in self.hidden[perspective].iter_mut().zip(&model.weights[id]) {
            *acc += sign * i32::from(w);
        }
    }
    pub fn white_residual(&self, model: &Network) -> i32 {
        let mut sum = 0i64;
        for i in 0..WIDTH {
            let difference =
                self.hidden[0][i].clamp(0, HIDDEN_CLIP) - self.hidden[1][i].clamp(0, HIDDEN_CLIP);
            sum += i64::from(difference) * i64::from(model.output[i]);
        }
        (sum / RESIDUAL_DIVISOR) as i32
    }
}
