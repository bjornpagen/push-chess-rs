//! Tiny, embedded NNUE residual, trained on outcomes of whole saved games.
//! Both color perspectives are incrementally maintained with integer arithmetic.
use crate::core::types::*;
use std::sync::LazyLock;

pub const WIDTH: usize = 32;
type FeatureWeights = [[i16; WIDTH]; 768];
pub struct Network {
    weights: FeatureWeights,
    bias: [i32; WIDTH],
    output: [i16; WIDTH],
    pub fingerprint: u64,
}
static MODEL: LazyLock<Network> = LazyLock::new(|| {
    Network::decode(include_bytes!("network.bin"))
        .expect("embedded network has the specified shape")
});

impl Network {
    pub fn embedded() -> &'static Self {
        &MODEL
    }
    pub fn decode(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        if bytes.len() != (768 * WIDTH + 2 * WIDTH) * 2 {
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
        let pt = piece.piece_type as usize - 1;
        for perspective in 0..2 {
            let color = piece.color as usize ^ perspective;
            let square = if perspective == 0 { sq } else { sq ^ 56 };
            let weights = &model.weights[(color * 6 + pt) * 64 + square as usize];
            for (acc, &w) in self.hidden[perspective].iter_mut().zip(weights) {
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
