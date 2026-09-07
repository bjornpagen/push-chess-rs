use std::sync::LazyLock;

pub struct Zobrist {
    pub piece_keys: [[[u64; 64]; 7]; 2],
    pub side_key: u64,
    pub castling_keys: [u64; 16],
    pub ep_keys: [u64; 8],
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

impl Zobrist {
    fn new() -> Self {
        let mut state: u64 = 0x50555348434845;
        let mut z = Zobrist {
            piece_keys: [[[0; 64]; 7]; 2],
            side_key: 0,
            castling_keys: [0; 16],
            ep_keys: [0; 8],
        };
        for color in &mut z.piece_keys {
            for piece in color {
                for square in piece {
                    *square = splitmix64(&mut state);
                }
            }
        }
        z.side_key = splitmix64(&mut state);
        for key in &mut z.castling_keys {
            *key = splitmix64(&mut state);
        }
        for key in &mut z.ep_keys {
            *key = splitmix64(&mut state);
        }
        z
    }
}

static ZOBRIST: LazyLock<Zobrist> = LazyLock::new(Zobrist::new);

pub fn zobrist_tables() -> &'static Zobrist {
    &ZOBRIST
}
