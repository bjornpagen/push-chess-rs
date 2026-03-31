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
        for c in 0..2 {
            for p in 0..7 {
                for sq in 0..64 {
                    z.piece_keys[c][p][sq] = splitmix64(&mut state);
                }
            }
        }
        z.side_key = splitmix64(&mut state);
        for i in 0..16 {
            z.castling_keys[i] = splitmix64(&mut state);
        }
        for i in 0..8 {
            z.ep_keys[i] = splitmix64(&mut state);
        }
        z
    }
}

static ZOBRIST: LazyLock<Zobrist> = LazyLock::new(Zobrist::new);

pub fn zobrist_tables() -> &'static Zobrist {
    &ZOBRIST
}
