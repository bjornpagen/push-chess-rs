//! Outcome-only NNUE trainer, implemented entirely in Rust. Sparse board inputs
//! are cheaper here than dense GPU tensors. No runtime training dependency.
//! Usage: train_cataclysm <model.bin> <report.json> <epochs> <max-game>
//!        [--warm-start <model.bin>] <db> [db...]
use push_chess::core::position::Position;
use rusqlite::{Connection, OpenFlags};
use std::collections::{BTreeMap, HashSet};
use std::fmt::Write;

const WIDTH: usize = 32;
const INPUTS: usize = 768;
const PARAMETERS: usize = INPUTS * WIDTH + 2 * WIDTH;
const BIAS: usize = INPUTS * WIDTH;
const OUTPUT: usize = BIAS + WIDTH;
const VALUES: [f32; 7] = [0., 100., 305., 365., 550., 1050., 0.];

struct Random(u64);
impl Random {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
}

struct Sample {
    features: [Vec<usize>; 2],
    material: f32,
    target: f32,
}

impl Sample {
    fn from_fen(fen: &str, target: f32) -> Self {
        let mut pos = Position::empty();
        pos.set_from_fen(fen);
        let mut s = Self {
            features: [Vec::new(), Vec::new()],
            material: 0.,
            target,
        };
        for (sq, p) in pos.board.iter().enumerate() {
            if p.is_empty() {
                continue;
            }
            let pt = p.piece_type as usize - 1;
            let c = p.color as usize;
            s.features[0].push((c * 6 + pt) * 64 + sq);
            s.features[1].push(((1 - c) * 6 + pt) * 64 + (sq ^ 56));
            s.material += VALUES[p.piece_type as usize] * if c == 0 { 1. } else { -1. };
        }
        s.material /= 400.;
        s
    }
}

#[derive(Clone)]
struct Model {
    p: Vec<f32>,
}

fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x.clamp(-30., 30.)).exp())
}

impl Model {
    fn read(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        if bytes.len() != PARAMETERS * 2 {
            return Err("invalid warm-start model shape".into());
        }
        let p = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .enumerate()
            .map(|(i, b)| {
                f32::from(i16::from_le_bytes(*b)) / if i < OUTPUT { 256. } else { 400. * 16. }
            })
            .collect();
        Ok(Self { p })
    }
    fn new(random: &mut Random) -> Self {
        let mut p = vec![0.; PARAMETERS];
        for v in &mut p[..BIAS] {
            *v = (random.unit() + random.unit() + random.unit() - 1.5) * 0.05;
        }
        p[BIAS..OUTPUT].fill(0.5);
        for v in &mut p[OUTPUT..] {
            *v = (random.unit() - 0.5) * 0.2;
        }
        Self { p }
    }
    fn forward(&self, s: &Sample) -> ([[f32; WIDTH]; 2], f32) {
        let mut z = [[0.; WIDTH]; 2];
        for (c, side) in z.iter_mut().enumerate() {
            side.copy_from_slice(&self.p[BIAS..OUTPUT]);
            for &feature in &s.features[c] {
                for (a, &weight) in side
                    .iter_mut()
                    .zip(&self.p[feature * WIDTH..(feature + 1) * WIDTH])
                {
                    *a += weight;
                }
            }
        }
        let mut logit = s.material;
        for i in 0..WIDTH {
            logit += (z[0][i].clamp(0., 1.) - z[1][i].clamp(0., 1.)) * self.p[OUTPUT + i];
        }
        (z, sigmoid(logit))
    }
    fn gradient(&self, s: &Sample, scale: f32, grad: &mut [f32]) {
        let (z, predicted) = self.forward(s);
        let error = (predicted - s.target) * scale;
        let mut delta = [[0.; WIDTH]; 2];
        for i in 0..WIDTH {
            grad[OUTPUT + i] += error * (z[0][i].clamp(0., 1.) - z[1][i].clamp(0., 1.));
            for c in 0..2 {
                if z[c][i] > 0. && z[c][i] < 1. {
                    delta[c][i] = error * self.p[OUTPUT + i] * if c == 0 { 1. } else { -1. };
                }
                grad[BIAS + i] += delta[c][i];
            }
        }
        for (c, deltas) in delta.iter().enumerate() {
            for &feature in &s.features[c] {
                for (g, d) in grad[feature * WIDTH..(feature + 1) * WIDTH]
                    .iter_mut()
                    .zip(deltas)
                {
                    *g += d;
                }
            }
        }
    }
    fn loss(&self, data: &[Sample]) -> f64 {
        data.iter()
            .map(|s| {
                let p = f64::from(self.forward(s).1).clamp(1e-7, 1. - 1e-7);
                -f64::from(s.target) * p.ln() - (1. - f64::from(s.target)) * (1. - p).ln()
            })
            .sum::<f64>()
            / data.len() as f64
    }
    fn quantized(&self) -> (Vec<u8>, Self) {
        let mut bytes = Vec::with_capacity(PARAMETERS * 2);
        let mut restored = self.clone();
        for (i, &value) in self.p.iter().enumerate() {
            let scale = if i < OUTPUT { 256. } else { 400. * 16. };
            let v = (value * scale).round();
            assert!(v.abs() < 32767., "network quantization overflow");
            bytes.extend_from_slice(&(v as i16).to_le_bytes());
            restored.p[i] = v / scale;
        }
        (bytes, restored)
    }
}

fn fingerprint(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
    })
}

type TrainingData = ([Vec<Sample>; 2], String);

fn load(paths: &[String], max_game: i64) -> Result<TrainingData, Box<dyn std::error::Error>> {
    let mut splits: [BTreeMap<String, (f32, usize)>; 2] = [BTreeMap::new(), BTreeMap::new()];
    let mut games = HashSet::new();
    let mut counts = [0; 2];
    let mut visited = 0;
    for path in paths {
        let db = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        let mut query = db.prepare(
            "SELECT game_id,result,termination FROM games WHERE game_id<=?1 ORDER BY game_id",
        )?;
        let records = query.query_map([max_game], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
            ))
        })?;
        for record in records {
            let (id, result, end) = record?;
            let target = match result.as_str() {
                "1-0" => 1.,
                "0-1" => 0.,
                "1/2-1/2" => 0.5,
                _ => continue,
            };
            if ![
                "checkmate",
                "stalemate",
                "threefold_repetition",
                "50_move_rule",
                "max_ply",
            ]
            .contains(&end.as_str())
            {
                continue;
            }
            let mut query=db.prepare("SELECT fen_before,fen_after,move_from,move_to,path_kind,promo_piece FROM moves WHERE game_id=?1 ORDER BY ply")?;
            let rows = query
                .query_map([id], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        format!(
                            "{}:{}:{}:{}:{}",
                            r.get::<_, String>(1)?,
                            r.get::<_, i64>(2)?,
                            r.get::<_, i64>(3)?,
                            r.get::<_, i64>(4)?,
                            r.get::<_, String>(5)?
                        ),
                    ))
                })?
                .collect::<Result<Vec<_>, _>>()?;
            if rows.is_empty() {
                continue;
            }
            let signature = rows
                .iter()
                .map(|(fen, mv)| format!("{fen}:{mv}\n"))
                .collect::<String>();
            if !games.insert(signature.clone()) {
                continue;
            }
            let split = usize::from(fingerprint(signature.as_bytes()).is_multiple_of(5));
            counts[split] += 1;
            let n = rows.len();
            for (i, (fen, _)) in rows.into_iter().enumerate() {
                let key = fen.split_whitespace().take(4).collect::<Vec<_>>().join(" ");
                let confidence = 0.45 + 0.55 * i as f32 / n as f32;
                let label = 0.5 + confidence * (target - 0.5);
                let entry = splits[split].entry(key).or_default();
                entry.0 += label;
                entry.1 += 1;
                visited += 1;
            }
        }
    }
    let shared: Vec<_> = splits[0]
        .keys()
        .filter(|fen| splits[1].contains_key(*fen))
        .cloned()
        .collect();
    for fen in &shared {
        splits[1].remove(fen);
    }
    let info = format!(
        "\"games\": {counts:?}, \"visited_move_rows\": {visited}, \"positions\": [{},{}], \"validation_overlap_removed\": {}",
        splits[0].len(),
        splits[1].len(),
        shared.len()
    );
    Ok((
        splits.map(|split| {
            split
                .into_iter()
                .map(|(fen, (sum, count))| Sample::from_fen(&fen, sum / count as f32))
                .collect()
        }),
        info,
    ))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 5 {
        return Err("usage: train_cataclysm <model.bin> <report.json> <epochs> <max-game> [--warm-start <model.bin>] <database> [database...]".into());
    }
    let epochs: usize = args[2].parse()?;
    let max_game: i64 = args[3].parse()?;
    if epochs == 0 || max_game < 0 {
        return Err("epochs must be positive and max-game nonnegative".into());
    }
    let warm = if args.get(4).is_some_and(|s| s == "--warm-start") {
        Some(args.get(5).ok_or("missing warm-start path")?)
    } else {
        None
    };
    let databases = &args[if warm.is_some() { 6 } else { 4 }..];
    if databases.is_empty() {
        return Err("at least one training database is required".into());
    }
    let ([train, valid], info) = load(databases, max_game)?;
    if train.is_empty() || valid.is_empty() {
        return Err("both training and held-out sets must be nonempty".into());
    }
    println!("{info}");
    let mut rng = Random(20260905);
    let mut model = if let Some(path) = warm {
        Model::read(path)?
    } else {
        Model::new(&mut rng)
    };
    let baseline = Model {
        p: vec![0.; PARAMETERS],
    }
    .loss(&valid);
    let mut moment = vec![0f32; PARAMETERS];
    let mut variance = vec![0f32; PARAMETERS];
    let mut gradient = vec![0f32; PARAMETERS];
    let mut indices: Vec<_> = (0..train.len()).collect();
    let mut best = (
        if warm.is_some() {
            model.loss(&valid)
        } else {
            f64::INFINITY
        },
        0,
        model.clone(),
    );
    let mut step = 0;
    let mut history = String::new();
    for epoch in 1..=epochs {
        for i in (1..indices.len()).rev() {
            let j = rng.next() as usize % (i + 1);
            indices.swap(i, j);
        }
        let lr = 0.0015 * (0.2 + 0.8 * (1. - (epoch - 1) as f32 / epochs as f32));
        for batch in indices.chunks(512) {
            gradient.fill(0.);
            for &i in batch {
                model.gradient(&train[i], 1. / batch.len() as f32, &mut gradient);
            }
            step += 1;
            let mfix = 1. - 0.9f32.powi(step);
            let vfix = 1. - 0.999f32.powi(step);
            for i in 0..PARAMETERS {
                let g = gradient[i] + 0.0001 * model.p[i];
                moment[i] = 0.9 * moment[i] + 0.1 * g;
                variance[i] = 0.999 * variance[i] + 0.001 * g * g;
                model.p[i] -= lr * (moment[i] / mfix) / ((variance[i] / vfix).sqrt() + 1e-8);
            }
        }
        let training = model.loss(&train);
        let validation = model.loss(&valid);
        println!("epoch {epoch}: training {training:.6}, held-out {validation:.6}");
        if !history.is_empty() {
            history.push(',');
        }
        write!(
            history,
            "{{\"epoch\":{epoch},\"training\":{training},\"validation\":{validation}}}"
        )?;
        if validation < best.0 {
            best = (validation, epoch, model.clone());
        }
        if epoch >= best.1 + 8 {
            println!("Stopping: eight epochs without held-out improvement");
            break;
        }
    }
    let (bytes, quantized) = best.2.quantized();
    std::fs::write(&args[0], &bytes)?;
    let report = format!(
        "{{\n{info},\n\"seed\":20260905,\"max_game\":{max_game},\"hidden\":{WIDTH},\"bytes\":{},\"model_fnv1a64\":\"{:016x}\",\n\"baseline_validation_cross_entropy\":{baseline},\"selected_validation_cross_entropy\":{},\"quantized_validation_cross_entropy\":{},\"selected_epoch\":{},\n\"labels\":\"smoothed final outcome only\",\"split\":\"FNV-1a whole-game signature modulo five, with duplicate games and overlapping validation positions removed\",\n\"epochs\":[{history}]\n}}\n",
        bytes.len(),
        fingerprint(&bytes),
        best.0,
        quantized.loss(&valid),
        best.1
    );
    std::fs::write(&args[1], report)?;
    println!(
        "Selected epoch {}; held-out loss {:.6}, material-only baseline {:.6}",
        best.1, best.0, baseline
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn analytic_gradient_matches_finite_difference() {
        let sample = Sample::from_fen("7k/8/8/8/3P4/8/8/K7 w - - 0 1", 0.8);
        let mut model = Model::new(&mut Random(42));
        let mut gradient = vec![0.; PARAMETERS];
        model.gradient(&sample, 1., &mut gradient);
        for at in [BIAS, OUTPUT, sample.features[0][0] * WIDTH] {
            let old = model.p[at];
            let eps = 0.002;
            model.p[at] = old + eps;
            let plus = model.loss(std::slice::from_ref(&sample));
            model.p[at] = old - eps;
            let minus = model.loss(std::slice::from_ref(&sample));
            model.p[at] = old;
            let numerical = (plus - minus) / (2. * f64::from(eps));
            assert!((numerical - f64::from(gradient[at])).abs() < 0.0001);
        }
    }
    #[test]
    fn color_flip_negates_prediction() {
        let model = Model::new(&mut Random(42));
        let a = Sample::from_fen("7k/8/8/8/3P4/8/8/K7 w - - 0 1", 0.5);
        let b = Sample::from_fen("k7/8/8/3p4/8/8/8/7K b - - 0 1", 0.5);
        assert!((model.forward(&a).1 + model.forward(&b).1 - 1.).abs() < 1e-6);
    }
}
