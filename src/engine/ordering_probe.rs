//! Opt-in, allocation-free timed comparisons on actual engine records.
//! The historical loop is a test instrument, never a production alternative.
use std::hint::black_box;
use std::time::Instant;

pub(crate) fn historical<T, const LAST: bool>(
    items: &mut [T],
    from: usize,
    score: impl Fn(&T) -> i32,
) {
    let mut best = from;
    for i in from + 1..items.len() {
        if score(&items[i]) > score(&items[best])
            || (LAST && score(&items[i]) == score(&items[best]))
        {
            best = i;
        }
    }
    items.swap(from, best);
}

fn two_pass<T, const LAST: bool>(items: &mut [T], from: usize, score: impl Fn(&T) -> i32) {
    let tail = &items[from..];
    let high = tail.iter().map(&score).max().unwrap();
    let best = if LAST {
        tail.iter().rposition(|x| score(x) == high)
    } else {
        tail.iter().position(|x| score(x) == high)
    }
    .unwrap();
    items.swap(from, from + best);
}

/// Many distinct lists avoid timing one predictor-memorized outcome sequence.
/// Copy/reset lies outside the timed region, with identical warm input for each
/// arm. Consumption caps cover early cutoffs as well as exhaustive traversal.
pub(crate) fn compare<T: Clone, const LAST: bool, F: Fn(&T) -> i32 + Copy>(
    name: &str,
    samples: &[Vec<T>],
    score: F,
) {
    assert!(!samples.is_empty() && samples.iter().all(|x| !x.is_empty()));
    type Pick<T, F> = fn(&mut [T], usize, F);
    let arms: [(&str, Pick<T, F>); 4] = [
        ("historical", historical::<T, LAST>),
        ("historical_repeat", historical::<T, LAST>),
        ("two_pass", two_pass::<T, LAST>),
        ("deployed", super::shared::pick_best::<T, LAST>),
    ];
    let mut scratch = samples.to_vec();
    println!(
        "{name}: records={} bytes/record={} scores={}",
        samples.len(),
        size_of::<T>(),
        samples.iter().map(Vec::len).sum::<usize>()
    );
    for cap in [1, 8, usize::MAX] {
        let picks: usize = samples.iter().map(|x| x.len().min(cap)).sum();
        let mut times: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
        for rep in 0..19 {
            for offset in 0..arms.len() {
                let arm = if rep % 2 == 0 {
                    offset
                } else {
                    arms.len() - 1 - offset
                };
                for (dst, src) in scratch.iter_mut().zip(samples) {
                    dst.clone_from(src);
                }
                let begin = Instant::now();
                for list in &mut scratch {
                    let count = list.len().min(cap);
                    for i in 0..count {
                        arms[arm].1(black_box(list), i, score);
                    }
                    black_box(&*list);
                }
                let elapsed = begin.elapsed().as_nanos() as f64 / picks as f64;
                if rep >= 4 {
                    times[arm].push(elapsed);
                }
            }
        }
        for values in &mut times {
            values.sort_by(f64::total_cmp);
        }
        let baseline = times[0][times[0].len() / 2];
        for (arm, (label, _)) in arms.iter().enumerate() {
            let values = &times[arm];
            let mid = values[values.len() / 2];
            println!(
                "{name} cap={cap} {label}: median={mid:.2} ns/pick, relative={:.3}, p10={:.2}, p90={:.2}",
                mid / baseline,
                values[1],
                values[13]
            );
        }
    }
}

pub(crate) fn positions() -> Vec<crate::core::position::Position> {
    let mut result = Vec::new();
    let mut state = crate::selfplay::State::default();
    let mut random = 187u64;
    while result.len() < 1024 {
        if state.white_value().is_some() {
            state = crate::selfplay::State::default();
        }
        result.push(state.position().clone());
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        let legal = state.legal_moves();
        state
            .play(legal[random as usize % legal.len()].id())
            .unwrap();
    }
    result
}
