//! Coarse native actor pool. No Python callbacks, per-edge atomics, or spinning.
//! Workers retain private arenas. The coordinator serves whichever workers are
//! ready, with a bounded coalescing delay instead of a slowest-worker barrier.
use super::{
    ACTION_FIELDS, BOARD_FLOATS, BatchSearch, EFFECT_FIELDS, Features, SearchMetrics,
    SearchOptions, SearchResult, SearchRoot,
};
use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

enum Job {
    Start {
        roots: Vec<SearchRoot>,
        noise: Vec<Vec<f32>>,
        simulations: usize,
        candidates: usize,
        options: SearchOptions,
    },
    Reply {
        id: u64,
        logits: Vec<f32>,
        values: Vec<f32>,
        width: usize,
    },
    Shutdown,
}
enum Event {
    Ready {
        worker: usize,
        id: u64,
        features: Features,
    },
    Done {
        worker: usize,
        results: Vec<SearchResult>,
        metrics: SearchMetrics,
    },
    Failed(&'static str),
}

fn worker_loop(
    worker: usize,
    jobs: mpsc::Receiver<Job>,
    ready: mpsc::Sender<Event>,
    cancel: Arc<AtomicBool>,
) {
    let mut search: Option<BatchSearch> = None;
    while let Ok(job) = jobs.recv() {
        let changed = match job {
            Job::Shutdown => break,
            Job::Start {
                roots,
                noise,
                simulations,
                candidates,
                options,
            } => {
                if let Some(search) = &mut search {
                    search.restart(roots, noise, simulations, candidates, options)
                } else {
                    BatchSearch::with_options(roots, noise, simulations, candidates, options).map(
                        |mut s| {
                            s.set_cancellation(cancel.clone());
                            search = Some(s);
                        },
                    )
                }
            }
            Job::Reply {
                id,
                logits,
                values,
                width,
            } => search
                .as_mut()
                .ok_or("worker has no search")
                .and_then(|s| s.submit_for(id, &logits, &values, width)),
        };
        let result = changed.and_then(|()| {
            let search = search.as_mut().ok_or("worker has no search")?;
            match search.request()? {
                Some(features) => Ok(Event::Ready {
                    worker,
                    id: search.request_id().expect("pending"),
                    features,
                }),
                None => Ok(Event::Done {
                    worker,
                    results: search.results()?,
                    metrics: search.metrics(),
                }),
            }
        });
        match result {
            Ok(event) => {
                if ready.send(event).is_err() {
                    break;
                }
            }
            Err(error) => {
                let _ = ready.send(Event::Failed(error));
                break;
            }
        }
    }
}

struct Pending {
    worker: usize,
    id: u64,
    start: usize,
    rows: usize,
}

pub struct SearchRuntime {
    ports: Vec<mpsc::SyncSender<Job>>,
    threads: Vec<JoinHandle<()>>,
    inbox: mpsc::Receiver<Event>,
    cancel: Arc<AtomicBool>,
    ready: VecDeque<(usize, u64, Features)>,
    pending: Vec<Pending>,
    pending_lengths: Vec<i32>,
    lane_map: Vec<Vec<usize>>,
    results: Vec<Option<SearchResult>>,
    active: usize,
    completed: usize,
    epoch: u64,
    awaiting: bool,
    failed: bool,
    metrics: SearchMetrics,
    coalesce: Duration,
}

impl SearchRuntime {
    /// Zero chooses available parallelism. This does not promise core affinity.
    pub fn new(workers: usize, coalesce_us: u64) -> Result<Self, &'static str> {
        let count = if workers == 0 {
            thread::available_parallelism().map_or(1, usize::from)
        } else {
            workers
        };
        if count > 256 || coalesce_us > 100_000 {
            return Err("invalid worker/coalescing budget");
        }
        let (sender, inbox) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let mut ports: Vec<mpsc::SyncSender<Job>> = Vec::with_capacity(count);
        let mut threads: Vec<JoinHandle<()>> = Vec::with_capacity(count);
        for i in 0..count {
            let (tx, rx) = mpsc::sync_channel(1);
            let ready = sender.clone();
            let flag = cancel.clone();
            let failure = sender.clone();
            let handle = thread::Builder::new()
                .name(format!("pushzero-{i}"))
                .spawn(move || {
                    // A panic must wake the coordinator, not leave it waiting forever.
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        worker_loop(i, rx, ready, flag)
                    }));
                    if result.is_err() {
                        let _ = failure.send(Event::Failed("search worker panicked"));
                    }
                });
            let handle = match handle {
                Ok(handle) => handle,
                Err(_) => {
                    for port in ports {
                        let _ = port.send(Job::Shutdown);
                    }
                    for worker in threads {
                        let _ = worker.join();
                    }
                    return Err("could not create search worker");
                }
            };
            ports.push(tx);
            threads.push(handle);
        }
        Ok(Self {
            ports,
            threads,
            inbox,
            cancel,
            ready: VecDeque::new(),
            pending: Vec::new(),
            pending_lengths: Vec::new(),
            lane_map: Vec::new(),
            results: Vec::new(),
            active: 0,
            completed: 0,
            epoch: 0,
            awaiting: false,
            failed: false,
            metrics: SearchMetrics::default(),
            coalesce: Duration::from_micros(coalesce_us),
        })
    }

    pub fn start(
        &mut self,
        roots: Vec<SearchRoot>,
        noise: Vec<Vec<f32>>,
        simulations: usize,
        candidates: usize,
        options: SearchOptions,
    ) -> Result<(), &'static str> {
        if self.failed || self.ports.is_empty() || self.awaiting || self.completed != self.active {
            return Err("runtime closed, failed or busy");
        }
        if roots.is_empty()
            || roots.len() != noise.len()
            || simulations == 0
            || simulations > 1_000_000
            || candidates == 0
            || options.max_nodes_per_tree == 0
            || options.max_nodes_per_tree > u32::MAX as usize
            || roots.iter().zip(&noise).any(|(r, n)| {
                r.action_count() == 0
                    || r.action_count() != n.len()
                    || n.iter().any(|x| !x.is_finite())
            })
        {
            return Err("invalid runtime roots, noise or budget");
        }
        self.active = roots.len().min(self.ports.len());
        self.completed = 0;
        self.ready.clear();
        self.pending.clear();
        self.pending_lengths.clear();
        self.metrics = SearchMetrics::default();
        self.cancel.store(false, Ordering::Relaxed);
        self.results = (0..roots.len()).map(|_| None).collect();
        self.lane_map = (0..self.active).map(|_| Vec::new()).collect();
        let mut groups: Vec<(Vec<SearchRoot>, Vec<Vec<f32>>)> =
            (0..self.active).map(|_| (Vec::new(), Vec::new())).collect();
        for (i, (root, noise)) in roots.into_iter().zip(noise).enumerate() {
            let worker = i % self.active;
            self.lane_map[worker].push(i);
            groups[worker].0.push(root);
            groups[worker].1.push(noise);
        }
        for (worker, (roots, noise)) in groups.into_iter().enumerate() {
            if self.ports[worker]
                .send(Job::Start {
                    roots,
                    noise,
                    simulations,
                    candidates,
                    options,
                })
                .is_err()
            {
                self.failed = true;
                return Err("search worker disconnected");
            }
        }
        Ok(())
    }

    pub fn stop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    fn accept(&mut self, event: Event) -> Result<(), &'static str> {
        match event {
            Event::Ready {
                worker,
                id,
                features,
            } => self.ready.push_back((worker, id, features)),
            Event::Done {
                worker,
                results,
                metrics,
            } => {
                if results.len() != self.lane_map[worker].len() {
                    self.failed = true;
                    return Err("worker result shape mismatch");
                }
                for (&lane, result) in self.lane_map[worker].iter().zip(results) {
                    self.results[lane] = Some(result);
                }
                self.completed += 1;
                self.metrics.nodes += metrics.nodes;
                self.metrics.edges += metrics.edges;
                self.metrics.arena_bytes += metrics.arena_bytes;
                self.metrics.neural_rounds += metrics.neural_rounds;
            }
            Event::Failed(error) => {
                self.failed = true;
                self.stop();
                return Err(error);
            }
        }
        Ok(())
    }

    pub fn request(&mut self) -> Result<Option<(u64, Features)>, &'static str> {
        if self.failed {
            return Err("runtime failed; close it");
        }
        if self.awaiting {
            return Err("submit the pending runtime batch first");
        }
        while self.ready.is_empty() && self.completed < self.active {
            let event = self.inbox.recv().map_err(|_| "all workers disconnected")?;
            self.accept(event)?;
        }
        if self.ready.is_empty() {
            return Ok(None);
        }
        let start = Instant::now();
        while self.ready.len() + self.completed < self.active {
            let elapsed = start.elapsed();
            if elapsed >= self.coalesce {
                break;
            }
            match self.inbox.recv_timeout(self.coalesce - elapsed) {
                Ok(event) => self.accept(event)?,
                Err(mpsc::RecvTimeoutError::Timeout) => break,
                Err(_) => return Err("all workers disconnected"),
            }
        }
        self.pending.clear();
        if self.ready.len() == 1 {
            let (worker, id, features) = self.ready.pop_front().expect("one ready worker");
            self.pending.push(Pending {
                worker,
                id,
                start: 0,
                rows: features.rows,
            });
            return self.publish(features).map(Some);
        }
        let rows = self.ready.iter().map(|(_, _, f)| f.rows).sum();
        let width = self.ready.iter().map(|(_, _, f)| f.width).max().unwrap();
        let effect_width = self
            .ready
            .iter()
            .map(|(_, _, f)| f.effect_width)
            .max()
            .unwrap();
        let mut output = Features::empty(rows, width);
        output.effect_width = effect_width;
        output
            .effects
            .resize(rows * effect_width * EFFECT_FIELDS, 0);
        let mut start = 0;
        while let Some((worker, id, f)) = self.ready.pop_front() {
            output.boards[start * BOARD_FLOATS..(start + f.rows) * BOARD_FLOATS]
                .copy_from_slice(&f.boards);
            output.lengths[start..start + f.rows].copy_from_slice(&f.lengths);
            for row in 0..f.rows {
                let dst = (start + row) * width * ACTION_FIELDS;
                let src = row * f.width * ACTION_FIELDS;
                output.actions[dst..dst + f.width * ACTION_FIELDS]
                    .copy_from_slice(&f.actions[src..src + f.width * ACTION_FIELDS]);
                let dst = (start + row) * effect_width * EFFECT_FIELDS;
                let src = row * f.effect_width * EFFECT_FIELDS;
                output.effects[dst..dst + f.effect_width * EFFECT_FIELDS]
                    .copy_from_slice(&f.effects[src..src + f.effect_width * EFFECT_FIELDS]);
            }
            self.pending.push(Pending {
                worker,
                id,
                start,
                rows: f.rows,
            });
            start += f.rows;
        }
        self.publish(output).map(Some)
    }

    fn publish(&mut self, output: Features) -> Result<(u64, Features), &'static str> {
        self.pending_lengths.clone_from(&output.lengths);
        self.epoch = super::search::next_request_id()?;
        self.awaiting = true;
        Ok((self.epoch, output))
    }

    pub fn submit(
        &mut self,
        epoch: u64,
        logits: &[f32],
        values: &[f32],
        width: usize,
    ) -> Result<(), &'static str> {
        if !self.awaiting || epoch != self.epoch {
            return Err("stale runtime reply");
        }
        if values.len() != self.pending_lengths.len()
            || values.len().checked_mul(width) != Some(logits.len())
            || values.iter().any(|v| !v.is_finite() || v.abs() > 1.00001)
            || self.pending_lengths.iter().enumerate().any(|(i, &n)| {
                n as usize > width
                    || logits[i * width..i * width + n as usize]
                        .iter()
                        .any(|x| !x.is_finite())
            })
        {
            return Err("invalid runtime reply");
        }
        // Worker messages own their memory. No borrowed NumPy slice crosses a
        // detached thread or survives this call, even if the caller mutates it.
        for p in &self.pending {
            let job = Job::Reply {
                id: p.id,
                logits: logits[p.start * width..(p.start + p.rows) * width].to_vec(),
                values: values[p.start..p.start + p.rows].to_vec(),
                width,
            };
            if self.ports[p.worker].send(job).is_err() {
                self.failed = true;
                return Err("search worker disconnected");
            }
        }
        self.awaiting = false;
        Ok(())
    }

    pub fn results(&self) -> Result<Vec<&SearchResult>, &'static str> {
        if self.failed || self.awaiting || self.active == 0 || self.completed != self.active {
            return Err("runtime not finished");
        }
        self.results
            .iter()
            .map(|r| r.as_ref().ok_or("missing worker result"))
            .collect()
    }

    pub fn metrics(&self) -> &SearchMetrics {
        &self.metrics
    }

    pub fn close(&mut self) {
        self.stop();
        for port in self.ports.drain(..) {
            let _ = port.send(Job::Shutdown);
        }
        for worker in self.threads.drain(..) {
            let _ = worker.join();
        }
        self.failed = true;
    }
}

impl Drop for SearchRuntime {
    fn drop(&mut self) {
        self.close();
    }
}
