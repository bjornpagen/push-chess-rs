//! Independent search lanes, a bounded CPU queue, and owned inference leases.
//! A lane is a small group of trees, not a thread and not a device batch.
use super::{
    ACTION_FIELDS, BOARD_FLOATS, BatchSearch, EFFECT_FIELDS, Features, SearchMetrics,
    SearchOptions, SearchResult, SearchRoot,
};
use std::{
    collections::{BTreeMap, VecDeque},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

enum Work {
    Start {
        retained: Option<Box<BatchSearch>>,
        roots: Vec<SearchRoot>,
        noise: Vec<Vec<f32>>,
        simulations: usize,
        candidates: usize,
        options: SearchOptions,
    },
    Reply {
        search: Box<BatchSearch>,
        id: u64,
        logits: Vec<f32>,
        values: Vec<f32>,
        width: usize,
    },
}
struct Task {
    lane: usize,
    work: Work,
}
enum Event {
    Ready {
        lane: usize,
        search: Box<BatchSearch>,
        features: Features,
        seconds: f64,
    },
    Done {
        lane: usize,
        search: Box<BatchSearch>,
        seconds: f64,
    },
    Failed(&'static str),
}
struct Ready {
    search: Box<BatchSearch>,
    features: Features,
}
enum Lane {
    Idle(Option<Box<BatchSearch>>),
    Working,
    Ready(Ready),
    Leased(Box<BatchSearch>),
}
struct Route {
    lane: usize,
    start: usize,
    rows: usize,
}
struct Lease {
    routes: Vec<Route>,
    lengths: Vec<i32>,
}

pub struct Completion {
    pub lane: usize,
    pub results: Vec<SearchResult>,
    pub metrics: SearchMetrics,
}
#[derive(Default)]
pub struct RuntimeMetrics {
    pub batches: usize,
    pub rows: usize,
    pub completed: usize,
    pub worker_seconds: f64,
    pub arena_bytes: usize,
}
pub struct Poll {
    pub request: Option<(u64, Features)>,
    pub completed: Vec<Completion>,
}

fn execute(
    work: Work,
    cancel: &Arc<AtomicBool>,
) -> Result<(Box<BatchSearch>, Option<Features>), &'static str> {
    let mut search = match work {
        Work::Start {
            retained,
            roots,
            noise,
            simulations,
            candidates,
            options,
        } => {
            if let Some(mut search) = retained {
                search.restart(roots, noise, simulations, candidates, options)?;
                search
            } else {
                let mut search = Box::new(BatchSearch::with_options(
                    roots,
                    noise,
                    simulations,
                    candidates,
                    options,
                )?);
                search.set_cancellation(cancel.clone());
                search
            }
        }
        Work::Reply {
            mut search,
            id,
            logits,
            values,
            width,
        } => {
            search.submit_for(id, &logits, &values, width)?;
            search
        }
    };
    let features = search.request()?;
    Ok((search, features))
}

/// Ownership crosses threads only in coarse jobs. The queue has at most one job
/// per lane; ready and leased lanes cannot also be running.
pub struct SearchRuntime {
    sender: Option<mpsc::SyncSender<Task>>,
    threads: Vec<JoinHandle<()>>,
    inbox: mpsc::Receiver<Event>,
    cancel: Arc<AtomicBool>,
    lanes: Vec<Lane>,
    ready: VecDeque<usize>,
    leases: BTreeMap<u64, Lease>,
    batch_rows: usize,
    failed: bool,
    metrics: RuntimeMetrics,
    arena_sizes: Vec<usize>,
}

impl SearchRuntime {
    /// Zero workers means available parallelism, not a core-affinity guarantee.
    pub fn new(workers: usize, lanes: usize, batch_rows: usize) -> Result<Self, &'static str> {
        let count = if workers == 0 {
            thread::available_parallelism().map_or(1, usize::from)
        } else {
            workers
        };
        if count > 256 || lanes == 0 || lanes > 4096 || batch_rows == 0 || batch_rows > 4096 {
            return Err("invalid worker/lane/batch capacity");
        }
        let (sender, jobs) = mpsc::sync_channel::<Task>(lanes);
        let jobs = Arc::new(Mutex::new(jobs));
        let (events, inbox) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let mut threads = Vec::new();
        for i in 0..count.min(lanes) {
            let jobs = jobs.clone();
            let events = events.clone();
            let flag = cancel.clone();
            let handle = thread::Builder::new()
                .name(format!("pushzero-{i}"))
                .spawn(move || {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        loop {
                            // The receiver lock protects dequeue only, never tree work.
                            let task = match jobs.lock().expect("job queue poisoned").recv() {
                                Ok(task) => task,
                                Err(_) => break,
                            };
                            let begin = Instant::now();
                            let event = match execute(task.work, &flag) {
                                Ok((search, Some(features))) => Event::Ready {
                                    lane: task.lane,
                                    search,
                                    features,
                                    seconds: begin.elapsed().as_secs_f64(),
                                },
                                Ok((search, None)) => Event::Done {
                                    lane: task.lane,
                                    search,
                                    seconds: begin.elapsed().as_secs_f64(),
                                },
                                Err(error) => Event::Failed(error),
                            };
                            let failed = matches!(event, Event::Failed(_));
                            if events.send(event).is_err() || failed {
                                break;
                            }
                        }
                    }));
                    if result.is_err() {
                        let _ = events.send(Event::Failed("search worker panicked"));
                    }
                });
            match handle {
                Ok(handle) => threads.push(handle),
                Err(_) => {
                    drop(sender);
                    for handle in threads {
                        let _ = handle.join();
                    }
                    return Err("could not create search worker");
                }
            }
        }
        Ok(Self {
            sender: Some(sender),
            threads,
            inbox,
            cancel,
            lanes: (0..lanes).map(|_| Lane::Idle(None)).collect(),
            ready: VecDeque::new(),
            leases: BTreeMap::new(),
            batch_rows,
            failed: false,
            metrics: RuntimeMetrics::default(),
            arena_sizes: vec![0; lanes],
        })
    }

    fn check_open(&self) -> Result<(), &'static str> {
        if self.failed || self.sender.is_none() {
            Err("runtime closed or failed")
        } else {
            Ok(())
        }
    }
    pub fn idle(&self) -> bool {
        self.lanes.iter().all(|s| matches!(s, Lane::Idle(_)))
    }
    pub fn lane_count(&self) -> usize {
        self.lanes.len()
    }
    pub fn metrics(&self) -> &RuntimeMetrics {
        &self.metrics
    }

    fn schedule(&mut self, lane: usize, work: Work) -> Result<(), &'static str> {
        // At most one queued/running task per lane: this bounded send cannot
        // wait for an inference reply or deadlock on a full job queue.
        if self
            .sender
            .as_ref()
            .ok_or("runtime closed")?
            .send(Task { lane, work })
            .is_err()
        {
            self.failed = true;
            return Err("search workers disconnected");
        }
        Ok(())
    }

    pub fn start(
        &mut self,
        lane: usize,
        roots: Vec<SearchRoot>,
        noise: Vec<Vec<f32>>,
        simulations: usize,
        candidates: usize,
        options: SearchOptions,
    ) -> Result<(), &'static str> {
        self.check_open()?;
        if lane >= self.lanes.len() || !matches!(self.lanes[lane], Lane::Idle(_)) {
            return Err("search lane is not idle");
        }
        if roots.is_empty()
            || roots.len() > self.batch_rows
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
            return Err("invalid search roots, noise, or capacity");
        }
        if self.idle() {
            self.cancel.store(false, Ordering::Relaxed);
        }
        let Lane::Idle(retained) = std::mem::replace(&mut self.lanes[lane], Lane::Working) else {
            unreachable!()
        };
        self.schedule(
            lane,
            Work::Start {
                retained,
                roots,
                noise,
                simulations,
                candidates,
                options,
            },
        )
    }

    fn accept(
        &mut self,
        event: Event,
        completed: &mut Vec<Completion>,
    ) -> Result<(), &'static str> {
        match event {
            Event::Ready {
                lane,
                search,
                features,
                seconds,
            } => {
                if !matches!(self.lanes[lane], Lane::Working) {
                    return Err("invalid ready lane transition");
                }
                self.metrics.worker_seconds += seconds;
                self.lanes[lane] = Lane::Ready(Ready { search, features });
                self.ready.push_back(lane);
            }
            Event::Done {
                lane,
                search,
                seconds,
            } => {
                if !matches!(self.lanes[lane], Lane::Working) {
                    return Err("invalid done lane transition");
                }
                self.metrics.worker_seconds += seconds;
                let metrics = search.metrics();
                self.arena_sizes[lane] = metrics.arena_bytes;
                self.metrics.arena_bytes =
                    self.metrics.arena_bytes.max(self.arena_sizes.iter().sum());
                completed.push(Completion {
                    lane,
                    results: search.results()?,
                    metrics,
                });
                self.lanes[lane] = Lane::Idle(Some(search));
                self.metrics.completed += 1;
            }
            Event::Failed(error) => {
                self.failed = true;
                return Err(error);
            }
        }
        Ok(())
    }

    /// Lease at most one device batch. Other independent leases may remain in
    /// flight, and replies may arrive in any order. Empty polls are not EOF.
    pub fn poll(&mut self, wait_us: u64) -> Result<Poll, &'static str> {
        self.check_open()?;
        if wait_us > 1_000_000 {
            return Err("poll wait exceeds one second");
        }
        let mut completed = Vec::new();
        while let Ok(event) = self.inbox.try_recv() {
            self.accept(event, &mut completed)?;
        }
        if self.ready.is_empty() && completed.is_empty() && !self.idle() && wait_us > 0 {
            match self.inbox.recv_timeout(Duration::from_micros(wait_us)) {
                Ok(event) => self.accept(event, &mut completed)?,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(_) => {
                    self.failed = true;
                    return Err("all search workers disconnected");
                }
            }
            while let Ok(event) = self.inbox.try_recv() {
                self.accept(event, &mut completed)?;
            }
        }
        let mut selected = Vec::new();
        let mut rows = 0;
        while let Some(&lane) = self.ready.front() {
            let Lane::Ready(r) = &self.lanes[lane] else {
                unreachable!()
            };
            if rows + r.features.rows > self.batch_rows {
                break;
            }
            rows += r.features.rows;
            self.ready.pop_front();
            let Lane::Ready(r) = std::mem::replace(&mut self.lanes[lane], Lane::Working) else {
                unreachable!()
            };
            self.lanes[lane] = Lane::Leased(r.search);
            selected.push((lane, r.features));
        }
        if selected.is_empty() {
            return Ok(Poll {
                request: None,
                completed,
            });
        }
        let mut routes = Vec::with_capacity(selected.len());
        let output = if selected.len() == 1 {
            let (lane, features) = selected.pop().expect("one selected");
            routes.push(Route {
                lane,
                start: 0,
                rows,
            });
            features
        } else {
            let width = selected.iter().map(|(_, f)| f.width).max().unwrap();
            let effect_width = selected.iter().map(|(_, f)| f.effect_width).max().unwrap();
            let mut out = Features::empty(rows, width);
            out.effect_width = effect_width;
            out.effects.resize(rows * effect_width * EFFECT_FIELDS, 0);
            let mut start = 0;
            for (lane, f) in selected {
                out.boards[start * BOARD_FLOATS..(start + f.rows) * BOARD_FLOATS]
                    .copy_from_slice(&f.boards);
                out.lengths[start..start + f.rows].copy_from_slice(&f.lengths);
                for row in 0..f.rows {
                    let dst = (start + row) * width * ACTION_FIELDS;
                    let src = row * f.width * ACTION_FIELDS;
                    out.actions[dst..dst + f.width * ACTION_FIELDS]
                        .copy_from_slice(&f.actions[src..src + f.width * ACTION_FIELDS]);
                    let dst = (start + row) * effect_width * EFFECT_FIELDS;
                    let src = row * f.effect_width * EFFECT_FIELDS;
                    out.effects[dst..dst + f.effect_width * EFFECT_FIELDS]
                        .copy_from_slice(&f.effects[src..src + f.effect_width * EFFECT_FIELDS]);
                }
                routes.push(Route {
                    lane,
                    start,
                    rows: f.rows,
                });
                start += f.rows;
            }
            out
        };
        let id = super::search::next_request_id()?;
        self.leases.insert(
            id,
            Lease {
                routes,
                lengths: output.lengths.clone(),
            },
        );
        self.metrics.batches += 1;
        self.metrics.rows += rows;
        Ok(Poll {
            request: Some((id, output)),
            completed,
        })
    }

    pub fn submit(
        &mut self,
        id: u64,
        logits: &[f32],
        values: &[f32],
        width: usize,
    ) -> Result<(), &'static str> {
        self.check_open()?;
        let lease = self
            .leases
            .get(&id)
            .ok_or("unknown or already consumed inference lease")?;
        if values.len() != lease.lengths.len()
            || values.len().checked_mul(width) != Some(logits.len())
            || values.iter().any(|v| !v.is_finite() || v.abs() > 1.00001)
            || lease.lengths.iter().enumerate().any(|(i, &n)| {
                n as usize > width
                    || logits[i * width..i * width + n as usize]
                        .iter()
                        .any(|v| !v.is_finite())
            })
        {
            return Err("invalid inference reply");
        }
        // Atomic boundary validation is complete before any worker is released.
        let lease = self.leases.remove(&id).expect("validated lease");
        for route in lease.routes {
            let Lane::Leased(search) =
                std::mem::replace(&mut self.lanes[route.lane], Lane::Working)
            else {
                unreachable!()
            };
            let id = search.request_id().expect("leased search");
            self.schedule(
                route.lane,
                Work::Reply {
                    search,
                    id,
                    logits: logits[route.start * width..(route.start + route.rows) * width]
                        .to_vec(),
                    values: values[route.start..route.start + route.rows].to_vec(),
                    width,
                },
            )?;
        }
        Ok(())
    }

    pub fn stop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn close(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        self.sender.take();
        for worker in self.threads.drain(..) {
            let _ = worker.join();
        }
        self.ready.clear();
        self.leases.clear();
        self.lanes.clear();
        while self.inbox.try_recv().is_ok() {}
    }
}
impl Drop for SearchRuntime {
    fn drop(&mut self) {
        self.close();
    }
}
