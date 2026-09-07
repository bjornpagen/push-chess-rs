//! Thin ownership boundary. Rules/search live in push_chess::selfplay.
//! Rust Vec allocations become NumPy-owned buffers; no element-wise boxing.
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods, ndarray::Array,
};
use push_chess::core::types::{Color, SearchBudget};
use push_chess::engines::cataclysm::learning as nnue;
use push_chess::selfplay::{self, ACTION_FIELDS, Encoded, Features};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::time::Instant;

type Observation<'py> = (
    Bound<'py, PyArray3<f32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray2<i32>>,
);
type NeuralBatch<'py> = (Bound<'py, PyArray4<f32>>, Bound<'py, PyArray3<i32>>);
type NnueBatch<'py> = (Bound<'py, PyArray3<u16>>, Bound<'py, PyArray1<i32>>);
type EvaluationRequest<'py> = (
    u64,
    Bound<'py, PyArray4<f32>>,
    Bound<'py, PyArray3<i32>>,
    Bound<'py, PyArray1<i32>>,
    Option<Bound<'py, PyArray3<i32>>>,
);
type EffectObservation<'py> = (
    Bound<'py, PyArray3<f32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray2<i32>>,
    Option<Bound<'py, PyArray2<i32>>>,
);
type ResultRow<'py> = (
    u32,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<u32>>,
    usize,
);
type DetailedRow<'py> = (
    u32,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<u32>>,
    usize,
    f32,
    f32,
);
type RuntimePoll<'py> = (
    Option<EvaluationRequest<'py>>,
    Vec<(usize, Vec<DetailedRow<'py>>)>,
);

fn detailed_results<'py>(
    py: Python<'py>,
    rows: impl IntoIterator<Item = selfplay::SearchResult>,
) -> Vec<DetailedRow<'py>> {
    rows.into_iter()
        .map(|r| {
            (
                r.mv,
                r.policy.into_pyarray(py),
                r.visits.into_pyarray(py),
                r.nodes,
                r.root_value,
                r.selected_value,
            )
        })
        .collect()
}

fn observation(py: Python<'_>, row: Encoded) -> Observation<'_> {
    let n = row.ids.len();
    (
        Array::from_shape_vec((32, 8, 8), row.board.to_vec())
            .unwrap()
            .into_pyarray(py),
        row.ids.into_pyarray(py),
        Array::from_shape_vec(
            (n, ACTION_FIELDS),
            row.actions.into_iter().flatten().collect(),
        )
        .unwrap()
        .into_pyarray(py),
    )
}
fn neural_batch(py: Python<'_>, f: Features) -> NeuralBatch<'_> {
    (
        Array::from_shape_vec((f.rows, 32, 8, 8), f.boards)
            .unwrap()
            .into_pyarray(py),
        Array::from_shape_vec((f.rows, f.width, ACTION_FIELDS), f.actions)
            .unwrap()
            .into_pyarray(py),
    )
}

fn nnue_batch(py: Python<'_>, rows: Vec<nnue::Input>) -> NnueBatch<'_> {
    let count = rows.len();
    let mut ids = Vec::with_capacity(count * 2 * nnue::SLOTS);
    let mut baselines = Vec::with_capacity(count);
    for row in rows {
        for perspective in row.ids {
            ids.extend_from_slice(&perspective);
        }
        baselines.push(row.baseline);
    }
    (
        Array::from_shape_vec((count, 2, nnue::SLOTS), ids)
            .unwrap()
            .into_pyarray(py),
        baselines.into_pyarray(py),
    )
}

#[pyfunction]
fn nnue_control(py: Python<'_>) -> Bound<'_, pyo3::types::PyBytes> {
    pyo3::types::PyBytes::new(py, nnue::CONTROL_BYTES)
}

#[pyfunction]
fn nnue_inputs<'py>(py: Python<'py>, states: Vec<PyRef<'_, State>>) -> PyResult<NnueBatch<'py>> {
    if states.len() > 4096 {
        return Err(PyValueError::new_err("NNUE batch exceeds 4096 positions"));
    }
    Ok(nnue_batch(
        py,
        states
            .iter()
            .map(|s| nnue::input(s.inner.position()))
            .collect(),
    ))
}

#[pyfunction]
fn nnue_evaluate<'py>(
    py: Python<'py>,
    model: &[u8],
    states: Vec<PyRef<'_, State>>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    if states.len() > 4096 {
        return Err(PyValueError::new_err("NNUE batch exceeds 4096 positions"));
    }
    let evaluator = nnue::Evaluator::decode(model).map_err(PyValueError::new_err)?;
    Ok(states
        .iter()
        .map(|s| evaluator.evaluate(s.inner.position()))
        .collect::<Vec<_>>()
        .into_pyarray(py))
}

fn evaluation_request(py: Python<'_>, id: u64, f: Features) -> EvaluationRequest<'_> {
    let effects = (f.effect_width != 0).then(|| {
        Array::from_shape_vec((f.rows, f.effect_width, selfplay::EFFECT_FIELDS), f.effects)
            .expect("native effect shape")
            .into_pyarray(py)
    });
    (
        id,
        Array::from_shape_vec((f.rows, 32, 8, 8), f.boards)
            .expect("native board shape")
            .into_pyarray(py),
        Array::from_shape_vec((f.rows, f.width, ACTION_FIELDS), f.actions)
            .expect("native action shape")
            .into_pyarray(py),
        f.lengths.into_pyarray(py),
        effects,
    )
}

fn effect_observation(py: Python<'_>, mut row: Encoded, effects: bool) -> EffectObservation<'_> {
    let tokens = effects.then(|| {
        let values = std::mem::take(&mut row.effects);
        Array::from_shape_vec(
            (values.len(), selfplay::EFFECT_FIELDS),
            values.into_iter().flatten().collect(),
        )
        .expect("native effect shape")
        .into_pyarray(py)
    });
    let (board, ids, actions) = observation(py, row);
    (board, ids, actions, tokens)
}

#[pyclass]
#[derive(Clone)]
struct State {
    inner: selfplay::State,
}
#[pymethods]
impl State {
    #[new]
    #[pyo3(signature = (fen=None))]
    fn new(fen: Option<&str>) -> PyResult<Self> {
        Ok(Self {
            inner: match fen {
                Some(f) => selfplay::State::from_fen(f).map_err(PyValueError::new_err)?,
                None => selfplay::State::default(),
            },
        })
    }
    fn copy(&self) -> Self {
        self.clone()
    }
    fn fen(&self) -> String {
        self.inner.position().to_fen()
    }
    fn turn(&self) -> u8 {
        self.inner.position().side_to_move as u8
    }
    fn outcome(&self) -> Option<f32> {
        self.inner.white_value()
    }
    fn legal_ids(&self) -> Vec<u32> {
        self.inner.legal_moves().iter().map(|m| m.id()).collect()
    }
    fn observation<'py>(&self, py: Python<'py>) -> Observation<'py> {
        observation(py, self.inner.encode())
    }
    fn observation_with_effects<'py>(&self, py: Python<'py>) -> EffectObservation<'py> {
        effect_observation(py, self.inner.encode_effects(), true)
    }
    fn play(&mut self, id: u32) -> PyResult<()> {
        self.inner.play(id).map_err(PyValueError::new_err)
    }
}

#[pyclass]
struct SearchBatch {
    inner: selfplay::BatchSearch,
    #[pyo3(get)]
    native_seconds: f64,
    #[pyo3(get)]
    ffi_calls: usize,
}
#[pymethods]
impl SearchBatch {
    #[new]
    #[pyo3(signature = (states, noise, simulations, candidates, effects=false, max_nodes=16384))]
    fn new(
        py: Python<'_>,
        states: Vec<PyRef<'_, State>>,
        noise: PyReadonlyArray2<'_, f32>,
        simulations: usize,
        candidates: usize,
        effects: bool,
        max_nodes: usize,
    ) -> PyResult<Self> {
        if states.len() != noise.shape()[0] {
            return Err(PyValueError::new_err("noise batch size mismatch"));
        }
        let width = noise.shape()[1];
        let data = noise.as_slice()?;
        if states.iter().any(|s| s.inner.legal_moves().len() > width) {
            return Err(PyValueError::new_err("noise action width mismatch"));
        }
        let noises = states
            .iter()
            .enumerate()
            .map(|(i, s)| data[i * width..i * width + s.inner.legal_moves().len()].to_vec())
            .collect();
        let roots: Vec<_> = states
            .iter()
            .map(|s| selfplay::SearchRoot::from_state(&s.inner))
            .collect();
        let inner = py
            .detach(|| {
                selfplay::BatchSearch::with_options(
                    roots,
                    noises,
                    simulations,
                    candidates,
                    selfplay::SearchOptions {
                        effects,
                        max_nodes_per_tree: max_nodes,
                    },
                )
            })
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            inner,
            native_seconds: 0.0,
            ffi_calls: 0,
        })
    }
    fn request<'py>(&mut self, py: Python<'py>) -> PyResult<Option<NeuralBatch<'py>>> {
        let start = Instant::now();
        // Expensive move generation/search runs without the Python GIL.
        let result = py
            .detach(|| self.inner.request())
            .map_err(PyValueError::new_err)?;
        self.native_seconds += start.elapsed().as_secs_f64();
        self.ffi_calls += 1;
        Ok(result.map(|f| neural_batch(py, f)))
    }
    fn submit(
        &mut self,
        logits: PyReadonlyArray2<'_, f32>,
        values: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<()> {
        let start = Instant::now();
        // Borrow contiguous arrays in place. Keep the GIL while reading caller
        // memory; no unsafe lifetime extension or concurrent mutable alias.
        if logits.shape()[0] != values.shape()[0] {
            return Err(PyValueError::new_err("evaluation rows mismatch"));
        }
        self.inner
            .submit(logits.as_slice()?, values.as_slice()?, logits.shape()[1])
            .map_err(PyValueError::new_err)?;
        self.native_seconds += start.elapsed().as_secs_f64();
        self.ffi_calls += 1;
        Ok(())
    }
    /// One boundary per neural round. Incoming memory is consumed before
    /// detaching; outgoing allocations are owned by the caller, never recycled.
    #[pyo3(signature = (reply_id=None, logits=None, values=None, stop=false))]
    fn advance<'py>(
        &mut self,
        py: Python<'py>,
        reply_id: Option<u64>,
        logits: Option<PyReadonlyArray2<'_, f32>>,
        values: Option<PyReadonlyArray1<'_, f32>>,
        stop: bool,
    ) -> PyResult<Option<EvaluationRequest<'py>>> {
        let start = Instant::now();
        match (reply_id, logits, values) {
            (Some(id), Some(logits), Some(values)) => {
                if logits.shape()[0] != values.shape()[0] {
                    return Err(PyValueError::new_err("reply row mismatch"));
                }
                self.inner
                    .submit_for(
                        id,
                        logits.as_slice()?,
                        values.as_slice()?,
                        logits.shape()[1],
                    )
                    .map_err(PyValueError::new_err)?;
            }
            (None, None, None) => {}
            _ => {
                return Err(PyValueError::new_err(
                    "reply ID, logits and values must be provided together",
                ));
            }
        }
        if stop {
            self.inner.stop();
        }
        let result = py
            .detach(|| self.inner.request())
            .map_err(PyValueError::new_err)?;
        self.native_seconds += start.elapsed().as_secs_f64();
        self.ffi_calls += 1;
        Ok(result
            .map(|f| evaluation_request(py, self.inner.request_id().expect("pending request"), f)))
    }

    fn finish<'py>(&self, py: Python<'py>) -> PyResult<Vec<DetailedRow<'py>>> {
        Ok(detailed_results(
            py,
            self.inner.results().map_err(PyValueError::new_err)?,
        ))
    }

    fn metrics(&self) -> std::collections::BTreeMap<&'static str, usize> {
        let m = self.inner.metrics();
        std::collections::BTreeMap::from([
            ("nodes", m.nodes),
            ("edges", m.edges),
            ("arena_bytes", m.arena_bytes),
            ("neural_rounds", m.neural_rounds as usize),
        ])
    }
    fn results<'py>(&self, py: Python<'py>) -> PyResult<Vec<ResultRow<'py>>> {
        Ok(self
            .inner
            .results()
            .map_err(PyValueError::new_err)?
            .into_iter()
            .map(|r| {
                (
                    r.mv,
                    r.policy.into_pyarray(py),
                    r.visits.into_pyarray(py),
                    r.nodes,
                )
            })
            .collect())
    }
}

/// Mutex makes the class Sync; PyO3's exclusive method borrow serializes use.
/// Detached operations borrow only the Send Rust runtime, never Python objects.
#[pyclass]
struct SearchRuntime {
    inner: std::sync::Mutex<selfplay::SearchRuntime>,
    #[pyo3(get)]
    native_seconds: f64,
    #[pyo3(get)]
    ffi_calls: usize,
}

#[pymethods]
impl SearchRuntime {
    #[new]
    #[pyo3(signature = (workers, lanes, batch_rows))]
    fn new(py: Python<'_>, workers: usize, lanes: usize, batch_rows: usize) -> PyResult<Self> {
        let inner = py
            .detach(|| selfplay::SearchRuntime::new(workers, lanes, batch_rows))
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            inner: std::sync::Mutex::new(inner),
            native_seconds: 0.0,
            ffi_calls: 0,
        })
    }

    #[pyo3(signature = (lane, states, noise, simulations, candidates, effects=false, max_nodes=16384))]
    #[allow(clippy::too_many_arguments)] // Explicit bulk Python boundary, including hidden Python token.
    fn start(
        &mut self,
        py: Python<'_>,
        lane: usize,
        states: Vec<PyRef<'_, State>>,
        noise: PyReadonlyArray2<'_, f32>,
        simulations: usize,
        candidates: usize,
        effects: bool,
        max_nodes: usize,
    ) -> PyResult<()> {
        if states.len() != noise.shape()[0] {
            return Err(PyValueError::new_err("noise batch size mismatch"));
        }
        let width = noise.shape()[1];
        let data = noise.as_slice()?;
        if states.iter().any(|s| s.inner.legal_moves().len() > width) {
            return Err(PyValueError::new_err("noise width mismatch"));
        }
        let roots = states
            .iter()
            .map(|s| selfplay::SearchRoot::from_state(&s.inner))
            .collect();
        let noises = states
            .iter()
            .enumerate()
            .map(|(i, s)| data[i * width..i * width + s.inner.legal_moves().len()].to_vec())
            .collect();
        let inner = self
            .inner
            .get_mut()
            .map_err(|_| PyValueError::new_err("runtime poisoned"))?;
        py.detach(|| {
            inner.start(
                lane,
                roots,
                noises,
                simulations,
                candidates,
                selfplay::SearchOptions {
                    effects,
                    max_nodes_per_tree: max_nodes,
                },
            )
        })
        .map_err(PyValueError::new_err)?;
        Ok(())
    }

    fn submit(
        &mut self,
        reply_id: u64,
        logits: PyReadonlyArray2<'_, f32>,
        values: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<()> {
        let start = Instant::now();
        let inner = self
            .inner
            .get_mut()
            .map_err(|_| PyValueError::new_err("runtime poisoned"))?;
        if logits.shape()[0] != values.shape()[0] {
            return Err(PyValueError::new_err("reply row mismatch"));
        }
        inner
            .submit(
                reply_id,
                logits.as_slice()?,
                values.as_slice()?,
                logits.shape()[1],
            )
            .map_err(PyValueError::new_err)?;
        self.native_seconds += start.elapsed().as_secs_f64();
        self.ffi_calls += 1;
        Ok(())
    }

    #[pyo3(signature = (wait_us=1000))]
    fn poll<'py>(&mut self, py: Python<'py>, wait_us: u64) -> PyResult<RuntimePoll<'py>> {
        let start = Instant::now();
        let inner = self
            .inner
            .get_mut()
            .map_err(|_| PyValueError::new_err("runtime poisoned"))?;
        let result = py
            .detach(|| inner.poll(wait_us))
            .map_err(PyValueError::new_err)?;
        self.native_seconds += start.elapsed().as_secs_f64();
        self.ffi_calls += 1;
        Ok((
            result.request.map(|(id, f)| evaluation_request(py, id, f)),
            result
                .completed
                .into_iter()
                .map(|c| (c.lane, detailed_results(py, c.results)))
                .collect(),
        ))
    }

    #[getter]
    fn idle(&mut self) -> PyResult<bool> {
        Ok(self
            .inner
            .get_mut()
            .map_err(|_| PyValueError::new_err("runtime poisoned"))?
            .idle())
    }

    fn stop(&mut self) -> PyResult<()> {
        self.inner
            .get_mut()
            .map_err(|_| PyValueError::new_err("runtime poisoned"))?
            .stop();
        Ok(())
    }

    #[getter]
    fn lane_count(&mut self) -> PyResult<usize> {
        Ok(self
            .inner
            .get_mut()
            .map_err(|_| PyValueError::new_err("runtime poisoned"))?
            .lane_count())
    }

    fn metrics(&mut self) -> PyResult<std::collections::BTreeMap<&'static str, f64>> {
        let m = self
            .inner
            .get_mut()
            .map_err(|_| PyValueError::new_err("runtime poisoned"))?
            .metrics();
        Ok(std::collections::BTreeMap::from([
            ("batches", m.batches as f64),
            ("rows", m.rows as f64),
            ("arena_bytes_peak", m.arena_bytes as f64),
            ("completed_search_groups", m.completed as f64),
            ("worker_seconds", m.worker_seconds),
        ]))
    }

    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        let inner = self
            .inner
            .get_mut()
            .map_err(|_| PyValueError::new_err("runtime poisoned"))?;
        py.detach(|| inner.close());
        Ok(())
    }
}

#[pyfunction]
#[pyo3(signature = (states, effects=false))]
fn observations<'py>(
    py: Python<'py>,
    states: Vec<PyRef<'_, State>>,
    effects: bool,
) -> Vec<EffectObservation<'py>> {
    states
        .iter()
        .map(|s| {
            effect_observation(
                py,
                if effects {
                    s.inner.encode_effects()
                } else {
                    s.inner.encode()
                },
                effects,
            )
        })
        .collect()
}

/// Classical search and isolated NNUE candidates, with whole-root FFI calls.
#[pyclass(unsendable)]
struct Opponent {
    engine: Box<dyn push_chess::engine::Engine>,
    fingerprint: Option<u64>,
}
impl Opponent {
    fn search(
        &mut self,
        py: Python<'_>,
        state: &State,
        time_ms: i64,
        nodes: i64,
        depth: i32,
    ) -> PyResult<(
        push_chess::core::types::Move,
        push_chess::core::types::SearchStats,
    )> {
        if state.inner.white_value().is_some()
            || !(0..=3_600_000).contains(&time_ms)
            || nodes < 0
            || (time_ms == 0 && nodes == 0)
            || !(0..=100).contains(&depth)
        {
            return Err(PyValueError::new_err("invalid position or budget"));
        }
        // Clone the exact history once, then release Python during all CPU
        // search. No Python objects, neural calls or atomics inside the tree.
        let mut position = state.inner.position().clone();
        let engine = &mut self.engine;
        let result = py.detach(|| {
            engine.choose_move(
                &mut position,
                &SearchBudget {
                    max_time_us: time_ms * 1000,
                    max_nodes: nodes,
                    max_depth: depth,
                    ..SearchBudget::default()
                },
            )
        });
        if !state.inner.legal_moves().contains(&result.0) {
            return Err(PyValueError::new_err("opponent returned illegal move"));
        }
        Ok(result)
    }
}
#[pymethods]
impl Opponent {
    #[new]
    #[pyo3(signature = (name, network=None))]
    fn new(name: &str, network: Option<&[u8]>) -> PyResult<Self> {
        if let Some(bytes) = network {
            let candidate = push_chess_lab::lab::Candidate::decode(name, bytes)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            return Ok(Self {
                engine: candidate.create(),
                fingerprint: Some(candidate.network_fingerprint()),
            });
        }
        let entry = push_chess::engines::find_engine(name)
            .ok_or_else(|| PyValueError::new_err("unknown opponent"))?;
        let mut engine = (entry.create)();
        engine.new_game(Color::White, 0);
        let fingerprint = push_chess::engines::info(name)
            .unwrap()
            .neural_accumulator
            .then(push_chess::engines::cataclysm::network_fingerprint);
        Ok(Self {
            engine,
            fingerprint,
        })
    }
    fn network_fingerprint(&self) -> Option<u64> {
        self.fingerprint
    }
    #[pyo3(signature = (side=0, seed=0))]
    fn new_game(&mut self, side: u8, seed: u64) -> PyResult<()> {
        let color = match side {
            0 => Color::White,
            1 => Color::Black,
            _ => return Err(PyValueError::new_err("invalid side")),
        };
        self.engine.new_game(color, seed);
        Ok(())
    }
    #[pyo3(signature = (state, time_ms=100, nodes=0))]
    fn choose(&mut self, py: Python<'_>, state: &State, time_ms: i64, nodes: i64) -> PyResult<u32> {
        Ok(self.search(py, state, time_ms, nodes, 0)?.0.id())
    }
    #[pyo3(signature = (state, time_ms=100, nodes=0, depth=0))]
    fn analyse<'py>(
        &mut self,
        py: Python<'py>,
        state: &State,
        time_ms: i64,
        nodes: i64,
        depth: i32,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let (mv, stats) = self.search(py, state, time_ms, nodes, depth)?;
        let row = pyo3::types::PyDict::new(py);
        row.set_item("move", mv.id())?;
        row.set_item("score", stats.eval_cp)?;
        row.set_item("nodes", stats.nodes)?;
        row.set_item("depth", stats.depth_reached)?;
        row.set_item("seldepth", stats.seldepth)?;
        row.set_item("wall_us", stats.time_used_us)?;
        row.set_item(
            "complete",
            stats.depth_reached > 0 || stats.diagnostics.mate_proof_plies.is_some_and(|n| n > 0),
        )?;
        row.set_item("qnodes", stats.diagnostics.qnodes)?;
        row.set_item("tt_hits", stats.diagnostics.tt_hits)?;
        row.set_item("proof_nodes", stats.diagnostics.proof_nodes)?;
        row.set_item("mate_proof_plies", stats.diagnostics.mate_proof_plies)?;
        row.set_item(
            "pv",
            stats
                .pv
                .into_iter()
                .map(|m| m.id())
                .collect::<Vec<_>>()
                .into_pyarray(py),
        )?;
        Ok(row)
    }
}

/// One FFI crossing per complete-game page; owned arrays, no JSON payloads.
type PythonCorpusPage<'py> = (Vec<Bound<'py, pyo3::types::PyDict>>, (u64, u64), bool);
#[pyclass(unsendable)]
struct CorpusReader {
    inner: Option<push_chess_lab::lab::Corpus>,
}
impl CorpusReader {
    fn corpus(&self) -> PyResult<&push_chess_lab::lab::Corpus> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("corpus reader is closed"))
    }
}
#[pymethods]
impl CorpusReader {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Some(
                push_chess_lab::lab::Corpus::open(std::path::Path::new(path))
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            ),
        })
    }
    #[pyo3(signature = (split="train", after=(0,0), limit=8, nnue=false))]
    fn page<'py>(
        &self,
        py: Python<'py>,
        split: &str,
        after: (u64, u64),
        limit: usize,
        nnue: bool,
    ) -> PyResult<PythonCorpusPage<'py>> {
        let page = self
            .corpus()?
            .page(split, after, limit)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut games = Vec::with_capacity(page.games.len());
        for game in page.games {
            let g = game.trajectory;
            let row = pyo3::types::PyDict::new(py);
            if nnue {
                let mut state =
                    selfplay::State::from_fen(&g.initial_fen).map_err(PyValueError::new_err)?;
                let mut rows = Vec::with_capacity(g.plies.len());
                let mut check = Vec::with_capacity(g.plies.len());
                for ply in &g.plies {
                    rows.push(nnue::input(state.position()));
                    check.push(state.position().in_check());
                    state.play(ply.action).map_err(PyValueError::new_err)?;
                }
                let (features, baselines) = nnue_batch(py, rows);
                row.set_item("nnue_features", features)?;
                row.set_item("nnue_baselines", baselines)?;
                row.set_item("in_check", check.into_pyarray(py))?;
            }
            row.set_item("run_id", game.run_id)?;
            row.set_item("game_index", g.index)?;
            row.set_item("initial_fen", g.initial_fen)?;
            row.set_item("final_fen", g.final_fen)?;
            row.set_item("white", g.white)?;
            row.set_item("black", g.black)?;
            row.set_item("opening_key", g.opening_key)?;
            row.set_item(
                "trajectory_key",
                pyo3::types::PyBytes::new(py, &game.trajectory_key),
            )?;
            row.set_item("white_value", g.white_value)?;
            row.set_item("split", g.split)?;
            row.set_item("termination", g.termination)?;
            row.set_item("rules", game.rules)?;
            row.set_item("binary", pyo3::types::PyBytes::new(py, &game.binary))?;
            let mut actions = Vec::with_capacity(g.plies.len());
            let mut columns = Vec::with_capacity(g.plies.len() * 6);
            let mut details = Vec::with_capacity(g.plies.len() * 5);
            let mut pv_offsets = Vec::with_capacity(g.plies.len() + 1);
            let mut pv_actions = Vec::new();
            let mut proofs = Vec::new();
            let mut mates = Vec::new();
            pv_offsets.push(0u32);
            for (ply, p) in g.plies.iter().enumerate() {
                actions.push(p.action);
                let nodes = i64::try_from(p.nodes)
                    .map_err(|_| PyValueError::new_err("node counter overflow"))?;
                columns.extend_from_slice(&[
                    p.side as i64,
                    p.score.unwrap_or(0) as i64,
                    i64::from(p.score.is_some()),
                    i64::from(p.search_complete),
                    nodes,
                    p.depth as i64,
                ]);
                details.extend_from_slice(&[
                    i64::from(p.seldepth),
                    p.wall_us,
                    p.reported_us,
                    i64::try_from(p.diagnostics.qnodes)
                        .map_err(|_| PyValueError::new_err("qnode counter overflow"))?,
                    i64::try_from(p.diagnostics.tt_hits)
                        .map_err(|_| PyValueError::new_err("TT counter overflow"))?,
                ]);
                pv_actions.extend_from_slice(&p.pv);
                pv_offsets.push(
                    u32::try_from(pv_actions.len())
                        .map_err(|_| PyValueError::new_err("PV page overflow"))?,
                );
                if let Some(nodes) = p.diagnostics.proof_nodes {
                    proofs.extend([ply as u64, nodes]);
                }
                if let Some(plies) = p.diagnostics.mate_proof_plies {
                    mates.extend([ply as u64, u64::from(plies)]);
                }
            }
            row.set_item("moves", actions.into_pyarray(py))?;
            row.set_item(
                "analysis",
                Array::from_shape_vec((g.plies.len(), 6), columns)
                    .unwrap()
                    .into_pyarray(py),
            )?;
            row.set_item(
                "search_details",
                Array::from_shape_vec((g.plies.len(), 5), details)
                    .unwrap()
                    .into_pyarray(py),
            )?;
            row.set_item("pv_offsets", pv_offsets.into_pyarray(py))?;
            row.set_item("pv_actions", pv_actions.into_pyarray(py))?;
            // Optional observations remain sparse (ply,value) rows at the
            // boundary, never ambiguous zero-filled proof claims.
            row.set_item(
                "proof_searches",
                Array::from_shape_vec((proofs.len() / 2, 2), proofs)
                    .unwrap()
                    .into_pyarray(py),
            )?;
            row.set_item(
                "mate_proofs",
                Array::from_shape_vec((mates.len() / 2, 2), mates)
                    .unwrap()
                    .into_pyarray(py),
            )?;
            games.push(row);
        }
        Ok((games, page.cursor, page.done))
    }
    fn state(&self, py: Python<'_>, run: u64, game: u64, ply: usize) -> PyResult<State> {
        let game = self
            .corpus()?
            .game(run, game)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if ply > game.trajectory.plies.len() {
            return Err(PyValueError::new_err("ply beyond saved game"));
        }
        py.detach(move || -> Result<State, String> {
            let mut state = selfplay::State::from_fen(&game.trajectory.initial_fen)?;
            for record in game.trajectory.plies.iter().take(ply) {
                state.play(record.action)?;
            }
            Ok(State { inner: state })
        })
        .map_err(PyValueError::new_err)
    }
    fn summary(&self) -> PyResult<String> {
        self.corpus()?
            .summary()
            .map(|v| v.to_string())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn close(&mut self) -> PyResult<()> {
        if let Some(corpus) = &self.inner {
            corpus
                .close()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
        // bumbledb's directory ownership belongs to the Db value. Release it
        // now, not when Python eventually collects this wrapper.
        self.inner = None;
        Ok(())
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<State>()?;
    m.add_class::<SearchBatch>()?;
    m.add_class::<SearchRuntime>()?;
    m.add_class::<Opponent>()?;
    m.add_class::<CorpusReader>()?;
    m.add_function(wrap_pyfunction!(observations, m)?)?;
    m.add_function(wrap_pyfunction!(nnue_control, m)?)?;
    m.add_function(wrap_pyfunction!(nnue_inputs, m)?)?;
    m.add_function(wrap_pyfunction!(nnue_evaluate, m)?)?;
    m.add("RULES_VERSION", selfplay::RULES_VERSION)?;
    m.add("ENCODING_VERSION", selfplay::ENCODING_VERSION)?;
    m.add("EFFECT_ENCODING_VERSION", selfplay::EFFECT_ENCODING_VERSION)?;
    Ok(())
}
