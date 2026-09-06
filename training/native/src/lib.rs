//! Thin ownership boundary. Rules/search live in push_chess::selfplay.
//! Rust Vec allocations become NumPy-owned buffers; no element-wise boxing.
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods, ndarray::Array,
};
use push_chess::core::types::{Color, SearchBudget};
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

/// Incumbents are evaluation opponents only; self-play never calls this class.
#[pyclass(unsendable)]
struct Opponent {
    engine: Box<dyn push_chess::engine::Engine>,
}
#[pymethods]
impl Opponent {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        let entry = push_chess::candidates::find_engine(name)
            .ok_or_else(|| PyValueError::new_err("unknown opponent"))?;
        let mut engine = (entry.create)();
        engine.new_game(Color::White, 0);
        Ok(Self { engine })
    }
    #[pyo3(signature = (state, time_ms=100, nodes=0))]
    fn choose(&mut self, state: &State, time_ms: i64, nodes: i64) -> PyResult<u32> {
        if state.inner.white_value().is_some()
            || !(0..=3_600_000).contains(&time_ms)
            || nodes < 0
            || (time_ms == 0 && nodes == 0)
        {
            return Err(PyValueError::new_err("invalid position or budget"));
        }
        let (mv, _) = self.engine.choose_move(
            &mut state.inner.position().clone(),
            &SearchBudget {
                max_time_us: time_ms * 1000,
                max_nodes: nodes,
                ..SearchBudget::default()
            },
        );
        if !state.inner.legal_moves().contains(&mv) {
            return Err(PyValueError::new_err("opponent returned illegal move"));
        }
        Ok(mv.id())
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<State>()?;
    m.add_class::<SearchBatch>()?;
    m.add_class::<SearchRuntime>()?;
    m.add_class::<Opponent>()?;
    m.add_function(wrap_pyfunction!(observations, m)?)?;
    m.add("RULES_VERSION", selfplay::RULES_VERSION)?;
    m.add("ENCODING_VERSION", selfplay::ENCODING_VERSION)?;
    m.add("EFFECT_ENCODING_VERSION", selfplay::EFFECT_ENCODING_VERSION)?;
    Ok(())
}
