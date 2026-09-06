use push_chess::game::{MovePreview, MoveResult, SavedGame, Snapshot};
use push_chess::session::{Analysis, AnalysisOptions, CataclysmSession, Checkpoint};
use tsify::{Ts, Tsify};
use wasm_bindgen::prelude::*;

fn error(message: String) -> JsValue {
    js_sys::Error::new(&message).into()
}

fn wire<T: Tsify + serde::Serialize>(value: T) -> Result<Ts<T>, JsValue> {
    value.into_ts().map_err(|e| error(e.to_string()))
}

fn integer(value: f64) -> Result<u32, JsValue> {
    if value.is_finite() && value.fract() == 0.0 && (0.0..=u32::MAX as f64).contains(&value) {
        Ok(value as u32)
    } else {
        Err(error("Expected an unsigned 32-bit integer".into()))
    }
}

/// Low-level synchronous API. Use the package's worker client on the web;
/// calling analyse on the browser's main thread will block that thread.
#[wasm_bindgen]
pub struct Session {
    inner: CataclysmSession,
}

#[wasm_bindgen]
impl Session {
    #[wasm_bindgen(constructor)]
    pub fn new(hash_mib: Option<f64>) -> Result<Session, JsValue> {
        Ok(Self {
            inner: CataclysmSession::new(integer(hash_mib.unwrap_or(8.0))?).map_err(error)?,
        })
    }

    pub fn snapshot(&self) -> Result<Ts<Snapshot>, JsValue> {
        wire(self.inner.snapshot())
    }
    pub fn save(&self) -> Result<Ts<SavedGame>, JsValue> {
        wire(self.inner.save())
    }

    pub fn restore(&mut self, json: &str) -> Result<Ts<Snapshot>, JsValue> {
        wire(self.inner.restore(json).map_err(error)?)
    }

    pub fn recover(&mut self, checkpoint: Ts<Checkpoint>) -> Result<Ts<Snapshot>, JsValue> {
        let checkpoint = checkpoint.to_rust().map_err(|e| error(e.to_string()))?;
        wire(self.inner.recover(checkpoint).map_err(error)?)
    }

    pub fn reset(&mut self, fen: Option<String>) -> Result<Ts<Snapshot>, JsValue> {
        wire(self.inner.reset(fen.as_deref()).map_err(error)?)
    }

    pub fn preview(&self, id: f64, revision: f64) -> Result<Ts<MovePreview>, JsValue> {
        wire(
            self.inner
                .preview(integer(id)?, integer(revision)?)
                .map_err(error)?,
        )
    }

    pub fn play(&mut self, id: f64, revision: f64) -> Result<Ts<MoveResult>, JsValue> {
        wire(
            self.inner
                .play(integer(id)?, integer(revision)?)
                .map_err(error)?,
        )
    }

    pub fn undo(&mut self, plies: f64, revision: f64) -> Result<Ts<Snapshot>, JsValue> {
        wire(
            self.inner
                .undo(integer(plies)?, integer(revision)?)
                .map_err(error)?,
        )
    }

    pub fn analyse(
        &mut self,
        options: Ts<AnalysisOptions>,
        revision: f64,
    ) -> Result<Ts<Analysis>, JsValue> {
        let options = options.to_rust().map_err(|e| error(e.to_string()))?;
        wire(
            self.inner
                .analyse(options, integer(revision)?)
                .map_err(error)?,
        )
    }
}
