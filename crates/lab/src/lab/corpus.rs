//! Sole durable source of games and observations. One game is one atomic
//! bumbledb change set; reads copy at most a page of complete trajectories.
use super::schema::*;
use super::{Result, Roster, RunConfig, Trajectory, relational};
use bumbledb::{
    Admission, ApplyExpected, ApplyOutcome, ChangeSet, ChangeSetBuilder, CloseReport, Db,
    ExecutionPolicy, Fact, WorkContext,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::io::Read;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

fn work() -> Result<WorkContext> {
    Ok(bumbledb::start_operation(ExecutionPolicy {
        input_bytes: 32 << 20,
        working_bytes: 256 << 20,
        scratch_bytes: 128 << 20,
        result_bytes: 64 << 20,
        rows: 2_000_000,
        work_units: 200_000_000,
        timeout: Duration::from_secs(60),
    })?)
}
pub(super) fn insert<'a, F: Fact<'a>>(draft: &mut ChangeSetBuilder<'_>, fact: &F) -> Result<()> {
    let mut values = Vec::new();
    fact.append_values(&mut values)?;
    draft.insert(F::RELATION, &values)?;
    Ok(())
}
pub fn digest_file(path: &Path) -> Result<[u8; 32]> {
    let mut file = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let len = file.read(&mut buffer)?;
        if len == 0 {
            break;
        }
        hash.update(&buffer[..len]);
    }
    Ok(hash.finalize().into())
}
pub(super) fn unhex(s: &str) -> Result<[u8; 32]> {
    if s.len() != 64 || !s.is_ascii() {
        return Err("invalid SHA256 digest".into());
    }
    let mut bytes = [0; 32];
    for (i, b) in bytes.iter_mut().enumerate() {
        *b = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)?;
    }
    Ok(bytes)
}
pub(super) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn now() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_micros()
        .try_into()?)
}
pub(super) fn split_id(s: &str) -> Result<SplitId> {
    Ok(match s {
        "train" => Split::Train.id(),
        "validation" => Split::Validation.id(),
        "test" => Split::Test.id(),
        "evaluation" => Split::Evaluation.id(),
        _ => return Err("unknown split".into()),
    })
}
pub(super) fn ending_id(s: &str) -> Result<EndingId> {
    Ok(match s {
        "checkmate" => Ending::Mate.id(),
        "stalemate" => Ending::Stalemate.id(),
        "50_move_rule" => Ending::FiftyMove.id(),
        "threefold_repetition" => Ending::Repetition.id(),
        "ply_limit" => Ending::PlyLimit.id(),
        "interrupted" => Ending::Interrupted.id(),
        _ => return Err("unknown ending".into()),
    })
}
pub(super) fn ending_name(id: EndingId) -> Result<&'static str> {
    for s in [
        "checkmate",
        "stalemate",
        "50_move_rule",
        "threefold_repetition",
        "ply_limit",
        "interrupted",
    ] {
        if ending_id(s)? == id {
            return Ok(s);
        }
    }
    Err("invalid stored ending".into())
}
pub(super) fn white_value(id: WhiteOutcomeId) -> Result<i32> {
    if id == WhiteOutcome::Win.id() {
        Ok(1)
    } else if id == WhiteOutcome::Draw.id() {
        Ok(0)
    } else if id == WhiteOutcome::Loss.id() {
        Ok(-1)
    } else {
        Err("invalid stored outcome".into())
    }
}
fn status_name(id: RunStatusId) -> &'static str {
    if id == RunStatus::Finished.id() {
        "finished"
    } else if id == RunStatus::Interrupted.id() {
        "interrupted"
    } else {
        "failed"
    }
}

pub struct CorpusGame {
    pub run_id: u64,
    pub trajectory: Trajectory,
    pub trajectory_key: [u8; 32],
    pub binary: [u8; 32],
}
pub struct CorpusPage {
    pub games: Vec<CorpusGame>,
    pub cursor: (u64, u64),
    pub done: bool,
}
pub struct Corpus {
    db: Db<TrainingGround>,
    reader: RefCell<Option<relational::GameReader>>,
}
impl Corpus {
    pub fn create(path: &Path) -> Result<Self> {
        match Db::create(path, TrainingGround, work()?)? {
            Admission::Accepted(db) => Ok(Self {
                db,
                reader: RefCell::new(None),
            }),
            Admission::Rejected(v) => Err(format!("empty schema rejected: {v:?}").into()),
        }
    }
    /// Requires an existing directory with exactly this schema; no conversion.
    pub fn open(path: &Path) -> Result<Self> {
        Ok(Self {
            db: Db::open(path, TrainingGround, work()?)?,
            reader: RefCell::new(None),
        })
    }
    pub fn close(&self) -> Result<()> {
        self.reader.borrow_mut().take();
        match self.db.close(&work()?) {
            CloseReport::Closed => Ok(()),
            report => Err(format!("database close incomplete: {report:?}").into()),
        }
    }

    fn read_game(
        &self,
        frame: &bumbledb::ReadFrame<'_, TrainingGround>,
        game: &Game,
    ) -> Result<Trajectory> {
        let mut reader = self.reader.borrow_mut();
        if reader.is_none() {
            *reader = Some(relational::GameReader::new(frame)?);
        }
        relational::read_game(
            frame,
            game,
            reader.as_mut().expect("initialized game reader"),
        )
    }
    pub fn disk_bytes(&self) -> Result<u64> {
        Ok(self.db.disk_size(work()?)?)
    }
    fn apply(&self, draft: ChangeSetBuilder<'_>, work: &WorkContext) -> Result<()> {
        match self.db.apply(&draft.finish()?, ApplyExpected::Any, work)? {
            ApplyOutcome::Accepted { .. } => Ok(()),
            ApplyOutcome::NoChange { .. } => Err("duplicate corpus write refused".into()),
            ApplyOutcome::InvariantRejected { violations } => {
                Err(format!("corpus invariant rejected: {violations:?}").into())
            }
            ApplyOutcome::Moved { .. } => Err("corpus changed under write".into()),
        }
    }
    #[cfg(test)]
    pub(super) fn start(&self, config: &RunConfig) -> Result<u64> {
        self.start_resolved(config, &Roster::builtins(&config.engines)?)
    }
    pub(super) fn start_resolved(&self, config: &RunConfig, roster: &Roster) -> Result<u64> {
        config.validate()?;
        roster.validate(&config.engines)?;
        let w = work()?;
        let id = {
            let snapshot = self.db.snapshot(&w)?;
            snapshot
                .frame(&w)
                .scan_facts::<Run>()?
                .map(|run| run.map(|run| run.id.0))
                .collect::<bumbledb::Result<Vec<_>>>()?
                .into_iter()
                .max()
                .unwrap_or(0)
                .checked_add(1)
                .ok_or("run id exhausted")?
        };
        let mut draft = ChangeSet::builder(self.db.schema(), w.clone());
        relational::write_config(
            &mut draft,
            RunId(id),
            config,
            roster,
            digest_file(&std::env::current_exe()?)?,
            now()?,
        )?;
        self.apply(draft, &w)?;
        Ok(id)
    }
    pub(super) fn finish(&self, run: u64, status: &str, error: Option<&str>) -> Result<()> {
        let status = match status {
            "finished" => RunStatus::Finished.id(),
            "interrupted" => RunStatus::Interrupted.id(),
            "failed" if error.is_some() => RunStatus::Failed.id(),
            _ => return Err("invalid run finalization".into()),
        };
        let w = work()?;
        let mut draft = ChangeSet::builder(self.db.schema(), w.clone());
        insert(
            &mut draft,
            &RunEnd {
                run: RunId(run),
                status,
                finished_us: now()?,
            },
        )?;
        if let Some(message) = error {
            insert(
                &mut draft,
                &RunFailure {
                    run: RunId(run),
                    message,
                },
            )?;
        }
        self.apply(draft, &w)
    }
    pub(super) fn save(&mut self, run: u64, game: &Trajectory) -> Result<()> {
        game.validate()?;
        let run = RunId(run);
        let w = work()?;
        let config = {
            let snapshot = self.db.snapshot(&w)?;
            let frame = snapshot.frame(&w);
            if frame.get(RunEndByRun { run })?.is_some() {
                return Err("run is sealed".into());
            }
            if frame
                .get(GameByRunIndex {
                    run,
                    index: game.index as u64,
                })?
                .is_some()
            {
                return Err("game already committed".into());
            }
            let meta = frame.get(RunById { id: run })?.ok_or("missing run")?;
            relational::config(&frame, &meta)?
        };
        let mut draft = ChangeSet::builder(self.db.schema(), w.clone());
        relational::write_game(&mut draft, run, game, &config)?;
        self.apply(draft, &w)
    }
    pub fn summary(&self) -> Result<Value> {
        let w = work()?;
        let snapshot = self.db.snapshot(&w)?;
        let frame = snapshot.frame(&w);
        let mut runs = Vec::new();
        for run in frame.scan_facts::<Run>()? {
            let run = run?;
            let status = frame.get(RunEndByRun { run: run.id })?;
            let error = frame.get(RunFailureByRun { run: run.id })?;
            runs.push(json!({"id":run.id.0,"status":status.map(|s|status_name(s.status)).unwrap_or("running"),
                "config":relational::config(&frame,&run)?,"binary_sha256":hex(&run.binary),
                "error":error.map(|e|e.message)}));
        }
        runs.sort_by_key(|r| r["id"].as_u64());
        Ok(
            json!({"backend":"bumbledb","disk_bytes":self.disk_bytes()?, "totals":{
            "games":frame.count(Game::RELATION)?,"moves":frame.count(Move::RELATION)?,"positions":frame.count(Position::RELATION)?,
            "analyses":frame.count(Analysis::RELATION)?,"terminal_games":frame.count(GameResult::RELATION)?},
            "runs":runs}),
        )
    }
    /// Paired fixed-sample evidence, not Elo or an automatic promotion gate.
    pub fn report(&self, run: u64) -> Result<Value> {
        let w = work()?;
        let snapshot = self.db.snapshot(&w)?;
        let frame = snapshot.frame(&w);
        let meta = frame
            .get(RunById { id: RunId(run) })?
            .ok_or("unknown run")?;
        let config: RunConfig = relational::config(&frame, &meta)?;
        let matchups = config.matchups();
        let mut results = vec![[0u64; 5]; matchups.len()];
        let mut outcomes = vec![[None; 2]; config.total_pairs()];
        for row in frame.scan_facts::<Game>()? {
            let game = row?;
            if game.run.0 != run {
                continue;
            }
            let tally = results
                .get_mut(game.pair as usize % matchups.len())
                .ok_or("unplanned matchup")?;
            tally[3] += 1;
            tally[4] += game.plies;
            if let Some(result) = frame.get(GameResultByRunGame {
                run: game.run,
                game: game.index,
            })? {
                let relative = white_value(result.outcome)? * if !game.swapped { 1 } else { -1 };
                outcomes
                    .get_mut(game.pair as usize)
                    .ok_or("unplanned pair")?[usize::from(game.swapped)] = Some(relative);
                tally[if relative > 0 {
                    0
                } else if relative == 0 {
                    1
                } else {
                    2
                }] += 1;
            }
        }
        let mut evidence: Vec<_> = (0..matchups.len())
            .map(|_| super::statistics::PairedScores::default())
            .collect();
        for (index, result) in outcomes.into_iter().enumerate() {
            if let [Some(first), Some(second)] = result {
                let pair = frame
                    .get(PairByRunIndex {
                        run: RunId(run),
                        index: index as u64,
                    })?
                    .ok_or("missing pair")?;
                evidence[index % matchups.len()].record(pair.opening, [first, second]);
            }
        }
        let status = frame.get(RunEndByRun { run: RunId(run) })?;
        let pairs: Vec<_> = matchups.iter().enumerate().map(|(index, &(a,b))| {
            let r = results[index];
            let known = r[0]+r[1]+r[2]; let scheduled = (config.pairs*2) as f64;
            let score = r[0] as f64 + r[1] as f64*0.5;
            json!({"a":config.engines[a],"b":config.engines[b],"wins":r[0],"draws":r[1],"losses":r[2],"saved":r[3],"positions":r[4],
                "saved_without_outcome":r[3]-known,"missing_games":config.pairs*2-r[3] as usize,
                "unknown_or_missing":config.pairs*2-known as usize,
                "score_bounds":[score/scheduled,(score+scheduled-known as f64)/scheduled],
                "paired":evidence[index].report(config.pairs,matchups.len())})
        }).collect();
        Ok(
            json!({"run":run,"binary_sha256":hex(&meta.binary),"status":status.map(|s|status_name(s.status)).unwrap_or("running"),
            "scheduled_games":config.total_pairs()*2,"matchups":pairs,"promotion_ready":false,
            "uncertainty":"Fixed-sample bounds assume independent opening families; exploratory, not sequential or a held-out promotion test. Incomplete pairs are excluded, not draws."}),
        )
    }
    /// Offline integrity audit. Includes quarantined and interrupted games;
    /// reports observations without changing their training eligibility.
    pub fn verify(&self, run: u64) -> Result<Value> {
        let config = {
            let w = work()?;
            let snapshot = self.db.snapshot(&w)?;
            let frame = snapshot.frame(&w);
            let meta = frame
                .get(RunById { id: RunId(run) })?
                .ok_or("unknown run")?;
            relational::config(&frame, &meta)?
        };
        let (mut games, mut moves, mut terminal) = (0u64, 0u64, 0u64);
        let mut cost = super::statistics::SearchCost::default();
        let mut engine_costs: Vec<_> = config
            .engines
            .iter()
            .map(|_| super::statistics::SearchCost::default())
            .collect();
        let budget_us = (config.time_ms > 0).then_some(config.time_ms * 1000);
        let mut castles = 0u64;
        for index in 0..(config.total_pairs() * 2) as u64 {
            let w = work()?;
            let snapshot = self.db.snapshot(&w)?;
            let frame = snapshot.frame(&w);
            let Some(g) = frame.get(GameByRunIndex {
                run: RunId(run),
                index,
            })?
            else {
                continue;
            };
            let game = self.read_game(&frame, &g)?;
            let audit = game.validate()?;
            castles += audit.castles;
            games += 1;
            moves += game.plies.len() as u64;
            terminal += u64::from(game.white_value.is_some());
            let mut slots = [0; 2];
            for (side, name) in [&game.white, &game.black].into_iter().enumerate() {
                slots[side] = config
                    .engines
                    .iter()
                    .position(|n| n == name)
                    .ok_or("unregistered game engine")?;
            }
            for p in &game.plies {
                cost.record(p, budget_us);
                engine_costs[slots[p.side as usize]].record(p, budget_us);
            }
        }
        let engine_costs: serde_json::Map<_, _> = config
            .engines
            .iter()
            .zip(engine_costs)
            .map(|(name, cost)| (name.clone(), cost.report()))
            .collect();
        Ok(
            json!({"run":run,"verified_games":games,"moves":moves,"positions":moves+games,
            "analyses":cost.searches,"completed_searches":cost.complete,"terminal_games":terminal,
            "search_wall_us":cost.wall_us,"search_nodes":cost.nodes,"time_overruns":cost.overruns,
            "search_cost":cost.report(),"search_by_engine":engine_costs,
            "search_cost_note":"Descriptive sums over actual searches, including incomplete searches; openings excluded. Node rates use total measured search time, not campaign elapsed time. Reported q/proof nodes are not time attribution or verified mate certificates.",
            "castling_audit":{"castles":castles}}),
        )
    }
    /// Short-lived snapshots: never pin an hours-long read across map growth.
    /// Failed/unsealed runs and interrupted games are excluded. Interrupted
    /// runs retain their already verified complete games. Explicit source runs
    /// are all checked before reading any trajectory; None selects all eligible
    /// runs. Filtering happens before replay and Python feature construction.
    pub fn page(
        &self,
        split: &str,
        after: (u64, u64),
        limit: usize,
        runs: Option<&[u64]>,
    ) -> Result<CorpusPage> {
        let selected = split_id(split)?;
        if !(1..=32).contains(&limit) {
            return Err("page limit must be 1..=32".into());
        }
        if runs.is_some_and(|ids| ids.is_empty() || ids.contains(&0)) {
            return Err("source runs must be nonempty positive IDs".into());
        }
        let w = work()?;
        let snapshot = self.db.snapshot(&w)?;
        let frame = snapshot.frame(&w);
        let mut ids = if let Some(ids) = runs {
            ids.to_vec()
        } else {
            frame
                .scan_facts::<Run>()?
                .map(|r| r.map(|r| r.id.0))
                .collect::<bumbledb::Result<Vec<_>>>()?
        };
        ids.sort_unstable();
        ids.dedup();
        let mut sources = Vec::with_capacity(ids.len());
        for id in ids {
            let run = frame
                .get(RunById { id: RunId(id) })?
                .ok_or_else(|| format!("unknown source run {id}"))?;
            let Some(end) = frame.get(RunEndByRun { run: RunId(id) })? else {
                if runs.is_some() {
                    return Err(format!("source run {id} is unsealed").into());
                }
                continue;
            };
            if end.status == RunStatus::Failed.id() {
                if runs.is_some() {
                    return Err(format!("source run {id} failed and is quarantined").into());
                }
                continue;
            }
            if id >= after.0 {
                let config: RunConfig = relational::config(&frame, &run)?;
                sources.push((run, config));
            }
        }
        let mut games = Vec::new();
        let mut cursor = after;
        for (run, config) in sources {
            let id = run.id.0;
            let start = if id == after.0 {
                after.1.saturating_add(1)
            } else {
                0
            };
            for index in start..(config.total_pairs() * 2) as u64 {
                cursor = (id, index);
                let Some(game) = frame.get(GameByRunIndex {
                    run: RunId(id),
                    index,
                })?
                else {
                    continue;
                };
                if game.split != selected || game.ending == Ending::Interrupted.id() {
                    continue;
                }
                let trajectory = self.read_game(&frame, &game)?;
                trajectory.validate()?;
                games.push(CorpusGame {
                    run_id: id,
                    trajectory,
                    trajectory_key: game.trajectory,
                    binary: run.binary,
                });
                if games.len() == limit {
                    return Ok(CorpusPage {
                        games,
                        cursor,
                        done: false,
                    });
                }
            }
        }
        Ok(CorpusPage {
            games,
            cursor,
            done: true,
        })
    }

    /// Explicit forensic access to one saved game, including quarantined runs.
    /// Training consumers must use the split/eligibility-filtered page API.
    pub fn game(&self, run: u64, index: u64) -> Result<CorpusGame> {
        let w = work()?;
        let snapshot = self.db.snapshot(&w)?;
        let frame = snapshot.frame(&w);
        let run = frame
            .get(RunById { id: RunId(run) })?
            .ok_or("unknown run")?;
        let game = frame
            .get(GameByRunIndex { run: run.id, index })?
            .ok_or("unknown saved game")?;
        let trajectory = self.read_game(&frame, &game)?;
        trajectory.validate()?;
        Ok(CorpusGame {
            run_id: run.id.0,
            trajectory,
            trajectory_key: game.trajectory,
            binary: run.binary,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::runner::tests::{config, fixture};
    use super::*;
    fn create() -> (tempfile::TempDir, Corpus) {
        let dir = tempfile::tempdir().unwrap();
        let corpus = Corpus::create(&dir.path().join("corpus")).unwrap();
        (dir, corpus)
    }
    #[test]
    fn atomic_roundtrip_preserves_every_observation_and_reopens() {
        let (dir, mut corpus) = create();
        let game = fixture();
        let run = corpus.start(&config()).unwrap();
        corpus.save(run, &game).unwrap();
        assert!(corpus.save(run, &game).is_err());
        assert_eq!(
            corpus
                .page(&game.split, (0, 0), 1, None)
                .unwrap()
                .games
                .len(),
            0
        );
        corpus.finish(run, "finished", None).unwrap();
        assert!(corpus.save(run, &game).is_err());
        let summary = corpus.summary().unwrap();
        assert_eq!(
            summary["totals"],
            json!({"games":1,"moves":8,"positions":9,"analyses":4,"terminal_games":0})
        );
        let page = corpus.page(&game.split, (0, 0), 1, None).unwrap();
        assert_eq!(page.games[0].trajectory.plies, game.plies);
        assert_eq!(page.games[0].trajectory.final_fen, game.final_fen);
        assert_eq!(corpus.verify(run).unwrap()["verified_games"], 1);
        let cursor = page.cursor;
        assert_eq!(
            corpus
                .page(&game.split, cursor, 1, None)
                .unwrap()
                .games
                .len(),
            0
        );
        corpus.close().unwrap();
        drop(corpus);
        let corpus = Corpus::open(&dir.path().join("corpus")).unwrap();
        assert_eq!(corpus.summary().unwrap()["totals"], summary["totals"]);
        corpus.close().unwrap();
    }
    #[test]
    fn bad_trajectories_cannot_partially_commit() {
        let (_dir, mut corpus) = create();
        let run = corpus.start(&config()).unwrap();
        let mut game = fixture();
        game.white_value = Some(0);
        assert!(corpus.save(run, &game).is_err());
        game.white_value = None;
        game.plies[0].action = u32::MAX;
        assert!(corpus.save(run, &game).is_err());
        assert_eq!(corpus.summary().unwrap()["totals"]["games"], 0);
        assert_eq!(corpus.summary().unwrap()["totals"]["positions"], 0);
        corpus.close().unwrap();
    }

    #[test]
    fn search_cost_attributes_observations_to_actual_entrants_after_color_swap() {
        let (_dir, mut corpus) = create();
        let mut config = config();
        config.engines.push("astra".into()); // An unplayed entrant stays empty.
        let run = corpus.start(&config).unwrap();
        let mut first = fixture();
        for p in &mut first.plies {
            if p.score.is_some() {
                p.nodes = if p.side == 0 { 10 } else { 20 };
            }
        }
        let mut second = first.clone();
        second.index = 1;
        std::mem::swap(&mut second.white, &mut second.black);
        for p in &mut second.plies {
            if p.score.is_some() {
                p.nodes = if p.side == 0 { 70 } else { 40 };
            }
        }
        corpus.save(run, &first).unwrap();
        corpus.save(run, &second).unwrap();
        corpus.finish(run, "finished", None).unwrap();
        let report = corpus.verify(run).unwrap();
        assert_eq!(report["analyses"], 8);
        assert_eq!(report["search_nodes"], 280);
        assert_eq!(report["search_cost"]["nodes"], 280);
        let engines = &report["search_by_engine"];
        assert_eq!(engines["cataclysm"]["searches"], 4);
        assert_eq!(engines["cataclysm"]["nodes"], 100);
        assert_eq!(engines["kinetic"]["searches"], 4);
        assert_eq!(engines["kinetic"]["nodes"], 180);
        assert_eq!(engines["astra"]["searches"], 0);
        assert!(engines["astra"]["mean_wall_us"].is_null());
        assert_eq!(report["time_overruns"], 0); // No wall-time limit in this fixture.
        corpus.close().unwrap();
    }

    #[test]
    fn schema_rejects_mixed_budgets_and_promotion_to_king() {
        let (_dir, corpus) = create();
        let run = corpus.start(&config()).unwrap();
        let w = work().unwrap();
        let mut draft = ChangeSet::builder(corpus.db.schema(), w.clone());
        insert(
            &mut draft,
            &TimeBudget {
                run: RunId(run),
                microseconds: 1000,
            },
        )
        .unwrap();
        assert!(corpus.apply(draft, &w).is_err());
        let mut draft = ChangeSet::builder(corpus.db.schema(), w.clone());
        let action = ActionId(1);
        insert(
            &mut draft,
            &Action {
                id: action,
                from: 48,
                to: 56,
                route: Route::Direct.id(),
                stop: 0,
                kind: MoveKind::Promotion.id(),
            },
        )
        .unwrap();
        insert(
            &mut draft,
            &ActionPromotion {
                action,
                piece: PieceKind::King.id(),
            },
        )
        .unwrap();
        assert!(corpus.apply(draft, &w).is_err());
        corpus.close().unwrap();
    }
    #[test]
    fn reader_rejects_missing_delta_even_when_moves_are_legal() {
        for (relation, error) in [
            (PieceRemoved::RELATION, "piece removal set"),
            (PiecePlaced::RELATION, "piece placement set"),
        ] {
            let (_dir, mut corpus) = create();
            let run = corpus.start(&config()).unwrap();
            corpus.save(run, &fixture()).unwrap();
            // Warm both query indexes, then invalidate each relation separately.
            assert_eq!(corpus.verify(run).unwrap()["verified_games"], 1);
            let w = work().unwrap();
            let mut values = Vec::new();
            {
                let snapshot = corpus.db.snapshot(&w).unwrap();
                let frame = snapshot.frame(&w);
                if relation == PieceRemoved::RELATION {
                    frame
                        .scan_facts::<PieceRemoved>()
                        .unwrap()
                        .next()
                        .unwrap()
                        .unwrap()
                        .append_values(&mut values)
                        .unwrap();
                } else {
                    frame
                        .scan_facts::<PiecePlaced>()
                        .unwrap()
                        .next()
                        .unwrap()
                        .unwrap()
                        .append_values(&mut values)
                        .unwrap();
                }
            }
            let mut draft = ChangeSet::builder(corpus.db.schema(), w.clone());
            draft.delete(relation, &values).unwrap();
            corpus.apply(draft, &w).unwrap();
            assert!(corpus.verify(run).unwrap_err().to_string().contains(error));
            corpus.close().unwrap();
        }
    }
    #[test]
    fn schema_rejects_orphan_analysis_and_unlabelled_terminal() {
        let (_dir, corpus) = create();
        let run = corpus.start(&config()).unwrap();
        let w = work().unwrap();
        let mut draft = ChangeSet::builder(corpus.db.schema(), w.clone());
        insert(
            &mut draft,
            &Analysis {
                run: RunId(run),
                game: 0,
                ply: 0,
                score_stm: 0,
                complete: true,
                nodes: 1,
                depth: 1,
                seldepth: 1,
                wall_us: 1,
                reported_us: 1,
                qnodes: 0,
                tt_hits: 0,
                pv_length: 0,
            },
        )
        .unwrap();
        assert!(corpus.apply(draft, &w).is_err());
        let mut game = fixture();
        game.initial_fen = "7k/6Q1/5K2/8/8/8/8/8 b - - 100 1".into();
        game.final_fen = game.initial_fen.clone();
        game.plies.clear();
        game.termination = "checkmate".into();
        game.white_value = None; // Deliberately bypass the replay boundary to test the theory.
        game.opening_key = super::super::opening_key(&game.initial_fen, std::iter::empty());
        game.split = super::super::split(&game.opening_key).into();
        let w = work().unwrap();
        let mut draft = ChangeSet::builder(corpus.db.schema(), w.clone());
        relational::write_game(&mut draft, RunId(run), &game, &config()).unwrap();
        assert!(corpus.apply(draft, &w).is_err());
        assert_eq!(corpus.summary().unwrap()["totals"]["analyses"], 0);
        corpus.close().unwrap();
    }
    #[test]
    fn failed_runs_are_quarantined_and_interrupted_runs_keep_valid_games() {
        let (_dir, mut corpus) = create();
        let run = corpus.start(&config()).unwrap();
        let game = fixture();
        corpus.save(run, &game).unwrap();
        corpus
            .finish(run, "failed", Some("fixture failure"))
            .unwrap();
        assert_eq!(
            corpus
                .page(&game.split, (0, 0), 8, None)
                .unwrap()
                .games
                .len(),
            0
        );
        let run = corpus.start(&config()).unwrap();
        corpus.save(run, &game).unwrap();
        corpus.finish(run, "interrupted", None).unwrap();
        assert_eq!(
            corpus
                .page(&game.split, (0, 0), 8, None)
                .unwrap()
                .games
                .len(),
            1
        );
        corpus.close().unwrap();
    }

    #[test]
    fn explicit_sources_are_validated_before_the_first_page() {
        let (_dir, mut corpus) = create();
        let game = fixture();
        let sealed = corpus.start(&config()).unwrap();
        corpus.save(sealed, &game).unwrap();
        corpus.finish(sealed, "finished", None).unwrap();
        let unsealed = corpus.start(&config()).unwrap();
        let failed = corpus.start(&config()).unwrap();
        corpus.finish(failed, "failed", Some("fixture")).unwrap();
        for (ids, message) in [
            (vec![], "nonempty positive"),
            (vec![sealed, 0], "nonempty positive"),
            (vec![sealed, 999], "unknown source run 999"),
            (vec![sealed, unsealed], "unsealed"),
            (vec![sealed, failed], "quarantined"),
        ] {
            // Even a one-game page must not yield the valid first run before
            // noticing a later invalid request. Cursor position cannot hide it.
            for cursor in [(0, 0), (999, 0)] {
                let error = corpus
                    .page(&game.split, cursor, 1, Some(&ids))
                    .err()
                    .expect("invalid explicit source");
                assert!(error.to_string().contains(message), "{error}");
            }
        }
        let page = corpus.page(&game.split, (0, 0), 8, None).unwrap();
        assert_eq!(page.games.len(), 1);
        assert_eq!(page.games[0].run_id, sealed);
        corpus.finish(unsealed, "interrupted", None).unwrap();
        let page = corpus
            .page(&game.split, (0, 0), 8, Some(&[unsealed, sealed]))
            .unwrap();
        assert_eq!(page.games.len(), 1);
        corpus.close().unwrap();
    }

    #[test]
    fn source_filter_skips_replay_and_pages_selected_runs_once() {
        let (_dir, mut corpus) = create();
        let game = fixture();
        let mut runs = Vec::new();
        for _ in 0..3 {
            let run = corpus.start(&config()).unwrap();
            corpus.save(run, &game).unwrap();
            corpus.finish(run, "finished", None).unwrap();
            runs.push(run);
        }
        // Poison an unselected run: if filtering moves after reconstruction,
        // the selected read will now fail instead of merely getting slower.
        let w = work().unwrap();
        let mut values = Vec::new();
        {
            let snapshot = corpus.db.snapshot(&w).unwrap();
            let frame = snapshot.frame(&w);
            let fact = frame
                .scan_facts::<PieceRemoved>()
                .unwrap()
                .map(|r| r.unwrap())
                .find(|r| r.run.0 == runs[1])
                .unwrap();
            fact.append_values(&mut values).unwrap();
        }
        let mut draft = ChangeSet::builder(corpus.db.schema(), w.clone());
        draft.delete(PieceRemoved::RELATION, &values).unwrap();
        corpus.apply(draft, &w).unwrap();
        let selection = [runs[2], runs[0], runs[2]];
        let mut cursor = (0, 0);
        for run in [runs[0], runs[2]] {
            let page = corpus
                .page(&game.split, cursor, 1, Some(&selection))
                .unwrap();
            assert_eq!(page.games.len(), 1);
            assert_eq!(page.games[0].run_id, run);
            assert_eq!(page.games[0].trajectory, game);
            assert!(!page.done);
            cursor = page.cursor;
        }
        let page = corpus
            .page(&game.split, cursor, 1, Some(&selection))
            .unwrap();
        assert!(page.done && page.games.is_empty());
        assert!(corpus.page(&game.split, (0, 0), 8, None).is_err());
        assert!(
            corpus
                .page(&game.split, (0, 0), 8, Some(&[runs[1]]))
                .is_err()
        );
        corpus.close().unwrap();
    }

    #[test]
    fn retained_query_work_is_flat_across_game_count_and_fresh_snapshots() {
        use bumbledb::work::Resource;
        let original = fixture();
        let mut costs = Vec::new();
        for size in [16, 128] {
            let (_dir, mut corpus) = create();
            let mut config = config();
            config.pairs = size;
            let run = corpus.start(&config).unwrap();
            for pair in 0..size {
                let mut game = original.clone();
                game.pair = Some(pair);
                game.index = pair * 2;
                corpus.save(run, &game).unwrap();
            }
            corpus.finish(run, "finished", None).unwrap();
            let read = |index, fresh| {
                // Separate snapshot AND operation budget on every call, just
                // like the real audit and paged reader. Only queries survive.
                let w = work().unwrap();
                let snapshot = corpus.db.snapshot(&w).unwrap();
                let frame = snapshot.frame(&w);
                let game = frame
                    .get(GameByRunIndex {
                        run: RunId(run),
                        index,
                    })
                    .unwrap()
                    .unwrap();
                let before = w.used(Resource::WorkUnits);
                let result = if fresh {
                    let mut query = relational::GameReader::new(&frame).unwrap();
                    relational::read_game(&frame, &game, &mut query).unwrap()
                } else {
                    corpus.read_game(&frame, &game).unwrap()
                };
                (result, w.used(Resource::WorkUnits) - before)
            };
            read(0, false);
            let index = (size as u64 - 1) * 2;
            let (warm, warm_work) = read(index, false);
            let (rebuilt, rebuilt_work) = read(index, true);
            assert_eq!(warm, rebuilt);
            assert_eq!(warm.plies, original.plies);
            eprintln!("READER_WORK games={size} retained={warm_work} rebuilt={rebuilt_work}");
            costs.push((warm_work, rebuilt_work));
            corpus.close().unwrap();
        }
        for &(warm, rebuilt) in &costs {
            assert!(
                warm < rebuilt,
                "retained query must use less work: {costs:?}"
            );
        }
        // Total work includes fixed per-game point reads and validation. Test
        // scaling, not a blanket speedup that small fixtures cannot deliver.
        assert!(
            costs[1].0 <= costs[0].0 * 2,
            "eight times as many games must not force another relation-sized pass: {costs:?}"
        );
        assert!(
            costs[1].1 - costs[1].0 >= (costs[0].1 - costs[0].0) * 4,
            "rebuilding must expose the relation-sized work avoided by retention: {costs:?}"
        );
    }

    #[test]
    fn corpus_rejects_unsafe_castles_and_roundtrips_safe_castles() {
        use push_chess::core::{position::Position as Board, types as c};
        let fixtures = [
            ("k7/8/8/8/8/8/7n/4K2R w K - 0 1", true),
            ("7k/8/8/8/8/8/1n6/R3K3 w Q - 0 1", true),
            ("4k2r/7N/8/8/8/8/8/K7 b k - 0 1", true),
            ("r3k3/1N6/8/8/8/8/8/7K b q - 0 1", true),
            ("k7/8/8/8/8/8/8/4K2R w K - 0 1", false),
            ("7k/8/8/8/8/8/8/R3K3 w Q - 0 1", false),
            ("4k2r/8/8/8/8/8/8/K7 b k - 0 1", false),
            ("r3k3/8/8/8/8/8/8/7K b q - 0 1", false),
            // Knight geometry alone is not control in push chess: the pawn
            // blocks one route and h1's enemy rook blocks the other.
            ("k7/8/8/8/8/8/6Pn/4K2R w K - 0 1", false),
        ];
        let (_dir, mut corpus) = create();
        let mut config = config();
        config.pairs = fixtures.len();
        let run = corpus.start(&config).unwrap();
        for (i, (fen, anomaly)) in fixtures.into_iter().enumerate() {
            let mut board = Board::try_from_fen(fen).unwrap();
            let mut legal = Vec::new();
            super::super::legal_moves(&mut board, &mut legal);
            let from = board.king_sq[board.side_to_move as usize];
            let mv = c::Move {
                from,
                to: if board.castling_rights & 5 != 0 {
                    from + 2
                } else {
                    from - 2
                },
                special: c::SpecialMove::Castle,
                ..Default::default()
            };
            assert_eq!(legal.contains(&mv), !anomaly);
            let record = super::super::Ply {
                action: mv.id(),
                origin: "opening",
                side: board.side_to_move as i32,
                score: None,
                score_perspective: "stm",
                search_complete: false,
                nodes: 0,
                depth: 0,
                seldepth: 0,
                wall_us: 0,
                reported_us: 0,
                pv: vec![],
                diagnostics: c::SearchDiagnostics::default(),
                pieces: board.board.iter().filter(|p| !p.is_empty()).count() as u32,
                halfmove_clock: board.halfmove_clock,
            };
            board.make_move(&mv);
            super::super::legal_moves(&mut board, &mut legal);
            let (ending, value) =
                super::super::terminal(&push_chess::game::adjudicate(&board, &legal));
            let key = super::super::opening_key(fen, std::iter::once(mv.id()));
            let game = Trajectory {
                index: i * 2,
                pair: Some(i),
                white: "cataclysm".into(),
                black: "kinetic".into(),
                initial_fen: fen.into(),
                final_fen: board.to_fen(),
                opening_key: key.clone(),
                split: super::super::split(&key).into(),
                termination: ending.into(),
                white_value: value,
                plies: vec![record],
            };
            if anomaly {
                assert!(
                    corpus
                        .save(run, &game)
                        .unwrap_err()
                        .to_string()
                        .contains("illegal stored action")
                );
            } else {
                assert_eq!(game.validate().unwrap().castles, 1);
                corpus.save(run, &game).unwrap();
                assert_eq!(
                    corpus.game(run, game.index as u64).unwrap().trajectory,
                    game
                );
            }
        }
        corpus.finish(run, "finished", None).unwrap();
        let report = corpus.verify(run).unwrap();
        assert_eq!(report["verified_games"], 5);
        assert_eq!(report["castling_audit"]["castles"], 5);
        corpus.close().unwrap();
    }

    #[test]
    fn run_ids_continue_after_preserved_gaps() {
        let (_dir, corpus) = create();
        let config = config();
        let roster = Roster::builtins(&config.engines).unwrap();
        let w = work().unwrap();
        let mut draft = ChangeSet::builder(corpus.db.schema(), w.clone());
        relational::write_config(&mut draft, RunId(6), &config, &roster, [1; 32], 1).unwrap();
        corpus.apply(draft, &w).unwrap();
        assert_eq!(corpus.start(&config).unwrap(), 7);
        assert_eq!(
            corpus.summary().unwrap()["runs"].as_array().unwrap().len(),
            2
        );
        corpus.close().unwrap();
    }

    #[test]
    fn complete_worker_pipeline_and_pre_stopped_shutdown() {
        let (_dir, mut corpus) = create();
        let run = super::super::generate(
            &mut corpus,
            &config(),
            &std::sync::atomic::AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(corpus.report(run).unwrap()["status"], "finished");
        assert_eq!(corpus.summary().unwrap()["totals"]["games"], 2);
        let run = super::super::generate(
            &mut corpus,
            &config(),
            &std::sync::atomic::AtomicBool::new(true),
        )
        .unwrap();
        assert_eq!(corpus.report(run).unwrap()["status"], "interrupted");
        assert_eq!(corpus.summary().unwrap()["totals"]["games"], 2);
        corpus.close().unwrap();
    }

    #[test]
    fn granite_tournament_records_neural_absence_without_changing_abacus() {
        let (_dir, mut corpus) = create();
        let mut config = config();
        config.engines = vec!["abacus".into(), "granite".into()];
        let run = super::super::generate(
            &mut corpus,
            &config,
            &std::sync::atomic::AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(corpus.verify(run).unwrap()["verified_games"], 2);
        let w = work().unwrap();
        {
            let snapshot = corpus.db.snapshot(&w).unwrap();
            let frame = snapshot.frame(&w);
            for slot in 0..2 {
                let entrant = frame
                    .get(EntrantByRunSlot {
                        run: RunId(run),
                        slot,
                    })
                    .unwrap()
                    .unwrap();
                let engine = frame
                    .get(EngineById { id: entrant.engine })
                    .unwrap()
                    .unwrap();
                let network = frame
                    .get(EngineNetworkByEngine {
                        engine: entrant.engine,
                    })
                    .unwrap();
                assert_eq!(engine.neural_accumulator, slot == 0);
                assert!(!engine.neural_evaluation);
                assert_eq!(network.is_some(), slot == 0);
                assert_eq!(engine.name, config.engines[slot as usize]);
            }
        }
        corpus.close().unwrap();
    }

    #[test]
    fn candidate_tournament_records_actual_identity_and_rejects_name_reuse() {
        use super::super::{Candidate, generate_controlled};
        let (_dir, mut corpus) = create();
        let mut config = config();
        config.engines = vec!["cataclysm".into(), "aurora-fixture".into()];
        config.pairs = 2;
        config.workers = 2;
        let bytes = push_chess::engines::cataclysm::Model::CONTROL_BYTES;
        let roster = Roster::resolve(
            &config.engines,
            vec![Candidate::decode("aurora-fixture", bytes).unwrap()],
        )
        .unwrap();
        let run = generate_controlled(
            &mut corpus,
            &config,
            &roster,
            &std::sync::atomic::AtomicBool::new(false),
            None,
        )
        .unwrap();
        assert_eq!(corpus.verify(run).unwrap()["verified_games"], 4);
        let report = corpus.report(run).unwrap();
        assert_eq!(report["matchups"][0]["a"], "cataclysm");
        assert_eq!(report["matchups"][0]["b"], "aurora-fixture");
        let w = work().unwrap();
        {
            let snapshot = corpus.db.snapshot(&w).unwrap();
            let frame = snapshot.frame(&w);
            let binary = digest_file(&std::env::current_exe().unwrap()).unwrap();
            for (slot, expected) in roster.entries.iter().enumerate() {
                let entrant = frame
                    .get(EntrantByRunSlot {
                        run: RunId(run),
                        slot: slot as u64,
                    })
                    .unwrap()
                    .unwrap();
                assert_eq!(entrant.engine.0, expected.identity(binary));
                let net = frame
                    .get(EngineNetworkByEngine {
                        engine: entrant.engine,
                    })
                    .unwrap()
                    .unwrap();
                assert_eq!(net.fingerprint, expected.network_fingerprint().unwrap());
            }
        }
        let mut changed = bytes.to_vec();
        changed[0] ^= 1;
        let different = Roster::resolve(
            &config.engines,
            vec![Candidate::decode("aurora-fixture", &changed).unwrap()],
        )
        .unwrap();
        // Engine(binary,name)->Engine prevents relabelling the same named
        // contender with different weights; even the Run insert rolls back.
        assert!(corpus.start_resolved(&config, &different).is_err());
        assert_eq!(
            corpus.summary().unwrap()["runs"].as_array().unwrap().len(),
            1
        );
        let second = corpus.start_resolved(&config, &roster).unwrap();
        assert_eq!(second, 2);
        corpus.finish(second, "finished", None).unwrap();
        corpus.close().unwrap();
    }

    #[test]
    fn report_joins_both_colors_and_keeps_unknown_and_missing_distinct() {
        let (_dir, mut corpus) = create();
        let mut config = config();
        config.pairs = 2;
        let run = corpus.start(&config).unwrap();
        let mut game = fixture();
        game.initial_fen = "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1".into();
        game.final_fen = game.initial_fen.clone();
        game.plies.clear();
        game.termination = "checkmate".into();
        game.white_value = Some(1);
        game.opening_key = super::super::opening_key(&game.initial_fen, std::iter::empty());
        game.split = super::super::split(&game.opening_key).into();
        corpus.save(run, &game).unwrap();
        game.index = 1;
        std::mem::swap(&mut game.white, &mut game.black);
        corpus.save(run, &game).unwrap();
        let mut capped = fixture();
        capped.index = 2;
        capped.pair = Some(1);
        corpus.save(run, &capped).unwrap();
        let report = corpus.report(run).unwrap();
        let matchup = &report["matchups"][0];
        assert_eq!(matchup["wins"], 1);
        assert_eq!(matchup["losses"], 1);
        assert_eq!(matchup["saved_without_outcome"], 1);
        assert_eq!(matchup["missing_games"], 1);
        assert_eq!(matchup["score_bounds"], json!([0.25, 0.75]));
        assert_eq!(matchup["paired"]["complete_pairs"], 1);
        assert_eq!(matchup["paired"]["incomplete_or_missing_pairs"], 1);
        assert_eq!(
            matchup["paired"]["pair_points_histogram"],
            json!([0, 0, 1, 0, 0])
        );
        assert_eq!(matchup["paired"]["pair_score"], 0.5);
        assert_eq!(report["promotion_ready"], false);
        corpus.close().unwrap();
    }

    #[test]
    fn relational_piece_differences_reconstruct_special_moves() {
        use push_chess::core::{position::Position as Board, types as c};
        let fixtures = [
            (
                "8/7k/4RB2/8/4N3/8/8/K7 w - - 0 1",
                28,
                45,
                1,
                c::SpecialMove::None,
            ),
            (
                "7k/P7/R7/8/8/8/8/K7 w - - 0 1",
                40,
                48,
                0,
                c::SpecialMove::Promotion,
            ),
            (
                "7k/P7/R7/8/8/8/8/K7 w - - 0 1",
                48,
                56,
                0,
                c::SpecialMove::Promotion,
            ),
            (
                "r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1",
                4,
                6,
                0,
                c::SpecialMove::Castle,
            ),
            (
                "r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1",
                36,
                43,
                0,
                c::SpecialMove::EnPassant,
            ),
        ];
        let (_dir, mut corpus) = create();
        let mut config = config();
        config.pairs = fixtures.len();
        let run = corpus.start(&config).unwrap();
        for (i, (fen, from, to, path, special)) in fixtures.iter().enumerate() {
            let mut board = Board::try_from_fen(fen).unwrap();
            let initial = board.clone();
            let mut legal = Vec::new();
            super::super::legal_moves(&mut board, &mut legal);
            let mv = *legal
                .iter()
                .find(|m| {
                    m.from == *from
                        && m.to == *to
                        && m.path_kind == *path
                        && m.special == *special
                        && (*special != c::SpecialMove::Promotion
                            || m.promo_piece == c::PieceType::Queen)
                })
                .expect("special fixture exists");
            let record = super::super::Ply {
                action: mv.id(),
                origin: "opening",
                side: 0,
                score: None,
                score_perspective: "stm",
                search_complete: false,
                nodes: 0,
                depth: 0,
                seldepth: 0,
                wall_us: 0,
                reported_us: 0,
                pv: vec![],
                diagnostics: c::SearchDiagnostics::default(),
                pieces: board.board.iter().filter(|p| !p.is_empty()).count() as u32,
                halfmove_clock: board.halfmove_clock,
            };
            board.make_move(&mv);
            super::super::legal_moves(&mut board, &mut legal);
            let (ending, value) =
                super::super::terminal(&push_chess::game::adjudicate(&board, &legal));
            let key = super::super::opening_key(fen, std::iter::once(mv.id()));
            let g = Trajectory {
                index: i * 2,
                pair: Some(i),
                white: "cataclysm".into(),
                black: "kinetic".into(),
                initial_fen: fen.to_string(),
                final_fen: board.to_fen(),
                opening_key: key.clone(),
                split: super::super::split(&key).into(),
                termination: ending.into(),
                white_value: value,
                plies: vec![record],
            };
            corpus.save(run, &g).unwrap();
            let w = work().unwrap();
            let snapshot = corpus.db.snapshot(&w).unwrap();
            let frame = snapshot.frame(&w);
            let stored = frame
                .get(GameByRunIndex {
                    run: RunId(run),
                    index: (i * 2) as u64,
                })
                .unwrap()
                .unwrap();
            assert_eq!(corpus.read_game(&frame, &stored).unwrap(), g);
            let mut squares = std::collections::BTreeMap::new();
            let kinds = [
                PieceKind::Pawn.id(),
                PieceKind::Knight.id(),
                PieceKind::Bishop.id(),
                PieceKind::Rook.id(),
                PieceKind::Queen.id(),
                PieceKind::King.id(),
            ];
            let fields = |p: c::Piece| {
                (
                    if p.color == c::Color::White {
                        Side::White.id()
                    } else {
                        Side::Black.id()
                    },
                    kinds[p.piece_type as usize - 1],
                )
            };
            for (sq, p) in initial
                .board
                .iter()
                .enumerate()
                .filter(|(_, p)| !p.is_empty())
            {
                squares.insert(sq as u64, fields(*p));
            }
            for r in frame.scan_facts::<PieceRemoved>().unwrap() {
                let r = r.unwrap();
                if r.game != g.index as u64 {
                    continue;
                }
                assert_eq!(squares.remove(&r.square), Some((r.side, r.kind)));
            }
            for r in frame.scan_facts::<PiecePlaced>().unwrap() {
                let r = r.unwrap();
                if r.game != g.index as u64 {
                    continue;
                }
                assert!(squares.insert(r.square, (r.side, r.kind)).is_none());
            }
            for (sq, p) in board.board.iter().enumerate() {
                assert_eq!(
                    squares.get(&(sq as u64)).copied(),
                    if p.is_empty() { None } else { Some(fields(*p)) }
                );
            }
        }
        corpus.close().unwrap();
    }
}
