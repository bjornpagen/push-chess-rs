//! Canonical facts at the storage boundary; operational board arrays remain
//! local to the engine. Text encodings exist only at external presentation.
use super::corpus::{ending_id, ending_name, insert, split_id, unhex, white_value};
use super::schema::*;
use super::{Ply, Result, Roster, RunConfig, Trajectory, legal_moves};
use bumbledb::{BindValue, ChangeSetBuilder, PreparedQuery, ReadFrame};
use push_chess::core::{position::Position as Board, types as core};
use sha2::{Digest, Sha256};

fn side(c: core::Color) -> SideId {
    if c == core::Color::White {
        Side::White.id()
    } else {
        Side::Black.id()
    }
}
fn color(c: SideId) -> Result<core::Color> {
    if c == Side::White.id() {
        Ok(core::Color::White)
    } else if c == Side::Black.id() {
        Ok(core::Color::Black)
    } else {
        Err("invalid side".into())
    }
}
fn kind(p: core::PieceType) -> Result<PieceKindId> {
    Ok(match p {
        core::PieceType::Pawn => PieceKind::Pawn.id(),
        core::PieceType::Knight => PieceKind::Knight.id(),
        core::PieceType::Bishop => PieceKind::Bishop.id(),
        core::PieceType::Rook => PieceKind::Rook.id(),
        core::PieceType::Queen => PieceKind::Queen.id(),
        core::PieceType::King => PieceKind::King.id(),
        core::PieceType::None => return Err("empty square has no piece fact".into()),
    })
}
fn piece(id: PieceKindId) -> Result<core::PieceType> {
    for p in [
        core::PieceType::Pawn,
        core::PieceType::Knight,
        core::PieceType::Bishop,
        core::PieceType::Rook,
        core::PieceType::Queen,
        core::PieceType::King,
    ] {
        if kind(p)? == id {
            return Ok(p);
        }
    }
    Err("invalid piece kind".into())
}
fn castles() -> [(u8, SideId, WingId); 4] {
    [
        (1, Side::White.id(), Wing::KingSide.id()),
        (2, Side::White.id(), Wing::QueenSide.id()),
        (4, Side::Black.id(), Wing::KingSide.id()),
        (8, Side::Black.id(), Wing::QueenSide.id()),
    ]
}
pub(super) fn config(frame: &ReadFrame<'_, TrainingGround>, run: &Run) -> Result<RunConfig> {
    let mut entrants = Vec::new();
    for e in frame.scan_facts::<Entrant>()? {
        let e = e?;
        if e.run == run.id {
            let engine = frame
                .get(EngineById { id: e.engine })?
                .ok_or("missing engine")?;
            entrants.push((e.slot, engine.name.to_owned()));
        }
    }
    entrants.sort_by_key(|e| e.0);
    if entrants.iter().enumerate().any(|(i, e)| i as u64 != e.0) {
        return Err("entrant slots are not contiguous".into());
    }
    let nodes = frame
        .get(NodeBudgetByRun { run: run.id })?
        .map_or(0, |b| b.nodes);
    let micros = frame
        .get(TimeBudgetByRun { run: run.id })?
        .map_or(0, |b| b.microseconds);
    if micros % 1000 != 0 {
        return Err("invalid millisecond budget".into());
    }
    Ok(RunConfig {
        engines: entrants.into_iter().map(|e| e.1).collect(),
        pairs: run.pairs.try_into()?,
        workers: run.workers.try_into()?,
        nodes,
        time_ms: micros / 1000,
        max_plies: run.max_plies.try_into()?,
        opening_plies: run.opening_plies.try_into()?,
        seed: run.seed,
        purpose: if run.purpose == Purpose::Corpus.id() {
            "corpus"
        } else {
            "arena"
        }
        .into(),
        max_seconds: run.max_seconds,
        max_bytes: run.max_bytes,
    })
}
pub(super) fn write_config(
    draft: &mut ChangeSetBuilder<'_>,
    id: RunId,
    c: &RunConfig,
    roster: &Roster,
    binary: [u8; 32],
    now: u64,
) -> Result<()> {
    insert(
        draft,
        &Run {
            id,
            binary,
            started_us: now,
            pairs: c.pairs as u64,
            workers: c.workers as u64,
            max_plies: c.max_plies as u64,
            opening_plies: c.opening_plies as u64,
            seed: c.seed,
            purpose: if c.purpose == "corpus" {
                Purpose::Corpus.id()
            } else {
                Purpose::Arena.id()
            },
            budget: if c.nodes > 0 {
                BudgetKind::Nodes.id()
            } else {
                BudgetKind::Time.id()
            },
            max_seconds: c.max_seconds,
            max_bytes: c.max_bytes,
        },
    )?;
    if c.nodes > 0 {
        insert(
            draft,
            &NodeBudget {
                run: id,
                nodes: c.nodes,
            },
        )?;
    } else {
        insert(
            draft,
            &TimeBudget {
                run: id,
                microseconds: c.time_ms * 1000,
            },
        )?;
    }
    for (slot, entrant) in roster.entries.iter().enumerate() {
        let info = entrant.info();
        let engine = EngineId(entrant.identity(binary));
        insert(
            draft,
            &Engine {
                id: engine,
                binary,
                name: &entrant.name,
                lineage: info.lineage,
                hypothesis: info.hypothesis,
                neural_accumulator: info.neural_accumulator,
                neural_evaluation: info.neural_evaluation,
            },
        )?;
        insert(
            draft,
            &Entrant {
                run: id,
                slot: slot as u64,
                engine,
            },
        )?;
        if let Some(fingerprint) = entrant.network_fingerprint() {
            insert(
                draft,
                &EngineNetwork {
                    engine,
                    fingerprint,
                },
            )?;
        }
    }
    Ok(())
}
fn write_action(draft: &mut ChangeSetBuilder<'_>, id: u32) -> Result<()> {
    let route = match (id >> 12) & 3 {
        0 => Route::Direct.id(),
        1 => Route::RankFirst.id(),
        2 => Route::FileFirst.id(),
        _ => return Err("invalid action route".into()),
    };
    let move_kind = match (id >> 18) & 3 {
        0 => MoveKind::Ordinary.id(),
        1 => MoveKind::Castle.id(),
        2 => MoveKind::EnPassant.id(),
        _ => MoveKind::Promotion.id(),
    };
    let promotion = id >> 20;
    if move_kind == MoveKind::Promotion.id() {
        let piece = match promotion {
            2 => PieceKind::Knight.id(),
            3 => PieceKind::Bishop.id(),
            4 => PieceKind::Rook.id(),
            5 => PieceKind::Queen.id(),
            _ => return Err("invalid promotion".into()),
        };
        insert(
            draft,
            &ActionPromotion {
                action: ActionId(id as u64),
                piece,
            },
        )?;
    } else if promotion != 0 {
        return Err("unexpected promotion bits".into());
    }
    insert(
        draft,
        &Action {
            id: ActionId(id as u64),
            from: (id & 63) as u64,
            to: ((id >> 6) & 63) as u64,
            route,
            stop: ((id >> 14) & 15) as u64,
            kind: move_kind,
        },
    )
}
fn write_setup(draft: &mut ChangeSetBuilder<'_>, pos: &Board) -> Result<SetupId> {
    let id = SetupId(Sha256::digest(pos.to_fen().as_bytes()).into());
    insert(
        draft,
        &Setup {
            id,
            side: side(pos.side_to_move),
            halfmove_clock: pos.halfmove_clock as u64,
            fullmove_number: pos.fullmove_number as u64,
        },
    )?;
    for (square, p) in pos.board.iter().enumerate().filter(|(_, p)| !p.is_empty()) {
        insert(
            draft,
            &SetupPiece {
                setup: id,
                square: square as u64,
                side: side(p.color),
                kind: kind(p.piece_type)?,
            },
        )?;
    }
    for (mask, side, wing) in castles() {
        if pos.castling_rights & mask != 0 {
            insert(
                draft,
                &SetupCastle {
                    setup: id,
                    side,
                    wing,
                },
            )?;
        }
    }
    if pos.ep_square < 64 {
        insert(
            draft,
            &SetupEp {
                setup: id,
                square: pos.ep_square as u64,
            },
        )?;
    }
    Ok(id)
}
fn write_position(
    draft: &mut ChangeSetBuilder<'_>,
    run: RunId,
    game: u64,
    ply: u64,
    pos: &Board,
) -> Result<()> {
    insert(
        draft,
        &Position {
            run,
            game,
            ply,
            side: side(pos.side_to_move),
            halfmove_clock: pos.halfmove_clock as u64,
            fullmove_number: pos.fullmove_number as u64,
            hash: pos.zobrist,
        },
    )?;
    for (mask, side, wing) in castles() {
        if pos.castling_rights & mask != 0 {
            insert(
                draft,
                &CastlingRight {
                    run,
                    game,
                    ply,
                    side,
                    wing,
                },
            )?;
        }
    }
    if pos.ep_square < 64 {
        insert(
            draft,
            &EnPassant {
                run,
                game,
                ply,
                square: pos.ep_square as u64,
            },
        )?;
    }
    Ok(())
}
pub(super) fn write_game(
    draft: &mut ChangeSetBuilder<'_>,
    run: RunId,
    g: &Trajectory,
    c: &RunConfig,
) -> Result<()> {
    if g.index >= c.total_pairs() * 2 {
        return Err("unscheduled game".into());
    }
    let (a, b) = c.matchup(g.index / 2);
    let swapped = g.index % 2 == 1;
    let (white, black) = if swapped {
        (&c.engines[b], &c.engines[a])
    } else {
        (&c.engines[a], &c.engines[b])
    };
    if (&g.white, &g.black) != (white, black) {
        return Err("wrong scheduled players".into());
    }
    let mut pos = Board::try_from_fen(&g.initial_fen)?;
    let setup = write_setup(draft, &pos)?;
    let mut digest = Sha256::new();
    digest.update(g.initial_fen.as_bytes());
    digest.update([0]);
    for m in &g.plies {
        digest.update(m.action.to_le_bytes());
    }
    insert(
        draft,
        &Pair {
            run,
            index: (g.index / 2) as u64,
            first: a as u64,
            second: b as u64,
            opening: unhex(&g.opening_key)?,
        },
    )?;
    insert(
        draft,
        &Game {
            run,
            index: g.index as u64,
            pair: (g.index / 2) as u64,
            swapped,
            setup,
            trajectory: digest.finalize().into(),
            split: split_id(&g.split)?,
            ending: ending_id(&g.termination)?,
            plies: g.plies.len() as u64,
        },
    )?;
    if let Some(v) = g.white_value {
        insert(
            draft,
            &GameResult {
                run,
                game: g.index as u64,
                outcome: match v {
                    1 => WhiteOutcome::Win.id(),
                    0 => WhiteOutcome::Draw.id(),
                    -1 => WhiteOutcome::Loss.id(),
                    _ => return Err("invalid outcome".into()),
                },
            },
        )?;
    }
    let mut legal = Vec::new();
    let game = g.index as u64;
    write_position(draft, run, game, 0, &pos)?;
    for (ply, p) in g.plies.iter().enumerate() {
        let ply = ply as u64;
        write_action(draft, p.action)?;
        insert(
            draft,
            &Move {
                run,
                game,
                ply,
                action: ActionId(p.action as u64),
            },
        )?;
        if let Some(score) = p.score {
            insert(
                draft,
                &Analysis {
                    run,
                    game,
                    ply,
                    score_stm: score as i64,
                    complete: p.search_complete,
                    nodes: p.nodes,
                    depth: p.depth as u64,
                    seldepth: p.seldepth as u64,
                    wall_us: p.wall_us.try_into()?,
                    reported_us: p.reported_us.try_into()?,
                    qnodes: p.diagnostics.qnodes,
                    tt_hits: p.diagnostics.tt_hits,
                    pv_length: p.pv.len() as u64,
                },
            )?;
            for (step, action) in p.pv.iter().enumerate() {
                write_action(draft, *action)?;
                insert(
                    draft,
                    &PvStep {
                        run,
                        game,
                        ply,
                        step: step as u64,
                        action: ActionId(*action as u64),
                    },
                )?;
            }
            if let Some(nodes) = p.diagnostics.proof_nodes {
                insert(
                    draft,
                    &ProofSearch {
                        run,
                        game,
                        ply,
                        nodes,
                    },
                )?;
            }
            if let Some(plies) = p.diagnostics.mate_proof_plies {
                insert(
                    draft,
                    &MateProof {
                        run,
                        game,
                        ply,
                        plies: plies as u64,
                    },
                )?;
            }
        }
        legal_moves(&mut pos, &mut legal);
        let mv = legal
            .iter()
            .find(|m| m.id() == p.action)
            .ok_or("illegal move at fact boundary")?;
        let before = pos.board;
        pos.make_move(mv);
        for (square, old) in before.iter().enumerate() {
            let new = pos.board[square];
            if *old == new {
                continue;
            }
            if !old.is_empty() {
                insert(
                    draft,
                    &PieceRemoved {
                        run,
                        game,
                        ply,
                        square: square as u64,
                        side: side(old.color),
                        kind: kind(old.piece_type)?,
                    },
                )?;
            }
            if !new.is_empty() {
                insert(
                    draft,
                    &PiecePlaced {
                        run,
                        game,
                        ply,
                        square: square as u64,
                        side: side(new.color),
                        kind: kind(new.piece_type)?,
                    },
                )?;
            }
        }
        write_position(draft, run, game, ply + 1, &pos)?;
    }
    Ok(())
}
pub(super) fn setup_board(frame: &ReadFrame<'_, TrainingGround>, setup: SetupId) -> Result<Board> {
    let s = frame.get(SetupById { id: setup })?.ok_or("missing setup")?;
    let mut pos = Board::empty();
    pos.side_to_move = color(s.side)?;
    pos.halfmove_clock = s.halfmove_clock.try_into()?;
    pos.fullmove_number = s.fullmove_number.try_into()?;
    for square in 0..64 {
        if let Some(p) = frame.get(SetupPieceBySetupSquare { setup, square })? {
            let p = core::Piece {
                color: color(p.side)?,
                piece_type: piece(p.kind)?,
            };
            pos.board[square as usize] = p;
            if p.piece_type == core::PieceType::King {
                pos.king_sq[p.color as usize] = square as u8;
            }
        }
    }
    for (mask, side, wing) in castles() {
        if frame
            .get(SetupCastleBySetupSideWing { setup, side, wing })?
            .is_some()
        {
            pos.castling_rights |= mask;
        }
    }
    if let Some(ep) = frame.get(SetupEpBySetup { setup })? {
        pos.ep_square = ep.square.try_into()?;
    }
    pos.compute_zobrist();
    if <[u8; 32]>::from(Sha256::digest(pos.to_fen().as_bytes())) != setup.0 {
        return Err("setup content identity mismatch".into());
    }
    Ok(pos)
}
fn read_action(frame: &ReadFrame<'_, TrainingGround>, id: ActionId) -> Result<u32> {
    let a = frame.get(ActionById { id })?.ok_or("missing action")?;
    let route = if a.route == Route::Direct.id() {
        0
    } else if a.route == Route::RankFirst.id() {
        1
    } else if a.route == Route::FileFirst.id() {
        2
    } else {
        return Err("invalid action route".into());
    };
    let special = if a.kind == MoveKind::Ordinary.id() {
        0
    } else if a.kind == MoveKind::Castle.id() {
        1
    } else if a.kind == MoveKind::EnPassant.id() {
        2
    } else if a.kind == MoveKind::Promotion.id() {
        3
    } else {
        return Err("invalid action kind".into());
    };
    let promotion = frame.get(ActionPromotionByAction { action: id })?;
    if a.from >= 64 || a.to >= 64 || a.stop >= 16 || (special == 3) != promotion.is_some() {
        return Err("invalid action fields".into());
    }
    let promotion = promotion
        .map(|p| piece(p.piece).map(|p| p as u64))
        .transpose()?
        .unwrap_or(0);
    let packed =
        a.from | (a.to << 6) | (route << 12) | (a.stop << 14) | (special << 18) | (promotion << 20);
    if packed != id.0 {
        return Err("action content identity mismatch".into());
    }
    Ok(packed.try_into()?)
}

type PieceDelta = (u64, u64, SideId, PieceKindId);
fn delta_rows(rows: bumbledb::Answers) -> Result<Vec<PieceDelta>> {
    use bumbledb::AnswerValue::U64;
    let mut result = rows
        .answers()
        .map(|r| match (r.get(0), r.get(1), r.get(2), r.get(3)) {
            (U64(ply), U64(square), U64(side), U64(kind)) => {
                Ok((ply, square, SideId(side), PieceKindId(kind)))
            }
            _ => Err("invalid piece delta projection".into()),
        })
        .collect::<Result<Vec<_>>>()?;
    result.sort_unstable();
    Ok(result)
}

/// Prepared queries own the reusable selection indexes. Dropping one for each
/// game discards that work even when the underlying relation image is cached.
/// bumbledb checks source identity and relation epochs on every execution.
pub(super) struct GameReader {
    removed: PreparedQuery<TrainingGround>,
    placed: PreparedQuery<TrainingGround>,
}

impl GameReader {
    pub(super) fn new(frame: &ReadFrame<'_, TrainingGround>) -> Result<Self> {
        let removed = bumbledb::query!(TrainingGround {
            (ply, square, side, kind) | PieceRemoved(run, game, ply, square, side, kind), run == ?run, game == ?game;
        });
        let placed = bumbledb::query!(TrainingGround {
            (ply, square, side, kind) | PiecePlaced(run, game, ply, square, side, kind), run == ?run, game == ?game;
        });
        Ok(Self {
            removed: frame.prepare(&removed)?,
            placed: frame.prepare(&placed)?,
        })
    }
}

pub(super) fn read_game(
    frame: &ReadFrame<'_, TrainingGround>,
    g: &Game,
    reader: &mut GameReader,
) -> Result<Trajectory> {
    // The first query may build a relation-sized selection index. Retain it
    // across games/pages; do not mistake a logical prefix for a physical seek.
    // Exact delta sets remain checked against the authoritative rules replay.
    let args = [BindValue::U64(g.run.0), BindValue::U64(g.index)];
    let removed = delta_rows(frame.execute_collect(&mut reader.removed, &args)?)?;
    let placed = delta_rows(frame.execute_collect(&mut reader.placed, &args)?)?;
    let mut removed = removed.into_iter();
    let mut placed = placed.into_iter();
    let pair = frame
        .get(PairByRunIndex {
            run: g.run,
            index: g.pair,
        })?
        .ok_or("missing pair")?;
    let first = frame
        .get(EntrantByRunSlot {
            run: g.run,
            slot: pair.first,
        })?
        .ok_or("missing entrant")?;
    let second = frame
        .get(EntrantByRunSlot {
            run: g.run,
            slot: pair.second,
        })?
        .ok_or("missing entrant")?;
    let first = frame
        .get(EngineById { id: first.engine })?
        .ok_or("missing engine")?;
    let second = frame
        .get(EngineById { id: second.engine })?
        .ok_or("missing engine")?;
    let (white, black) = if g.swapped {
        (second.name, first.name)
    } else {
        (first.name, second.name)
    };
    let mut pos = setup_board(frame, g.setup)?;
    let initial_fen = pos.to_fen();
    let mut records = Vec::new();
    let mut legal = Vec::new();
    for ply in 0..=g.plies {
        let state = frame
            .get(PositionByRunGamePly {
                run: g.run,
                game: g.index,
                ply,
            })?
            .ok_or("missing position")?;
        if state.hash != pos.zobrist
            || state.side != side(pos.side_to_move)
            || state.halfmove_clock != pos.halfmove_clock as u64
            || state.fullmove_number != pos.fullmove_number as u64
        {
            return Err("position reconstruction mismatch".into());
        }
        for (mask, side, wing) in castles() {
            let stored = frame
                .get(CastlingRightByRunGamePlySideWing {
                    run: g.run,
                    game: g.index,
                    ply,
                    side,
                    wing,
                })?
                .is_some();
            if stored != (pos.castling_rights & mask != 0) {
                return Err("castling reconstruction mismatch".into());
            }
        }
        let ep = frame.get(EnPassantByRunGamePly {
            run: g.run,
            game: g.index,
            ply,
        })?;
        if ep.map(|p| p.square) != (pos.ep_square < 64).then_some(pos.ep_square as u64) {
            return Err("en passant reconstruction mismatch".into());
        }
        if ply == g.plies {
            break;
        }
        let m = frame
            .get(MoveByRunGamePly {
                run: g.run,
                game: g.index,
                ply,
            })?
            .ok_or("missing move")?;
        let a = frame.get(AnalysisByRunGamePly {
            run: g.run,
            game: g.index,
            ply,
        })?;
        let mut pv = Vec::new();
        if let Some(a) = &a {
            for step in 0..a.pv_length {
                pv.push(read_action(
                    frame,
                    frame
                        .get(PvStepByRunGamePlyStep {
                            run: g.run,
                            game: g.index,
                            ply,
                            step,
                        })?
                        .ok_or("missing PV step")?
                        .action,
                )?);
            }
        }
        let proof = frame.get(ProofSearchByRunGamePly {
            run: g.run,
            game: g.index,
            ply,
        })?;
        let mate = frame.get(MateProofByRunGamePly {
            run: g.run,
            game: g.index,
            ply,
        })?;
        records.push(Ply {
            action: read_action(frame, m.action)?,
            origin: if a.is_some() { "search" } else { "opening" },
            side: pos.side_to_move as i32,
            score: a.as_ref().map(|x| x.score_stm.try_into()).transpose()?,
            score_perspective: "stm",
            search_complete: a.as_ref().is_some_and(|x| x.complete),
            nodes: a.as_ref().map_or(0, |x| x.nodes),
            depth: a.as_ref().map_or(0, |x| x.depth).try_into()?,
            seldepth: a.as_ref().map_or(0, |x| x.seldepth).try_into()?,
            wall_us: a.as_ref().map_or(0, |x| x.wall_us).try_into()?,
            reported_us: a.as_ref().map_or(0, |x| x.reported_us).try_into()?,
            pv,
            diagnostics: core::SearchDiagnostics {
                qnodes: a.as_ref().map_or(0, |x| x.qnodes),
                tt_hits: a.as_ref().map_or(0, |x| x.tt_hits),
                proof_nodes: proof.map(|p| p.nodes),
                mate_proof_plies: mate.map(|p| p.plies.try_into()).transpose()?,
            },
            pieces: pos.board.iter().filter(|p| !p.is_empty()).count() as u32,
            halfmove_clock: pos.halfmove_clock,
        });
        legal_moves(&mut pos, &mut legal);
        let mv = legal
            .iter()
            .find(|mv| mv.id() as u64 == m.action.0)
            .ok_or("illegal corpus move")?;
        let before = pos.board;
        pos.make_move(mv);
        for (square, old) in before.iter().enumerate() {
            let new = pos.board[square];
            if *old == new {
                continue;
            }
            if !old.is_empty()
                && removed.next()
                    != Some((ply, square as u64, side(old.color), kind(old.piece_type)?))
            {
                return Err("piece removal set differs from legal transition".into());
            }
            if !new.is_empty()
                && placed.next()
                    != Some((ply, square as u64, side(new.color), kind(new.piece_type)?))
            {
                return Err("piece placement set differs from legal transition".into());
            }
        }
    }
    if removed.next().is_some() || placed.next().is_some() {
        return Err("extraneous transition facts".into());
    }
    let mut digest = Sha256::new();
    digest.update(initial_fen.as_bytes());
    digest.update([0]);
    for p in &records {
        digest.update(p.action.to_le_bytes());
    }
    if <[u8; 32]>::from(digest.finalize()) != g.trajectory {
        return Err("trajectory content identity mismatch".into());
    }
    let result = frame.get(GameResultByRunGame {
        run: g.run,
        game: g.index,
    })?;
    Ok(Trajectory {
        index: g.index.try_into()?,
        pair: Some(g.pair.try_into()?),
        white: white.into(),
        black: black.into(),
        initial_fen,
        final_fen: pos.to_fen(),
        opening_key: super::corpus::hex(&pair.opening),
        split: if g.split == Split::Train.id() {
            "train"
        } else if g.split == Split::Validation.id() {
            "validation"
        } else if g.split == Split::Test.id() {
            "test"
        } else {
            "evaluation"
        }
        .into(),
        termination: ending_name(g.ending)?.into(),
        white_value: result.map(|r| white_value(r.outcome)).transpose()?,
        plies: records,
    })
}
