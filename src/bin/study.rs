//! Read-only, exhaustive history audit. Reconstructs each saved transition
//! independently, so a pre-refactor mismatch cannot contaminate later analysis.
use push_chess::core::movegen::generate_legal_moves;
use push_chess::core::position::Position;
use push_chess::core::push::{resolve_knight_push, resolve_push};
use push_chess::core::types::*;
use rusqlite::{Connection, OpenFlags};
use std::collections::{BTreeMap, HashSet};
use std::fmt::Write;

#[derive(Default)]
struct Totals {
    moves: usize,
    legal: usize,
    exact: usize,
    captures: usize,
    false_captures: usize,
    pushes: usize,
    empty_pushes: usize,
    king_pushes: usize,
    promotions: usize,
    pushed_promotions: usize,
    checks: usize,
    push_checks: usize,
    quiet_checks: usize,
    nodes: u64,
    time: u64,
    depth: u64,
    cliffs: usize,
}

fn special(s: &str) -> SpecialMove {
    match s {
        "none" => SpecialMove::None,
        "castle" => SpecialMove::Castle,
        "en_passant" => SpecialMove::EnPassant,
        "promotion" => SpecialMove::Promotion,
        _ => panic!("unknown stored special: {s}"),
    }
}

fn piece(s: &str) -> PieceType {
    match s {
        "none" => PieceType::None,
        "queen" => PieceType::Queen,
        "rook" => PieceType::Rook,
        "bishop" => PieceType::Bishop,
        "knight" => PieceType::Knight,
        _ => panic!("unknown stored promotion: {s}"),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        return Err("usage: study <new-report.md> <database> [database ...]".into());
    }
    // Refuse overwrites: reports are snapshots, not live database mutations.
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[0])?;
    let mut engines: BTreeMap<String, Totals> = BTreeMap::new();
    let mut endings = BTreeMap::<String, usize>::new();
    let mut ledger = String::from(
        "\n## Every game\n\nLegal/exact counts compare independently reconstructed saved moves with today's rules. Empty game rows are retained. A cliff is a recorded self-evaluation drop of at least 300 cp between consecutive turns, excluding mate scores; it is a lead for investigation, not a proven blunder.\n\n| Database | Game | White / Black | Result | End | Plies saved/reported | Legal/exact | Push/check/promo | Eval cliffs |\n|---|---:|---|---|---|---|---|---|---:|\n",
    );
    let mut games = 0;
    let mut no_moves = 0;
    let mut unique_positions = HashSet::new();
    let mut unique_games = HashSet::new();
    let mut discontinuities = 0;
    for path in &args[1..] {
        let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        let mut gs = conn.prepare("SELECT game_id, white_id, black_id, result, termination, ply_count FROM games ORDER BY game_id")?;
        let records = gs.query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, i64>(5)?,
            ))
        })?;
        let mut ms = conn.prepare("SELECT m.ply,m.candidate_id,m.fen_before,m.fen_after,m.move_from,m.move_to,m.path_kind,m.stop_index,m.special,m.promo_piece,m.is_capture,COALESCE(s.nodes,0),COALESCE(s.time_us,0),COALESCE(s.depth,0),COALESCE(s.eval_cp,0) FROM moves m LEFT JOIN search s USING(move_id) WHERE game_id=?1 ORDER BY ply")?;
        for record in records {
            let (id, white, black, result, end, reported) = record?;
            games += 1;
            *endings.entry(end.clone()).or_default() += 1;
            let mut rows = ms.query([id])?;
            let (
                mut count,
                mut legal_count,
                mut exact_count,
                mut pushes,
                mut checks,
                mut promos,
                mut cliffs,
            ) = (0, 0, 0, 0, 0, 0, 0);
            let mut previous = String::new();
            let mut signature = Vec::new();
            let mut evals = [None; 2];
            while let Some(row) = rows.next()? {
                let ply: usize = row.get(0)?;
                let engine: String = row.get(1)?;
                let before: String = row.get(2)?;
                let after: String = row.get(3)?;
                if (count > 0 && before != previous) || ply != count {
                    discontinuities += 1;
                }
                previous = after.clone();
                let mv = Move {
                    from: row.get(4)?,
                    to: row.get(5)?,
                    path_kind: row.get(6)?,
                    stop_index: row.get(7)?,
                    special: special(&row.get::<_, String>(8)?),
                    promo_piece: piece(&row.get::<_, String>(9)?),
                };
                signature.push((
                    mv.from,
                    mv.to,
                    mv.path_kind,
                    mv.promo_piece as u8,
                    before
                        .split_whitespace()
                        .take(4)
                        .collect::<Vec<_>>()
                        .join(" "),
                ));
                unique_positions.insert(
                    before
                        .split_whitespace()
                        .take(4)
                        .collect::<Vec<_>>()
                        .join(" "),
                );
                let t = engines.entry(engine).or_default();
                t.moves += 1;
                count += 1;
                t.nodes += row.get::<_, u64>(11)?;
                t.time += row.get::<_, u64>(12)?;
                t.depth += row.get::<_, u64>(13)?;
                let ev: i32 = row.get(14)?;
                if let Some(old) = evals[ply % 2]
                    && ev.abs() < 20_000
                    && i32::abs(old) < 20_000
                    && old - ev >= 300
                {
                    t.cliffs += 1;
                    cliffs += 1;
                }
                evals[ply % 2] = Some(ev);
                let mut pos = Position::empty();
                pos.set_from_fen(&before);
                if pos.king_sq.iter().any(|&s| s >= 64) {
                    continue;
                }
                let mover = pos.board[mv.from as usize];
                let target = pos.board[mv.to as usize];
                let cap = target.is_color(opponent(pos.side_to_move))
                    || mv.special == SpecialMove::EnPassant;
                t.captures += usize::from(cap);
                t.false_captures += usize::from(row.get::<_, bool>(10)? && !cap);
                let plan =
                    if mv.special == SpecialMove::Castle || mv.special == SpecialMove::EnPassant {
                        None
                    } else if mover.piece_type == PieceType::Knight {
                        resolve_knight_push(&pos, mv.from, mv.to, mv.path_kind == 1)
                    } else {
                        resolve_push(
                            &pos,
                            mv.from,
                            mv.to,
                            (rank_of(mv.to) - rank_of(mv.from)).signum(),
                            (file_of(mv.to) - file_of(mv.from)).signum(),
                        )
                    };
                let push = plan.as_ref().is_some_and(|p| p.displacements().len() > 1);
                if push {
                    t.pushes += 1;
                    pushes += 1;
                    t.empty_pushes += usize::from(target.is_empty());
                    t.king_pushes +=
                        usize::from(plan.as_ref().unwrap().displacements().iter().any(
                            |&(f, _)| {
                                f != mv.from && pos.board[f as usize].piece_type == PieceType::King
                            },
                        ));
                }
                if mv.special == SpecialMove::Promotion {
                    t.promotions += 1;
                    promos += 1;
                    t.pushed_promotions += usize::from(
                        mover.piece_type != PieceType::Pawn
                            || rank_of(mv.to) != if mover.color == Color::White { 7 } else { 0 },
                    );
                }
                let mut legal = Vec::new();
                generate_legal_moves(&mut pos, &mut legal);
                if !legal.contains(&mv) {
                    continue;
                }
                t.legal += 1;
                legal_count += 1;
                pos.make_move(&mv);
                let exact = pos.to_fen() == after;
                t.exact += usize::from(exact);
                exact_count += usize::from(exact);
                if pos.in_check() {
                    t.checks += 1;
                    checks += 1;
                    t.push_checks += usize::from(push);
                    t.quiet_checks += usize::from(!cap && mv.special != SpecialMove::Promotion);
                }
            }
            no_moves += usize::from(count == 0);
            if count > 0 {
                unique_games.insert(signature);
            }
            writeln!(
                ledger,
                "| {path} | {id} | {white} / {black} | {result} | {end} | {count}/{reported} | {legal_count}/{exact_count} | {pushes}/{checks}/{promos} | {cliffs} |"
            )?;
            if games % 100 == 0 {
                eprintln!("Audited {games} games");
            }
        }
    }
    let total = |f: fn(&Totals) -> usize| engines.values().map(f).sum::<usize>();
    let mut report = format!(
        "# Complete saved-game audit\n\nRead-only scan of {} database(s). No sampling: all {games} game rows and {} move rows were visited. {no_moves} games have no saved moves; {} distinct nonempty complete move/position sequences, {} distinct positions (ignoring move counters). {discontinuities} discontinuities or missing ply indices.\n\n## Reconstructed observations\n\n- Legal under current rules: {} moves; exact saved-board reproductions: {}. Historical differences are not silently treated as present-day tactics.\n- Actual captures: {}; incorrectly capture-labelled moves: {}.\n- Pushes: {}, including {} to empty destinations and {} displacing the friendly king.\n- Promotions: {}, including {} promotions of a pushed pawn.\n- Checks from legal reconstructed moves: {}, of which {} are pushes and {} are noncapture, nonpromotion moves.\n- Recorded self-evaluation cliffs: {} (not independently verified blunders).\n\nTerminations: {endings:?}.\n\n## By engine\n\nTiming/depth are historical self-reports across different versions and budgets, not a fair speed benchmark. Push counts exclude castling. Check counts exclude historical moves no longer legal.\n\n| Engine | Moves | Pushes | King pushes | Promotions by push | Checks | Quiet checks | False capture labels | Cliffs | Avg depth | Nodes/second |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        args.len() - 1,
        total(|t| t.moves),
        unique_games.len(),
        unique_positions.len(),
        total(|t| t.legal),
        total(|t| t.exact),
        total(|t| t.captures),
        total(|t| t.false_captures),
        total(|t| t.pushes),
        total(|t| t.empty_pushes),
        total(|t| t.king_pushes),
        total(|t| t.promotions),
        total(|t| t.pushed_promotions),
        total(|t| t.checks),
        total(|t| t.push_checks),
        total(|t| t.quiet_checks),
        total(|t| t.cliffs)
    );
    for (name, t) in engines {
        writeln!(
            report,
            "| {name} | {} | {} | {} | {} | {} | {} | {} | {} | {:.1} | {} |",
            t.moves,
            t.pushes,
            t.king_pushes,
            t.pushed_promotions,
            t.checks,
            t.quiet_checks,
            t.false_captures,
            t.cliffs,
            t.depth as f64 / t.moves as f64,
            t.nodes.saturating_mul(1_000_000) / t.time.max(1)
        )?;
    }
    report.push_str(&ledger);
    use std::io::Write as _;
    output.write_all(report.as_bytes())?;
    println!("Audited {games} games into {}", args[0]);
    Ok(())
}
