//! Explicit, bounded work. bumbledb is the sole durable game/analysis store.
use push_chess_lab::lab::{Corpus, RunConfig, control, generate_controlled};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
type Result<T> = push_chess_lab::lab::Result<T>;

const HELP: &str = "lab list
lab init --db NEW-DIRECTORY
lab tournament --db DIRECTORY --engines all|cataclysm,kinetic,astra
  --pairs N --workers N (--nodes N | --time-ms N)
  [--purpose corpus|arena] [--seed 1] [--opening-plies 6] [--max-plies 512]
  [--max-seconds 86400] [--max-gib 50]
lab summary --db DIRECTORY
lab status --db DIRECTORY [--run N]  (live tournament, same database owner)
lab report --db DIRECTORY --run N
lab verify --db DIRECTORY --run N  (offline relational + rules audit)

pairs is the number of color-swapped pairs PER matchup; all matchups interleave.
SIGINT/SIGTERM stop at move boundaries and commit verified partial progress.
The Python reader transfers typed game pages and NumPy arrays, not JSON.";

fn take(args: &mut BTreeMap<String, String>, key: &str) -> Result<String> {
    args.remove(key)
        .ok_or_else(|| format!("missing --{key}").into())
}
fn number<T: std::str::FromStr>(
    args: &mut BTreeMap<String, String>,
    key: &str,
    default: Option<&str>,
) -> Result<T> {
    let v = args
        .remove(key)
        .or_else(|| default.map(str::to_owned))
        .ok_or_else(|| format!("missing --{key}"))?;
    v.parse().map_err(|_| format!("invalid --{key}").into())
}
fn no_extra(args: &BTreeMap<String, String>) -> Result<()> {
    if args.is_empty() {
        Ok(())
    } else {
        Err(format!("unknown options: {:?}", args.keys()).into())
    }
}
fn main() -> Result<()> {
    let mut input = std::env::args().skip(1);
    let command = input.next().unwrap_or_else(|| "help".into());
    let mut args = BTreeMap::new();
    while let Some(key) = input.next() {
        let key = key
            .strip_prefix("--")
            .ok_or("expected --option value")?
            .to_owned();
        let value = input.next().ok_or("missing option value")?;
        if args.insert(key, value).is_some() {
            return Err("duplicate option".into());
        }
    }
    if command == "help" {
        no_extra(&args)?;
        println!("{HELP}");
        return Ok(());
    }
    if command == "list" {
        no_extra(&args)?;
        let engines: Vec<_> = push_chess::engines::ENGINE_REGISTRY
            .iter()
            .map(|e| push_chess::engines::info(e.name).unwrap())
            .collect();
        println!("{}", serde_json::to_string_pretty(&engines)?);
        return Ok(());
    }
    let path = take(&mut args, "db")?;
    let output = match command.as_str() {
        "init" => {
            no_extra(&args)?;
            let corpus = Corpus::create(Path::new(&path))?;
            let summary = corpus.summary()?;
            corpus.close()?;
            summary
        }
        "tournament" => {
            let engines = take(&mut args, "engines")?;
            let config = RunConfig {
                engines: if engines == "all" {
                    push_chess::engines::ENGINE_REGISTRY
                        .iter()
                        .map(|e| e.name.to_owned())
                        .collect()
                } else {
                    engines.split(',').map(str::to_owned).collect()
                },
                pairs: number(&mut args, "pairs", None)?,
                workers: number(&mut args, "workers", None)?,
                nodes: number(&mut args, "nodes", Some("0"))?,
                time_ms: number(&mut args, "time-ms", Some("0"))?,
                max_plies: number(&mut args, "max-plies", Some("512"))?,
                opening_plies: number(&mut args, "opening-plies", Some("6"))?,
                seed: number(&mut args, "seed", Some("1"))?,
                purpose: args.remove("purpose").unwrap_or_else(|| "corpus".into()),
                max_seconds: number(&mut args, "max-seconds", Some("86400"))?,
                max_bytes: number::<u64>(&mut args, "max-gib", Some("50"))?
                    .checked_mul(1 << 30)
                    .ok_or("disk cap overflow")?,
            };
            no_extra(&args)?;
            config.validate()?;
            let stop = Arc::new(AtomicBool::new(false));
            let signal = stop.clone();
            ctrlc::set_handler(move || signal.store(true, Ordering::Relaxed))?;
            let mut corpus = Corpus::open(Path::new(&path))?;
            let control = control::Control::bind(Path::new(&path))?;
            let result = generate_controlled(&mut corpus, &config, &stop, Some(&control))
                .and_then(|run| corpus.report(run));
            corpus.close()?;
            result?
        }
        "status" => {
            let run = if args.contains_key("run") {
                Some(number(&mut args, "run", None)?)
            } else {
                None
            };
            no_extra(&args)?;
            control::status(Path::new(&path), run)?
        }
        "summary" | "report" | "verify" => {
            let run: Option<u64> = if command != "summary" {
                Some(number(&mut args, "run", None)?)
            } else {
                None
            };
            no_extra(&args)?;
            let corpus = Corpus::open(Path::new(&path))?;
            let result = if let Some(run) = run {
                if command == "verify" {
                    corpus.verify(run)
                } else {
                    corpus.report(run)
                }
            } else {
                corpus.summary()
            };
            corpus.close()?;
            result?
        }
        _ => return Err(format!("unknown command\n{HELP}").into()),
    };
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
