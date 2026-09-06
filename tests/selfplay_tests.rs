use push_chess::selfplay::{BatchSearch, State, considered_visits};

fn finish(search: &mut BatchSearch) {
    while let Some(batch) = search.request().unwrap() {
        search
            .submit(
                &vec![0.0; batch.rows * batch.width],
                &vec![0.0; batch.rows],
                batch.width,
            )
            .unwrap();
    }
}

#[test]
fn budgets_and_state_machine_are_exact() {
    let state = State::default();
    for sims in [1, 2, 3, 7, 16, 31, 64] {
        for candidates in [1, 2, 3, 8, 16, 100] {
            let mut search = BatchSearch::new(
                vec![state.clone()],
                vec![vec![0.0; state.legal_moves().len()]],
                sims,
                candidates,
            )
            .unwrap();
            assert!(search.results().is_err());
            let batch = search.request().unwrap().unwrap();
            assert!(search.request().is_err());
            assert!(search.submit(&[], &[], 0).is_err());
            assert!(
                search
                    .submit(&vec![f32::NAN; batch.width], &[0.0], batch.width)
                    .is_err()
            );
            search
                .submit(&vec![0.0; batch.width], &[0.0], batch.width)
                .unwrap();
            assert!(search.submit(&[], &[], 0).is_err());
            finish(&mut search);
            let result = search.results().unwrap().remove(0);
            assert_eq!(result.visits.iter().sum::<u32>(), sims as u32);
            assert!(result.nodes <= sims + 1);
            assert!((result.policy.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            assert!(result.policy.iter().all(|p| p.is_finite() && *p >= 0.0));
            assert!(search.request().unwrap().is_none());
        }
    }
    assert_eq!(considered_visits(4, 8), [0, 0, 0, 0, 1, 1, 2, 2]);
}

#[test]
fn terminal_backup_finds_mate_and_keeps_source_intact() {
    let state = State::from_fen("7k/8/5KQ1/8/8/8/8/8 w - - 0 1").unwrap();
    let before = state.position().to_fen();
    let n = state.legal_moves().len();
    let mut search = BatchSearch::new(vec![state.clone()], vec![vec![0.0; n]], n, n).unwrap();
    finish(&mut search);
    let result = search.results().unwrap().remove(0);
    let mut after = state.clone();
    after.play(result.mv).unwrap();
    assert_eq!(
        after.white_value(),
        Some(1.0),
        "search must prefer a proven win from the correct perspective"
    );
    assert_eq!(state.position().to_fen(), before);
    assert!(state.position().undo_stack.is_empty());
}

#[test]
fn lossless_actions_and_board_history_match_rules() {
    for fen in [
        "7k/8/8/4R3/4N3/8/8/K7 w - - 0 1",
        "7k/P7/R7/8/8/8/8/K7 w - - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "7k/8/8/3pP3/8/8/8/K7 w - d6 0 1",
    ] {
        let state = State::from_fen(fen).unwrap();
        let row = state.encode();
        let unique: std::collections::HashSet<_> = row.actions.iter().collect();
        assert_eq!(unique.len(), row.ids.len());
        for mv in state.legal_moves() {
            let mut after = state.clone();
            after.play(mv.id()).unwrap();
            assert_eq!(
                after.position().previous_board().unwrap(),
                state.position().board
            );
            let previous = after.encode().board[12 * 64..24 * 64].to_vec();
            assert_eq!(
                previous.iter().sum::<f32>(),
                state
                    .position()
                    .board
                    .iter()
                    .filter(|p| !p.is_empty())
                    .count() as f32
            );
        }
    }
}

#[test]
fn repetition_history_is_not_approximated_by_fen() {
    let mut state = State::from_fen("7k/8/8/8/8/8/8/K7 w - - 0 1").unwrap();
    for _ in 0..2 {
        for (from, to) in [(0, 1), (63, 62), (1, 0), (62, 63)] {
            let id = state
                .legal_moves()
                .iter()
                .find(|m| m.from == from && m.to == to)
                .unwrap()
                .id();
            state.play(id).unwrap();
        }
    }
    assert_eq!(state.white_value(), Some(0.0));
    assert_eq!(state.clone().white_value(), Some(0.0));
    assert_eq!(
        State::from_fen(&state.position().to_fen())
            .unwrap()
            .white_value(),
        None
    );
    assert!(state.legal_moves().is_empty());
    assert!(BatchSearch::new(vec![state], vec![vec![]], 8, 8).is_err());
}

#[test]
fn request_ids_reject_cross_batch_and_duplicate_replies() {
    let state = State::default();
    let make = || {
        BatchSearch::new(
            vec![state.clone()],
            vec![vec![0.0; state.legal_moves().len()]],
            2,
            2,
        )
        .unwrap()
    };
    let (mut a, mut b) = (make(), make());
    let fa = a.request().unwrap().unwrap();
    let fb = b.request().unwrap().unwrap();
    let id = a.request_id().unwrap();
    assert_ne!(Some(id), b.request_id());
    assert!(
        b.submit_for(id, &vec![0.; fb.width], &[0.], fb.width)
            .is_err()
    );
    a.submit_for(id, &vec![0.; fa.width], &[0.], fa.width)
        .unwrap();
    assert!(
        a.submit_for(id, &vec![0.; fa.width], &[0.], fa.width)
            .is_err()
    );
    b.submit_for(
        b.request_id().unwrap(),
        &vec![0.; fb.width],
        &[0.],
        fb.width,
    )
    .unwrap();
    finish(&mut a);
    finish(&mut b);
    assert_eq!(
        a.results().unwrap()[0].visits,
        b.results().unwrap()[0].visits
    );
}

#[test]
fn capacity_is_an_error_and_stop_still_evaluates_the_root() {
    use push_chess::selfplay::{SearchOptions, SearchRoot};
    let state = State::default();
    let make = || {
        BatchSearch::with_options(
            vec![SearchRoot::from_state(&state)],
            vec![vec![0.; state.legal_moves().len()]],
            4,
            4,
            SearchOptions {
                effects: true,
                max_nodes_per_tree: 1,
            },
        )
        .unwrap()
    };
    let mut search = make();
    let batch = search.request().unwrap().unwrap();
    assert!(batch.effect_width > 0);
    search
        .submit(&vec![0.; batch.width], &[0.], batch.width)
        .unwrap();
    assert!(search.request().is_err());
    assert!(search.results().is_err());
    let mut stopped = make();
    stopped.stop();
    let batch = stopped.request().unwrap().unwrap();
    stopped
        .submit(&vec![0.; batch.width], &[0.25], batch.width)
        .unwrap();
    assert!(stopped.request().unwrap().is_none());
    let result = stopped.results().unwrap().remove(0);
    assert_eq!(result.visits.iter().sum::<u32>(), 0);
    assert_eq!(result.root_value, 0.25);
    assert!((result.policy.iter().sum::<f32>() - 1.).abs() < 1e-5);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn persistent_pool_matches_serial_and_restarts_cleanly() {
    use push_chess::selfplay::{SearchOptions, SearchRoot, SearchRuntime};
    let states = [
        State::default(),
        State::from_fen("7k/8/5KQ1/8/8/8/8/8 w - - 0 1").unwrap(),
    ];
    let roots = || states.iter().map(SearchRoot::from_state).collect();
    let noise = || {
        states
            .iter()
            .map(|s| vec![0.; s.legal_moves().len()])
            .collect()
    };
    let mut serial =
        BatchSearch::with_options(roots(), noise(), 16, 8, SearchOptions::default()).unwrap();
    finish(&mut serial);
    let reference = serial.results().unwrap();
    let mut runtime = SearchRuntime::new(2, 2, 1).unwrap();
    for _ in 0..3 {
        for (lane, state) in states.iter().enumerate() {
            runtime
                .start(
                    lane,
                    vec![SearchRoot::from_state(state)],
                    vec![vec![0.; state.legal_moves().len()]],
                    16,
                    8,
                    SearchOptions::default(),
                )
                .unwrap();
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let mut done = 0;
        while !runtime.idle() {
            assert!(std::time::Instant::now() < deadline, "runtime stalled");
            let poll = runtime.poll(1000).unwrap();
            if let Some((id, f)) = poll.request {
                assert!(
                    runtime
                        .submit(
                            u64::MAX,
                            &vec![0.; f.rows * f.width],
                            &vec![0.; f.rows],
                            f.width
                        )
                        .is_err()
                );
                assert!(
                    runtime
                        .submit(
                            id,
                            &vec![0.; f.rows * f.width],
                            &vec![f32::NAN; f.rows],
                            f.width
                        )
                        .is_err()
                );
                runtime
                    .submit(id, &vec![0.; f.rows * f.width], &vec![0.; f.rows], f.width)
                    .unwrap();
                assert!(
                    runtime
                        .submit(id, &vec![0.; f.rows * f.width], &vec![0.; f.rows], f.width)
                        .is_err()
                );
            }
            for c in poll.completed {
                let (a, b) = (&c.results[0], &reference[c.lane]);
                assert_eq!(a.mv, b.mv);
                assert_eq!(a.visits, b.visits);
                assert_eq!(a.policy, b.policy);
                done += 1;
            }
        }
        assert_eq!(done, states.len());
    }
    // Closing with unanswered inference must not wait for a Python reply.
    runtime
        .start(
            0,
            vec![SearchRoot::from_state(&states[0])],
            vec![vec![0.; states[0].legal_moves().len()]],
            16,
            8,
            SearchOptions::default(),
        )
        .unwrap();
    while runtime.poll(1000).unwrap().request.is_none() {}
    runtime.close();
    runtime.close();
}
