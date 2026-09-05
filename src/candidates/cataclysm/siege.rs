//! A separate AND/OR proof search for forcing checks. The attacking side may
//! choose any checking move; EVERY legal defender reply must be refuted.
//! A failed or interrupted proof says nothing about the position's value.
use super::{Board, Cataclysm, MAX};
use crate::core::types::*;

struct Limits {
    attacker: Color,
    deadline: u128,
    nodes: u64,
}

impl Cataclysm {
    pub(super) fn siege(&mut self, b: &mut Board, max_depth: i32) -> Option<Vec<Move>> {
        let limits = Limits {
            attacker: b.pos.side_to_move,
            deadline: if self.micros == 0 {
                u128::MAX
            } else {
                self.micros / 12
            },
            nodes: if self.limit == 0 {
                20_000
            } else {
                (self.limit / 12).max(1)
            },
        };
        for depth in (1..=max_depth.min(11)).step_by(2) {
            if let Some(line) = self.prove(b, depth, 0, &limits) {
                return Some(line);
            }
            if self.nodes >= limits.nodes || self.start.elapsed().as_micros() >= limits.deadline {
                break;
            }
        }
        None
    }

    fn prove(
        &mut self,
        b: &mut Board,
        depth: i32,
        ply: usize,
        limits: &Limits,
    ) -> Option<Vec<Move>> {
        if ply >= MAX - 2
            || self.nodes >= limits.nodes
            || self.start.elapsed().as_micros() >= limits.deadline
            || self.tick(ply)
        {
            return None;
        }
        let us = b.pos.side_to_move;
        let attacking = us == limits.attacker;
        if attacking && (depth <= 0 || self.draw(b, ply)) {
            return None;
        }
        let key = b.pos.zobrist;
        let mut actions = self.prepare(b, ply, self.probe(key).map_or(0, |e| e.mv));
        let mut replies = 0;
        let mut longest = Vec::new();
        let mut proof = None;
        for i in 0..actions.len() {
            Self::pick(&mut actions, i);
            let a = &actions[i];
            let undo = b.make(a);
            if b.checked(us) || (attacking && !b.checked(opponent(us))) {
                b.unmake(undo);
                continue;
            }
            replies += 1;
            self.path.push(key);
            let child = if depth > 0 {
                self.prove(b, depth - 1, ply + 1, limits)
            } else {
                None
            };
            self.path.pop();
            b.unmake(undo);
            if attacking {
                if let Some(mut line) = child {
                    line.insert(0, a.mv);
                    proof = Some(line);
                    break;
                }
            } else {
                let Some(mut line) = child else {
                    self.buffers[ply] = actions;
                    return None;
                };
                line.insert(0, a.mv);
                if line.len() > longest.len() {
                    longest = line;
                }
            }
        }
        self.buffers[ply] = actions;
        if !attacking {
            if replies == 0 {
                return b.checked(us).then(Vec::new);
            }
            if self.draw(b, ply) {
                return None;
            }
            return Some(longest);
        }
        proof
    }
}
