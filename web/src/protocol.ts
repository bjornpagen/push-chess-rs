import type { Analysis, AnalysisOptions, Checkpoint, MovePreview, MoveResult, SavedGame, Snapshot } from "./wasm/cataclysm_wasm.js";

export const PROTOCOL = 1;
export type Command =
  | { kind: "init"; hashMiB: number; checkpoint?: Checkpoint }
  | { kind: "reset"; fen?: string }
  | { kind: "restore"; saved: string }
  | { kind: "preview" | "play"; moveId: number; revision: number }
  | { kind: "undo"; plies: number; revision: number }
  | { kind: "analyse"; options: AnalysisOptions; revision: number };

export type Result = Snapshot | MovePreview | MoveResult | Analysis;
export interface Envelope {
  value: Result;
  snapshot: Snapshot;
  saved: SavedGame;
  memoryBytes: number;
}
export interface Request { protocol: number; id: number; command: Command }
export type Response = { id: number; ok: true; envelope: Envelope } | { id: number; ok: false; error: string; fatal: boolean };

export function unsigned(value: unknown, label: string): asserts value is number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) {
    throw new Error(`${label} must be an unsigned 32-bit integer`);
  }
}
