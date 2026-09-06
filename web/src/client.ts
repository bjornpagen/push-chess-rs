import { PROTOCOL, unsigned } from "./protocol.js";
import type { Command, Envelope, Request, Response, Result } from "./protocol.js";
import type { Analysis, AnalysisOptions, MovePreview, MoveResult, SavedGame, Snapshot } from "./wasm/cataclysm_wasm.js";

export type { Analysis, AnalysisOptions, AnimationPhase, BoardPiece, Color, Displacement, MoveOption, MovePreview, MoveResult, Outcome, PieceType, Promotion, SavedGame, Snapshot, SpecialMove, Square } from "./wasm/cataclysm_wasm.js";

export interface ClientOptions {
  /** Same-origin public directory populated by copy-assets, e.g. /cataclysm/. */
  assetBase: string | URL;
  /** Mobile default: 8 MiB. This is the search table, not total WASM memory. */
  hashMiB?: 4 | 8 | 16 | 32;
  /** Cancels work and releases WASM memory when the page is hidden. Default true. */
  suspendWhenHidden?: boolean;
}

export class CancelledError extends Error {
  constructor(message = "Operation cancelled") { super(message); this.name = "CancelledError"; }
}

interface Pending {
  id: number;
  resolve: (result: Result) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}
type Channel = { kind: "idle" } | { kind: "waiting"; pending: Pending };
interface Connection { worker: Worker; channel: Channel }
type Lifecycle =
  | { kind: "sleeping" }
  | { kind: "loading"; token: object; ready: Promise<void>; abort: AbortController }
  | { kind: "initializing"; token: object; ready: Promise<void>; abort: AbortController; connection: Connection }
  | { kind: "connected"; connection: Connection }
  | { kind: "disposed" };

/** SSR-safe module; create a client only in a browser effect. Dispose on unmount.
 * No rule logic runs in JavaScript. Acknowledged saves allow worker recovery.
 */
export class CataclysmClient {
  private lifecycle: Lifecycle = { kind: "sleeping" };
  private envelope: Envelope | undefined;
  private sequence = 0;
  private base: URL;
  private hashMiB: number;
  private suspendWhenHidden: boolean;

  constructor(options: ClientOptions) {
    if (typeof window === "undefined" || typeof Worker === "undefined") throw new Error("Create CataclysmClient in a browser, not during server rendering");
    this.base = new URL(options.assetBase, window.location.href);
    if (this.base.origin !== window.location.origin) throw new Error("Serve Cataclysm worker assets from the same origin");
    if (!this.base.pathname.endsWith("/")) this.base.pathname += "/";
    this.base.search = "";
    this.base.hash = "";
    this.hashMiB = options.hashMiB ?? 8;
    if (![4, 8, 16, 32].includes(this.hashMiB)) throw new Error("Unsupported hashMiB");
    this.suspendWhenHidden = options.suspendWhenHidden ?? true;
    document.addEventListener("visibilitychange", this.visibilityChanged);
    window.addEventListener("pagehide", this.pageHidden);
  }

  private visibilityChanged = () => { if (this.suspendWhenHidden && document.hidden) this.cancel(); };
  private pageHidden = () => { if (this.suspendWhenHidden) this.cancel(); };

  /** Read-only copies: mutating UI data cannot corrupt recovery state. */
  get snapshot(): Snapshot | undefined { return this.envelope && structuredClone(this.envelope.snapshot); }
  get memoryBytes(): number { return this.connection() ? this.envelope?.memoryBytes ?? 0 : 0; }
  get busy(): boolean {
    const state = this.lifecycle;
    return state.kind === "loading" || state.kind === "initializing"
      || (state.kind === "connected" && state.connection.channel.kind === "waiting");
  }
  exportGame(): SavedGame {
    if (!this.envelope) throw new Error("Call ready() before exporting a game");
    return structuredClone(this.envelope.saved);
  }

  async ready(): Promise<Snapshot> {
    await this.ensureWorker();
    return this.snapshot!;
  }

  private connection(): Connection | undefined {
    const state = this.lifecycle;
    return state.kind === "connected" || state.kind === "initializing" ? state.connection : undefined;
  }

  private ensureWorker(): Promise<void> {
    if (this.suspendWhenHidden && document.hidden) return Promise.reject(new CancelledError("Page is hidden; engine is suspended"));
    switch (this.lifecycle.kind) {
      case "disposed": return Promise.reject(new Error("CataclysmClient has been disposed"));
      case "connected": return Promise.resolve();
      case "loading": case "initializing": return this.lifecycle.ready;
      case "sleeping": {
        const token = {};
        const abort = new AbortController();
        let resolve!: () => void;
        let reject!: (error: unknown) => void;
        const ready = new Promise<void>((yes, no) => { resolve = yes; reject = no; });
        this.lifecycle = { kind: "loading", token, ready, abort };
        void this.startWorker(token, abort, ready).then(resolve, reject);
        return ready;
      }
    }
  }

  private isStarting(token: object): boolean {
    return (this.lifecycle.kind === "loading" || this.lifecycle.kind === "initializing") && this.lifecycle.token === token;
  }

  private async startWorker(token: object, abort: AbortController, ready: Promise<void>): Promise<void> {
    const timeout = setTimeout(() => abort.abort(), 30_000);
    try {
      const response = await fetch(new URL("manifest.json", this.base), { signal: abort.signal, cache: "no-cache" });
      if (!response.ok) throw new Error(`Cannot load engine manifest (${response.status})`);
      const manifest = await response.json() as { protocol?: unknown; worker?: unknown };
      if (manifest.protocol !== PROTOCOL || typeof manifest.worker !== "string" || !/^[a-f0-9]{16}[/]worker[.]js$/.test(manifest.worker)) throw new Error("Invalid or incompatible engine manifest");
      if (!this.isStarting(token)) throw new CancelledError();
      const worker = new Worker(new URL(manifest.worker, this.base), { type: "module", name: "cataclysm" });
      const connection: Connection = { worker, channel: { kind: "idle" } };
      this.lifecycle = { kind: "initializing", token, abort, ready, connection };
      worker.onmessage = ({ data }: MessageEvent<Response>) => {
        if (this.connection() !== connection || connection.channel.kind !== "waiting") return;
        const pending = connection.channel.pending;
        if (pending.id !== data.id) return;
        connection.channel = { kind: "idle" };
        clearTimeout(pending.timer);
        if (data.ok) {
          this.envelope = data.envelope;
          pending.resolve(data.envelope.value);
        } else {
          pending.reject(new Error(data.error));
          if (data.fatal) this.stop(new Error(data.error));
        }
      };
      worker.onerror = (event) => { event.preventDefault(); if (this.connection() === connection) this.stop(new Error(event.message || "Engine worker failed")); };
      worker.onmessageerror = () => { if (this.connection() === connection) this.stop(new Error("Invalid engine response")); };
      const command: Command = { kind: "init", hashMiB: this.hashMiB };
      if (this.envelope) command.checkpoint = { saved: this.envelope.saved, revision: this.envelope.snapshot.revision };
      await this.send(connection, command, 30_000);
      if (!this.isStarting(token)) throw new CancelledError();
      this.lifecycle = { kind: "connected", connection };
    } catch (error) {
      if (this.isStarting(token)) this.stop(error instanceof Error ? error : new Error(String(error)));
      throw error;
    } finally { clearTimeout(timeout); }
  }

  private send(connection: Connection, command: Command, timeoutMs = 10_000): Promise<Result> {
    if (connection.channel.kind === "waiting") return Promise.reject(new Error("Engine is busy; await the current operation or cancel it"));
    const id = this.sequence = (this.sequence + 1) >>> 0;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => this.stop(new Error("Engine operation timed out; saved game retained")), timeoutMs);
      connection.channel = { kind: "waiting", pending: { id, resolve, reject, timer } };
      const request: Request = { protocol: PROTOCOL, id, command };
      try { connection.worker.postMessage(request); }
      catch (error) { this.stop(error instanceof Error ? error : new Error(String(error))); }
    });
  }

  private async request(command: (revision: number) => Command): Promise<Result> {
    await this.ensureWorker();
    const state = this.lifecycle;
    if (state.kind !== "connected") throw new CancelledError();
    return this.send(state.connection, command(this.envelope!.snapshot.revision));
  }

  async preview(moveId: number, expectedRevision?: number): Promise<MovePreview> {
    unsigned(moveId, "moveId");
    if (expectedRevision !== undefined) unsigned(expectedRevision, "revision");
    return await this.request(revision => ({ kind: "preview", moveId, revision: expectedRevision ?? revision })) as MovePreview;
  }
  async play(moveId: number, expectedRevision?: number): Promise<MoveResult> {
    unsigned(moveId, "moveId");
    if (expectedRevision !== undefined) unsigned(expectedRevision, "revision");
    return await this.request(revision => ({ kind: "play", moveId, revision: expectedRevision ?? revision })) as MoveResult;
  }
  async analyse(options: AnalysisOptions = {}): Promise<Analysis> {
    return await this.request(revision => ({ kind: "analyse", options, revision })) as Analysis;
  }
  async undo(plies = 1): Promise<Snapshot> {
    unsigned(plies, "plies");
    if (this.busy) this.cancel();
    return await this.request(revision => ({ kind: "undo", plies, revision })) as Snapshot;
  }
  async reset(fen?: string): Promise<Snapshot> {
    if (this.busy) this.cancel();
    return await this.request(() => fen === undefined ? { kind: "reset" } : { kind: "reset", fen }) as Snapshot;
  }
  async restore(saved: SavedGame | string): Promise<Snapshot> {
    if (this.busy) this.cancel();
    const json = typeof saved === "string" ? saved : JSON.stringify(saved);
    if (json.length > 65_536) throw new Error("Saved game exceeds 64 KiB");
    return await this.request(() => ({ kind: "restore", saved: json })) as Snapshot;
  }

  /** Hard cancellation is immediate even inside synchronous WASM search.
   * Next use lazily recreates the worker from the last acknowledged move log.
   */
  cancel(): void { this.stop(new CancelledError()); }

  private stop(error: Error): void {
    const state = this.lifecycle;
    const connection = this.connection();
    if (state.kind === "loading" || state.kind === "initializing") state.abort.abort();
    this.lifecycle = state.kind === "disposed" ? state : { kind: "sleeping" };
    if (connection) {
      connection.worker.terminate();
      if (connection.channel.kind === "waiting") {
        clearTimeout(connection.channel.pending.timer);
        connection.channel.pending.reject(error);
        connection.channel = { kind: "idle" };
      }
    }
  }

  dispose(): void {
    this.stop(new CancelledError("Client disposed"));
    this.lifecycle = { kind: "disposed" };
    document.removeEventListener("visibilitychange", this.visibilityChanged);
    window.removeEventListener("pagehide", this.pageHidden);
  }
}
