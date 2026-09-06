import init, { Session } from "./wasm/cataclysm_wasm.js";
import type { Checkpoint } from "./wasm/cataclysm_wasm.js";
import { PROTOCOL, unsigned } from "./protocol.js";
import type { Request, Response, Result } from "./protocol.js";

// Imported only by the dedicated worker, never by React/SSR.
const scope = globalThis as unknown as {
  onmessage: ((event: MessageEvent<Request>) => void) | null;
  postMessage: (message: Response) => void;
};
interface Runtime { session: Session; memory: WebAssembly.Memory }
type WorkerState = { kind: "cold" } | { kind: "loading" } | { kind: "ready"; runtime: Runtime };
let state: WorkerState = { kind: "cold" };

async function initialize(hashMiB: number, checkpoint?: Checkpoint): Promise<Runtime> {
  if (state.kind !== "cold") throw new Error("Worker already initialized or initializing");
  if (![4, 8, 16, 32].includes(hashMiB)) throw new Error("Unsupported hashMiB");
  state = { kind: "loading" };
  try {
    const wasm = await init();
    const session = new Session(hashMiB);
    try { if (checkpoint !== undefined) session.recover(checkpoint); }
    catch (error) { session.free(); throw error; }
    const runtime = { session, memory: wasm.memory };
    state = { kind: "ready", runtime };
    return runtime;
  } catch (error) { state = { kind: "cold" }; throw error; }
}

scope.onmessage = async ({ data }) => {
  const id = data?.id;
  try {
    unsigned(id, "request id");
    if (data.protocol !== PROTOCOL) throw new Error("Worker/client version mismatch; refresh the page");
    const command = data.command;
    if (!command || typeof command.kind !== "string") throw new Error("Missing command");
    let value: Result;
    let runtime: Runtime;
    if (command.kind === "init") {
      runtime = await initialize(command.hashMiB, command.checkpoint);
      value = runtime.session.snapshot();
    } else {
      if (state.kind !== "ready") throw new Error("Worker is not initialized");
      runtime = state.runtime;
      const { session } = runtime;
      if ("revision" in command) unsigned(command.revision, "revision");
      switch (command.kind) {
        case "reset": value = session.reset(command.fen); break;
        case "restore": value = session.restore(command.saved); break;
        case "preview":
        case "play":
          unsigned(command.moveId, "moveId");
          value = session[command.kind](command.moveId, command.revision);
          break;
        case "undo":
          unsigned(command.plies, "plies");
          value = session.undo(command.plies, command.revision);
          break;
        case "analyse": value = session.analyse(command.options, command.revision); break;
        default: throw new Error("Unknown command");
      }
    }
    scope.postMessage({ id, ok: true, envelope: { value, snapshot: runtime.session.snapshot(), saved: runtime.session.save(), memoryBytes: runtime.memory.buffer.byteLength } });
  } catch (error) {
    scope.postMessage({ id, ok: false, error: error instanceof Error ? error.message : String(error), fatal: error instanceof WebAssembly.RuntimeError });
  }
};
