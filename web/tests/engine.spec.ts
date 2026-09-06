import { expect, test } from "@playwright/test";

test.beforeEach(async ({ page }) => { await page.goto("/"); });

test("actual browser WASM matches native search, rules, and animation", async ({ page }) => {
  const errors = await page.evaluate(async () => {
    const { CataclysmClient } = await import("/dist/client.js");
    const fixtures = await (await fetch("/tests/native-fixtures.json")).json();
    const normalize = (value: unknown) => JSON.stringify(value, (_key, val) => {
      if (val === undefined) return null;
      if (val && typeof val === "object" && !Array.isArray(val)) return Object.fromEntries(Object.entries(val).sort(([a], [b]) => a.localeCompare(b)));
      return val;
    });
    const errors: string[] = [];
    for (const fixture of fixtures.fixtures) {
      const client = new CataclysmClient({ assetBase: "/dist/assets/", suspendWhenHidden: false });
      try {
        await client.ready();
        const initial = await client.reset(fixture.fen);
        if (normalize(initial) !== normalize(fixture.initial)) errors.push(`initial: ${fixture.fen}`);
        for (const turn of fixture.turns) {
          const actual = await client.analyse(fixtures.options);
          actual.timeMs = turn.analysis.timeMs;
          if (normalize(actual) !== normalize(turn.analysis)) errors.push(`analysis: ${fixture.fen}: ${normalize(actual)} != ${normalize(turn.analysis)}`);
          const result = await client.play(actual.mv.id);
          if (normalize(result) !== normalize(turn.result)) errors.push(`move: ${fixture.fen}`);
        }
      } finally { client.dispose(); }
    }
    return errors;
  });
  expect(errors).toEqual([]);
});

test("cancellation, undo, restart and recovery retain the acknowledged game", async ({ page }) => {
  const result = await page.evaluate(async () => {
    const { CataclysmClient } = await import("/dist/client.js");
    const client = new CataclysmClient({ assetBase: "/dist/assets/", suspendWhenHidden: false });
    try {
      const start = await client.ready();
      const move = start.legalMoves.find((m: { from: number; to: number }) => m.from === 12 && m.to === 28)!;
      await client.play(move.id);
      const saved = client.exportGame();
      const fen = client.snapshot!.fen;
      const search = client.analyse({ timeMs: 5000, maxNodes: 2_000_000, maxDepth: 32 }).then(() => "finished", (e: Error) => e.name);
      await new Promise(resolve => setTimeout(resolve, 20));
      client.cancel();
      const cancellation = await search;
      const recovered = await client.ready();
      const same = recovered.fen === fen && JSON.stringify(client.exportGame()) === JSON.stringify(saved);
      const undone = await client.undo();
      client.cancel();
      const resumedUndo = await client.ready();
      if (resumedUndo.revision !== undone.revision) throw new Error("Recovery lost the undo revision");
      await client.restore(saved);
      const restored = client.snapshot!.fen === fen;
      await client.reset();
      return { cancellation, same, restored, undone: undone.fen === start.fen, reset: client.snapshot!.fen === start.fen };
    } finally { client.dispose(); }
  });
  expect(result).toEqual({ cancellation: "CancelledError", same: true, restored: true, undone: true, reset: true });
});

test("bad input is rejected without damaging the game or leaking memory", async ({ page }) => {
  const result = await page.evaluate(async () => {
    const { CataclysmClient } = await import("/dist/client.js");
    const client = new CataclysmClient({ assetBase: "/dist/assets/", suspendWhenHidden: false });
    let rejected = 0;
    try {
      const start = await client.ready();
      const memoryBefore = client.memoryBytes;
      for (let i = 0; i < 200; i++) {
        try { await client.analyse({ maxNodes: -1 } as never); } catch { rejected++; }
      }
      for (const action of [() => client.play(-1), () => client.play(1.5), () => client.play(0xffff_ffff), () => client.reset("8/8/8/8/8/8/8/8 w - - 0 1"), () => client.restore("{}")]) {
        try { await action(); } catch { rejected++; }
      }
      await client.preview(start.legalMoves[0]!.id);
      return { rejected, unchanged: client.snapshot!.fen === start.fen, growth: client.memoryBytes - memoryBefore, memory: client.memoryBytes };
    } finally { client.dispose(); }
  });
  expect(result.rejected).toBe(205);
  expect(result.unchanged).toBe(true);
  expect(result.growth).toBeLessThan(2 * 1024 * 1024);
  expect(result.memory).toBeLessThan(32 * 1024 * 1024);
});

test("engine search leaves the main thread responsive and stays within a mobile memory budget", async ({ page }) => {
  const result = await page.evaluate(async () => {
    const { CataclysmClient } = await import("/dist/client.js");
    const client = new CataclysmClient({ assetBase: "/dist/assets/", suspendWhenHidden: false });
    try {
      await client.ready();
      let ticks = 0;
      const timer = setInterval(() => ticks++, 10);
      const started = performance.now();
      const analysis = await client.analyse({ timeMs: 250, maxNodes: 0, maxDepth: 32 });
      clearInterval(timer);
      return { ticks, elapsed: performance.now() - started, memory: client.memoryBytes, nodes: analysis.nodes };
    } finally { client.dispose(); }
  });
  expect(result.ticks).toBeGreaterThan(2);
  expect(result.elapsed).toBeLessThan(2000);
  expect(result.memory).toBeLessThan(48 * 1024 * 1024);
  expect(result.nodes).toBeGreaterThan(0);
  console.log(JSON.stringify(result));
});

test("dispose and import work across React-style mount/unmount and initialization races", async ({ page }) => {
  const result = await page.evaluate(async () => {
    const { CataclysmClient } = await import("/dist/client.js");
    let stopped = 0;
    for (let i = 0; i < 5; i++) {
      const client = new CataclysmClient({ assetBase: "/dist/assets/", suspendWhenHidden: false });
      const ready = client.ready().then(() => false, () => true);
      client.dispose();
      if (await ready) stopped++;
      try { await client.ready(); } catch { stopped++; }
    }
    const client = new CataclysmClient({ assetBase: "/dist/assets/", suspendWhenHidden: false });
    const [a, b] = await Promise.all([client.ready(), client.ready()]);
    client.dispose();
    return { stopped, same: a.fen === b.fen };
  });
  expect(result).toEqual({ stopped: 10, same: true });
});

test("hiding a mobile page immediately suspends search and resumes from its save", async ({ page }) => {
  const result = await page.evaluate(async () => {
    const { CataclysmClient } = await import("/dist/client.js");
    let hidden = false;
    Object.defineProperty(document, "hidden", { configurable: true, get: () => hidden });
    const client = new CataclysmClient({ assetBase: "/dist/assets/" });
    try {
      const before = await client.ready();
      const searching = client.analyse({ timeMs: 5000, maxNodes: 2_000_000 }).then(() => "finished", (e: Error) => e.name);
      await new Promise(resolve => setTimeout(resolve, 20));
      hidden = true;
      document.dispatchEvent(new Event("visibilitychange"));
      const stopped = await searching;
      const memory = client.memoryBytes;
      const background = await client.ready().then(() => "running", (e: Error) => e.name);
      hidden = false;
      document.dispatchEvent(new Event("visibilitychange"));
      const after = await client.ready();
      return { stopped, memory, background, same: before.fen === after.fen };
    } finally { client.dispose(); delete (document as unknown as { hidden?: boolean }).hidden; }
  });
  expect(result).toEqual({ stopped: "CancelledError", memory: 0, background: "CancelledError", same: true });
});

test("stale revisions and failed initialization cannot corrupt a later game", async ({ page }) => {
  await page.route("**/dist/assets/manifest.json", route => route.fulfill({ status: 503, body: "unavailable" }), { times: 1 });
  const result = await page.evaluate(async () => {
    const { CataclysmClient } = await import("/dist/client.js");
    const client = new CataclysmClient({ assetBase: "/dist/assets/", suspendWhenHidden: false });
    try {
      const failed = await client.ready().then(() => false, () => true);
      const before = await client.ready();
      const current = await client.reset();
      const rejected = await client.play(before.legalMoves[0]!.id, before.revision).then(() => false, () => true);
      return { failed, rejected, unchanged: current.fen === client.snapshot!.fen, advanced: current.revision > before.revision };
    } finally { client.dispose(); }
  });
  expect(result).toEqual({ failed: true, rejected: true, unchanged: true, advanced: true });
});

test("raw WASM input conversion is fallible and cannot coerce an invalid move", async ({ page }) => {
  const result = await page.evaluate(async () => {
    const { default: init, Session } = await import("/dist/wasm/cataclysm_wasm.js");
    const wasm = await init();
    const session = new Session(4);
    let rejected = 0;
    try {
      const before = session.snapshot();
      const legal = before.legalMoves[0]!.id;
      for (const invalid of [-1, NaN, Infinity, legal + 0.5, legal + 2 ** 32]) {
        try { session.play(invalid, before.revision); } catch { rejected++; }
      }
      const memory = wasm.memory.buffer.byteLength;
      for (let i = 0; i < 1000; i++) {
        try { session.analyse({ maxNodes: "wrong" }, before.revision); } catch { rejected++; }
      }
      return { rejected, same: before.fen === session.snapshot().fen, growth: wasm.memory.buffer.byteLength - memory };
    } finally { session.free(); }
  });
  expect(result.rejected).toBe(1005);
  expect(result.same).toBe(true);
  expect(result.growth).toBeLessThan(1024 * 1024);
});
