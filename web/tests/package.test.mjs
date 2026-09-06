import { test } from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtemp, readFile, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const web = fileURLToPath(new URL("../", import.meta.url));

test("package imports without DOM globals and copies a complete static worker deployment", async () => {
  const { CataclysmClient } = await import("@push-chess/cataclysm");
  assert.equal(typeof CataclysmClient, "function");
  const destination = await mkdtemp(path.join(tmpdir(), "cataclysm-package-test-"));
  try {
    for (let i = 0; i < 2; i++) {
      const result = spawnSync(process.execPath, [path.join(web, "scripts/copy-assets.mjs"), destination], { encoding: "utf8" });
      assert.equal(result.status, 0, result.stderr);
    }
    const manifest = JSON.parse(await readFile(path.join(destination, "manifest.json"), "utf8"));
    assert.equal(manifest.protocol, 1);
    const version = path.dirname(manifest.worker);
    for (const file of ["worker.js", "protocol.js", "wasm/cataclysm_wasm.js", "wasm/cataclysm_wasm_bg.wasm"]) {
      assert.ok((await stat(path.join(destination, version, file))).size > 0);
    }
    const bytes = await readFile(path.join(destination, version, "wasm/cataclysm_wasm_bg.wasm"));
    const module = await WebAssembly.compile(bytes);
    assert.ok(!WebAssembly.Module.imports(module).some(item => item.module.startsWith("wasi")));
  } finally { await rm(destination, { recursive: true, force: true }); }
});
