import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { cp, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const web = fileURLToPath(new URL("../", import.meta.url));
const root = path.resolve(web, "..");
function run(command, args, cwd = root) {
  const result = spawnSync(command, args, { cwd, stdio: "inherit" });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${command} failed (${result.status})`);
}
const version = spawnSync("wasm-bindgen", ["--version"], { encoding: "utf8" });
if (version.stdout?.trim() !== "wasm-bindgen 0.2.128") throw new Error("Install the matching tool: cargo install wasm-bindgen-cli --version 0.2.128 --locked");
run("cargo", ["build", "--locked", "--release", "--target", "wasm32-unknown-unknown", "-p", "cataclysm-wasm"]);
const metadata = JSON.parse(spawnSync("cargo", ["metadata", "--format-version", "1", "--no-deps"], { cwd: root, encoding: "utf8" }).stdout);
const wasm = path.join(metadata.target_directory, "wasm32-unknown-unknown/release/cataclysm_wasm.wasm");
run("wasm-bindgen", [wasm, "--target", "web", "--out-dir", path.join(web, "src/wasm"), "--out-name", "cataclysm_wasm"]);
// Only generated package output. Deployment copy-assets retains old versions.
await rm(path.join(web, "dist"), { recursive: true, force: true });
run("pnpm", ["exec", "tsc"], web);
await cp(path.join(web, "src/wasm"), path.join(web, "dist/wasm"), { recursive: true });
// Hash the complete worker runtime, not just the WASM, to avoid mixed releases.
const hash = createHash("sha256");
for (const file of ["worker.js", "protocol.js", ... (await readdir(path.join(web, "dist/wasm"))).sort().map(f => `wasm/${f}`)]) {
  hash.update(file); hash.update(await readFile(path.join(web, "dist", file)));
}
const versionId = hash.digest("hex").slice(0, 16);
const assets = path.join(web, "dist/assets", versionId);
await mkdir(assets, { recursive: true });
for (const file of ["worker.js", "protocol.js", "wasm"]) await cp(path.join(web, "dist", file), path.join(assets, file), { recursive: true });
const { PROTOCOL } = await import(new URL("../dist/protocol.js", import.meta.url));
await writeFile(path.join(web, "dist/assets/manifest.json"), JSON.stringify({ protocol: PROTOCOL, engine: "Cataclysm 002", worker: `${versionId}/worker.js` }) + "\n");
console.log(`Built Cataclysm worker ${versionId}; WASM ${(await readFile(path.join(web, "src/wasm/cataclysm_wasm_bg.wasm"))).length} bytes`);
