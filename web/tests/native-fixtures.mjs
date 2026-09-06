import { spawnSync } from "node:child_process";
import { writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../../", import.meta.url));
const result = spawnSync("cargo", ["run", "--locked", "--release", "-p", "push-chess", "--example", "wasm_parity"], { cwd: root, encoding: "utf8", maxBuffer: 16 * 1024 * 1024 });
if (result.error || result.status !== 0) throw result.error ?? new Error(result.stderr);
await writeFile(new URL("native-fixtures.json", import.meta.url), result.stdout);
