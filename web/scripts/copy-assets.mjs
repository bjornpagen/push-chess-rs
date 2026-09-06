import { cp, mkdir, rename } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

if (process.argv.length !== 3) throw new Error("Usage: pnpm copy-assets /absolute/path/to/next-app/public/cataclysm");
const source = fileURLToPath(new URL("../dist/assets/", import.meta.url));
const target = path.resolve(process.argv[2]);
await mkdir(target, { recursive: true });
// Publish immutable version directories before switching the manifest. Keep old
// versions so already-open tabs can finish loading during a deployment.
await cp(source, target, { recursive: true, filter: file => !file.endsWith("manifest.json") });
await cp(path.join(source, "manifest.json"), path.join(target, "manifest.json.next"));
await rename(path.join(target, "manifest.json.next"), path.join(target, "manifest.json"));
console.log(`Cataclysm assets copied to ${target}`);
