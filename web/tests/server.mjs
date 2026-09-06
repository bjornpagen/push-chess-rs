import http from "node:http";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(fileURLToPath(new URL("../", import.meta.url)));
http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");
  if (url.pathname === "/") { res.setHeader("Content-Type", "text/html"); res.end("<!doctype html><html><head><meta charset=utf-8></head><body></body></html>"); return; }
  const file = path.resolve(root, "." + decodeURIComponent(url.pathname));
  if (!file.startsWith(root + path.sep) || !(url.pathname.startsWith("/dist/") || url.pathname === "/tests/native-fixtures.json")) { res.writeHead(404); res.end(); return; }
  try {
    const bytes = await readFile(file);
    res.setHeader("Content-Type", file.endsWith(".wasm") ? "application/wasm" : file.endsWith(".json") ? "application/json" : "text/javascript");
    res.setHeader("Cache-Control", "no-store");
    res.end(bytes);
  } catch { res.writeHead(404); res.end(); }
}).listen(4173, "127.0.0.1");
