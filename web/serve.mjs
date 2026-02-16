#!/usr/bin/env node
/**
 * HTTPS dev server for Kyutai STT browser demo.
 *
 * Serves the web app, WASM pkg, and model shards.
 * WebGPU requires HTTPS — run scripts/gen-cert.sh first.
 *
 * Usage: bun web/serve.mjs [--port 8443]
 */

import { createServer } from "node:https";
import { readFileSync, createReadStream, existsSync, statSync, readdirSync } from "node:fs";
import { join, extname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(fileURLToPath(import.meta.url), "../..");
const PORT = parseInt(process.argv.find((_, i, a) => a[i - 1] === "--port") ?? "8443");

const MIME = {
    ".html": "text/html",
    ".js":   "text/javascript",
    ".mjs":  "text/javascript",
    ".wasm": "application/wasm",
    ".json": "application/json",
    ".wav":  "audio/wav",
    ".css":  "text/css",
    ".bin":  "application/octet-stream",
};

const TLS = {
    key:  readFileSync("/tmp/stt-key.pem"),
    cert: readFileSync("/tmp/stt-cert.pem"),
};

// Discover model shards
const SHARD_DIR = join(ROOT, "models/stt-q4-shards");
const shardNames = existsSync(SHARD_DIR)
    ? readdirSync(SHARD_DIR).filter(f => f.startsWith("shard-")).sort()
    : [];

const server = createServer(TLS, (req, res) => {
    const url = new URL(req.url, `https://${req.headers.host}`);
    const pathname = decodeURIComponent(url.pathname);

    // CORS headers (needed for WASM + WebGPU in cross-origin workers)
    res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
    res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");

    // API: list available shards
    if (pathname === "/api/shards") {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ shards: shardNames }));
        return;
    }

    // Route to file
    let filePath;
    if (pathname === "/" || pathname === "/index.html") {
        filePath = join(ROOT, "web/index.html");
    } else if (pathname.startsWith("/pkg/")) {
        // WASM build output from crates/stt-wasm/pkg/
        filePath = join(ROOT, "crates/stt-wasm", pathname);
    } else if (pathname.startsWith("/mimi-pkg/")) {
        filePath = join(ROOT, "crates/mimi-wasm/pkg", pathname.replace("/mimi-pkg/", ""));
    } else if (pathname.startsWith("/models/")) {
        filePath = join(ROOT, pathname);
    } else {
        filePath = join(ROOT, "web", pathname);
    }

    if (!existsSync(filePath) || !statSync(filePath).isFile()) {
        res.writeHead(404);
        res.end("Not found: " + pathname);
        return;
    }

    const ext = extname(filePath);
    const mime = MIME[ext] ?? "application/octet-stream";
    const stat = statSync(filePath);

    // Support range requests for large shard files
    const range = req.headers.range;
    if (range && stat.size > 1_000_000) {
        const match = range.match(/bytes=(\d+)-(\d*)/);
        if (match) {
            const start = parseInt(match[1]);
            const end = match[2] ? parseInt(match[2]) : stat.size - 1;
            res.writeHead(206, {
                "Content-Type": mime,
                "Content-Range": `bytes ${start}-${end}/${stat.size}`,
                "Content-Length": end - start + 1,
                "Accept-Ranges": "bytes",
            });
            createReadStream(filePath, { start, end }).pipe(res);
            return;
        }
    }

    res.writeHead(200, {
        "Content-Type": mime,
        "Content-Length": stat.size,
        "Accept-Ranges": "bytes",
    });
    createReadStream(filePath).pipe(res);
});

server.listen(PORT, "0.0.0.0", () => {
    console.log(`\nKyutai STT dev server running:`);
    console.log(`  Local:   https://localhost:${PORT}`);
    console.log(`\nModel shards: ${shardNames.length} (${SHARD_DIR})`);
    console.log(`\nNote: Accept the self-signed certificate in your browser.\n`);
});
