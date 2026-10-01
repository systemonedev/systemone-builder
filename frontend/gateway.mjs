// Dashboard gateway: the only process listening on the dashboard port.
//
//   browser ──► gateway ──► /api/v1/*  ──► API (http://api:8000), X-API-Key added here
//                      └──► everything else ──► Next.js server (127.0.0.1, private port)
//
// The browser only ever talks to this origin and never sees S1_API_KEY. Because
// the gateway attaches the key for its callers, it must make sure the caller is
// the dashboard itself:
//
// * Same-origin only. Requests (and WebSocket handshakes) carrying an Origin or
//   Sec-Fetch-Site from another site are refused, so a web page you visit cannot
//   drive the API through your dashboard (CSRF / cross-site WebSocket hijacking).
// * Loopback bind (S1_BIND_ADDR=127.0.0.1, the default): no sign-in, like the API
//   itself on loopback, but the Host header must be a known name for this
//   machine. That stops DNS-rebinding pages from posing as same-origin.
// * Any other bind: the dashboard requires a sign-in with S1_API_KEY. A correct
//   key gets an HttpOnly, SameSite=Strict session cookie (a random token, not
//   the key); /api/v1 requests without a valid session get 401.
//
// Node built-ins only, so the runtime image needs no extra packages.

import { spawn } from "node:child_process";
import { randomBytes, timingSafeEqual, createHash } from "node:crypto";
import http from "node:http";

const PORT = Number(process.env.PORT || 3000);
const HOST = process.env.HOSTNAME || "0.0.0.0";
const API = new URL(process.env.S1_API_INTERNAL_URL || "http://api:8000");
const KEY = process.env.S1_API_KEY || "";
const BIND = (process.env.S1_BIND_ADDR || "127.0.0.1").trim();
const LOOPBACK = new Set(["127.0.0.1", "localhost", "::1", "[::1]"]);
const LOGIN_REQUIRED = !LOOPBACK.has(BIND);
const SESSION_TTL_MS = Number(process.env.S1_DASHBOARD_SESSION_HOURS || 12) * 3600_000;
const COOKIE = "s1_session";
const ALLOWED_HOSTS = new Set(
  ["localhost", "127.0.0.1", "[::1]", process.env.S1_PUBLIC_HOST, ...(process.env.S1_DASHBOARD_ALLOWED_HOSTS || "").split(",")]
    .map((h) => (h || "").trim().toLowerCase())
    .filter(Boolean),
);
// Hop-by-hop headers are per connection and must not be forwarded.
const HOP = new Set(["connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade"]);

// --------------------------------------------------------------- Next.js server
// S1_NEXT_URL points at an already running Next server (e.g. `npm run dev`);
// otherwise start the standalone server.js next to this file on a private port.
let NEXT;
if (process.env.S1_NEXT_URL) {
  NEXT = new URL(process.env.S1_NEXT_URL);
} else {
  const nextPort = Number(process.env.S1_NEXT_PORT || 3001);
  NEXT = new URL(`http://127.0.0.1:${nextPort}`);
  const child = spawn(process.execPath, ["server.js"], {
    env: { ...process.env, PORT: String(nextPort), HOSTNAME: "127.0.0.1" },
    stdio: "inherit",
  });
  child.on("exit", (code, signal) => {
    console.error(`[gateway] Next.js server exited (${signal ?? code}); stopping`);
    process.exit(code ?? 1);
  });
  for (const sig of ["SIGTERM", "SIGINT"]) process.on(sig, () => child.kill(sig));
}

// --------------------------------------------------------------------- helpers
const sessions = new Map(); // token -> expiry (ms); in memory, so a restart signs everyone out

function hostOnly(hostHeader) {
  const h = (hostHeader || "").toLowerCase();
  if (h.startsWith("[")) return h.slice(0, h.indexOf("]") + 1);
  return h.split(":")[0];
}

function cookies(req) {
  const out = {};
  for (const part of (req.headers.cookie || "").split(";")) {
    const i = part.indexOf("=");
    if (i > 0) out[part.slice(0, i).trim()] = decodeURIComponent(part.slice(i + 1).trim());
  }
  return out;
}

function hasSession(req) {
  const token = cookies(req)[COOKIE];
  const exp = token && sessions.get(token);
  if (!exp) return false;
  if (exp < Date.now()) {
    sessions.delete(token);
    return false;
  }
  return true;
}

function keyMatches(candidate) {
  // Hash both sides so the comparison is constant-time regardless of length.
  const a = createHash("sha256").update(String(candidate)).digest();
  const b = createHash("sha256").update(KEY).digest();
  return KEY !== "" && timingSafeEqual(a, b);
}

/** Why this request may not use the gateway's credentials, or null if it may. */
function refusal(req) {
  const site = req.headers["sec-fetch-site"];
  if (site && site !== "same-origin" && site !== "none") return "cross-site request refused";
  const origin = req.headers.origin;
  if (origin) {
    let o;
    try {
      o = new URL(origin);
    } catch {
      return "bad Origin header";
    }
    if (o.host.toLowerCase() !== (req.headers.host || "").toLowerCase()) return "cross-origin request refused";
  }
  if (!LOGIN_REQUIRED && !ALLOWED_HOSTS.has(hostOnly(req.headers.host))) {
    return `unknown host '${hostOnly(req.headers.host)}' (add it to S1_DASHBOARD_ALLOWED_HOSTS)`;
  }
  return null;
}

function sendJson(res, status, body, extraHeaders = {}) {
  res.writeHead(status, { "Content-Type": "application/json", "Cache-Control": "no-store", ...extraHeaders });
  res.end(JSON.stringify(body));
}

function forwardHeaders(req, target, toApi) {
  const h = {};
  for (const [k, v] of Object.entries(req.headers)) if (!HOP.has(k)) h[k] = v;
  h.host = target.host;
  if (toApi) {
    // Never pass browser credentials through; the API only sees the gateway's key.
    delete h["x-api-key"];
    delete h.cookie;
    delete h.authorization;
    if (KEY) h["x-api-key"] = KEY;
  }
  return h;
}

function readBody(req, limit = 4096) {
  return new Promise((resolve, reject) => {
    let data = "";
    req.on("data", (c) => {
      data += c;
      if (data.length > limit) reject(new Error("body too large"));
    });
    req.on("end", () => resolve(data));
    req.on("error", reject);
  });
}

const isApi = (url) => url === "/api/v1" || url.startsWith("/api/v1/") || url.startsWith("/api/v1?");

// ------------------------------------------------------------- sign-in routes
async function handleAuth(req, res) {
  const why = refusal(req);
  if (why) return sendJson(res, 403, { detail: why });
  const path = req.url.split("?")[0];
  if (path === "/s1-auth/status" && req.method === "GET") {
    return sendJson(res, 200, { required: LOGIN_REQUIRED, authenticated: !LOGIN_REQUIRED || hasSession(req) });
  }
  if (path === "/s1-auth/login" && req.method === "POST") {
    if (!LOGIN_REQUIRED) return sendJson(res, 200, { authenticated: true });
    let key = "";
    try {
      key = JSON.parse((await readBody(req)) || "{}").key || "";
    } catch {
      return sendJson(res, 400, { detail: "expected JSON {\"key\": ...}" });
    }
    if (!keyMatches(key)) {
      await new Promise((r) => setTimeout(r, 500)); // slow down guessing
      return sendJson(res, 401, { detail: "wrong API key" });
    }
    const token = randomBytes(32).toString("base64url");
    sessions.set(token, Date.now() + SESSION_TTL_MS);
    const cookie = `${COOKIE}=${token}; HttpOnly; SameSite=Strict; Path=/; Max-Age=${Math.floor(SESSION_TTL_MS / 1000)}`;
    return sendJson(res, 200, { authenticated: true }, { "Set-Cookie": cookie });
  }
  if (path === "/s1-auth/logout" && req.method === "POST") {
    const token = cookies(req)[COOKIE];
    if (token) sessions.delete(token);
    return sendJson(res, 200, { authenticated: false }, { "Set-Cookie": `${COOKIE}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0` });
  }
  return sendJson(res, 404, { detail: "not found" });
}

// ------------------------------------------------------------------ HTTP proxy
function proxy(req, res, target, toApi) {
  const up = http.request(
    { host: target.hostname, port: target.port || 80, method: req.method, path: req.url, headers: forwardHeaders(req, target, toApi) },
    (upRes) => {
      const headers = {};
      for (const [k, v] of Object.entries(upRes.headers)) if (!HOP.has(k)) headers[k] = v;
      res.writeHead(upRes.statusCode || 502, headers);
      upRes.pipe(res);
    },
  );
  up.on("error", (err) => {
    if (res.headersSent) return res.destroy();
    const what = toApi ? `the API at ${API.origin}` : "the dashboard's Next.js server";
    sendJson(res, 502, { detail: `dashboard cannot reach ${what}: ${err.code || err.message}` });
  });
  req.pipe(up);
}

const server = http.createServer((req, res) => {
  if (req.url.startsWith("/s1-auth/")) {
    handleAuth(req, res).catch(() => sendJson(res, 500, { detail: "sign-in failed" }));
    return;
  }
  if (isApi(req.url)) {
    const why = refusal(req);
    if (why) return sendJson(res, 403, { detail: why });
    if (LOGIN_REQUIRED && !hasSession(req)) return sendJson(res, 401, { detail: "sign in to the dashboard" });
    return proxy(req, res, API, true);
  }
  proxy(req, res, NEXT, false);
});

// ------------------------------------------------------------ WebSocket proxy
server.on("upgrade", (req, socket, head) => {
  const toApi = isApi(req.url);
  if (toApi) {
    const why = refusal(req) || (LOGIN_REQUIRED && !hasSession(req) ? "sign in to the dashboard" : null);
    if (why) {
      socket.end(`HTTP/1.1 403 Forbidden\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\n${why}`);
      return;
    }
  }
  const target = toApi ? API : NEXT; // Next only upgrades in dev (hot reload)
  const headers = forwardHeaders(req, target, toApi);
  headers.connection = "Upgrade";
  headers.upgrade = req.headers.upgrade;
  const up = http.request({ host: target.hostname, port: target.port || 80, method: req.method, path: req.url, headers });
  up.on("upgrade", (upRes, upSocket, upHead) => {
    let head0 = `HTTP/1.1 ${upRes.statusCode} ${upRes.statusMessage}\r\n`;
    for (let i = 0; i < upRes.rawHeaders.length; i += 2) head0 += `${upRes.rawHeaders[i]}: ${upRes.rawHeaders[i + 1]}\r\n`;
    socket.write(head0 + "\r\n");
    if (upHead?.length) socket.write(upHead);
    if (head?.length) upSocket.write(head);
    upSocket.on("error", () => socket.destroy());
    socket.on("error", () => upSocket.destroy());
    upSocket.pipe(socket).pipe(upSocket);
  });
  up.on("response", (upRes) => {
    // Upstream answered without upgrading (e.g. 404): relay the status and close.
    socket.end(`HTTP/1.1 ${upRes.statusCode} ${upRes.statusMessage}\r\nConnection: close\r\n\r\n`);
  });
  up.on("error", () => socket.destroy());
  up.end();
});

server.listen(PORT, HOST, () => {
  const mode = LOGIN_REQUIRED
    ? `sign-in required (S1_BIND_ADDR=${BIND})`
    : `loopback: no sign-in, hosts ${[...ALLOWED_HOSTS].join(", ")}`;
  console.log(`[gateway] listening on ${HOST}:${PORT}; API ${API.origin}${KEY ? " (key set)" : " (no key)"}; ${mode}`);
  if (LOGIN_REQUIRED && !KEY) console.error("[gateway] WARNING: S1_BIND_ADDR is not loopback but S1_API_KEY is empty");
});
