// REST client for the systemone LAN API.
//
// The API base defaults to port 8000 on the same host that serves the
// dashboard, so opening http://<linux-host>:3000 from any LAN machine works.
// NEXT_PUBLIC_S1_API_URL overrides it; the X-API-Key is kept per browser.

export function apiBase(): string {
  const env = process.env.NEXT_PUBLIC_S1_API_URL;
  if (env) return env.replace(/\/$/, "");
  if (typeof window === "undefined") return "http://localhost:8000";
  return `${window.location.protocol}//${window.location.hostname}:8000`;
}

export function apiKey(): string | null {
  try {
    return window.localStorage.getItem("s1.apiKey");
  } catch {
    return null;
  }
}

export function setApiKey(key: string) {
  try {
    window.localStorage.setItem("s1.apiKey", key);
  } catch {
    /* storage unavailable: key lasts for this page only */
  }
}

export class ApiError extends Error {
  constructor(public status: number, public detail: unknown) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
}

export async function api<T = unknown>(path: string, init: RequestInit & { json?: unknown } = {}): Promise<T> {
  const headers: Record<string, string> = { ...(init.headers as Record<string, string>) };
  const key = apiKey();
  if (key) headers["X-API-Key"] = key;
  let body = init.body;
  if (init.json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(init.json);
  }
  const res = await fetch(`${apiBase()}/api/v1${path}`, { ...init, headers, body, cache: "no-store" });
  if (res.status === 204) return undefined as T;
  const data = await res.json().catch(() => null);
  if (!res.ok) throw new ApiError(res.status, data?.detail ?? data ?? res.statusText);
  return data as T;
}

export function wsUrl(channels?: string[]): string {
  const base = apiBase().replace(/^http/, "ws");
  const qs = new URLSearchParams();
  if (channels?.length) qs.set("channels", channels.join(","));
  const key = apiKey();
  if (key) qs.set("api_key", key);
  return `${base}/api/v1/ws?${qs.toString()}`;
}
