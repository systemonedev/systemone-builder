// REST client for the systemone API.
//
// The browser only talks to the dashboard's own origin. Its gateway
// (frontend/gateway.mjs) forwards /api/v1/* to the API and adds S1_API_KEY
// server-side, so the key never reaches the browser. When the dashboard is
// exposed beyond loopback, the gateway requires a sign-in (HttpOnly session
// cookie) first; see authStatus/login/logout.

export class ApiError extends Error {
  constructor(public status: number, public detail: unknown) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
}

export async function api<T = unknown>(path: string, init: RequestInit & { json?: unknown } = {}): Promise<T> {
  const headers: Record<string, string> = { ...(init.headers as Record<string, string>) };
  let body = init.body;
  if (init.json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(init.json);
  }
  const res = await fetch(`/api/v1${path}`, { ...init, headers, body, cache: "no-store", credentials: "same-origin" });
  if (res.status === 204) return undefined as T;
  const data = await res.json().catch(() => null);
  if (!res.ok) throw new ApiError(res.status, data?.detail ?? data ?? res.statusText);
  return data as T;
}

export function wsUrl(channels?: string[]): string {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const qs = new URLSearchParams();
  if (channels?.length) qs.set("channels", channels.join(","));
  return `${proto}://${window.location.host}/api/v1/ws?${qs.toString()}`;
}

export type AuthStatus = { required: boolean; authenticated: boolean };

async function authCall(path: string, init?: RequestInit): Promise<AuthStatus & { detail?: string }> {
  const res = await fetch(`/s1-auth/${path}`, { cache: "no-store", credentials: "same-origin", ...init });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new ApiError(res.status, data?.detail ?? res.statusText);
  return data;
}

export const authStatus = () => authCall("status");

export const login = (key: string) =>
  authCall("login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ key }) });

export const logout = () => authCall("logout", { method: "POST" });
