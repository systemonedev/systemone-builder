"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, wsUrl } from "./api";

export type BusEvent = { channel: string; type: string; ts: number; data: Record<string, any> };
export type TelemetrySample = Record<string, any> & { ts: number };

/** Live event stream with automatic reconnect. */
export function useEvents(channels: string[], onEvent: (e: BusEvent) => void, onHello?: (hello: any) => void) {
  const [connected, setConnected] = useState(false);
  const cb = useRef(onEvent);
  const hello = useRef(onHello);
  cb.current = onEvent;
  hello.current = onHello;
  const key = channels.join(",");
  useEffect(() => {
    let ws: WebSocket | null = null;
    let closed = false;
    let retry: ReturnType<typeof setTimeout>;
    const connect = () => {
      ws = new WebSocket(wsUrl(key ? key.split(",") : undefined));
      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        if (!closed) retry = setTimeout(connect, 1500);
      };
      ws.onmessage = (m) => {
        const msg = JSON.parse(m.data);
        if (msg.type === "hello") hello.current?.(msg);
        else cb.current(msg as BusEvent);
      };
    };
    connect();
    return () => {
      closed = true;
      clearTimeout(retry);
      ws?.close();
    };
  }, [key]);
  return connected;
}

/** Poll a REST endpoint. */
export function usePoll<T>(path: string | null, intervalMs = 3000) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const load = useCallback(async () => {
    if (!path) return;
    try {
      setData(await api<T>(path));
      setError(null);
    } catch (e: any) {
      setError(e.message ?? String(e));
    }
  }, [path]);
  useEffect(() => {
    load();
    if (!intervalMs) return;
    const t = setInterval(load, intervalMs);
    return () => clearInterval(t);
  }, [load, intervalMs]);
  return { data, error, reload: load };
}
