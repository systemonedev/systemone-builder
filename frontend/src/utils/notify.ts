// Minimal global notifications: notify() from anywhere, <Toasts/> renders them.

export type Toast = { id: number; text: string; kind: "ok" | "error" | "info" };

type Listener = (toasts: Toast[]) => void;
let toasts: Toast[] = [];
let next = 1;
const listeners = new Set<Listener>();

function emit() {
  listeners.forEach((l) => l(toasts));
}

export function notify(text: string, kind: Toast["kind"] = "ok", ttlMs = kind === "error" ? 10000 : 5000) {
  const t = { id: next++, text, kind };
  toasts = [...toasts, t].slice(-5);
  emit();
  setTimeout(() => dismiss(t.id), ttlMs);
}

export function dismiss(id: number) {
  toasts = toasts.filter((t) => t.id !== id);
  emit();
}

export function subscribeToasts(l: Listener): () => void {
  listeners.add(l);
  l(toasts);
  return () => listeners.delete(l);
}
