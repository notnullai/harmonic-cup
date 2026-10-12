import { useCallback, useEffect, useRef, useState } from "react";
import { api, type HEvent, type Health, type Mark, type Point } from "./api";

const HEALTH_MS = 1000;
const HISTORY_MS = 1000;
const EVENTS_MS = 1500;
const MAX_POINTS = 4000;

export interface Live {
  health: Health | null;
  points: Point[];
  events: HEvent[];
  /** Where the brain changed (a re-baseline), to mark on the timeline. */
  marks: Mark[];
  connected: boolean;
  /** Changes whenever the machine restarts or the source changes. */
  epoch: number;
  refresh: () => void;
}

/**
 * Polls the API. The history is fetched incrementally — only readings newer
 * than the last tick seen — so a week of recording is sent once, not every
 * second. A restart or a source switch shows up as the tick going backwards
 * or the source key changing, and the local copy is dropped and refetched.
 */
export function useLive(): Live {
  const [health, setHealth] = useState<Health | null>(null);
  const [points, setPoints] = useState<Point[]>([]);
  const [events, setEvents] = useState<HEvent[]>([]);
  const [marks, setMarks] = useState<Mark[]>([]);
  const [connected, setConnected] = useState(true);
  const [epoch, setEpoch] = useState(0);

  const lastTick = useRef(0);
  const sourceKey = useRef<string | null>(null);

  const reset = useCallback(() => {
    lastTick.current = 0;
    setPoints([]);
    setEvents([]);
    setEpoch((e) => e + 1);
  }, []);

  const pullHistory = useCallback(async () => {
    const res = await api.history(lastTick.current);
    if (sourceKey.current !== null && res.source !== sourceKey.current) {
      sourceKey.current = res.source;
      reset();
      return;
    }
    sourceKey.current = res.source;
    const m = res.marks ?? [];
    setMarks((prev) => (prev.length === m.length ? prev : m));
    if (!res.points.length) return;
    const first = res.points[0].tick;
    if (lastTick.current > 0 && first <= lastTick.current) {
      // The engine restarted underneath us.
      reset();
      return;
    }
    lastTick.current = res.points[res.points.length - 1].tick;
    setPoints((prev) => {
      const next = prev.concat(res.points);
      return next.length > MAX_POINTS ? next.slice(next.length - MAX_POINTS) : next;
    });
  }, [reset]);

  const pullHealth = useCallback(async () => {
    try {
      const h = await api.health();
      setHealth(h);
      setConnected(true);
      const tick = h.machine?.tick ?? 0;
      if (lastTick.current > 0 && tick > 0 && tick < lastTick.current) reset();
    } catch {
      setConnected(false);
    }
  }, [reset]);

  const pullEvents = useCallback(async () => {
    try {
      const r = await api.events();
      setEvents(r.events);
    } catch {
      /* the health poll reports the connection */
    }
  }, []);

  useEffect(() => {
    let alive = true;
    const loop = (fn: () => Promise<void>, ms: number) => {
      let t: number | undefined;
      const run = async () => {
        try { await fn(); } catch { /* reported by health */ }
        if (alive) t = window.setTimeout(run, ms);
      };
      run();
      return () => window.clearTimeout(t);
    };
    const stops = [
      loop(pullHealth, HEALTH_MS),
      loop(pullHistory, HISTORY_MS),
      loop(pullEvents, EVENTS_MS),
    ];
    return () => { alive = false; stops.forEach((s) => s()); };
  }, [pullHealth, pullHistory, pullEvents]);

  const refresh = useCallback(() => {
    reset();
    void pullHealth();
    void pullHistory();
    void pullEvents();
  }, [reset, pullHealth, pullHistory, pullEvents]);

  return { health, points, events, marks, connected, epoch, refresh };
}
