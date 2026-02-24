const hits = new Map<string, { n: number; ts: number }>();
const WINDOW_MS = 60_000; // 1 minute
const LIMIT = 10; // 10 emails/min per IP

export function limit(ip: string) {
  const now = Date.now();
  const record = hits.get(ip) ?? { n: 0, ts: now };

  if (now - record.ts > WINDOW_MS) {
    record.n = 0;
    record.ts = now;
  }

  record.n += 1;
  hits.set(ip, record);

  return record.n <= LIMIT;
}
