// Thin fetch layer over the EXISTING FastAPI endpoints. The console
// renders backend data only -- there is no mock/demo data anywhere.
async function request(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body.detail) detail = `${detail} — ${JSON.stringify(body.detail)}`;
    } catch { /* keep default detail */ }
    throw new Error(`${path} → ${detail}`);
  }
  return res.json();
}

export const api = {
  health: () => request("/api/health"),
  reports: () => request("/api/reports"),
  report: (id) => request(`/api/reports/${encodeURIComponent(id)}`),
  findings: () => request("/api/findings"),
  auditLog: () => request("/api/audit-log"),
  runAssurance: (payload) =>
    request("/api/assurance/run", { method: "POST", body: JSON.stringify(payload) }),
  browse: (path) =>
    request(`/api/fs/browse${path ? `?path=${encodeURIComponent(path)}` : ""}`),
  decide: (id, payload) =>
    request(`/api/reports/${encodeURIComponent(id)}/decision`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};

export const shortHash = (h, n = 12) => (h ? `${String(h).slice(0, n)}…` : "—");
export const fmt = (v) => (v === null || v === undefined ? "—" : String(v));
