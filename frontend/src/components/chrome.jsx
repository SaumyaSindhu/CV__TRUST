import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity, Bell, FileSearch, LayoutDashboard, ListChecks, Moon, Plus, RefreshCw,
  ScrollText, Search, ServerCog, ShieldCheck, Sun, WifiOff, X,
} from "lucide-react";
import { useTheme } from "../theme.jsx";
import { IconButton, Pill, PrimaryButton, useOutside } from "./ui.jsx";

/* Navigation hierarchy mirrors the reference's grouped rail. Every entry
 * maps 1:1 onto a view the original console already had. */
const NAV_GROUPS = [
  {
    group: "Dashboard",
    items: [{ id: "overview", label: "Overview", icon: LayoutDashboard }],
  },
  {
    group: "Evidence",
    items: [
      { id: "findings", label: "Findings", icon: ListChecks },
      { id: "audit", label: "Audit Trail", icon: ScrollText },
    ],
  },
  {
    group: "Archive",
    items: [{ id: "reports", label: "Reports", icon: FileSearch }],
  },
];

/* ------------------------------------------------------------------ *
 * Sidebar
 * ------------------------------------------------------------------ */
export function Sidebar({ view, setView, health, chainValid, counts, onNew }) {
  const { dark, setDark } = useTheme();

  return (
    <aside
      className="flex h-full w-16 shrink-0 flex-col border-r border-line bg-panel
                 lg:w-60"
    >
      <div className="flex items-center gap-2.5 px-3 py-4 lg:px-4">
        <div
          className="grid h-8 w-8 shrink-0 place-items-center rounded-lg"
          style={{ background: "linear-gradient(135deg,#f97316,#b45309)" }}
        >
          <ShieldCheck className="h-4.5 w-4.5 text-white" />
        </div>
        <div className="hidden min-w-0 lg:block">
          <div className="truncate text-[13px] font-semibold tracking-tight text-head">
            CV Assurance
          </div>
          <div className="truncate text-[10.5px] text-muted">Integrity console</div>
        </div>
      </div>

      <nav className="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
        {NAV_GROUPS.map((g) => (
          <div key={g.group} className="mb-3">
            <p className="hidden px-3 pt-2 pb-1.5 text-[11px] text-muted lg:block">
              {g.group}
            </p>
            {g.items.map(({ id, label, icon: Icon }) => {
              const active = view === id;
              const badge = counts?.[id];
              return (
                <button
                  key={id}
                  onClick={() => setView(id)}
                  title={label}
                  className={`mb-0.5 flex w-full items-center gap-2.5 rounded-lg px-3 py-2
                              text-left text-[12.5px] transition-colors ${
                                active
                                  ? "border border-line bg-panel2 text-head"
                                  : "border border-transparent text-muted hover:bg-panel2 hover:text-ink"
                              }`}
                >
                  <Icon className={`h-4 w-4 shrink-0 ${active ? "text-accent" : ""}`} />
                  <span className="hidden min-w-0 flex-1 truncate lg:inline">{label}</span>
                  {badge != null && badge > 0 && (
                    <span className="mono hidden text-[10.5px] text-muted lg:inline">
                      {badge}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        ))}

        <div className="mb-3">
          <p className="hidden px-3 pt-2 pb-1.5 text-[11px] text-muted lg:block">Actions</p>
          <button
            onClick={onNew}
            title="New assessment"
            className="flex w-full items-center gap-2.5 rounded-lg border border-transparent
                       px-3 py-2 text-left text-[12.5px] text-muted transition-colors
                       hover:bg-panel2 hover:text-accent"
          >
            <Plus className="h-4 w-4 shrink-0" />
            <span className="hidden lg:inline">New assessment</span>
          </button>
        </div>
      </nav>

      {/* backend + audit-chain health (preserved from the original rail) */}
      <div className="hidden border-t border-line px-3 py-3 text-[11px] text-muted lg:block">
        <div className="mb-1.5 flex items-center gap-2">
          <span
            className={`live-dot inline-block h-2 w-2 shrink-0 rounded-full ${
              health ? "bg-ok" : "bg-bad"
            }`}
          />
          <span className="truncate">
            backend {health ? `v${health.version}` : "offline"}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <Activity className="h-3.5 w-3.5 shrink-0" />
          <span>audit chain</span>
          <span
            className={`mono ${chainValid === false ? "text-bad" : "text-ok"}`}
          >
            {chainValid === false ? "BROKEN" : chainValid === true ? "intact" : "—"}
          </span>
        </div>
      </div>

      <div className="m-2 grid grid-cols-1 gap-1 rounded-xl border border-line bg-panel2 p-1
                      lg:grid-cols-2">
        <button
          onClick={() => setDark(false)}
          title="Light theme"
          className={`flex items-center justify-center gap-1.5 rounded-lg py-1.5
                      text-[11.5px] transition-colors ${
                        !dark
                          ? "border border-line bg-panel text-head"
                          : "text-muted hover:text-ink"
                      }`}
        >
          <Sun className="h-3.5 w-3.5" />
          <span className="hidden lg:inline">Light</span>
        </button>
        <button
          onClick={() => setDark(true)}
          title="Dark theme"
          className={`flex items-center justify-center gap-1.5 rounded-lg py-1.5
                      text-[11.5px] transition-colors ${
                        dark
                          ? "border border-line bg-panel text-head"
                          : "text-muted hover:text-ink"
                      }`}
        >
          <Moon className="h-3.5 w-3.5" />
          <span className="hidden lg:inline">Dark</span>
        </button>
      </div>
    </aside>
  );
}

/* ------------------------------------------------------------------ *
 * Global search — searches the data already loaded from the backend.
 * ------------------------------------------------------------------ */
function GlobalSearch({ findings, reports, audit, onOpenFinding, onOpenReport, onGoAudit }) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  const inputRef = useRef(null);
  useOutside(ref, () => setOpen(false));

  useEffect(() => {
    function key(e) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        inputRef.current?.focus();
        setOpen(true);
      }
    }
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, []);

  const results = useMemo(() => {
    const term = q.trim().toLowerCase();
    if (!term) return null;
    const f = findings
      .filter((x) =>
        `${x.finding_id} ${x.affected_asset} ${x.category} ${x.source_phase} ${x.severity} ${x.explanation}`
          .toLowerCase()
          .includes(term)
      )
      .slice(0, 6);
    const r = reports
      .filter((x) => `${x.report_id} ${x.kind}`.toLowerCase().includes(term))
      .slice(0, 5);
    const a = (audit?.entries || [])
      .filter((e) => `${e.record_id} ${e.entry_hash}`.toLowerCase().includes(term))
      .slice(0, 4);
    return { f, r, a, total: f.length + r.length + a.length };
  }, [q, findings, reports, audit]);

  return (
    <div className="relative w-full max-w-md" ref={ref}>
      <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-muted" />
      <input
        ref={inputRef}
        value={q}
        onChange={(e) => {
          setQ(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        placeholder="Search findings, reports, audit entries…"
        className="w-full rounded-lg border border-line bg-panel2 py-2 pr-16 pl-9
                   text-[12.5px] text-ink outline-none transition-colors
                   placeholder:text-muted focus:border-accent/50"
      />
      {q ? (
        <button
          onClick={() => {
            setQ("");
            inputRef.current?.focus();
          }}
          title="Clear search"
          className="absolute right-2 top-2 grid h-5 w-5 place-items-center rounded
                     text-muted hover:text-bad"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      ) : (
        <kbd
          className="pointer-events-none absolute right-2 top-1.5 rounded-md border
                     border-line px-1.5 py-0.5 text-[10.5px] text-muted"
        >
          ⌘K
        </kbd>
      )}

      {open && results && (
        <div className="panel absolute left-0 right-0 z-40 mt-2 max-h-96 overflow-y-auto p-1
                        shadow-2xl">
          {results.total === 0 && (
            <div className="px-3 py-3 text-[12px] text-muted">
              Nothing in the loaded backend data matches “{q}”.
            </div>
          )}
          {results.f.length > 0 && (
            <p className="px-3 pt-2 pb-1 text-[10.5px] text-muted">Findings</p>
          )}
          {results.f.map((x) => (
            <button
              key={`${x.report_id || ""}${x.finding_id}`}
              onClick={() => {
                onOpenFinding(x);
                setOpen(false);
              }}
              className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-left
                         transition-colors hover:bg-panel2"
            >
              <span className="mono shrink-0 text-[11.5px] text-accent">{x.finding_id}</span>
              <span className="min-w-0 flex-1 truncate text-[12px] text-ink">
                {x.explanation}
              </span>
              <Pill value={x.severity} />
            </button>
          ))}
          {results.r.length > 0 && (
            <p className="px-3 pt-2 pb-1 text-[10.5px] text-muted">Reports</p>
          )}
          {results.r.map((x) => (
            <button
              key={x.report_id}
              onClick={() => {
                onOpenReport(x);
                setOpen(false);
              }}
              className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-left
                         transition-colors hover:bg-panel2"
            >
              <span className="mono min-w-0 flex-1 truncate text-[11.5px] text-ink">
                {x.report_id}
              </span>
              <Pill value={x.kind} />
            </button>
          ))}
          {results.a.length > 0 && (
            <p className="px-3 pt-2 pb-1 text-[10.5px] text-muted">Audit entries</p>
          )}
          {results.a.map((e) => (
            <button
              key={e.entry_index}
              onClick={() => {
                onGoAudit();
                setOpen(false);
              }}
              className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-left
                         transition-colors hover:bg-panel2"
            >
              <span className="mono shrink-0 text-[11.5px] text-muted">#{e.entry_index}</span>
              <span className="mono min-w-0 flex-1 truncate text-[12px] text-ink">
                {e.record_id}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * Alerts bell — driven by real high-severity findings + chain state.
 * ------------------------------------------------------------------ */
function Alerts({ findings, audit, onOpenFinding, onGoAudit }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useOutside(ref, () => setOpen(false));

  const high = findings.filter((f) => f.severity === "high");
  const chainBroken = audit ? audit.chain_valid === false : false;
  const count = high.length + (chainBroken ? 1 : 0);

  return (
    <div className="relative" ref={ref}>
      <IconButton icon={Bell} title="Alerts" onClick={() => setOpen((v) => !v)} active={open} />
      {count > 0 && (
        <span className="pointer-events-none absolute right-1.5 top-1.5 h-1.5 w-1.5
                         rounded-full bg-accent" />
      )}
      {open && (
        <div className="panel absolute right-0 z-40 mt-2 w-80 max-h-96 overflow-y-auto p-1
                        shadow-2xl">
          <p className="px-3 pt-2 pb-1 text-[10.5px] text-muted">
            Alerts from stored assessments
          </p>
          {count === 0 && (
            <div className="px-3 py-3 text-[12px] text-muted">
              No high-severity findings and the audit chain verifies.
            </div>
          )}
          {chainBroken && (
            <button
              onClick={() => {
                onGoAudit();
                setOpen(false);
              }}
              className="w-full rounded-md px-3 py-2 text-left transition-colors hover:bg-panel2"
            >
              <div className="text-[12.5px] text-bad">Audit hash chain broken</div>
              <div className="text-[11px] text-muted">
                {audit.chain_issues.length} issue(s) — open the audit trail
              </div>
            </button>
          )}
          {high.slice(0, 12).map((f) => (
            <button
              key={`${f.report_id || ""}${f.finding_id}`}
              onClick={() => {
                onOpenFinding(f);
                setOpen(false);
              }}
              className="w-full rounded-md px-3 py-2 text-left transition-colors hover:bg-panel2"
            >
              <div className="truncate text-[12.5px] text-ink">{f.explanation}</div>
              <div className="mono truncate text-[11px] text-muted">
                {f.finding_id} · {f.category}
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * Backend identity — replaces the reference's user block. There is no
 * auth in this product, so it surfaces the real /api/health payload.
 * ------------------------------------------------------------------ */
function BackendIdentity({ health }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useOutside(ref, () => setOpen(false));

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 rounded-lg px-2 py-1.5 transition-colors
                   hover:bg-panel2"
      >
        <span
          className="grid h-8 w-8 shrink-0 place-items-center rounded-full border
                     border-line bg-panel2"
        >
          <ServerCog className="h-4 w-4 text-muted" />
        </span>
        <span className="hidden text-left leading-tight md:block">
          <span className="block text-[11.5px] font-medium text-head">
            {health ? health.service : "connecting…"}
          </span>
          <span className="mono block text-[10.5px] text-muted">
            {health ? `v${health.version}` : "no backend"}
          </span>
        </span>
      </button>
      {open && (
        <div className="panel absolute right-0 z-40 mt-2 w-72 p-3 shadow-2xl">
          <div className="mb-2 flex items-center gap-2">
            <span
              className={`h-2 w-2 rounded-full ${health ? "bg-ok" : "bg-bad"}`}
            />
            <span className="text-[12.5px] text-head">
              {health ? "Backend reachable" : "Backend unreachable"}
            </span>
          </div>
          {health ? (
            <>
              <div className="mono mb-2 break-all text-[11.5px] text-muted">
                {health.service} · v{health.version} · {health.status}
              </div>
              <p className="mb-1.5 text-[10.5px] text-muted">Phases exposed</p>
              <div className="flex flex-wrap gap-1">
                {health.phases.map((p) => (
                  <Pill key={p} value={p} />
                ))}
              </div>
            </>
          ) : (
            <p className="text-[12px] text-muted">
              Start the API with{" "}
              <span className="mono text-accent">uvicorn backend.app:app</span>.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * Top bar
 * ------------------------------------------------------------------ */
export function TopBar({
  health, findings, reports, audit, onRefresh, refreshing, onNew,
  onOpenFinding, onOpenReport, onGoAudit,
}) {
  return (
    <header className="flex items-center gap-3 border-b border-line px-3 py-2.5 sm:px-5">
      <GlobalSearch
        findings={findings}
        reports={reports}
        audit={audit}
        onOpenFinding={onOpenFinding}
        onOpenReport={onOpenReport}
        onGoAudit={onGoAudit}
      />

      <div className="ml-auto flex items-center gap-2">
        <div
          className="hidden items-center gap-1.5 rounded-lg border border-ok/40 bg-ok/10
                     px-2.5 py-1.5 text-[11px] text-ok xl:flex"
          title="No network calls are made — every path is server-local"
        >
          <WifiOff className="h-3.5 w-3.5" />
          Air-gapped
        </div>

        <PrimaryButton icon={Plus} onClick={onNew} className="hidden sm:flex">
          New assessment
        </PrimaryButton>

        <BackendIdentity health={health} />
        <IconButton
          icon={RefreshCw}
          title="Refresh from backend"
          onClick={onRefresh}
          spinning={refreshing}
        />
        <Alerts
          findings={findings}
          audit={audit}
          onOpenFinding={onOpenFinding}
          onGoAudit={onGoAudit}
        />
      </div>
    </header>
  );
}
