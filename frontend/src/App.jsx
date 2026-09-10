import React, { useCallback, useEffect, useMemo, useState } from "react";
import { ScrollText } from "lucide-react";
import { api } from "./api.js";
import { Sidebar, TopBar } from "./components/chrome.jsx";
import Overview from "./components/Overview.jsx";
import Findings from "./components/Findings.jsx";
import Reports from "./components/Reports.jsx";
import RunAssessment from "./components/RunAssessment.jsx";
import { AuditTimeline } from "./components/panels.jsx";
import { ErrorState, LoadingState, Panel } from "./components/ui.jsx";

export default function App() {
  const [view, setView] = useState("overview");
  const [health, setHealth] = useState(null);
  const [reports, setReports] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [env, setEnv] = useState(null);
  const [findings, setFindings] = useState([]);
  const [audit, setAudit] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [busy, setBusy] = useState(false);
  const [showRun, setShowRun] = useState(false);
  const [error, setError] = useState(null);
  const [findingsQuery, setFindingsQuery] = useState("");

  const assessments = useMemo(
    () => reports.filter((r) => r.kind === "assurance_assessment"),
    [reports]
  );

  const refreshAll = useCallback(async (keepSelection = true) => {
    setRefreshing(true);
    setError(null);
    try {
      const [h, r, f, a] = await Promise.all([
        api.health(), api.reports(), api.findings(), api.auditLog(),
      ]);
      setHealth(h);
      setReports(r.reports);
      setFindings(f.findings);
      setAudit(a);
      const list = r.reports.filter((x) => x.kind === "assurance_assessment");
      setSelectedId((cur) =>
        keepSelection && cur && list.some((x) => x.report_id === cur)
          ? cur
          : list.length ? list[list.length - 1].report_id : null);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => { refreshAll(false); }, [refreshAll]);

  useEffect(() => {
    if (!selectedId) { setEnv(null); return; }
    let cancelled = false;
    api.report(selectedId)
      .then((e) => { if (!cancelled) setEnv(e); })
      .catch((e) => { if (!cancelled) setError(e.message || String(e)); });
    return () => { cancelled = true; };
  }, [selectedId]);

  const onCreated = async (created) => {
    // real assessment just returned from the backend: open it and refresh
    setShowRun(false);
    setEnv(created);
    setSelectedId(created.report_id);
    setView("overview");
    await refreshAll();
  };

  const onDecide = async (payload) => {
    setBusy(true);
    try {
      await api.decide(selectedId, payload);
      const [e, a] = await Promise.all([api.report(selectedId), api.auditLog()]);
      setEnv(e); setAudit(a);
    } finally {
      setBusy(false);
    }
  };

  /* --- navigation helpers shared by the chrome and the pages --- */
  const goFindings = (f) => {
    setFindingsQuery(f && f.finding_id ? f.finding_id : "");
    setView("findings");
  };
  const goAudit = () => setView("audit");
  const openAssessment = (reportId) => {
    setSelectedId(reportId);
    setView("overview");
  };
  const openReportEntry = (r) => {
    if (r.kind === "assurance_assessment") openAssessment(r.report_id);
    else setView("reports");
  };

  const counts = {
    findings: findings.length,
    reports: reports.length,
    audit: audit ? audit.num_entries : 0,
  };

  return (
    <div className="h-full p-2 sm:p-3">
      <div className="ring-soft flex h-full overflow-hidden rounded-2xl border border-line
                      bg-panel">
        <Sidebar
          view={view}
          setView={setView}
          health={health}
          chainValid={audit ? audit.chain_valid : null}
          counts={counts}
          onNew={() => setShowRun(true)}
        />
        <div className="flex min-w-0 flex-1 flex-col">
          <TopBar
            health={health}
            findings={findings}
            reports={reports}
            audit={audit}
            onRefresh={() => refreshAll()}
            refreshing={refreshing}
            onNew={() => setShowRun(true)}
            onOpenFinding={goFindings}
            onOpenReport={openReportEntry}
            onGoAudit={goAudit}
          />

          <main className="min-w-0 flex-1 overflow-y-auto p-4 lg:p-5">
            {showRun && (
              <RunAssessment onClose={() => setShowRun(false)} onCreated={onCreated} />
            )}
            {error && <ErrorState error={error} onRetry={() => refreshAll()} />}

            {loading ? (
              <LoadingState />
            ) : view === "overview" ? (
              <Overview
                env={env}
                onDecide={onDecide}
                busy={busy}
                onNew={() => setShowRun(true)}
                assessments={assessments}
                selectedId={selectedId}
                onSelect={setSelectedId}
                audit={audit}
                onGoFindings={goFindings}
                onGoAudit={goAudit}
              />
            ) : view === "findings" ? (
              <Findings
                findings={findings}
                query={findingsQuery}
                setQuery={setFindingsQuery}
                onOpenReport={openAssessment}
              />
            ) : view === "audit" ? (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <h1 className="text-[19px] font-semibold tracking-tight text-head">
                    Audit Trail
                  </h1>
                  <span className="flex items-center gap-2 text-[12.5px] text-muted">
                    <ScrollText className="h-3.5 w-3.5" />
                    {audit ? `${audit.num_entries} chained entries` : "—"}
                  </span>
                </div>
                <Panel title="Tamper-evident audit timeline" className="fade-up">
                  <AuditTimeline log={audit} />
                </Panel>
              </div>
            ) : (
              <Reports
                reports={reports}
                selectedId={selectedId}
                onOpen={openReportEntry}
              />
            )}

            <footer className="mt-6 border-t border-line pt-3 pb-2 text-[11px]
                               leading-relaxed text-muted">
              We detect integrity violations and suspicious indicators using multiple layers of
              evidence. Statistical anomalies are not automatically treated as malicious
              activity. Hashing proves content identity, not authorship; valid provenance does
              not prove model accuracy; distribution shift can be legitimate.
            </footer>
          </main>
        </div>
      </div>
    </div>
  );
}
