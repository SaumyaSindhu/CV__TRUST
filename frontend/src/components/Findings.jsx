import React, { useMemo, useRef, useState } from "react";
import { Download, Search, SlidersHorizontal, X } from "lucide-react";
import { Dropdown, EmptyState, Panel, useOutside } from "./ui.jsx";
import { FindingsTable, } from "./panels.jsx";
import { phaseLabel } from "./charts.jsx";

const SEVERITIES = ["all", "low", "medium", "high"];

function download(name, mime, text) {
  try {
    const url = URL.createObjectURL(new Blob([text], { type: mime }));
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    return true;
  } catch {
    return false;
  }
}

const csvCell = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;

function ExportMenu({ rows }) {
  const [open, setOpen] = useState(false);
  const [note, setNote] = useState(null);
  const ref = useRef(null);
  useOutside(ref, () => setOpen(false));

  const asCsv = () => {
    const head = [
      "finding_id", "report_id", "timestamp", "source_phase", "category", "severity",
      "confidence", "recommendation", "affected_asset", "explanation", "evidence",
    ];
    const body = rows.map((f) =>
      [
        f.finding_id, f.report_id ?? "", f.timestamp, f.source_phase, f.category,
        f.severity, f.confidence, f.recommendation, f.affected_asset, f.explanation,
        JSON.stringify(f.evidence ?? {}),
      ]
        .map(csvCell)
        .join(",")
    );
    const ok = download(
      "cv-assurance-findings.csv",
      "text/csv;charset=utf-8",
      [head.join(","), ...body].join("\n")
    );
    setNote(ok ? null : "Download blocked by the browser.");
    setOpen(false);
  };

  const asJson = () => {
    const ok = download(
      "cv-assurance-findings.json",
      "application/json",
      JSON.stringify(rows, null, 2)
    );
    setNote(ok ? null : "Download blocked by the browser.");
    setOpen(false);
  };

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 rounded-lg border border-line bg-panel2 px-3 py-2
                   text-[12.5px] text-ink transition-colors hover:border-accent/50"
      >
        <Download className="h-3.5 w-3.5 text-muted" />
        Export
      </button>
      {open && (
        <div className="panel absolute right-0 z-40 mt-2 w-44 p-1 shadow-2xl">
          <button
            onClick={asCsv}
            className="w-full rounded-md px-3 py-2 text-left text-[12.5px] text-muted
                       transition-colors hover:bg-panel2 hover:text-ink"
          >
            CSV ({rows.length} rows)
          </button>
          <button
            onClick={asJson}
            className="w-full rounded-md px-3 py-2 text-left text-[12.5px] text-muted
                       transition-colors hover:bg-panel2 hover:text-ink"
          >
            JSON ({rows.length} rows)
          </button>
        </div>
      )}
      {note && <p className="absolute right-0 top-11 text-[11px] text-bad">{note}</p>}
    </div>
  );
}

export default function Findings({ findings, query, setQuery, onOpenReport }) {
  const [sev, setSev] = useState("all");
  const [rec, setRec] = useState("all");
  const [type, setType] = useState("all");
  const [phase, setPhase] = useState("all");
  const [report, setReport] = useState("all");
  const [showFilters, setShowFilters] = useState(false);

  const categories = useMemo(
    () => [...new Set(findings.map((f) => f.category))].sort(),
    [findings]
  );
  const phases = useMemo(
    () => [...new Set(findings.map((f) => f.source_phase))].sort(),
    [findings]
  );
  const reportIds = useMemo(
    () => [...new Set(findings.map((f) => f.report_id).filter(Boolean))],
    [findings]
  );

  const filtered = useMemo(() => {
    const term = query.trim().toLowerCase();
    return findings.filter((f) => {
      if (sev !== "all" && f.severity !== sev) return false;
      if (rec !== "all" && f.recommendation !== rec) return false;
      if (type !== "all" && f.category !== type) return false;
      if (phase !== "all" && f.source_phase !== phase) return false;
      if (report !== "all" && f.report_id !== report) return false;
      if (term) {
        const hay = `${f.finding_id} ${f.affected_asset} ${f.category} ${f.source_phase} ${f.explanation} ${f.recommendation} ${f.report_id || ""}`;
        if (!hay.toLowerCase().includes(term)) return false;
      }
      return true;
    });
  }, [findings, sev, rec, type, phase, report, query]);

  const activeFilters =
    (sev !== "all" ? 1 : 0) + (rec !== "all" ? 1 : 0) + (type !== "all" ? 1 : 0) +
    (phase !== "all" ? 1 : 0) + (report !== "all" ? 1 : 0) + (query.trim() ? 1 : 0);

  const clearAll = () => {
    setSev("all");
    setRec("all");
    setType("all");
    setPhase("all");
    setReport("all");
    setQuery("");
  };

  return (
    <div className="space-y-4">
      {/* header + severity counters, in the reference's Active/Closed slot */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-[19px] font-semibold tracking-tight text-head">Findings</h1>
        <div className="flex flex-wrap items-center gap-4 text-[12.5px]">
          {SEVERITIES.map((s) => {
            const n =
              s === "all" ? findings.length : findings.filter((f) => f.severity === s).length;
            const active = sev === s;
            const tone =
              s === "high" ? "text-bad" : s === "medium" ? "text-warn" : s === "low" ? "text-ok" : "text-accent";
            return (
              <button
                key={s}
                onClick={() => setSev(s)}
                className={`transition-colors ${active ? tone : "text-muted hover:text-ink"}`}
              >
                {s === "all" ? "All" : s.charAt(0).toUpperCase() + s.slice(1)} {n}
              </button>
            );
          })}
        </div>
      </div>

      {/* control bar */}
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative min-w-[220px] flex-1">
          <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-muted" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search findings by id, asset, category or explanation…"
            className="w-full rounded-lg border border-line bg-panel2 py-2 pr-9 pl-9
                       text-[12.5px] text-ink outline-none transition-colors
                       placeholder:text-muted focus:border-accent/50"
          />
          {query && (
            <button
              onClick={() => setQuery("")}
              title="Clear search"
              className="absolute right-2 top-2 grid h-5 w-5 place-items-center rounded
                         text-muted hover:text-bad"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>

        <button
          onClick={() => setShowFilters((v) => !v)}
          className={`flex items-center gap-2 rounded-lg border bg-panel2 px-3 py-2
                      text-[12.5px] transition-colors ${
                        showFilters
                          ? "border-accent/50 text-accent"
                          : "border-line text-ink hover:border-accent/50"
                      }`}
        >
          Filter
          <SlidersHorizontal className="h-3.5 w-3.5" />
          {activeFilters > 0 && (
            <span className="mono text-[10.5px] text-accent">{activeFilters}</span>
          )}
        </button>

        <div className="ml-auto flex flex-wrap items-center gap-2">
          <Dropdown
            value={rec}
            onChange={setRec}
            className="w-40"
            options={[
              { value: "all", label: "Recommendation" },
              { value: "ACCEPT", label: "ACCEPT" },
              { value: "REVIEW", label: "REVIEW" },
              { value: "QUARANTINE", label: "QUARANTINE" },
            ]}
          />
          <Dropdown
            value={type}
            onChange={setType}
            className="w-44"
            options={[
              { value: "all", label: "Type" },
              ...categories.map((c) => ({ value: c, label: c })),
            ]}
          />
          <ExportMenu rows={filtered} />
        </div>
      </div>

      {showFilters && (
        <div className="panel flex flex-wrap items-center gap-2 p-3">
          <span className="text-[11.5px] text-muted">Narrow by</span>
          <Dropdown
            value={phase}
            onChange={setPhase}
            className="w-44"
            options={[
              { value: "all", label: "Any evidence layer" },
              ...phases.map((p) => ({ value: p, label: phaseLabel(p) })),
            ]}
          />
          <Dropdown
            value={report}
            onChange={setReport}
            className="w-56"
            options={[
              { value: "all", label: "Any assessment" },
              ...reportIds.map((r) => ({ value: r, label: r.slice(0, 20) })),
            ]}
          />
          {activeFilters > 0 && (
            <button
              onClick={clearAll}
              className="ml-auto flex items-center gap-1.5 rounded-lg border border-line
                         px-3 py-1.5 text-[12px] text-muted transition-colors
                         hover:border-bad hover:text-bad"
            >
              <X className="h-3.5 w-3.5" /> Clear all
            </button>
          )}
        </div>
      )}

      <Panel title={`${filtered.length} findings`}>
        {findings.length === 0 ? (
          <EmptyState title="No governance findings stored">
            Run an assurance assessment — findings are collected across every stored
            assessment.
          </EmptyState>
        ) : filtered.length === 0 ? (
          <EmptyState title="No findings match these filters">
            Clear the search or widen the severity, type and layer filters.
          </EmptyState>
        ) : (
          <FindingsTable
            findings={filtered}
            showReport
            toolbar={false}
            onOpenReport={onOpenReport}
          />
        )}
      </Panel>
    </div>
  );
}
