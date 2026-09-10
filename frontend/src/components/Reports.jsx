import React, { useMemo, useState } from "react";
import { ArrowRight, FileSearch, Search, X } from "lucide-react";
import { Dropdown, EmptyState, KV, Panel, Pill } from "./ui.jsx";

export default function Reports({ reports, selectedId, onOpen }) {
  const [q, setQ] = useState("");
  const [kind, setKind] = useState("all");

  const kinds = useMemo(() => [...new Set(reports.map((r) => r.kind))].sort(), [reports]);

  const filtered = useMemo(() => {
    const term = q.trim().toLowerCase();
    return reports.filter((r) => {
      if (kind !== "all" && r.kind !== kind) return false;
      if (term && !`${r.report_id} ${r.kind}`.toLowerCase().includes(term)) return false;
      return true;
    });
  }, [reports, q, kind]);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-[19px] font-semibold tracking-tight text-head">Reports</h1>
        <span className="text-[12.5px] text-muted">
          {filtered.length} of {reports.length} stored
        </span>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <div className="relative min-w-[220px] flex-1">
          <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-muted" />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search report id or kind…"
            className="w-full rounded-lg border border-line bg-panel2 py-2 pr-9 pl-9
                       text-[12.5px] text-ink outline-none transition-colors
                       placeholder:text-muted focus:border-accent/50"
          />
          {q && (
            <button
              onClick={() => setQ("")}
              title="Clear search"
              className="absolute right-2 top-2 grid h-5 w-5 place-items-center rounded
                         text-muted hover:text-bad"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
        <Dropdown
          value={kind}
          onChange={setKind}
          className="w-52"
          options={[
            { value: "all", label: "All report kinds" },
            ...kinds.map((k) => ({ value: k, label: k })),
          ]}
        />
      </div>

      <Panel title="Report store">
        {reports.length === 0 ? (
          <EmptyState title="No reports stored">
            Run any audit through the API to populate the store.
          </EmptyState>
        ) : filtered.length === 0 ? (
          <EmptyState title="No reports match this filter">
            Clear the search or pick another report kind.
          </EmptyState>
        ) : (
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 2xl:grid-cols-3">
            {filtered.map((r) => {
              const openable = r.kind === "assurance_assessment";
              const active = r.report_id === selectedId;
              return (
                <button
                  key={r.report_id}
                  onClick={() => openable && onOpen(r)}
                  className={`group rounded-xl border bg-panel2/60 p-3 text-left
                              transition-colors ${
                                active
                                  ? "border-accent/50"
                                  : "border-line hover:border-accent/40"
                              } ${openable ? "" : "cursor-default"}`}
                >
                  <div className="mb-2 flex items-center justify-between gap-2">
                    <span className="flex items-center gap-2 text-[12px] text-muted">
                      <FileSearch className="h-3.5 w-3.5" />
                      {active ? "open in overview" : "stored report"}
                    </span>
                    <Pill value={r.kind} />
                  </div>
                  <KV rows={[["report id", r.report_id.slice(0, 20)]]} />
                  {openable && (
                    <div className="mt-2 flex items-center gap-1 text-[11px] text-accent">
                      open in overview
                      <ArrowRight
                        className="h-3 w-3 transition-transform group-hover:translate-x-0.5"
                      />
                    </div>
                  )}
                </button>
              );
            })}
          </div>
        )}
      </Panel>
    </div>
  );
}
