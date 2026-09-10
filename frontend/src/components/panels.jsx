import React, { useMemo, useRef, useState } from "react";
import {
  Box, CheckCircle2, ChevronDown, ChevronRight, Clipboard, Cog, ExternalLink,
  FileInput, FileOutput, Fingerprint, Gavel, KeyRound, Link2, MoreHorizontal,
  RotateCcw, XCircle,
} from "lucide-react";
import {
  EmptyState, KV, Pill, REC_STYLES, SEV_STYLES, useOutside,
} from "./ui.jsx";
import { phaseLabel } from "./charts.jsx";
import { shortHash, fmt } from "../api.js";

const copy = (text) => {
  try {
    navigator.clipboard?.writeText(text);
  } catch {
    /* clipboard unavailable — non-fatal */
  }
};

const shortDate = (ts) => {
  if (!ts) return "—";
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return String(ts).slice(0, 19).replace("T", " ");
  return d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
};

/* ------------------------------------------------------------------ *
 * Provenance hash chain
 * ------------------------------------------------------------------ */
function ChainNode({ icon: Icon, label, hash, status }) {
  const ok = ["valid", "new", "VALID"].includes(String(status));
  const warn = ["unsigned", "not_checked"].includes(String(status));
  const border =
    status === undefined
      ? "border-line"
      : ok
      ? "border-ok/50"
      : warn
      ? "border-warn/50"
      : "border-bad/60";
  return (
    <div
      className={`flex min-w-[140px] flex-1 flex-col gap-1.5 rounded-xl border ${border}
                  bg-panel2 px-3 py-2.5`}
    >
      <div className="flex items-center gap-2 text-[11px] text-muted">
        <Icon className="h-3.5 w-3.5" /> {label}
      </div>
      <div className="mono text-[11px] text-ink" title={hash || ""}>
        {hash ? shortHash(hash, 16) : "—"}
      </div>
      {status !== undefined && <Pill value={status} className="self-start" />}
    </div>
  );
}

export function ProvenanceChain({ section }) {
  if (!section) {
    return (
      <EmptyState title="Provenance not run">
        No inference record was supplied to this assessment.
      </EmptyState>
    );
  }
  const s = section.summary || {};
  const h = section.report?.recorded_hashes || {};
  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <Pill value={s.overall_status} />
        <span className="text-[12px] text-muted">
          canonical digests as recorded in the verified payload
        </span>
      </div>
      <div className="flex flex-col items-stretch gap-2 xl:flex-row xl:items-center">
        <ChainNode icon={FileInput} label="Input" hash={h.input_sha256} />
        <Link2 className="mx-auto h-4 w-4 shrink-0 rotate-90 text-muted xl:rotate-0" />
        <ChainNode icon={Box} label="Model" hash={h.model_sha256} />
        <Link2 className="mx-auto h-4 w-4 shrink-0 rotate-90 text-muted xl:rotate-0" />
        <ChainNode icon={Cog} label="Config" hash={h.config_sha256} />
        <Link2 className="mx-auto h-4 w-4 shrink-0 rotate-90 text-muted xl:rotate-0" />
        <ChainNode icon={FileOutput} label="Output" hash={h.output_sha256} />
      </div>
      <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-3">
        <ChainNode icon={Fingerprint} label="Hash integrity" status={s.hash_integrity} />
        <ChainNode icon={KeyRound} label="Signature / HMAC" status={s.signature_status} />
        <ChainNode icon={RotateCcw} label="Replay" status={s.replay_status} />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * Findings table — laid out exactly like the reference vulnerabilities
 * table (risk / type / description / status / change history / action).
 * ------------------------------------------------------------------ */
function RowActions({ f, showReport, onOpenReport, onToggle, expanded }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useOutside(ref, () => setOpen(false));

  const items = [
    {
      label: expanded ? "Collapse evidence" : "Expand evidence",
      icon: expanded ? ChevronDown : ChevronRight,
      run: onToggle,
    },
    { label: "Copy finding ID", icon: Clipboard, run: () => copy(f.finding_id) },
    {
      label: "Copy evidence JSON",
      icon: Clipboard,
      run: () => copy(JSON.stringify(f.evidence ?? {}, null, 2)),
    },
  ];
  if (showReport && f.report_id && onOpenReport) {
    items.push({
      label: "Open assessment",
      icon: ExternalLink,
      run: () => onOpenReport(f.report_id),
    });
  }

  return (
    <div className="relative flex justify-end" ref={ref}>
      <button
        onClick={(e) => {
          e.stopPropagation();
          setOpen((v) => !v);
        }}
        title="Actions"
        className="grid h-7 w-7 place-items-center rounded-md text-muted transition-colors
                   hover:bg-panel2 hover:text-ink"
      >
        <MoreHorizontal className="h-4 w-4" />
      </button>
      {open && (
        <div className="panel absolute right-0 top-8 z-30 w-52 p-1 shadow-2xl">
          {items.map((it) => (
            <button
              key={it.label}
              onClick={(e) => {
                e.stopPropagation();
                it.run();
                setOpen(false);
              }}
              className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-left
                         text-[12.5px] text-muted transition-colors hover:bg-panel2
                         hover:text-ink"
            >
              <it.icon className="h-3.5 w-3.5" />
              {it.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function FindingRow({ f, showReport, onOpenReport }) {
  const [open, setOpen] = useState(false);
  const toggle = () => setOpen((v) => !v);

  return (
    <>
      <tr
        onClick={toggle}
        className="cursor-pointer border-b border-line transition-colors hover:bg-panel2/60"
      >
        <td className="px-3 py-3 sm:px-4">
          <span
            className={`inline-flex items-center rounded-md border px-2 py-0.5 text-[11px]
                        font-medium ${SEV_STYLES[f.severity] || ""}`}
          >
            {f.severity}
          </span>
        </td>
        <td className="px-3 py-3 text-[12.5px] text-muted sm:px-4">{f.category}</td>
        <td className="px-3 py-3 sm:px-4">
          <p
            className="max-w-[420px] truncate text-[12.5px] text-head"
            title={f.explanation}
          >
            {f.explanation}
          </p>
          <p
            className="mono max-w-[420px] truncate text-[11px] text-muted"
            title={f.affected_asset}
          >
            {String(f.affected_asset || "").split("/").pop() || "—"}
          </p>
        </td>
        <td className="px-3 py-3 sm:px-4">
          <span
            className={`inline-flex items-center rounded-md border px-2 py-0.5 text-[11px]
                        font-medium ${REC_STYLES[f.recommendation] || ""}`}
          >
            {f.recommendation}
          </span>
        </td>
        <td className="px-3 py-3 text-right sm:px-4">
          <p className="text-[11px] text-muted">
            {phaseLabel(f.source_phase)}
            {showReport && f.report_id ? ` · ${String(f.report_id).slice(0, 8)}` : ""}
          </p>
          <p className="text-[12.5px] text-head">{shortDate(f.timestamp)}</p>
        </td>
        <td className="px-3 py-3 sm:px-4">
          <RowActions
            f={f}
            showReport={showReport}
            onOpenReport={onOpenReport}
            onToggle={toggle}
            expanded={open}
          />
        </td>
      </tr>
      {open && (
        <tr className="border-b border-line bg-panel2/40">
          <td colSpan={6} className="px-3 py-4 sm:px-4">
            <div className="grid gap-4 lg:grid-cols-2">
              <div>
                <div className="mb-1.5 text-[11px] text-muted">Explanation</div>
                <p className="text-[12.5px] leading-relaxed text-ink">{f.explanation}</p>
                {f.limitations?.length > 0 && (
                  <>
                    <div className="mt-3 mb-1.5 text-[11px] text-muted">Limitations</div>
                    <ul className="list-disc space-y-1 pl-4 text-[12px] text-muted">
                      {f.limitations.map((l, i) => (
                        <li key={i}>{l}</li>
                      ))}
                    </ul>
                  </>
                )}
              </div>
              <div>
                <div className="mb-1.5 text-[11px] text-muted">Evidence (raw)</div>
                <pre
                  className="mono max-h-56 overflow-auto rounded-xl border border-line bg-bg
                             p-3 text-[11px] leading-relaxed text-ink/90"
                >
{JSON.stringify(f.evidence, null, 2)}
                </pre>
                <div className="mono mt-2 break-all text-[11px] text-muted">
                  {f.finding_id} · {f.timestamp} · confidence {f.confidence} · asset{" "}
                  {f.affected_asset}
                </div>
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

export function FindingsTable({
  findings, showReport = false, toolbar = true, onOpenReport,
}) {
  const [sev, setSev] = useState("all");
  const rows = useMemo(
    () => (toolbar ? findings.filter((f) => sev === "all" || f.severity === sev) : findings),
    [findings, sev, toolbar]
  );

  if (!findings.length) {
    return (
      <EmptyState title="No findings recorded">
        Executed phases produced no findings for this scope.
      </EmptyState>
    );
  }

  return (
    <div>
      {toolbar && (
        <div className="mb-3 flex flex-wrap items-center gap-1.5">
          {["all", "low", "medium", "high"].map((s) => {
            const n =
              s === "all" ? findings.length : findings.filter((f) => f.severity === s).length;
            return (
              <button
                key={s}
                onClick={() => setSev(s)}
                className={`rounded-lg border px-3 py-1 text-[11.5px] transition-colors ${
                  sev === s
                    ? "border-accent/50 bg-accent/10 text-accent"
                    : "border-line text-muted hover:text-ink"
                }`}
              >
                {s} {n}
              </button>
            );
          })}
        </div>
      )}
      <div className="overflow-x-auto">
        <table className="w-full min-w-[860px] border-collapse text-left">
          <thead>
            <tr className="border-y border-line text-[11px] font-normal text-muted">
              <th className="px-3 py-2.5 font-normal sm:px-4">Risk level</th>
              <th className="px-3 py-2.5 font-normal sm:px-4">Type</th>
              <th className="px-3 py-2.5 font-normal sm:px-4">Finding</th>
              <th className="px-3 py-2.5 font-normal sm:px-4">Status</th>
              <th className="px-3 py-2.5 text-right font-normal sm:px-4">Change history</th>
              <th className="px-3 py-2.5 text-right font-normal sm:px-4">Action</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((f, i) => (
              <FindingRow
                key={`${f.report_id || ""}${f.finding_id}${i}`}
                f={f}
                showReport={showReport}
                onOpenReport={onOpenReport}
              />
            ))}
          </tbody>
        </table>
      </div>
      {rows.length === 0 && (
        <EmptyState title="No findings match this filter">
          Widen the severity filter to see the rest of the evidence.
        </EmptyState>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * Analyst decision workflow (POST /api/reports/{id}/decision)
 * ------------------------------------------------------------------ */
export function DecisionPanel({ governance, onDecide, busy }) {
  const [analyst, setAnalyst] = useState("");
  const [note, setNote] = useState("");
  const [error, setError] = useState(null);
  const decided = governance.analyst_decision;

  const submit = async (decision) => {
    setError(null);
    if (!analyst.trim()) {
      setError("Analyst identifier is required.");
      return;
    }
    try {
      await onDecide({ decision, analyst: analyst.trim(), note: note.trim() });
    } catch (e) {
      setError(String(e.message || e));
    }
  };

  if (decided) {
    return (
      <div className="flex items-start gap-3 rounded-xl border border-line bg-panel2 p-3">
        <Gavel className="mt-0.5 h-4 w-4 shrink-0 text-accent" />
        <div className="text-[12.5px]">
          <div className="flex flex-wrap items-center gap-1.5">
            Final analyst decision:
            <span
              className={`rounded-md border px-2 py-0.5 text-[11px] font-medium
                          ${REC_STYLES[decided.decision] || ""}`}
            >
              {decided.decision}
            </span>
            by <b className="text-head">{decided.analyst}</b>
          </div>
          <div className="mono mt-1 text-[11px] text-muted">
            {decided.decided_at}
            {decided.note ? ` — ${decided.note}` : ""} · audit-logged
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-2.5">
      <div className="grid gap-2 sm:grid-cols-2">
        <input
          value={analyst}
          onChange={(e) => setAnalyst(e.target.value)}
          placeholder="analyst id (required)"
          className="rounded-lg border border-line bg-panel2 px-3 py-2 text-[12.5px]
                     text-ink outline-none transition-colors placeholder:text-muted
                     focus:border-accent/50"
        />
        <input
          value={note}
          onChange={(e) => setNote(e.target.value)}
          placeholder="note (optional)"
          className="rounded-lg border border-line bg-panel2 px-3 py-2 text-[12.5px]
                     text-ink outline-none transition-colors placeholder:text-muted
                     focus:border-accent/50"
        />
      </div>
      <div className="flex flex-wrap gap-2">
        {[
          ["ACCEPT", "border-ok/50 text-ok hover:bg-ok/10", CheckCircle2],
          ["REVIEW", "border-warn/50 text-warn hover:bg-warn/10", Gavel],
          ["QUARANTINE", "border-bad/50 text-bad hover:bg-bad/10", XCircle],
        ].map(([d, cls, Icon]) => (
          <button
            key={d}
            disabled={busy}
            onClick={() => submit(d)}
            className={`flex items-center gap-2 rounded-lg border px-4 py-2 text-[12px]
                        font-medium transition-colors disabled:opacity-40 ${cls}`}
          >
            <Icon className="h-4 w-4" /> {d}
          </button>
        ))}
      </div>
      <p className="text-[11.5px] leading-relaxed text-muted">
        The decision is written into the stored report and appended to the tamper-evident
        audit chain.
      </p>
      {error && <div className="mono text-[12px] text-bad">{error}</div>}
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * Audit timeline
 * ------------------------------------------------------------------ */
export function AuditTimeline({ log }) {
  if (!log) return null;
  if (!log.entries.length) {
    return (
      <EmptyState title="Audit log is empty">
        Run an assurance assessment to create the first chained entry.
      </EmptyState>
    );
  }
  const badIdx = new Set((log.chain_issues || []).map((i) => i.entry_index));
  return (
    <div>
      <div
        className={`mono mb-4 inline-flex items-center gap-2 rounded-lg border px-3 py-1.5
                    text-[12px] ${
                      log.chain_valid
                        ? "border-ok/50 bg-ok/10 text-ok"
                        : "border-bad/50 bg-bad/10 text-bad"
                    }`}
      >
        <span className={`h-2 w-2 rounded-full ${log.chain_valid ? "bg-ok" : "bg-bad"}`} />
        {log.chain_valid
          ? `HASH CHAIN INTACT — ${log.num_entries} entries`
          : `HASH CHAIN BROKEN — ${log.chain_issues.length} issue(s)`}
      </div>
      <ol className="relative ml-3 border-l border-line">
        {log.entries.map((e) => {
          const bad = badIdx.has(e.entry_index);
          return (
            <li key={e.entry_index} className="mb-5 ml-5">
              <span
                className={`absolute -left-[7px] mt-1 grid h-3.5 w-3.5 place-items-center
                            rounded-full border-2 ${
                              bad ? "border-bad bg-bad/30" : "border-ok bg-panel"
                            }`}
              />
              <div className="flex flex-wrap items-center gap-2">
                <span className="mono text-[11px] text-muted">#{e.entry_index}</span>
                <span className="mono text-[12px] text-head">{e.record_id}</span>
                <Pill
                  value={
                    String(e.record_id).endsWith("-decision")
                      ? "analyst_decision"
                      : "assessment"
                  }
                />
                {bad && <Pill value="chain violation" />}
              </div>
              <div className="mono mt-1 text-[11px] text-muted">{e.logged_at}</div>
              <div className="mt-2 max-w-xl">
                <KV
                  rows={[
                    ["record digest", shortHash(e.record_digest, 22)],
                    ["entry hash", shortHash(e.entry_hash, 22)],
                    ["previous", shortHash(e.previous_entry_hash, 22)],
                  ]}
                />
              </div>
            </li>
          );
        })}
      </ol>
      <div className="mono text-[11px] break-all text-muted">log: {fmt(log.log_path)}</div>
    </div>
  );
}
