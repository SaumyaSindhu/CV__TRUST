import React, { useEffect, useRef, useState } from "react";
import {
  AlertTriangle, ArrowDown, ArrowUp, Check, ChevronDown, Inbox, Loader2, RefreshCw,
} from "lucide-react";

/* ------------------------------------------------------------------ *
 * Status vocabulary (unchanged semantics from the original console)
 * ------------------------------------------------------------------ */
export const SEV_STYLES = {
  low: "text-ok border-ok/40 bg-ok/10",
  medium: "text-warn border-warn/40 bg-warn/10",
  high: "text-bad border-bad/40 bg-bad/10",
};

export const REC_STYLES = {
  ACCEPT: "text-ok border-ok/40 bg-ok/10",
  REVIEW: "text-warn border-warn/40 bg-warn/10",
  QUARANTINE: "text-bad border-bad/40 bg-bad/10",
};

const STATUS_STYLE = (s) => {
  const v = String(s || "").toLowerCase();
  if (["low", "clean", "valid", "accept", "new", "ok"].includes(v)) return SEV_STYLES.low;
  if (["medium", "review", "file_changed_only", "unsigned", "not_checked"].includes(v))
    return SEV_STYLES.medium;
  if (v) return SEV_STYLES.high;
  return "text-muted border-line bg-panel2";
};

/* ------------------------------------------------------------------ *
 * Hooks
 * ------------------------------------------------------------------ */
export function useOutside(ref, onClose) {
  useEffect(() => {
    function handler(e) {
      if (ref.current && !ref.current.contains(e.target)) onClose();
    }
    function esc(e) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("mousedown", handler);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", handler);
      document.removeEventListener("keydown", esc);
    };
  }, [ref, onClose]);
}

/* ------------------------------------------------------------------ *
 * Atoms
 * ------------------------------------------------------------------ */
export function Pill({ value, className = "", title }) {
  return (
    <span
      title={title}
      className={`inline-flex items-center rounded-md border px-2 py-0.5 text-[11px]
                  leading-4 font-medium ${STATUS_STYLE(value)} ${className}`}
    >
      {value ?? "—"}
    </span>
  );
}

/** Percentage/level delta with the reference's arrow treatment. */
export function Delta({ value, tone = "up" }) {
  const good = tone === "up";
  const Icon = good ? ArrowUp : ArrowDown;
  return (
    <span
      className={`inline-flex items-center gap-0.5 text-[11px] font-medium
                  ${good ? "text-ok" : "text-bad"}`}
    >
      {value}
      <Icon className="h-3 w-3" />
    </span>
  );
}

export function Panel({ title, right, children, className = "", id }) {
  return (
    <section id={id} className={`panel ring-soft p-4 sm:p-5 ${className}`}>
      {(title || right) && (
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <h2 className="panel-title">{title}</h2>
          {right}
        </div>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, tone = "text-head" }) {
  return (
    <div className="min-w-[92px]">
      <div className="subtle">{label}</div>
      <div className={`mono mt-0.5 text-[15px] ${tone}`}>{value ?? "—"}</div>
    </div>
  );
}

export function KV({ rows }) {
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-[12.5px]">
      {rows.map(([k, v]) => (
        <React.Fragment key={k}>
          <dt className="text-muted">{k}</dt>
          <dd className="mono break-all text-right text-ink">{v ?? "—"}</dd>
        </React.Fragment>
      ))}
    </dl>
  );
}

/** Day / Monthly / Yearly-style segmented control. */
export function SegTabs({ options, value, onChange, size = "sm" }) {
  return (
    <div className="inline-flex rounded-lg border border-line bg-panel2 p-0.5">
      {options.map((o) => {
        const val = typeof o === "string" ? o : o.value;
        const label = typeof o === "string" ? o : o.label;
        const active = val === value;
        return (
          <button
            key={val}
            onClick={() => onChange(val)}
            className={`rounded-md transition-colors ${
              size === "sm" ? "px-3 py-1 text-[11.5px]" : "px-3.5 py-1.5 text-[12.5px]"
            } ${
              active
                ? "border border-line bg-panel text-head"
                : "text-muted hover:text-ink"
            }`}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}

/** Reference-style dropdown ("All applications", "Risk Level", …). */
export function Dropdown({
  label, value, options, onChange, icon: Icon, align = "right",
  compact = false, className = "", emptyLabel = "none",
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useOutside(ref, () => setOpen(false));

  const current = options.find((o) => o.value === value);
  const text = current ? current.label : value || label || emptyLabel;

  return (
    <div className={`relative ${className}`} ref={ref}>
      <button
        onClick={() => setOpen((v) => !v)}
        title={text}
        className={`flex w-full items-center gap-2 rounded-lg border border-line bg-panel2
                    text-ink transition-colors hover:border-accent/50 ${
                      compact ? "px-2.5 py-1.5 text-[11.5px]" : "px-3 py-2 text-[12.5px]"
                    }`}
      >
        {Icon && <Icon className="h-3.5 w-3.5 shrink-0 text-muted" />}
        <span className="min-w-0 flex-1 truncate text-left">{text}</span>
        <ChevronDown
          className={`h-3.5 w-3.5 shrink-0 text-muted transition-transform ${
            open ? "rotate-180" : ""
          }`}
        />
      </button>
      {open && (
        <div
          className={`panel absolute z-40 mt-2 max-h-72 min-w-full overflow-y-auto p-1
                      shadow-2xl ${align === "right" ? "right-0" : "left-0"}`}
        >
          {options.length === 0 && (
            <div className="px-3 py-2 text-[12px] text-muted">{emptyLabel}</div>
          )}
          {options.map((o) => (
            <button
              key={o.value}
              onClick={() => {
                onChange(o.value);
                setOpen(false);
              }}
              className={`flex w-full items-center justify-between gap-6 rounded-md px-3 py-2
                          text-left text-[12.5px] transition-colors hover:bg-panel2 ${
                            o.value === value ? "text-head" : "text-muted"
                          }`}
            >
              <span className="mono min-w-0 flex-1 truncate">{o.label}</span>
              {o.value === value && <Check className="h-3.5 w-3.5 shrink-0 text-accent" />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export function IconButton({ icon: Icon, onClick, title, active, spinning, className = "" }) {
  return (
    <button
      onClick={onClick}
      title={title}
      aria-label={title}
      className={`grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-line
                  bg-panel2 text-muted transition-colors hover:border-accent/50
                  hover:text-accent ${active ? "border-accent/50 text-accent" : ""} ${className}`}
    >
      <Icon className={`h-4 w-4 ${spinning ? "animate-spin" : ""}`} />
    </button>
  );
}

export function PrimaryButton({ icon: Icon, children, onClick, disabled, className = "" }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`flex items-center gap-1.5 rounded-lg border border-accent/50 bg-accent/10
                  px-3 py-2 text-[12.5px] font-medium text-accent transition-colors
                  hover:bg-accent/20 disabled:cursor-not-allowed disabled:opacity-40
                  ${className}`}
    >
      {Icon && <Icon className="h-4 w-4" />}
      {children}
    </button>
  );
}

/* ------------------------------------------------------------------ *
 * States
 * ------------------------------------------------------------------ */
export function LoadingState({ label = "Querying backend…" }) {
  return (
    <div className="flex h-64 flex-col items-center justify-center gap-3 text-muted">
      <Loader2 className="h-7 w-7 animate-spin text-accent" />
      <div className="text-sm">{label}</div>
    </div>
  );
}

export function ErrorState({ error, onRetry }) {
  return (
    <div className="panel mb-4 flex items-start gap-3 border-bad/40 bg-bad/5 p-4">
      <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-bad" />
      <div className="min-w-0 flex-1">
        <div className="text-[13px] font-medium text-bad">Backend request failed</div>
        <div className="mono mt-1 break-all text-[12px] text-ink/90">{String(error)}</div>
      </div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="flex shrink-0 items-center gap-1.5 rounded-lg border border-line
                     px-3 py-1.5 text-[12px] text-ink transition-colors
                     hover:border-accent hover:text-accent"
        >
          <RefreshCw className="h-3.5 w-3.5" /> Retry
        </button>
      )}
    </div>
  );
}

export function EmptyState({ title, children }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-12 text-center">
      <div className="grid h-11 w-11 place-items-center rounded-full border border-line bg-panel2">
        <Inbox className="h-5 w-5 text-muted" />
      </div>
      <div className="text-[13.5px] font-medium text-head">{title}</div>
      <div className="max-w-xl text-[12.5px] leading-relaxed text-muted">{children}</div>
    </div>
  );
}

export function SkeletonRows({ n = 4 }) {
  return (
    <div className="space-y-2">
      {Array.from({ length: n }).map((_, i) => (
        <div key={i} className="h-9 animate-pulse rounded-lg bg-panel2" />
      ))}
    </div>
  );
}
