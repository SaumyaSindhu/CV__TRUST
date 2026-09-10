import React, { useMemo, useRef, useState } from "react";
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, PolarAngleAxis,
  RadialBar, RadialBarChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { useTheme, tooltipProps } from "../theme.jsx";
import { EmptyState, Pill, SegTabs } from "./ui.jsx";

const SEV_ORDER = { high: 3, medium: 2, low: 1 };
const PHASE_LABEL = {
  phase2_dataset_integrity: "Dataset",
  phase3_model_integrity: "Model",
  phase4_inference_provenance: "Provenance",
  phase5_distribution_shift: "Distribution",
};
export const phaseLabel = (p) => PHASE_LABEL[p] || String(p || "—");

const sevColor = (C, s) => (s === "high" ? C.bad : s === "medium" ? C.warn : C.ok);
const riskColor = (C, risk) =>
  risk === "HIGH" ? C.bad : risk === "MEDIUM" ? C.warn : C.ok;

/* ------------------------------------------------------------------ *
 * Overall governance gauge (unchanged data contract)
 * ------------------------------------------------------------------ */
export function RiskGauge({ governance }) {
  const { C } = useTheme();
  const value = Math.round((governance.overall_confidence || 0) * 100);
  const color = riskColor(C, governance.overall_risk);
  return (
    <div className="relative h-44 w-44 shrink-0">
      <ResponsiveContainer>
        <RadialBarChart
          data={[{ name: "confidence", value, fill: color }]}
          innerRadius="74%"
          outerRadius="100%"
          startAngle={225}
          endAngle={-45}
        >
          <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
          <RadialBar dataKey="value" cornerRadius={10} background={{ fill: C.panel2 }} />
        </RadialBarChart>
      </ResponsiveContainer>
      <div className="pointer-events-none absolute inset-0 flex flex-col items-center
                      justify-center text-center">
        <div className="text-[10.5px] text-muted">overall risk</div>
        <div className="text-2xl font-semibold" style={{ color }}>
          {governance.overall_risk}
        </div>
        <div className="mono mt-0.5 text-[11px] text-muted">confidence {value}%</div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * Signal chart — the reference's "Normal and malicious traffic" card,
 * driven entirely by the real governance findings of the assessment.
 * ------------------------------------------------------------------ */
function SignalTooltip({ active, payload, C }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  const rows = d.finding
    ? [
        ["Severity", d.finding.severity],
        ["Confidence", d.finding.confidence],
        ["Phase", phaseLabel(d.finding.source_phase)],
        ["Recommendation", d.finding.recommendation],
      ]
    : [["Findings", d.value]];
  return (
    <div
      className="rounded-xl border px-3 py-2 text-[11.5px] shadow-2xl"
      style={{ background: C.tooltipBg, borderColor: C.line }}
    >
      <p className="mb-1.5 flex items-center gap-1.5 font-medium text-head">
        <span className="h-1.5 w-1.5 rounded-full" style={{ background: C.accent }} />
        {d.label}
      </p>
      {rows.map(([k, v]) => (
        <div key={k} className="flex items-center justify-between gap-8 py-0.5">
          <span className="text-muted">{k}</span>
          <span className="mono text-ink">{String(v)}</span>
        </div>
      ))}
      {d.finding && (
        <p className="mono mt-1.5 max-w-[240px] truncate text-[10.5px] text-muted">
          {String(d.finding.affected_asset || "").split("/").pop()}
        </p>
      )}
    </div>
  );
}

export function SignalChart({ findings }) {
  const { C } = useTheme();
  const [mode, setMode] = useState("confidence");
  const [hidden, setHidden] = useState({});

  const counts = useMemo(
    () => ({
      low: findings.filter((f) => f.severity === "low").length,
      medium: findings.filter((f) => f.severity === "medium").length,
      high: findings.filter((f) => f.severity === "high").length,
    }),
    [findings]
  );

  const visible = useMemo(
    () => findings.filter((f) => !hidden[f.severity]),
    [findings, hidden]
  );

  const series = useMemo(() => {
    if (mode === "confidence") {
      return [...visible]
        .sort((a, b) => String(a.timestamp).localeCompare(String(b.timestamp)))
        .map((f) => ({
          label: f.finding_id,
          value: Number(f.confidence ?? 0),
          fill: sevColor(C, f.severity),
          finding: f,
        }));
    }
    if (mode === "severity") {
      return ["low", "medium", "high"]
        .filter((s) => !hidden[s])
        .map((s) => ({
          label: s,
          value: counts[s],
          fill: sevColor(C, s),
        }));
    }
    const byPhase = new Map();
    visible.forEach((f) => {
      const k = phaseLabel(f.source_phase);
      byPhase.set(k, (byPhase.get(k) || 0) + 1);
    });
    return [...byPhase.entries()].map(([label, value]) => ({
      label,
      value,
      fill: C.accent,
    }));
  }, [mode, visible, counts, hidden, C]);

  const legend = [
    { key: "low", label: "Low", color: C.ok },
    { key: "medium", label: "Medium", color: C.warn },
    { key: "high", label: "High", color: C.bad },
  ];

  return (
    <div className="panel ring-soft overflow-hidden">
      <div className="flex flex-wrap items-start justify-between gap-3 px-4 pt-4 sm:px-5">
        <div>
          <h2 className="panel-title">Governance finding signal</h2>
          <div className="mt-2 flex flex-wrap items-center gap-4">
            {legend.map((l) => (
              <button
                key={l.key}
                onClick={() => setHidden((h) => ({ ...h, [l.key]: !h[l.key] }))}
                title={`Toggle ${l.label} severity`}
                className={`flex items-center gap-1.5 text-[11.5px] transition-opacity ${
                  hidden[l.key] ? "text-muted opacity-50" : "text-ink"
                }`}
              >
                <span className="flex items-center">
                  <span className="h-px w-2" style={{ background: l.color }} />
                  <span
                    className="h-1.5 w-1.5 rounded-full"
                    style={{ background: l.color }}
                  />
                  <span className="h-px w-2" style={{ background: l.color }} />
                </span>
                {l.label} {counts[l.key]}
              </button>
            ))}
          </div>
        </div>
        <SegTabs
          value={mode}
          onChange={setMode}
          options={[
            { value: "confidence", label: "Confidence" },
            { value: "severity", label: "Severity" },
            { value: "phase", label: "Phase" },
          ]}
        />
      </div>

      {series.length === 0 ? (
        <div className="px-5">
          <EmptyState title="No findings in this view">
            The executed evidence layers produced no findings at the selected severities.
          </EmptyState>
        </div>
      ) : (
        <>
          <div className="glow-accent relative mt-3 h-52">
            <ResponsiveContainer width="100%" height="100%">
              {mode === "confidence" ? (
                <AreaChart data={series} margin={{ top: 12, right: 16, left: 16, bottom: 0 }}>
                  <defs>
                    <linearGradient id="signalFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={C.accent} stopOpacity={0.4} />
                      <stop offset="100%" stopColor={C.accent} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="label" hide />
                  <YAxis hide domain={[0, 1]} />
                  <Tooltip
                    content={<SignalTooltip C={C} />}
                    cursor={{ stroke: C.muted, strokeDasharray: "3 3" }}
                  />
                  <Area
                    type="monotone"
                    dataKey="value"
                    stroke={C.accent}
                    strokeWidth={1.8}
                    fill="url(#signalFill)"
                    dot={{ r: 2.5, fill: C.accent, strokeWidth: 0 }}
                    activeDot={{ r: 5, fill: C.accent, stroke: C.panel, strokeWidth: 2 }}
                    isAnimationActive={false}
                  />
                </AreaChart>
              ) : (
                <BarChart data={series} margin={{ top: 16, right: 16, left: 8, bottom: 4 }}>
                  <CartesianGrid stroke={C.grid} strokeDasharray="3 3" vertical={false} />
                  <XAxis
                    dataKey="label"
                    tick={{ fill: C.muted, fontSize: 11 }}
                    axisLine={{ stroke: C.line }}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fill: C.muted, fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                    allowDecimals={false}
                  />
                  <Tooltip content={<SignalTooltip C={C} />} cursor={{ fill: `${C.accent}14` }} />
                  <Bar dataKey="value" radius={[6, 6, 0, 0]} isAnimationActive={false}>
                    {series.map((d, i) => (
                      <Cell key={i} fill={d.fill} />
                    ))}
                  </Bar>
                </BarChart>
              )}
            </ResponsiveContainer>
          </div>

          {/* dense confidence strip, mirroring the reference's mini bars */}
          <div className="h-12 px-4 pb-4 sm:px-5">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={visible.map((f) => ({
                  v: Number(f.confidence ?? 0),
                  fill: sevColor(C, f.severity),
                }))}
                margin={{ top: 0, right: 0, left: 0, bottom: 0 }}
                barCategoryGap={2}
              >
                <YAxis hide domain={[0, 1]} />
                <Bar dataKey="v" isAnimationActive={false} radius={[2, 2, 0, 0]}>
                  {visible.map((f, i) => (
                    <Cell key={i} fill={sevColor(C, f.severity)} fillOpacity={0.55} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * Finding distribution field — occupies the reference's map slot. The
 * dotted grid and orange markers are the same visual language, but the
 * marker positions are real finding confidence / severity values.
 * ------------------------------------------------------------------ */
export function FindingField({ findings, onOpenFinding }) {
  const { C } = useTheme();
  const [mode, setMode] = useState("confidence");
  const [hover, setHover] = useState(null);
  const box = useRef(null);

  const W = 600;
  const H = 250;
  const PAD_X = 46;
  const BANDS = { high: 62, medium: 125, low: 188 };

  const grid = useMemo(() => {
    const dots = [];
    for (let x = 18; x < W - 8; x += 14) {
      for (let y = 22; y < H - 14; y += 14) dots.push([x, y]);
    }
    return dots;
  }, []);

  const phases = useMemo(
    () => [...new Set(findings.map((f) => phaseLabel(f.source_phase)))],
    [findings]
  );

  const points = useMemo(
    () =>
      findings.map((f, i) => {
        const band = BANDS[f.severity] ?? BANDS.medium;
        const jitter = ((i % 5) - 2) * 7;
        let x;
        if (mode === "confidence") {
          x = PAD_X + Number(f.confidence ?? 0) * (W - PAD_X * 2);
        } else {
          const idx = Math.max(0, phases.indexOf(phaseLabel(f.source_phase)));
          const step = (W - PAD_X * 2) / Math.max(1, phases.length);
          x = PAD_X + step * (idx + 0.5) + ((i % 7) - 3) * 5;
        }
        return { f, x, y: band + jitter };
      }),
    [findings, mode, phases]
  );

  return (
    <div className="panel ring-soft p-4 sm:p-5">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <h2 className="panel-title">Finding distribution</h2>
        <SegTabs
          value={mode}
          onChange={setMode}
          options={[
            { value: "confidence", label: "Confidence" },
            { value: "phase", label: "Phase" },
          ]}
        />
      </div>

      {findings.length === 0 ? (
        <EmptyState title="No findings to plot">
          Every executed evidence layer returned a clean result.
        </EmptyState>
      ) : (
        <div ref={box} className="relative h-64 w-full">
          <svg
            viewBox={`0 0 ${W} ${H}`}
            className="h-full w-full"
            preserveAspectRatio="xMidYMid meet"
          >
            {grid.map(([cx, cy], i) => (
              <circle key={i} cx={cx} cy={cy} r={1.6} fill={C.line} />
            ))}

            {Object.entries(BANDS).map(([sev, y]) => (
              <g key={sev}>
                <line
                  x1={PAD_X - 12}
                  x2={W - 14}
                  y1={y}
                  y2={y}
                  stroke={C.line}
                  strokeDasharray="4 6"
                />
                <text x={4} y={y + 4} fontSize="9.5" fill={C.muted}>
                  {sev}
                </text>
              </g>
            ))}

            {mode === "confidence"
              ? [0, 0.25, 0.5, 0.75, 1].map((t) => (
                  <text
                    key={t}
                    x={PAD_X + t * (W - PAD_X * 2)}
                    y={H - 4}
                    fontSize="9.5"
                    fill={C.muted}
                    textAnchor="middle"
                  >
                    {t}
                  </text>
                ))
              : phases.map((p, i) => {
                  const step = (W - PAD_X * 2) / Math.max(1, phases.length);
                  return (
                    <text
                      key={p}
                      x={PAD_X + step * (i + 0.5)}
                      y={H - 4}
                      fontSize="9.5"
                      fill={C.muted}
                      textAnchor="middle"
                    >
                      {p}
                    </text>
                  );
                })}

            {points.map((p, i) => {
              const on = hover?.f.finding_id === p.f.finding_id;
              return (
                <g
                  key={`${p.f.report_id || ""}${p.f.finding_id}${i}`}
                  style={{ cursor: "pointer" }}
                  onMouseEnter={(e) => {
                    const r = box.current.getBoundingClientRect();
                    setHover({ ...p, mx: e.clientX - r.left, my: e.clientY - r.top });
                  }}
                  onMouseMove={(e) => {
                    const r = box.current.getBoundingClientRect();
                    setHover({ ...p, mx: e.clientX - r.left, my: e.clientY - r.top });
                  }}
                  onMouseLeave={() => setHover(null)}
                  onClick={() => onOpenFinding?.(p.f)}
                >
                  <circle
                    cx={p.x}
                    cy={p.y}
                    r={on ? 13 : 10}
                    fill={sevColor(C, p.f.severity)}
                    opacity={on ? 0.26 : 0.14}
                  />
                  <circle
                    cx={p.x}
                    cy={p.y}
                    r={on ? 5.5 : 4}
                    fill={sevColor(C, p.f.severity)}
                  />
                </g>
              );
            })}
          </svg>

          {hover && (
            <div
              className="panel pointer-events-none absolute z-30 w-56 p-3 shadow-2xl"
              style={{
                left: Math.min(hover.mx + 14, (box.current?.clientWidth || 400) - 240),
                top: Math.max(hover.my - 84, 2),
              }}
            >
              <div className="mono mb-1 text-[11.5px] text-accent">
                {hover.f.finding_id}
              </div>
              <div className="mb-2 line-clamp-2 text-[12px] text-ink">
                {hover.f.category}
              </div>
              <div className="flex items-center justify-between gap-2">
                <Pill value={hover.f.severity} />
                <span className="mono text-[11.5px] text-head">
                  {hover.f.confidence}
                </span>
              </div>
              <div className="mono mt-2 truncate text-[10.5px] text-muted">
                {String(hover.f.affected_asset || "").split("/").pop()}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * Phase 3 behavioural charts (data contract unchanged)
 * ------------------------------------------------------------------ */
export function ModelComparisonChart({ comparison }) {
  const { C } = useTheme();
  const data = [
    { name: "Matched", value: comparison.total_matched, fill: C.ok },
    { name: "Class flips", value: comparison.total_class_flips, fill: C.warn },
    { name: "Added", value: comparison.total_added, fill: C.bad },
    { name: "Missing", value: comparison.total_missing, fill: C.violet },
  ];
  return (
    <ResponsiveContainer width="100%" height={190}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
        <CartesianGrid stroke={C.grid} strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="name"
          tick={{ fill: C.muted, fontSize: 11 }}
          axisLine={{ stroke: C.line }}
          tickLine={false}
        />
        <YAxis
          tick={{ fill: C.muted, fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          allowDecimals={false}
        />
        <Tooltip {...tooltipProps(C)} />
        <Bar dataKey="value" radius={[6, 6, 0, 0]}>
          {data.map((d) => (
            <Cell key={d.name} fill={d.fill} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function ModelPerImageChart({ perImage }) {
  const { C } = useTheme();
  const data = (perImage || []).map((r) => ({
    name: r.image,
    Added: r.num_added,
    Missing: r.num_missing,
    "Class flips": r.num_class_flips,
  }));
  if (!data.length) return null;
  return (
    <ResponsiveContainer width="100%" height={190}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
        <CartesianGrid stroke={C.grid} strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="name"
          tick={{ fill: C.muted, fontSize: 10 }}
          axisLine={{ stroke: C.line }}
          tickLine={false}
        />
        <YAxis
          tick={{ fill: C.muted, fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          allowDecimals={false}
        />
        <Tooltip {...tooltipProps(C)} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Bar dataKey="Added" stackId="a" fill={C.bad} />
        <Bar dataKey="Missing" stackId="a" fill={C.violet} />
        <Bar dataKey="Class flips" stackId="a" fill={C.warn} radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

/* ------------------------------------------------------------------ *
 * Phase 5 distribution charts (data contract unchanged)
 * ------------------------------------------------------------------ */
export function DistributionSamplesChart({ perSample }) {
  const { C } = useTheme();
  const data = (perSample || []).map((s) => ({
    name: String(s.image).split("/").pop(),
    z: Number(s.z?.toFixed ? s.z.toFixed(3) : s.z),
    anomalous: s.anomalous,
  }));
  if (!data.length) return null;
  return (
    <ResponsiveContainer width="100%" height={210}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -14 }}>
        <CartesianGrid stroke={C.grid} strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="name"
          tick={{ fill: C.muted, fontSize: 9 }}
          interval={0}
          angle={-30}
          height={52}
          textAnchor="end"
          axisLine={{ stroke: C.line }}
          tickLine={false}
        />
        <YAxis
          tick={{ fill: C.muted, fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          label={{
            value: "z vs baseline",
            angle: -90,
            position: "insideLeft",
            fill: C.muted,
            fontSize: 10,
          }}
        />
        <Tooltip
          {...tooltipProps(C)}
          formatter={(v, _n, item) => [v, item?.payload?.anomalous ? "z (ANOMALOUS)" : "z"]}
        />
        <ReferenceLine y={0} stroke={C.muted} />
        <Bar dataKey="z" radius={[4, 4, 0, 0]}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.anomalous ? C.bad : C.accent} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function ScoreDial({ label, value, tone }) {
  const { C } = useTheme();
  const pct = Math.round((value || 0) * 100);
  const fill = tone || (pct >= 60 ? C.bad : pct >= 30 ? C.warn : C.ok);
  return (
    <div className="relative h-24 w-24">
      <ResponsiveContainer>
        <RadialBarChart
          data={[{ value: pct, fill }]}
          innerRadius="72%"
          outerRadius="100%"
          startAngle={90}
          endAngle={-270}
        >
          <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
          <RadialBar dataKey="value" cornerRadius={8} background={{ fill: C.panel2 }} />
        </RadialBarChart>
      </ResponsiveContainer>
      <div className="pointer-events-none absolute inset-0 flex flex-col items-center
                      justify-center">
        <div className="mono text-[15px] text-head">{(value ?? 0).toFixed(2)}</div>
        <div className="px-1 text-center text-[9.5px] leading-tight text-muted">{label}</div>
      </div>
    </div>
  );
}

export { SEV_ORDER };
