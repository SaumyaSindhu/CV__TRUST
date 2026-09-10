import React, { useMemo, useState } from "react";
import {
  ArrowDown, ArrowRight, ArrowUp, Boxes, Database, Gauge, Gavel, GitCompareArrows,
  Layers, ListChecks, Plus, Radar, ScrollText,
} from "lucide-react";
import { fmt, shortHash } from "../api.js";
import { useTheme } from "../theme.jsx";
import {
  DistributionSamplesChart, FindingField, ModelComparisonChart, ModelPerImageChart,
  RiskGauge, ScoreDial, SignalChart, phaseLabel,
} from "./charts.jsx";
import { DecisionPanel, FindingsTable, ProvenanceChain } from "./panels.jsx";
import {
  Delta, Dropdown, EmptyState, KV, Panel, Pill, PrimaryButton, SegTabs, Stat,
} from "./ui.jsx";

const LAYERS = [
  { key: "dataset", label: "Dataset Integrity", phase: "phase2_dataset_integrity", icon: Database },
  { key: "model", label: "Model Integrity", phase: "phase3_model_integrity", icon: Boxes },
  { key: "provenance", label: "Provenance", phase: "phase4_inference_provenance", icon: GitCompareArrows },
  { key: "distribution", label: "Distribution Shift", phase: "phase5_distribution_shift", icon: Radar },
];

const scrollTo = (id) =>
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });

/* ------------------------------------------------------------------ *
 * "Summary for a period" analogue — every row is a real metric.
 * ------------------------------------------------------------------ */
function SummaryRow({ icon: Icon, tone, value, delta, deltaTone, label, onClick }) {
  return (
    <button
      onClick={onClick}
      className="group flex w-full items-center gap-3 rounded-xl border border-line
                 bg-panel2/60 p-3 text-left transition-colors hover:border-accent/40"
    >
      <span
        className="grid h-9 w-9 shrink-0 place-items-center rounded-full border"
        style={{ borderColor: tone, background: `${tone}1a` }}
      >
        <Icon className="h-4 w-4" style={{ color: tone }} />
      </span>
      <span className="min-w-0">
        <span className="flex items-center gap-2">
          <span className="truncate text-[17px] leading-none font-semibold text-head">
            {value}
          </span>
          {delta && <Delta value={delta} tone={deltaTone} />}
        </span>
        <span className="mt-1 block truncate text-[11.5px] text-muted">{label}</span>
      </span>
      <ArrowRight
        className="ml-auto h-4 w-4 shrink-0 text-muted transition-transform
                   group-hover:translate-x-0.5"
      />
    </button>
  );
}

function SummaryCard({ rep, findings, audit, onGoFindings, onGoAudit }) {
  const { C } = useTheme();
  const g = rep.governance;
  const high = findings.filter((f) => f.severity === "high").length;
  const ran = LAYERS.filter((l) => rep.sections[l.key]).length;
  const chainOk = audit ? audit.chain_valid : null;
  const riskTone = g.overall_risk === "HIGH" ? C.bad : g.overall_risk === "MEDIUM" ? C.warn : C.ok;

  return (
    <Panel title="Summary for this assessment" className="fade-up-1">
      <div className="space-y-3">
        <SummaryRow
          icon={Gauge}
          tone={riskTone}
          value={g.overall_risk}
          delta={`${Math.round((g.overall_confidence || 0) * 100)}%`}
          deltaTone={g.overall_risk === "LOW" ? "up" : "down"}
          label="Overall risk · governance confidence"
          onClick={() => scrollTo("verdict")}
        />
        <SummaryRow
          icon={ListChecks}
          tone={high > 0 ? C.bad : C.accent}
          value={findings.length}
          delta={high > 0 ? `${high} high` : null}
          deltaTone="down"
          label="Findings in this assessment"
          onClick={onGoFindings}
        />
        <SummaryRow
          icon={Layers}
          tone={C.info}
          value={`${ran} / 4`}
          label="Evidence layers executed"
          onClick={() => scrollTo("layers")}
        />
        <SummaryRow
          icon={ScrollText}
          tone={chainOk === false ? C.bad : C.ok}
          value={`#${fmt(rep.audit?.entry_index)}`}
          delta={chainOk === false ? "broken" : chainOk ? "intact" : null}
          deltaTone={chainOk === false ? "down" : "up"}
          label="Tamper-evident audit entry"
          onClick={onGoAudit}
        />
      </div>
    </Panel>
  );
}

/* ------------------------------------------------------------------ *
 * "Attack targets" analogue — the four evidence layers, sortable.
 * ------------------------------------------------------------------ */
function EvidenceLayersTable({ rep, findings }) {
  const [sort, setSort] = useState({ key: "findings", dir: "desc" });

  const rows = useMemo(() => {
    const list = LAYERS.map((l) => {
      const section = rep.sections[l.key];
      const mine = findings.filter((f) => f.source_phase === l.phase);
      return {
        ...l,
        section,
        findings: mine.length,
        high: mine.filter((f) => f.severity === "high").length,
        status: section ? section.status : null,
        ran: Boolean(section),
      };
    });
    list.sort((a, b) => {
      const [av, bv] =
        sort.key === "layer"
          ? [a.label, b.label]
          : sort.key === "high"
          ? [a.high, b.high]
          : [a.findings, b.findings];
      if (av < bv) return sort.dir === "asc" ? -1 : 1;
      if (av > bv) return sort.dir === "asc" ? 1 : -1;
      return 0;
    });
    return list;
  }, [rep, findings, sort]);

  const Head = ({ k, label, right }) => (
    <th
      onClick={() =>
        setSort((s) => ({ key: k, dir: s.key === k && s.dir === "desc" ? "asc" : "desc" }))
      }
      className={`cursor-pointer px-3 py-2 text-[11px] font-normal text-muted select-none ${
        right ? "text-right" : "text-left"
      }`}
    >
      <span className="inline-flex items-center gap-1">
        {label}
        {sort.key === k &&
          (sort.dir === "desc" ? (
            <ArrowDown className="h-3 w-3" />
          ) : (
            <ArrowUp className="h-3 w-3" />
          ))}
      </span>
    </th>
  );

  return (
    <Panel title="Evidence layers" className="fade-up-2">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[420px] border-collapse">
          <thead>
            <tr className="border-b border-line">
              <Head k="layer" label="Layer" />
              <Head k="findings" label="Findings" right />
              <Head k="high" label="High" right />
              <th className="px-3 py-2 text-right text-[11px] font-normal text-muted">
                Status
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr
                key={r.key}
                onClick={() => scrollTo(`layer-${r.key}`)}
                className="cursor-pointer border-b border-line/70 transition-colors
                           hover:bg-panel2/60"
              >
                <td className="px-3 py-2.5">
                  <span className="flex items-center gap-2">
                    <r.icon
                      className={`h-3.5 w-3.5 ${r.ran ? "text-accent" : "text-muted"}`}
                    />
                    <span
                      className={`rounded px-1.5 py-0.5 text-[12.5px] ${
                        r.ran ? "bg-panel2 text-head" : "text-muted"
                      }`}
                    >
                      {r.label}
                    </span>
                  </span>
                </td>
                <td className="px-3 py-2.5 text-right text-[12.5px] text-ink">
                  {r.ran ? r.findings : <span className="text-muted">--</span>}
                </td>
                <td className="px-3 py-2.5 text-right text-[12.5px]">
                  {r.high > 0 ? (
                    <span className="text-bad">{r.high}</span>
                  ) : (
                    <span className="text-muted">--</span>
                  )}
                </td>
                <td className="px-3 py-2.5 text-right">
                  <Pill value={r.status || "not run"} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-[11.5px] leading-relaxed text-muted">
        Select a layer to jump to its evidence card. Layers with no input are skipped by the
        engine rather than failed.
      </p>
    </Panel>
  );
}

/* ------------------------------------------------------------------ *
 * Layer evidence card
 * ------------------------------------------------------------------ */
function LayerCard({ id, icon: Icon, title, section, children }) {
  return (
    <Panel
      id={id}
      className="fade-up-2"
      title={
        <span className="flex items-center gap-2">
          <Icon className="h-4 w-4 text-accent" /> {title}
        </span>
      }
      right={<Pill value={section ? section.status : "not run"} />}
    >
      {section ? (
        children
      ) : (
        <p className="py-4 text-[12.5px] text-muted">
          This evidence layer was not part of the assessment.
        </p>
      )}
    </Panel>
  );
}

/* ------------------------------------------------------------------ *
 * Overview page
 * ------------------------------------------------------------------ */
export default function Overview({
  env, onDecide, busy, onNew, assessments, selectedId, onSelect, audit,
  onGoFindings, onGoAudit,
}) {
  const { C } = useTheme();
  const [layer, setLayer] = useState("all");

  if (!env) {
    return (
      <div className="panel ring-soft fade-up p-6">
        <EmptyState title="No assurance assessments stored yet">
          <div className="mb-3 flex justify-center">
            <PrimaryButton icon={Plus} onClick={onNew}>
              New assessment
            </PrimaryButton>
          </div>
          …or from the CLI:{" "}
          <code className="mono text-accent">python run_assurance.py</code> /{" "}
          <code className="mono text-accent">POST /api/assurance/run</code>.
        </EmptyState>
      </div>
    );
  }

  const rep = env.report;
  const { governance: g, sections, assets, findings } = rep;
  const model = sections.model;
  const dist = sections.distribution;
  const distEv = dist?.report?.evidence || {};

  const layerPhase = LAYERS.find((l) => l.key === layer)?.phase;
  const scoped = layerPhase ? findings.filter((f) => f.source_phase === layerPhase) : findings;

  return (
    <div className="space-y-4">
      {/* page header — mirrors the reference's title + two selectors */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-[19px] font-semibold tracking-tight text-head">
            Assurance Overview
          </h1>
          <p className="mono mt-0.5 text-[11px] text-muted">
            assessment {String(rep.assessment_id).slice(0, 16)} ·{" "}
            {rep.run_metadata.finished_at}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Dropdown
            icon={Layers}
            value={layer}
            onChange={setLayer}
            options={[
              { value: "all", label: "All layers" },
              ...LAYERS.map((l) => ({ value: l.key, label: l.label })),
            ]}
            className="w-44"
          />
          <Dropdown
            icon={ScrollText}
            value={selectedId || ""}
            onChange={onSelect}
            emptyLabel="no assessments stored"
            options={assessments.map((r) => ({
              value: r.report_id,
              label: r.report_id.slice(0, 16),
            }))}
            className="w-52"
          />
        </div>
      </div>

      {/* row A — signal chart + summary */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <div className="fade-up xl:col-span-2">
          <SignalChart findings={scoped} />
        </div>
        <SummaryCard
          rep={rep}
          findings={findings}
          audit={audit}
          onGoFindings={onGoFindings}
          onGoAudit={onGoAudit}
        />
      </div>

      {/* row B — distribution field + evidence layer table */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <div className="fade-up-2">
          <FindingField findings={scoped} onOpenFinding={onGoFindings} />
        </div>
        <EvidenceLayersTable rep={rep} findings={findings} />
      </div>

      {/* row C — governance verdict + analyst decision */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <Panel id="verdict" className="fade-up-2 xl:col-span-2">
          <div className="flex flex-col items-center gap-6 lg:flex-row">
            <RiskGauge governance={g} />
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-[11px] text-muted">recommendation</span>
                <Pill value={g.recommendation} className="px-3 py-1 text-[12.5px]" />
              </div>
              <ul className="mt-3 space-y-1.5">
                {g.decision_reasons.map((r, i) => (
                  <li
                    key={i}
                    className="border-l-2 border-line pl-3 text-[12.5px] leading-relaxed
                               text-ink"
                  >
                    {r}
                  </li>
                ))}
              </ul>
              <div className="mt-4 flex flex-wrap gap-x-8 gap-y-3">
                <Stat label="findings" value={findings.length} />
                <Stat label="dataset hash" value={shortHash(assets.dataset_hash, 10)} />
                <Stat label="model hash" value={shortHash(assets.model_hash, 10)} />
                <Stat label="config hash" value={shortHash(assets.configuration_hash, 10)} />
                <Stat label="audit entry" value={`#${fmt(rep.audit?.entry_index)}`} />
              </div>
            </div>
          </div>
        </Panel>

        <Panel
          title={
            <span className="flex items-center gap-2">
              <Gavel className="h-4 w-4 text-accent" /> Analyst decision
            </span>
          }
          className="fade-up-2"
        >
          <DecisionPanel governance={g} onDecide={onDecide} busy={busy} />
        </Panel>
      </div>

      {/* row D — the four evidence layers */}
      <div id="layers" className="grid grid-cols-1 gap-4 md:grid-cols-2 2xl:grid-cols-4">
        <LayerCard
          id="layer-dataset"
          icon={Database}
          title="Dataset Integrity"
          section={sections.dataset}
        >
          {sections.dataset && (
            <KV
              rows={[
                ["valid samples", fmt(sections.dataset.summary.valid_samples)],
                ["invalid samples", fmt(sections.dataset.summary.invalid_samples)],
                ["exact dup groups", fmt(sections.dataset.summary.exact_duplicate_groups)],
                ["near dup groups", fmt(sections.dataset.summary.near_duplicate_groups)],
                ["label anomalies", fmt(sections.dataset.summary.label_anomalies)],
                [
                  "OOD anomalies",
                  fmt(sections.dataset.summary.potential_ood_anomalous_samples),
                ],
              ]}
            />
          )}
        </LayerCard>

        <LayerCard id="layer-model" icon={Boxes} title="Model Integrity" section={model}>
          {model && (
            <KV
              rows={[
                [
                  "fingerprint",
                  model.report.fingerprint.changed === false
                    ? "unchanged"
                    : model.report.fingerprint.changed === true
                    ? "CHANGED"
                    : "—",
                ],
                ["agreement rate", fmt(model.report.comparison?.agreement_rate)],
                ["mean conf diff", fmt(model.report.comparison?.mean_confidence_diff)],
                ["class flips", fmt(model.report.comparison?.total_class_flips)],
                [
                  "added / missing",
                  `${fmt(model.report.comparison?.total_added)} / ${fmt(
                    model.report.comparison?.total_missing
                  )}`,
                ],
                [
                  "trojan indicator",
                  model.status === "possible_trojan_indicator" ? "INDICATED" : "none",
                ],
              ]}
            />
          )}
        </LayerCard>

        <LayerCard
          id="layer-provenance"
          icon={GitCompareArrows}
          title="Provenance"
          section={sections.provenance}
        >
          {sections.provenance && (
            <KV
              rows={[
                ["overall", fmt(sections.provenance.summary.overall_status)],
                ["hash integrity", fmt(sections.provenance.summary.hash_integrity)],
                ["signature / HMAC", fmt(sections.provenance.summary.signature_status)],
                ["replay", fmt(sections.provenance.summary.replay_status)],
                ["findings", fmt(sections.provenance.summary.num_findings)],
              ]}
            />
          )}
        </LayerCard>

        <LayerCard
          id="layer-distribution"
          icon={Radar}
          title="Distribution Shift"
          section={dist}
        >
          {dist && (
            <KV
              rows={[
                ["baseline images", fmt(dist.report.baseline.num_samples)],
                ["incoming images", fmt(dist.report.current_batch.num_images)],
                ["pattern", fmt(distEv.shift_pattern)],
                ["batch shift z", fmt(distEv.batch_shift_z)],
                ["MMD ratio vs null", fmt(distEv.mmd_ratio_vs_null)],
                ["recommendation", fmt(dist.report.recommendation)],
              ]}
            />
          )}
        </LayerCard>
      </div>

      {/* row E — behavioural + distribution analysis */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Panel
          title="Model behavioural comparison"
          className="fade-up-3"
          right={model && <Pill value={model.status} />}
        >
          {model?.report?.comparison ? (
            <>
              <ModelComparisonChart comparison={model.report.comparison} />
              <div className="mt-3 mb-1 text-[11px] text-muted">Per test image</div>
              <ModelPerImageChart perImage={model.report.comparison.per_image} />
            </>
          ) : (
            <EmptyState title="No behavioural comparison">
              Model audit was not run, or no reference behaviour was available.
            </EmptyState>
          )}
        </Panel>

        <Panel
          title="Distribution shift analysis"
          className="fade-up-3"
          right={dist && <Pill value={dist.status} />}
        >
          {dist ? (
            <>
              <div className="mb-2 flex flex-wrap items-center justify-around gap-2">
                <ScoreDial label="shift score" value={dist.report.shift_score} />
                <ScoreDial label="anomalous fraction" value={dist.report.anomalous_fraction} />
                <ScoreDial label="confidence" value={dist.report.confidence} tone={C.accent} />
              </div>
              <DistributionSamplesChart perSample={distEv.per_sample} />
              <p className="mt-2 border-l-2 border-line pl-3 text-[12.5px] leading-relaxed
                            text-muted">
                {dist.report.interpretation}
              </p>
            </>
          ) : (
            <EmptyState title="No distribution analysis">
              No incoming batch / baseline supplied to this assessment.
            </EmptyState>
          )}
        </Panel>
      </div>

      {/* row F — provenance chain */}
      <Panel title="Provenance hash chain" className="fade-up-3">
        <ProvenanceChain section={sections.provenance} />
      </Panel>

      {/* row G — this assessment's findings */}
      <Panel
        title={
          layer === "all"
            ? `Findings — this assessment (${findings.length})`
            : `Findings — ${phaseLabel(layerPhase)} (${scoped.length})`
        }
        className="fade-up-3"
        right={
          <SegTabs
            value={layer}
            onChange={setLayer}
            options={[
              { value: "all", label: "All" },
              ...LAYERS.map((l) => ({ value: l.key, label: phaseLabel(l.phase) })),
            ]}
          />
        }
      >
        <FindingsTable findings={scoped} />
      </Panel>
    </div>
  );
}
