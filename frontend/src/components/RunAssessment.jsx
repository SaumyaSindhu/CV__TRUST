import React, { useEffect, useMemo, useState } from "react";
import {
  ArrowUp, Box, CheckCircle2, ChevronRight, CircleSlash, Database, File, FileJson2,
  Folder, FolderOpen, Loader2, Play, Radar, Settings2, X, XCircle,
} from "lucide-react";
import { api } from "../api.js";
import { Pill } from "./ui.jsx";

/* ------------------------------------------------------------------ *
 * Field definitions — every input is OPTIONAL by design.
 * pick: 'dir' | 'file' controls what the browser lets you select.
 * ------------------------------------------------------------------ */
const FIELDS = [
  { key: "dataset_dir", label: "Dataset", pick: "dir",
    hint: "yolo_folder root (images/ + labels/)" },
  { key: "candidate_model_path", label: "Candidate Model", pick: "file",
    hint: ".pth / .pt checkpoint under audit" },
  { key: "reference_model_path", label: "Reference Model", pick: "file",
    hint: "trusted checkpoint (needed with a candidate)" },
  { key: "incoming_batch", label: "Incoming Batch", pick: "dir",
    apiKey: "incoming_dir", hint: "directory of incoming images" },
  { key: "distribution_baseline_path", label: "Distribution Baseline", pick: "file",
    hint: "baseline.json from build-baseline" },
  { key: "inference_record_path", label: "Inference Record", pick: "file",
    hint: "protected provenance record (.json)" },
];
const apiKeyOf = (f) => f.apiKey || f.key;

/* Which layers run for a given selection — DISPLAY logic only; the
 * backend engine itself decides what executes (same skip semantics). */
export function planLayers(v) {
  const has = (k) => Boolean(v[k] && v[k].trim());
  const layers = [
    { name: "Dataset Integrity", phase: "P1–2", active: has("dataset_dir") },
    { name: "Model Integrity", phase: "P3", active: has("candidate_model_path"),
      missing: has("candidate_model_path") && !has("reference_model_path")
        ? "Candidate Model requires a Reference Model for the Phase 3 behavioral comparison."
        : null },
    { name: "Provenance", phase: "P4", active: has("inference_record_path") },
    { name: "Distribution Shift", phase: "P5", active: has("incoming_batch"),
      missing: has("incoming_batch") && !has("distribution_baseline_path")
        ? "Incoming Batch requires a Distribution Baseline for the Phase 5 comparison."
        : null },
  ];
  const notes = [];
  if (has("reference_model_path") && !has("candidate_model_path"))
    notes.push("A Reference Model alone runs nothing — it is only used to compare a Candidate Model against.");
  if (has("distribution_baseline_path") && !has("incoming_batch"))
    notes.push("A Distribution Baseline alone runs nothing — it is only used to audit an Incoming Batch against.");
  const errors = layers.filter((l) => l.missing).map((l) => l.missing);
  const activeCount = layers.filter((l) => l.active && !l.missing).length;
  if (activeCount === 0 && errors.length === 0)
    errors.push("Select at least one input — every field is optional, but an assessment needs something to assess.");
  return { layers, errors, notes, activeCount };
}

/* ------------------------------------------------------------------ *
 * Server-path browser (read-only /api/fs/browse)
 * ------------------------------------------------------------------ */
function PathBrowser({ pick, onSelect, onClose }) {
  const [state, setState] = useState({ loading: true, err: null, data: null });

  const load = (path) => {
    setState((s) => ({ ...s, loading: true, err: null }));
    api
      .browse(path)
      .then((data) => setState({ loading: false, err: null, data }))
      .catch((e) =>
        setState((s) => ({ ...s, loading: false, err: String(e.message || e) }))
      );
  };
  useEffect(() => {
    load(undefined);
  }, []);

  const d = state.data;
  return (
    <div className="panel absolute inset-0 z-10 flex flex-col">
      <div className="flex items-center gap-2 border-b border-line px-3 py-2">
        <FolderOpen className="h-4 w-4 shrink-0 text-accent" />
        <span
          className="mono min-w-0 flex-1 truncate text-[11.5px] text-muted"
          title={d?.path}
        >
          {d?.path || "…"}
        </span>
        {d?.parent && (
          <button
            onClick={() => load(d.parent)}
            title="Up one level"
            className="grid h-7 w-7 place-items-center rounded-lg border border-line
                       text-muted transition-colors hover:border-accent hover:text-accent"
          >
            <ArrowUp className="h-3.5 w-3.5" />
          </button>
        )}
        {pick === "dir" && d && (
          <button
            onClick={() => onSelect(d.path)}
            className="rounded-lg border border-ok/50 px-2.5 py-1 text-[11px] text-ok
                       transition-colors hover:bg-ok/10"
          >
            select this folder
          </button>
        )}
        <button
          onClick={onClose}
          className="grid h-7 w-7 place-items-center rounded-lg border border-line
                     text-muted transition-colors hover:border-bad hover:text-bad"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-1.5">
        {state.loading && (
          <div className="flex items-center gap-2 p-3 text-[12px] text-muted">
            <Loader2 className="h-4 w-4 animate-spin" /> listing…
          </div>
        )}
        {state.err && <div className="mono p-3 text-[12px] text-bad">{state.err}</div>}
        {d?.entries.map((e) => (
          <button
            key={e.path}
            onClick={() => (e.is_dir ? load(e.path) : pick === "file" && onSelect(e.path))}
            disabled={!e.is_dir && pick !== "file"}
            className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left
                       text-[12.5px] transition-colors hover:bg-panel2
                       disabled:cursor-not-allowed disabled:opacity-40"
          >
            {e.is_dir ? (
              <Folder className="h-4 w-4 shrink-0 text-accent" />
            ) : e.name.endsWith(".json") ? (
              <FileJson2 className="h-4 w-4 shrink-0 text-warn" />
            ) : (
              <File className="h-4 w-4 shrink-0 text-muted" />
            )}
            <span className="min-w-0 flex-1 truncate">{e.name}</span>
            {e.is_dir && <ChevronRight className="h-3.5 w-3.5 text-muted" />}
          </button>
        ))}
        {d && d.entries.length === 0 && !state.loading && (
          <div className="p-3 text-[12px] italic text-muted">empty directory</div>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * The modal
 * ------------------------------------------------------------------ */
export default function RunAssessment({ onClose, onCreated }) {
  const [values, setValues] = useState(
    Object.fromEntries(FIELDS.map((f) => [f.key, ""]))
  );
  const [advanced, setAdvanced] = useState(false);
  const [adv, setAdv] = useState({
    architecture: "fasterrcnn_mobilenet_v3_large_320_fpn",
    test_images_dir: "data/sample_images",
    detector_score_thresh: "",
    provenance_key_path: "",
  });
  const [browsing, setBrowsing] = useState(null); // field key being browsed
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [apiError, setApiError] = useState(null);

  const plan = useMemo(() => planLayers(values), [values]);

  useEffect(() => {
    if (!running) return;
    const t0 = Date.now();
    const id = setInterval(() => setElapsed(Math.round((Date.now() - t0) / 1000)), 500);
    return () => clearInterval(id);
  }, [running]);

  useEffect(() => {
    function esc(e) {
      if (e.key === "Escape" && !running) onClose();
    }
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [onClose, running]);

  const set = (k, v) => setValues((s) => ({ ...s, [k]: v }));

  const submit = async () => {
    if (plan.errors.length) return;
    setApiError(null);
    setRunning(true);
    try {
      // Build the payload for the EXISTING /api/assurance/run contract:
      // only fields the user actually provided are sent.
      const payload = {};
      for (const f of FIELDS) {
        const v = values[f.key].trim();
        if (v) payload[apiKeyOf(f)] = v;
      }
      if (payload.candidate_model_path) {
        payload.architecture = adv.architecture;
        payload.test_images_dir = adv.test_images_dir.trim() || "data/sample_images";
        if (
          adv.detector_score_thresh !== "" &&
          !Number.isNaN(Number(adv.detector_score_thresh))
        )
          payload.detector_score_thresh = Number(adv.detector_score_thresh);
      }
      if (payload.inference_record_path && adv.provenance_key_path.trim())
        payload.provenance_key_path = adv.provenance_key_path.trim();

      const env = await api.runAssurance(payload);
      onCreated(env); // App selects the new assessment + refreshes everything
    } catch (e) {
      setApiError(String(e.message || e));
      setRunning(false);
    }
  };

  const architectures = [
    "fasterrcnn_mobilenet_v3_large_320_fpn",
    "fasterrcnn_resnet50_fpn",
    "fasterrcnn_mobilenet_v3_large_fpn",
    "retinanet_resnet50_fpn",
    "yolov5",
  ];

  const inputCls =
    "mono rounded-lg border border-line bg-panel2 px-3 py-2 text-[12px] text-ink " +
    "outline-none transition-colors placeholder:text-muted focus:border-accent/50 " +
    "disabled:opacity-50";

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/70 p-4 backdrop-blur-sm">
      <div
        className="panel fade-up relative flex max-h-[92vh] w-full max-w-3xl flex-col
                   overflow-hidden shadow-2xl"
      >
        <div className="flex items-center gap-3 border-b border-line px-5 py-3.5">
          <span
            className="grid h-8 w-8 shrink-0 place-items-center rounded-lg border
                       border-accent/40 bg-accent/10"
          >
            <Play className="h-4 w-4 text-accent" />
          </span>
          <div className="min-w-0">
            <div className="text-[13.5px] font-semibold text-head">
              New assurance assessment
            </div>
            <div className="text-[11px] text-muted">
              Every input is optional — run any single layer or any combination.
            </div>
          </div>
          <button
            onClick={onClose}
            disabled={running}
            className="ml-auto grid h-8 w-8 place-items-center rounded-lg border border-line
                       text-muted transition-colors hover:border-bad hover:text-bad
                       disabled:opacity-40"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="relative min-h-0 flex-1 overflow-y-auto p-5">
          {browsing && (
            <PathBrowser
              pick={FIELDS.find((f) => f.key === browsing)?.pick}
              onSelect={(p) => {
                set(browsing, p);
                setBrowsing(null);
              }}
              onClose={() => setBrowsing(null)}
            />
          )}

          <div className="space-y-2.5">
            {FIELDS.map((f) => (
              <div
                key={f.key}
                className="grid items-center gap-2 sm:grid-cols-[170px_1fr_auto]"
              >
                <label className="text-[12.5px] text-ink">
                  {f.label}
                  <span className="ml-1.5 text-[10.5px] text-muted">optional</span>
                </label>
                <input
                  value={values[f.key]}
                  onChange={(e) => set(f.key, e.target.value)}
                  placeholder={f.hint}
                  disabled={running}
                  className={inputCls}
                />
                <div className="flex gap-1.5">
                  <button
                    onClick={() => setBrowsing(f.key)}
                    disabled={running}
                    className="rounded-lg border border-line px-3 py-2 text-[12px] text-muted
                               transition-colors hover:border-accent hover:text-accent
                               disabled:opacity-40"
                  >
                    Browse
                  </button>
                  {values[f.key] && (
                    <button
                      onClick={() => set(f.key, "")}
                      disabled={running}
                      title="Clear"
                      className="grid w-9 place-items-center rounded-lg border border-line
                                 text-muted transition-colors hover:border-bad hover:text-bad
                                 disabled:opacity-40"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>

          <button
            onClick={() => setAdvanced(!advanced)}
            disabled={running}
            className="mt-4 flex items-center gap-1.5 text-[12px] text-muted
                       transition-colors hover:text-accent"
          >
            <Settings2 className="h-3.5 w-3.5" />
            advanced options {advanced ? "▾" : "▸"}
          </button>
          {advanced && (
            <div className="mt-2 grid gap-2.5 rounded-xl border border-line bg-panel2/60 p-3
                            sm:grid-cols-2">
              <label className="text-[11.5px] text-muted">
                architecture (model audit)
                <select
                  value={adv.architecture}
                  disabled={running}
                  onChange={(e) => setAdv({ ...adv, architecture: e.target.value })}
                  className="mono mt-1 w-full rounded-lg border border-line bg-panel px-2
                             py-1.5 text-[12px] text-ink"
                >
                  {architectures.map((a) => (
                    <option key={a}>{a}</option>
                  ))}
                </select>
              </label>
              <label className="text-[11.5px] text-muted">
                test images dir (model audit)
                <input
                  value={adv.test_images_dir}
                  disabled={running}
                  onChange={(e) => setAdv({ ...adv, test_images_dir: e.target.value })}
                  className="mono mt-1 w-full rounded-lg border border-line bg-panel px-2
                             py-1.5 text-[12px] text-ink"
                />
              </label>
              <label className="text-[11.5px] text-muted">
                detector score thresh (blank = default; 0.0 for untrained weights)
                <input
                  value={adv.detector_score_thresh}
                  disabled={running}
                  placeholder="e.g. 0.0"
                  onChange={(e) =>
                    setAdv({ ...adv, detector_score_thresh: e.target.value })
                  }
                  className="mono mt-1 w-full rounded-lg border border-line bg-panel px-2
                             py-1.5 text-[12px] text-ink"
                />
              </label>
              <label className="text-[11.5px] text-muted">
                provenance HMAC key path (optional)
                <input
                  value={adv.provenance_key_path}
                  disabled={running}
                  onChange={(e) =>
                    setAdv({ ...adv, provenance_key_path: e.target.value })
                  }
                  className="mono mt-1 w-full rounded-lg border border-line bg-panel px-2
                             py-1.5 text-[12px] text-ink"
                />
              </label>
            </div>
          )}

          <div className="mt-4 rounded-xl border border-line bg-panel2/60 p-3">
            <div className="mb-2 text-[11px] text-muted">Layers that will run</div>
            <ul className="space-y-1.5">
              {plan.layers.map((l) => (
                <li key={l.name} className="flex items-center gap-2 text-[12.5px]">
                  {l.missing ? (
                    <XCircle className="h-4 w-4 shrink-0 text-bad" />
                  ) : l.active ? (
                    <CheckCircle2 className="h-4 w-4 shrink-0 text-ok" />
                  ) : (
                    <CircleSlash className="h-4 w-4 shrink-0 text-muted/60" />
                  )}
                  <span className={l.active && !l.missing ? "text-ink" : "text-muted"}>
                    {l.name}
                  </span>
                  <Pill value={l.phase} />
                  <span className="mono ml-auto text-[11px] text-muted">
                    {l.missing ? "blocked" : l.active ? "will run" : "skipped"}
                  </span>
                </li>
              ))}
            </ul>
            {plan.errors.map((e, i) => (
              <div
                key={i}
                className="mono mt-2 flex items-start gap-1.5 text-[11.5px] text-bad"
              >
                <XCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" /> {e}
              </div>
            ))}
            {plan.notes.map((n, i) => (
              <div key={i} className="mt-2 text-[11.5px] text-warn">
                {n}
              </div>
            ))}
          </div>

          {apiError && (
            <div className="mono mt-3 rounded-xl border border-bad/50 bg-bad/10 p-2.5
                            text-[12px] text-bad">
              {apiError}
            </div>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-3 border-t border-line px-5 py-3.5">
          <div className="flex items-center gap-2 text-[11px] text-muted">
            <Database className="h-3.5 w-3.5" />
            <Box className="h-3.5 w-3.5" />
            <Radar className="h-3.5 w-3.5" />
            executes the real backend engine — results are never mocked
          </div>
          <button
            onClick={submit}
            disabled={running || plan.errors.length > 0}
            className="ml-auto flex items-center gap-2 rounded-lg border border-accent/60
                       bg-accent/10 px-5 py-2.5 text-[12.5px] font-semibold text-accent
                       transition-colors hover:bg-accent/20 disabled:cursor-not-allowed
                       disabled:opacity-40"
          >
            {running ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Running… {elapsed}s ({plan.activeCount} layer
                {plan.activeCount === 1 ? "" : "s"})
              </>
            ) : (
              <>
                <Play className="h-4 w-4" /> Run assurance
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
