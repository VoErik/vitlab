// Intervention: select patches on the model's exact image, then either ablate (pixel or
// concept) or run a dose-response steer — sweep a feature's scale α (0=ablate, 1=identity,
// >1=amplify) and plot the target probability (and optional class-margin) vs α.
import { useEffect, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";
import { api, AblateResult, DoseResult } from "../lib/api";
import { useSetup } from "../lib/useSetup";
import { SetupPanel } from "../components/SetupPanel";
import { PatchGrid } from "../components/PatchGrid";
import { ProbBars } from "../components/ProbBars";

type Mode = "patch" | "concept" | "dose";

export function InterveneView() {
  const s = useSetup();
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [mode, setMode] = useState<Mode>("patch");
  const [patchMethod, setPatchMethod] = useState("road");
  const [features, setFeatures] = useState("");           // comma-separated feature ids
  const [contrastClass, setContrastClass] = useState<number | null>(null);
  const [res, setRes] = useState<AblateResult | null>(null);
  const [dose, setDose] = useState<DoseResult | null>(null);
  const [busy, setBusy] = useState(false);
  const chart = useRef<HTMLDivElement>(null);
  const result = s.result;
  const names = result?.classes.slice().sort((a, b) => a.idx - b.idx).map((c) => c.name);
  const nameOf = (i: number | null | undefined) => (i == null ? "" : names?.[i] ?? i);
  const featIds = () => features.split(",").map((x) => parseInt(x.trim(), 10)).filter((x) => !isNaN(x));
  const bankSites = s.banks.find((b) => b.id === s.bankId)?.sites ?? [];

  async function run() {
    if (!result) return;
    setBusy(true);
    try {
      if (mode === "patch") {
        setRes(await api.patchAblate({
          image_token: result.image_token, model_id: s.modelId, task: s.task,
          patches: [...selected], method: patchMethod,
        }));
      } else if (mode === "concept") {
        setRes(await api.conceptAblate({
          image_token: result.image_token, model_id: s.modelId, task: s.task, bank_id: s.bankId,
          sites: [s.site], features: featIds(), patches: selected.size ? [...selected] : null,
        }));
      } else {
        setDose(await api.doseResponse({
          image_token: result.image_token, model_id: s.modelId, task: s.task, bank_id: s.bankId,
          sites: [s.site], features: featIds(), patches: selected.size ? [...selected] : null,
          contrast_class: contrastClass,
        }));
      }
    } finally { setBusy(false); }
  }

  useEffect(() => {
    if (mode !== "dose" || !dose || !chart.current) return;
    const x = dose.alphas;
    const traces: any[] = [{
      x, y: dose.target_prob, mode: "lines+markers", type: "scatter",
      name: `p(${nameOf(dose.target_class)})`, line: { color: "#5e4fa2" },
    }];
    if (dose.margin) traces.push({
      x, y: dose.margin, mode: "lines+markers", type: "scatter", yaxis: "y2",
      name: "logit margin", line: { color: "#e07a5f" },
    });
    const shapes: any[] = [{
      type: "line", x0: 1, x1: 1, y0: 0, y1: 1, yref: "paper",
      line: { dash: "dot", color: "#9aa1ac", width: 1 },
    }];
    if (dose.flip_alpha != null) shapes.push({
      type: "line", x0: dose.flip_alpha, x1: dose.flip_alpha, y0: 0, y1: 1, yref: "paper",
      line: { color: "#e07a5f", width: 1.5 },
    });
    Plotly.react(chart.current, traces, {
      margin: { l: 46, r: 46, t: 10, b: 42 }, font: { color: "#1f2733" },
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", shapes,
      showlegend: true, legend: { orientation: "h", y: 1.12 },
      xaxis: { title: "steer coefficient α  (0 = ablate · 1 = identity)" },
      yaxis: { title: "target probability", range: [0, 1] },
      yaxis2: dose.margin ? { title: "margin", overlaying: "y", side: "right", zeroline: true } : undefined,
    }, { responsive: true, displaylogo: false });
  }, [dose, mode]);

  const canRun = !!result && !busy && (
    (mode === "patch" && selected.size > 0) ||
    (mode !== "patch" && !!s.bankId && featIds().length > 0));

  return (
    <div className="cols">
      <SetupPanel s={s} showSite={false} />
      <div className="panel">
        <h3>Select patches</h3>
        {result
          ? <PatchGrid src={`/api/image/${result.image_token}`} side={s.side} selected={selected} onChange={setSelected} />
          : <p>Upload an image in Setup.</p>}
        <p className="note">{selected.size} patches selected · grid {s.side}×{s.side}
          {mode !== "patch" && " · empty = all patches"}</p>
        <label>Mode
          <select value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
            <option value="patch">Patch ablation (pixel)</option>
            <option value="concept">Concept ablation (latent)</option>
            <option value="dose">Dose-response (steer)</option>
          </select>
        </label>
        {mode === "patch" && (
          <label>Fill
            <select value={patchMethod} onChange={(e) => setPatchMethod(e.target.value)}>
              <option value="road">ROAD diffusion</option>
              <option value="mean">Mean</option>
              <option value="noise">Noise</option>
            </select>
          </label>
        )}
        {mode !== "patch" && (
          <label>Ablation site
            <select value={s.site} onChange={(e) => s.setSite(e.target.value)}>
              {bankSites.length === 0 && <option value={s.site}>{s.site}</option>}
              {bankSites.map((st) => <option key={st} value={st}>{st}</option>)}
            </select>
          </label>
        )}
        {mode !== "patch" && (
          <label>Feature ids (comma-sep)
            <input value={features} onChange={(e) => setFeatures(e.target.value)} placeholder="e.g. 1847, 902" />
          </label>
        )}
        {mode === "dose" && (
          <label>Track margin vs (optional)
            <select value={contrastClass ?? ""}
              onChange={(e) => setContrastClass(e.target.value === "" ? null : Number(e.target.value))}>
              <option value="">— target probability only</option>
              {names?.map((nm, i) => <option key={i} value={i}>vs {nm}</option>)}
            </select>
          </label>
        )}
        <button onClick={run} disabled={!canRun}>
          {busy ? "Running…" : mode === "dose" ? "Run sweep" : "Run intervention"}
        </button>
      </div>

      <div className="panel grow">
        {mode === "dose" ? (
          <>
            <h3>Dose-response</h3>
            {dose ? (
              <>
                <div ref={chart} style={{ height: 380 }} />
                <p className="note">
                  {dose.flip_alpha != null
                    ? <>Prediction flips at α = <b>{dose.flip_alpha}</b>.</>
                    : "Prediction never flips across the sweep."}
                  {" "}A flat line means the feature has little causal effect here; a steep,
                  monotone one means it drives the decision.
                </p>
              </>
            ) : <p className="note">Enter feature id(s) and run a sweep.</p>}
          </>
        ) : (
          <>
            <h3>Clean → corrupted</h3>
            {res ? (
              <>
                <p>predicted {nameOf(res.clean.predicted)} → <b>{nameOf(res.corrupted.predicted)}</b>
                  {res.clean.predicted !== res.corrupted.predicted && <span className="flip"> (flipped)</span>}</p>
                <ProbBars probs={res.clean.probs} compare={res.corrupted.probs}
                  names={names} highlight={res.clean.predicted} />
                {res.corrupted_image && <img className="shown" style={{ marginTop: 10 }}
                  src={`/api${res.corrupted_image.replace("/api", "")}`} width={220} height={220} alt="corrupted" />}
              </>
            ) : <p className="note">Run an intervention to compare.</p>}
          </>
        )}
      </div>
    </div>
  );
}
