// Logit lens: CLS trajectory across layers (line chart) or per-patch overlay at one layer.
import { useEffect, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";
import { api } from "../lib/api";
import { useSetup } from "../lib/useSetup";
import { SetupPanel } from "../components/SetupPanel";
import { PatchGrid } from "../components/PatchGrid";

export function LogitLensView() {
  const s = useSetup();
  const [mode, setMode] = useState<"cls" | "per_patch">("cls");
  const [layer, setLayer] = useState(9);
  const [res, setRes] = useState<any>(null);
  const chart = useRef<HTMLDivElement>(null);
  const result = s.result;
  const nLayers = s.model?.n_layers ?? 12;

  async function run() {
    if (!result) return;
    setRes(await api.logitLens({
      image_token: result.image_token, model_id: s.modelId, task: s.task, mode, layer,
    }));
  }

  // CLS trajectory -> line per class (prob vs layer)
  useEffect(() => {
    if (mode !== "cls" || !res?.trajectory || !chart.current) return;
    const traj: number[][] = res.trajectory;       // (n_layers, n_classes)
    const nC = traj[0].length;
    const traces = Array.from({ length: nC }, (_, c) => ({
      x: traj.map((_, l) => l), y: traj.map((row) => row[c]),
      mode: "lines+markers", name: `class ${c}`, type: "scatter",
    }));
    Plotly.react(chart.current, traces, {
      margin: { l: 40, r: 10, t: 10, b: 40 },
      xaxis: { title: "layer" }, yaxis: { title: "probability", range: [0, 1] },
    }, { responsive: true });
  }, [res, mode]);

  const overlay = mode === "per_patch" && res?.argmax
    ? (i: number) => `hsla(${(res.argmax[i] * 47) % 360},70%,50%,${0.2 + 0.55 * res.confidence[i]})`
    : undefined;

  return (
    <div className="cols">
      <SetupPanel s={s} showBank={false} showSite={false} />
      <div className="panel">
        <h3>Controls</h3>
        <label>Mode
          <select value={mode} onChange={(e) => setMode(e.target.value as any)}>
            <option value="cls">CLS trajectory</option>
            <option value="per_patch">Per-patch (one layer)</option>
          </select>
        </label>
        {mode === "per_patch" && (
          <label>Layer: {layer}
            <input type="range" min={0} max={nLayers - 1} value={layer}
              onChange={(e) => setLayer(+e.target.value)} />
          </label>
        )}
        <button onClick={run} disabled={!result}>Run lens</button>
        <p className="note">Applying the head to earlier layers is the logit-lens approximation,
          not the trained pooled path.</p>
      </div>
      <div className="panel grow">
        <h3>Result</h3>
        {mode === "cls" && <div ref={chart} style={{ height: 420 }} />}
        {mode === "per_patch" && result &&
          <PatchGrid src={`/api/image/${result.image_token}`} side={s.side}
            selected={new Set()} onChange={() => {}} overlay={overlay} />}
        {mode === "per_patch" && res && <p className="note">Each patch colored by argmax class · opacity = confidence.</p>}
      </div>
    </div>
  );
}
