// Token-group flow: per-layer share of the decision carried by CLS vs registers vs patches.
// gradient-mass view = fraction of gradient mass (stacked to 1); ablation view = logit drop
// when each group is mean-ablated at that layer. Needs no SAE — just the model + image.
import { useEffect, useMemo, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";
import { TokenGroups } from "../lib/api";
import { api } from "../lib/api";
import { useSetup } from "../lib/useSetup";
import { SetupPanel } from "../components/SetupPanel";
import { ProbBars } from "../components/ProbBars";

const GROUP_COLOR: Record<string, string> = { cls: "#5e4fa2", registers: "#e0a35f", patches: "#2f9e78" };
const avg = (a: number[]) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);

export function TokenGroupsView() {
  const s = useSetup();
  const [tg, setTg] = useState<TokenGroups | null>(null);
  const [targetClass, setTargetClass] = useState<number | null>(null);
  const [view, setView] = useState<"mass" | "ablation">("mass");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const chart = useRef<HTMLDivElement>(null);
  const result = s.result;
  const names = result?.classes.slice().sort((a, b) => a.idx - b.idx).map((c) => c.name);
  const effTarget = targetClass ?? result?.predicted ?? null;

  useEffect(() => { setTargetClass(null); }, [result?.image_token]);

  useEffect(() => {
    const cls = effTarget;
    if (!result || !s.modelId || !s.task || cls == null) return;
    let cancel = false;
    setLoading(true); setError(null);
    api.tokenGroups({ image_token: result.image_token, model_id: s.modelId, task: s.task, target_class: cls })
      .then((r) => { if (!cancel) setTg(r); })
      .catch((e) => { if (!cancel) { setError(String(e?.message ?? e)); setTg(null); } })
      .finally(() => { if (!cancel) setLoading(false); });
    return () => { cancel = true; };
  }, [result?.image_token, s.modelId, s.task, targetClass]);

  useEffect(() => {
    if (!tg || !chart.current) return;
    const traces = tg.groups.map((g) => view === "mass" ? ({
      x: tg.layers, y: tg.gradient_mass[g], name: g, type: "scatter",
      mode: "none", stackgroup: "one", fillcolor: GROUP_COLOR[g] ?? "#888",
    }) : ({
      x: tg.layers, y: tg.ablation_drop[g], name: g, type: "scatter",
      mode: "lines+markers", line: { color: GROUP_COLOR[g] ?? "#888" },
    }));
    Plotly.react(chart.current, traces as any, {
      margin: { l: 54, r: 12, t: 10, b: 44 }, font: { color: "#1f2733" },
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      legend: { orientation: "h", y: 1.12 },
      xaxis: { title: "layer" },
      yaxis: view === "mass"
        ? { title: "gradient-mass share", range: [0, 1] }
        : { title: "logit drop when group ablated", zeroline: true },
    }, { responsive: true, displaylogo: false });
  }, [tg, view]);

  const patchMean = useMemo(() => (tg ? avg(tg.gradient_mass["patches"] ?? []) : null), [tg]);

  return (
    <div className="cols">
      <SetupPanel s={s} showBank={false} showSite={false} />
      <div className="panel">
        <h3>Prediction</h3>
        {result ? (
          <>
            <ProbBars probs={result.classes.slice().sort((a, b) => a.idx - b.idx).map((c) => c.prob)}
              names={names} highlight={effTarget ?? undefined} onPick={(i) => setTargetClass(i)} />
            <p className="note">Click a class to decompose its decision.</p>
          </>
        ) : <p className="note">Upload an image in Setup.</p>}
      </div>

      <div className="panel grow">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
          <h3 style={{ margin: 0 }}>
            Token-group flow {tg && <small className="note">· {names?.[tg.target_class] ?? tg.target_class}</small>}
          </h3>
          <select value={view} onChange={(e) => setView(e.target.value as any)} style={{ width: "auto" }}>
            <option value="mass">Gradient-mass share</option>
            <option value="ablation">Ablation drop (causal)</option>
          </select>
        </div>
        {loading && <p className="note">Computing token-group flow…</p>}
        {error && <div className="warn">{error}</div>}
        {tg && <div ref={chart} style={{ height: 400 }} />}
        {tg && patchMean != null && view === "mass" &&
          <p className="note">Patch tokens carry on average <b>{(patchMean * 100).toFixed(0)}%</b> of the
            gradient mass — an approximate ceiling on what patch-level SAE features can explain.</p>}
        {tg && view === "ablation" &&
          <p className="note">Bars/lines show the logit drop when each group's residual is mean-ablated
            at that layer — the causal counterpart to the gradient share.</p>}
        {!tg && !loading && <p className="note">Upload and classify to see the decomposition.</p>}
      </div>
    </div>
  );
}