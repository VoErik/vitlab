// Pick model+task+bank, upload, classify, attribute. Heatmap overlays + class switching.
import { useState } from "react";
import { AttrResult } from "../lib/api";
import { api } from "../lib/api";
import { useSetup } from "../lib/useSetup";
import { SetupPanel } from "../components/SetupPanel";
import { Heatmap } from "../components/Heatmap";
import { ProbBars } from "../components/ProbBars";

export function ClassifyView() {
  const s = useSetup();
  const [attr, setAttr] = useState<AttrResult | null>(null);
  const [targetClass, setTargetClass] = useState<number | null>(null);
  const [activeFeature, setActiveFeature] = useState<number | null>(null);

  const result = s.result;
  const names = result?.classes.slice().sort((a, b) => a.idx - b.idx).map((c) => c.name);

  async function runAttr(cls: number | null) {
    if (!result || !s.modelId || !s.task || !s.bankId) return;
    setTargetClass(cls); setActiveFeature(null);
    setAttr(await api.attribute({
      image_token: result.image_token, model_id: s.modelId, task: s.task,
      bank_id: s.bankId, site: s.site, target_class: cls, method: "two_stage", top_k: 5,
    }));
  }

  const activeMap = attr?.features.find((f) => f.feature === activeFeature)?.grid_map ?? null;

  return (
    <div className="cols">
      <SetupPanel s={s} />

      <div className="panel">
        <h3>Prediction</h3>
        {result && (
          <div className="imgwrap" style={{ width: 320, height: 320, position: "relative" }}>
            <img className="shown" src={`/api/image/${result.image_token}`} width={320} height={320} alt="input" />
            {activeMap && <Heatmap grid={activeMap} size={320} />}
          </div>
        )}
        {result && (
          <ProbBars
            probs={result.classes.slice().sort((a, b) => a.idx - b.idx).map((c) => c.prob)}
            names={names} highlight={targetClass ?? result.predicted}
            onPick={(i) => runAttr(i)} />
        )}
        {result && <button onClick={() => runAttr(result.predicted)} disabled={!s.bankId}>
          Explain predicted class</button>}
      </div>

      <div className="panel">
        <h3>Attribution {attr && <small>({attr.method}, class {attr.target_class})</small>}</h3>
        {attr?.warning && <div className="warn">{attr.warning}</div>}
        {attr?.features.map((f) => (
          <div key={`${f.site}-${f.feature}`}
            className={"featcard clickable" + (f.feature === activeFeature ? " sel" : "")}
            onClick={() => setActiveFeature(f.feature === activeFeature ? null : f.feature)}>
            <b>{f.site} · F{f.feature}</b> <span>Δ={f.score.toFixed(4)}</span>
            <div className="hint">{f.feature === activeFeature ? "hiding" : "click to overlay heatmap"}</div>
          </div>
        ))}
        {attr && !attr.features.length && <p>No features returned.</p>}
      </div>
    </div>
  );
}
