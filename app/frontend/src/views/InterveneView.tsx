// Intervention: select patches on the model's exact image, ablate (pixel or concept),
// compare clean vs corrupted. Reuses the shared setup + PatchGrid.
import { useState } from "react";
import { api, AblateResult } from "../lib/api";
import { useSetup } from "../lib/useSetup";
import { SetupPanel } from "../components/SetupPanel";
import { PatchGrid } from "../components/PatchGrid";
import { ProbBars } from "../components/ProbBars";

export function InterveneView() {
  const s = useSetup();
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [mode, setMode] = useState<"patch" | "concept">("patch");
  const [patchMethod, setPatchMethod] = useState("road");
  const [features, setFeatures] = useState("");   // comma-separated feature ids
  const [res, setRes] = useState<AblateResult | null>(null);
  const result = s.result;

  async function run() {
    if (!result) return;
    if (mode === "patch") {
      setRes(await api.patchAblate({
        image_token: result.image_token, model_id: s.modelId, task: s.task,
        patches: [...selected], method: patchMethod,
      }));
    } else {
      const feats = features.split(",").map((x) => parseInt(x.trim(), 10)).filter((x) => !isNaN(x));
      setRes(await api.conceptAblate({
        image_token: result.image_token, model_id: s.modelId, task: s.task, bank_id: s.bankId,
        sites: [s.site], features: feats, patches: selected.size ? [...selected] : null,
      }));
    }
  }

  return (
    <div className="cols">
      <SetupPanel s={s} />
      <div className="panel">
        <h3>Select patches</h3>
        {result
          ? <PatchGrid src={`/api/image/${result.image_token}`} side={s.side} selected={selected} onChange={setSelected} />
          : <p>Upload an image in Setup.</p>}
        <p className="note">{selected.size} patches selected · grid {s.side}×{s.side}</p>
        <label>Mode
          <select value={mode} onChange={(e) => setMode(e.target.value as any)}>
            <option value="patch">Patch ablation (pixel)</option>
            <option value="concept">Concept ablation (latent)</option>
          </select>
        </label>
        {mode === "patch" ? (
          <label>Fill
            <select value={patchMethod} onChange={(e) => setPatchMethod(e.target.value)}>
              <option value="road">ROAD diffusion</option>
              <option value="mean">Mean</option>
              <option value="noise">Noise</option>
            </select>
          </label>
        ) : (
          <label>Feature ids (comma-sep)
            <input value={features} onChange={(e) => setFeatures(e.target.value)} placeholder="e.g. 1847, 902" />
          </label>
        )}
        <button onClick={run} disabled={!result || (mode === "patch" && !selected.size) || (mode === "concept" && !s.bankId)}>
          Run intervention
        </button>
      </div>

      <div className="panel">
        <h3>Clean → corrupted</h3>
        {res ? (
          <>
            <p>predicted {res.clean.predicted} → <b>{res.corrupted.predicted}</b>
              {res.clean.predicted !== res.corrupted.predicted && <span className="flip"> (flipped)</span>}</p>
            <ProbBars probs={res.clean.probs} compare={res.corrupted.probs} highlight={res.clean.predicted} />
            {res.corrupted_image && <img className="shown" style={{ marginTop: 10 }}
              src={`/api${res.corrupted_image.replace("/api", "")}`} width={220} height={220} alt="corrupted" />}
          </>
        ) : <p className="note">Run an intervention to compare.</p>}
      </div>
    </div>
  );
}
