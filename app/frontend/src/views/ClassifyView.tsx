// Classify + attribute. Upload -> classify -> auto-explain across ALL layers.
// Method is selectable; an optional "vs" class turns the target into the logit margin
// (contrastive attribution — what separates two confused classes). Top features per layer
// are chips whose border encodes relevance; click a chip to overlay its heatmap.
import { useEffect, useMemo, useState } from "react";
import { AttrAllResult } from "../lib/api";
import { api } from "../lib/api";
import { useSetup } from "../lib/useSetup";
import { SetupPanel } from "../components/SetupPanel";
import { Heatmap } from "../components/Heatmap";
import { ProbBars } from "../components/ProbBars";

const METHODS = [
  { value: "two_stage", label: "Two-stage (patching → ablation)" },
  { value: "attribution_patching", label: "Attribution patching" },
  { value: "dla", label: "Direct logit attribution" },
];

const keyOf = (site: string, feature: number) => `${site}::${feature}`;

export function ClassifyView() {
  const s = useSetup();
  const [attr, setAttr] = useState<AttrAllResult | null>(null);
  const [method, setMethod] = useState("two_stage");
  const [targetClass, setTargetClass] = useState<number | null>(null); // null = predicted
  const [contrastClass, setContrastClass] = useState<number | null>(null); // null = single-class
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const result = s.result;
  const names = result?.classes.slice().sort((a, b) => a.idx - b.idx).map((c) => c.name);
  const effectiveTarget = targetClass ?? result?.predicted ?? null;

  useEffect(() => {
    if (!s.bankId && s.banks.length) s.setBankId(s.banks[0].id);
  }, [s.banks.length, s.bankId]);

  useEffect(() => { setTargetClass(null); setContrastClass(null); setActiveKey(null); }, [result?.image_token]);

  // a contrast equal to the target is meaningless; drop it
  useEffect(() => {
    if (contrastClass != null && contrastClass === effectiveTarget) setContrastClass(null);
  }, [contrastClass, effectiveTarget]);

  useEffect(() => {
    const cls = effectiveTarget;
    if (!result || !s.modelId || !s.task || !s.bankId || cls == null) return;
    let cancelled = false;
    setLoading(true); setError(null); setActiveKey(null);
    api.attributeAll({
      image_token: result.image_token, model_id: s.modelId, task: s.task, bank_id: s.bankId,
      target_class: cls, contrast_class: contrastClass, method, top_k: 5,
    })
      .then((r) => { if (!cancelled) setAttr(r); })
      .catch((e) => { if (!cancelled) { setError(String(e?.message ?? e)); setAttr(null); } })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [result?.image_token, s.modelId, s.task, s.bankId, method, targetClass, contrastClass]);

  const activeMap = useMemo(() => {
    if (!attr || !activeKey) return null;
    for (const L of attr.layers)
      for (const f of L.features)
        if (keyOf(f.site, f.feature) === activeKey) return f.grid_map;
    return null;
  }, [attr, activeKey]);

  const maxAbs = useMemo(() => {
    let m = 0;
    attr?.layers.forEach((L) => L.features.forEach((f) => { m = Math.max(m, Math.abs(f.score)); }));
    return m || 1;
  }, [attr]);

  const topKey = attr?.top ? keyOf(attr.top.site, attr.top.feature) : null;
  const nameOf = (i: number | null) => (i == null ? null : names?.[i] ?? i);
  const header = attr
    ? (attr.contrast_class != null
      ? `${nameOf(attr.target_class)} vs ${nameOf(attr.contrast_class)}`
      : `class ${nameOf(attr.target_class)}`)
    : null;

  return (
    <div className="cols">
      <SetupPanel s={s} showSite={false} />

      <div className="panel">
        <h3>Prediction</h3>
        {result ? (
          <>
            <div className="imgwrap" style={{ width: 320, height: 320, position: "relative" }}>
              <img className="shown" src={`/api/image/${result.image_token}`} width={320} height={320} alt="input" />
              {activeMap && <Heatmap grid={activeMap} size={320} />}
            </div>
            <button onClick={() => setActiveKey(null)} disabled={!activeKey}>Clear overlay</button>
            <ProbBars
              probs={result.classes.slice().sort((a, b) => a.idx - b.idx).map((c) => c.prob)}
              names={names} highlight={effectiveTarget ?? undefined}
              onPick={(i) => setTargetClass(i)} />
            <p className="note">Click a class to explain it — attribution runs automatically.</p>
          </>
        ) : <p className="note">Upload an image in Setup.</p>}
      </div>

      <div className="panel grow">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          <h3 style={{ margin: 0 }}>
            Attribution {header && <small className="note">· {header}</small>}
          </h3>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <select value={contrastClass ?? ""} style={{ width: "auto" }}
              onChange={(e) => setContrastClass(e.target.value === "" ? null : Number(e.target.value))}
              title="Contrast class — attribute the logit margin (A vs B)">
              <option value="">vs — (single class)</option>
              {names?.map((nm, i) => i !== effectiveTarget &&
                <option key={i} value={i}>vs {nm}</option>)}
            </select>
            <select value={method} onChange={(e) => setMethod(e.target.value)} style={{ width: "auto" }}>
              {METHODS.map((m) => <option key={m.value} value={m.value}>{m.label}</option>)}
            </select>
          </div>
        </div>

        {contrastClass != null &&
          <p className="note">Contrastive: positive scores push toward {nameOf(effectiveTarget)} over {nameOf(contrastClass)}.</p>}
        {loading && <p className="note">Computing attribution across all layers…</p>}
        {error && <div className="warn">{error}</div>}
        {attr?.warning && <div className="warn">{attr.warning}</div>}

        {!loading && attr?.layers.map((L) => (
          <div key={L.site} className="featcard">
            <div className="note" style={{ marginBottom: 6 }}>Layer {L.layer} · {L.site}</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {L.features.map((f) => {
                const key = keyOf(f.site, f.feature);
                const strength = Math.abs(f.score) / maxAbs;
                const isActive = key === activeKey;
                const isTop = key === topKey;
                return (
                  <button key={key} onClick={() => setActiveKey(isActive ? null : key)}
                    title={`Δ=${f.score.toFixed(4)}`}
                    style={{
                      margin: 0, padding: "4px 8px", fontSize: 12, cursor: "pointer", borderRadius: 6,
                      background: isActive ? "var(--accent)" : "transparent",
                      color: isActive ? "#fff" : "inherit",
                      border: `2px solid ${isTop ? "var(--flip)" : `rgba(94,79,162,${(0.2 + 0.8 * strength).toFixed(2)})`}`,
                    }}>
                    F{f.feature} <span style={{ opacity: 0.7 }}>Δ{f.score >= 0 ? "+" : ""}{f.score.toFixed(3)}</span>
                  </button>
                );
              })}
              {!L.features.length && <span className="note">no features</span>}
            </div>
          </div>
        ))}
        {!loading && attr && !attr.layers.length && <p className="note">No layers in this bank.</p>}
        {!loading && !attr && !error && <p className="note">Attribution appears here after classifying.</p>}
      </div>
    </div>
  );
}