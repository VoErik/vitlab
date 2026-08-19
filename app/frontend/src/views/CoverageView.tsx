// Coverage meter: at each SAE site, what fraction of the patch tokens' causal pull on the
// decision survives when we keep only what the SAE reconstructs.
//   logit coverage = (L_recon - L_ablate) / (L_clean - L_ablate)
// Shown per layer alongside plain reconstruction fidelity (explained variance), with a
// headline mean — the quantitative "how much can the patch-SAE explain" answer.
import { useEffect, useMemo, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";
import { Coverage } from "../lib/api";
import { api } from "../lib/api";
import { useSetup } from "../lib/useSetup";
import { SetupPanel } from "../components/SetupPanel";
import { ProbBars } from "../components/ProbBars";

export function CoverageView() {
  const s = useSetup();
  const [cov, setCov] = useState<Coverage | null>(null);
  const [targetClass, setTargetClass] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const chart = useRef<HTMLDivElement>(null);
  const result = s.result;
  const names = result?.classes.slice().sort((a, b) => a.idx - b.idx).map((c) => c.name);
  const effTarget = targetClass ?? result?.predicted ?? null;

  useEffect(() => {
    if (!s.bankId && s.banks.length) s.setBankId(s.banks[0].id);
  }, [s.banks.length, s.bankId]);
  useEffect(() => { setTargetClass(null); }, [result?.image_token]);

  useEffect(() => {
    const cls = effTarget;
    if (!result || !s.modelId || !s.task || !s.bankId || cls == null) return;
    let cancel = false;
    setLoading(true); setError(null);
    api.coverage({ image_token: result.image_token, model_id: s.modelId, task: s.task, bank_id: s.bankId, target_class: cls })
      .then((r) => { if (!cancel) setCov(r); })
      .catch((e) => { if (!cancel) { setError(String(e?.message ?? e)); setCov(null); } })
      .finally(() => { if (!cancel) setLoading(false); });
    return () => { cancel = true; };
  }, [result?.image_token, s.modelId, s.task, s.bankId, targetClass]);

  useEffect(() => {
    if (!cov || !chart.current) return;
    const x = cov.sites.map((d) => d.layer);
    const traces = [
      {
        x, y: cov.sites.map((d) => d.logit_coverage), name: "logit coverage",
        mode: "lines+markers", type: "scatter", line: { color: "#5e4fa2" },
        connectgaps: false,
      },
      {
        x, y: cov.sites.map((d) => d.explained_var), name: "explained variance",
        mode: "lines+markers", type: "scatter", line: { color: "#2f9e78", dash: "dot" },
      },
    ];
    Plotly.react(chart.current, traces as any, {
      margin: { l: 54, r: 12, t: 10, b: 44 }, font: { color: "#1f2733" },
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      legend: { orientation: "h", y: 1.12 },
      xaxis: { title: "layer" },
      yaxis: { title: "fraction", range: [0, 1.05] },
      shapes: [{ type: "line", x0: Math.min(...x), x1: Math.max(...x), y0: 1, y1: 1, line: { dash: "dot", color: "#9aa1ac", width: 1 } }],
    }, { responsive: true, displaylogo: false });
  }, [cov]);

  const pct = (v: number | null | undefined) => (v == null ? "—" : `${(v * 100).toFixed(0)}%`);
  const meanCov = useMemo(() => cov?.overall.mean_logit_coverage ?? null, [cov]);

  return (
    <div className="cols">
      <SetupPanel s={s} showSite={false} />
      <div className="panel">
        <h3>Prediction</h3>
        {result ? (
          <>
            <ProbBars probs={result.classes.slice().sort((a, b) => a.idx - b.idx).map((c) => c.prob)}
              names={names} highlight={effTarget ?? undefined} onPick={(i) => setTargetClass(i)} />
            <p className="note">Click a class to measure coverage for it.</p>
          </>
        ) : <p className="note">Upload an image in Setup.</p>}
      </div>

      <div className="panel grow">
        <h3 style={{ margin: 0 }}>
          Patch-SAE coverage {cov && <small className="note">· {names?.[cov.target_class] ?? cov.target_class}</small>}
        </h3>
        {loading && <p className="note">Measuring coverage across layers…</p>}
        {error && <div className="warn">{error}</div>}
        {cov && (
          <>
            <div className="stat-grid" style={{ maxWidth: 420, margin: "12px 0" }}>
              <div><span>mean logit coverage</span><b>{pct(meanCov)}</b></div>
              <div><span>mean explained var</span><b>{pct(cov.overall.mean_explained_var)}</b></div>
            </div>
            <div ref={chart} style={{ height: 380 }} />
            <p className="note">
              <b>Logit coverage</b> = fraction of the patch tokens' causal pull on the decision that
              survives keeping only the SAE reconstruction ((L<sub>recon</sub>−L<sub>ablate</sub>)/(L<sub>clean</sub>−L<sub>ablate</sub>)).
              High = the patch-SAE explains the decision; a gap to <b>explained variance</b> means the
              reconstruction is faithful in norm but loses decision-relevant directions.
            </p>
          </>
        )}
        {!cov && !loading && <p className="note">Upload and classify to measure coverage.</p>}
      </div>
    </div>
  );
}