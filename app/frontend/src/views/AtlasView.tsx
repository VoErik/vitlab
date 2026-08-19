// Feature Atlas: scatter (UMAP / freq-vs-strength) with lasso/click -> feature detail modal.
// Named features accumulate in a per-atlas Notebook (name + note + thumbnail), persisted
// server-side to labels.json and exportable to JSON. Search matches ids, names, and notes.
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";
import { api, NotebookEntry } from "../lib/api";

const FILTERS = ["alive", "dead", "rare", "common", "high_strength"];

export function AtlasView() {
  const { data: list } = useQuery({ queryKey: ["atlasList"], queryFn: api.atlasList });
  const [atlasId, setAtlasId] = useState("");
  const [scatter, setScatter] = useState<"umap" | "freq_strength">("umap");
  const [filter, setFilter] = useState("alive");
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<number[]>([]);
  const [heatmaps, setHeatmaps] = useState(true);
  const [zoom, setZoom] = useState<string | null>(null);
  const [notebookOpen, setNotebookOpen] = useState(false);

  const host = useRef<HTMLDivElement>(null);
  const plot = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(0);
  const H = Math.min(typeof window !== "undefined" ? Math.round(window.innerHeight * 0.72) : 560, 640);

  useEffect(() => {
    const el = host.current; if (!el) return;
    const ro = new ResizeObserver(() => setW(el.clientWidth));
    ro.observe(el); setW(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  const { data: feats } = useQuery({
    queryKey: ["atlasFeatures", atlasId, scatter, filter, search],
    queryFn: () => api.atlasFeatures(atlasId, {
      scatter, filter, limit: 8000, ...(search ? { search } : {}),
    }),
    enabled: !!atlasId,
  });
  const { data: notebook } = useQuery({
    queryKey: ["notebook", atlasId],
    queryFn: () => api.atlasNotebook(atlasId),
    enabled: !!atlasId,
  });

  useEffect(() => {
    if (!feats || !plot.current || !w) return;
    const rows = feats.rows as any[];
    const isFreq = scatter === "freq_strength";
    Plotly.react(plot.current, [{
      x: rows.map((r) => r.x), y: rows.map((r) => r.y),
      customdata: rows.map((r) => r.feature),
      text: rows.map((r) =>
        `F${r.feature}${r.label ? " · " + r.label : ""} · fire ${r.firing_rate.toFixed(3)} · str ${r.mean_act.toFixed(2)}`),
      hoverinfo: "text", mode: "markers", type: "scattergl",
      marker: {
        size: 6, opacity: 0.85,
        color: rows.map((r) => r.firing_rate), colorscale: "Viridis", showscale: true,
        colorbar: { title: "firing rate", thickness: 12, len: 0.6 }, line: { width: 0 },
      },
    }], {
      width: w, height: H, autosize: false, margin: { l: 56, r: 10, t: 10, b: 48 },
      dragmode: "lasso", paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#1f2733", size: 12 },
      xaxis: { title: isFreq ? "firing rate (log)" : "UMAP-1", type: isFreq ? "log" : "linear", zeroline: false },
      yaxis: { title: isFreq ? "mean activation (log)" : "UMAP-2", type: isFreq ? "log" : "linear", zeroline: false },
    }, { responsive: false, displaylogo: false });

    const el = plot.current as any;
    el.removeAllListeners?.("plotly_selected");
    el.removeAllListeners?.("plotly_click");
    el.on?.("plotly_selected", (ev: any) =>
      setSelected(ev ? ev.points.map((p: any) => p.customdata) : []));
    el.on?.("plotly_click", (ev: any) => setSelected([ev.points[0].customdata]));
  }, [feats, scatter, w, H]);

  const meta = list?.atlases.find((a) => a.id === atlasId);
  const shown = selected.slice(0, 12);
  const cols = Math.max(1, Math.min(shown.length, 4));       // ≤ 4 features per row
  const locate = (f: number) => { setNotebookOpen(false); setSelected([f]); };

  return (
    <div className="atlas">
      <aside className="panel atlas-controls">
        <h3>Atlas</h3>
        <label>Atlas
          <select value={atlasId} onChange={(e) => { setAtlasId(e.target.value); setSelected([]); }}>
            <option value="">—</option>
            {list?.atlases.map((a) => <option key={a.id} value={a.id}>{a.id}</option>)}
          </select>
        </label>
        <label>Scatter
          <select value={scatter} onChange={(e) => setScatter(e.target.value as any)}>
            <option value="umap">UMAP (dictionary)</option>
            <option value="freq_strength">Frequency vs strength</option>
          </select>
        </label>
        <label>Filter
          <select value={filter} onChange={(e) => setFilter(e.target.value)}>
            {FILTERS.map((f) => <option key={f} value={f}>{f}</option>)}
          </select>
        </label>
        <label>Search id, name or note
          <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="e.g. watermark or 1847" />
        </label>
        <button disabled={!atlasId} onClick={() => setNotebookOpen(true)}>
          Notebook ({notebook?.count ?? 0})
        </button>
        <p className="note">{feats?.total ?? 0} shown · {selected.length} selected</p>
        {meta && <p className="note">{meta.dataset} · {meta.site}<br />{meta.n_features} features · {meta.n_dead} dead</p>}
        {search && <p className="note">Search ignores the alive/dead filter.</p>}
        {scatter === "freq_strength" && <p className="note">Log axes; dead features (rate 0) are omitted.</p>}
        <p className="note">Lasso or click points to inspect.</p>
      </aside>

      <section className="panel atlas-plot">
        {!atlasId && <p className="note">Pick an atlas to explore its features.</p>}
        <div className="atlas-canvas" ref={host}><div ref={plot} /></div>
      </section>

      {selected.length > 0 && (
        <div className="atlas-modal-backdrop" onClick={() => setSelected([])}>
          <div className="atlas-modal" onClick={(e) => e.stopPropagation()}>
            <div className="atlas-modal-head">
              <b>{selected.length} feature{selected.length > 1 ? "s" : ""} selected</b>
              <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
                <label className="row" style={{ margin: 0 }}>
                  <input type="checkbox" checked={heatmaps} onChange={(e) => setHeatmaps(e.target.checked)} />
                  heatmaps
                </label>
                <button className="ghost" onClick={() => setSelected([])}>Close</button>
              </div>
            </div>
            <div className="atlas-modal-body"
              style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 260px))` }}>
              {shown.map((f) => (
                <FeatureCard key={f} atlasId={atlasId} feature={f} heatmaps={heatmaps} onZoom={setZoom} />
              ))}
            </div>
            {selected.length > 12 && <p className="note" style={{ padding: "0 18px 14px" }}>
              +{selected.length - 12} more not shown — lasso a smaller group to see them.</p>}
          </div>
        </div>
      )}

      {notebookOpen && (
        <NotebookModal atlasId={atlasId} entries={notebook?.entries ?? []}
          heatmaps={heatmaps} onZoom={setZoom} onLocate={locate} onClose={() => setNotebookOpen(false)} />
      )}

      {zoom && (
        <div className="lightbox" onClick={() => setZoom(null)}>
          <img src={zoom} alt="feature example enlarged" />
        </div>
      )}
    </div>
  );
}

// ---- shared name+note editor ----------------------------------------------
function LabelEditor({ atlasId, feature, name, note }:
  { atlasId: string; feature: number; name: string; note: string }) {
  const qc = useQueryClient();
  const [n, setN] = useState(name);
  const [t, setT] = useState(note);
  const [saving, setSaving] = useState(false);
  useEffect(() => { setN(name); setT(note); }, [name, note]);

  const dirty = n !== name || t !== note;
  const commit = async (nn: string, tt: string) => {
    setSaving(true);
    try {
      await api.atlasSetLabel(atlasId, feature, nn, tt);
      qc.invalidateQueries({ queryKey: ["feat", atlasId, feature] });
      qc.invalidateQueries({ queryKey: ["atlasFeatures", atlasId] });
      qc.invalidateQueries({ queryKey: ["notebook", atlasId] });
    } finally { setSaving(false); }
  };

  return (
    <div className="label-editor">
      <div className="rename">
        <input value={n} onChange={(e) => setN(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") commit(n, t); }}
          placeholder="name (e.g. watermark)" />
        <button onClick={() => commit(n, t)} disabled={saving || !dirty}>Save</button>
      </div>
      <textarea className="note-input" value={t} onChange={(e) => setT(e.target.value)}
        placeholder="note — why this feature matters" rows={2} />
      {(name || note) && (
        <button className="ghost small" onClick={() => commit("", "")} disabled={saving}>Remove from notebook</button>
      )}
    </div>
  );
}

// ---- feature detail card (in the selection modal) --------------------------
function FeatureCard({ atlasId, feature, heatmaps, onZoom }:
  { atlasId: string; feature: number; heatmaps: boolean; onZoom: (u: string) => void }) {
  const { data } = useQuery({
    queryKey: ["feat", atlasId, feature],
    queryFn: () => api.atlasFeature(atlasId, feature),
  });
  if (!data) return <div className="feature-detail"><b>F{feature}</b> <span className="note">loading…</span></div>;
  const st = data.stats;
  return (
    <div className="feature-detail">
      <div className="feature-detail-head">
        <b>F{feature}</b>
        {data.name && <span className="feature-tag">{data.name}</span>}
        {st.dead && <span className="dead">dead</span>}
      </div>
      <LabelEditor atlasId={atlasId} feature={feature} name={data.name ?? ""} note={data.note ?? ""} />
      <div className="stat-grid">
        <div><span>firing rate</span><b>{st.firing_rate.toFixed(4)}</b></div>
        <div><span>mean act</span><b>{st.mean_act.toFixed(3)}</b></div>
        <div><span>max act</span><b>{st.max_act.toFixed(3)}</b></div>
        <div><span>top images</span><b>{data.top_images.length}</b></div>
      </div>
      {data.top_images.length > 0 ? (
        <div className="thumbs">
          {data.top_images.slice(0, 12).map((t: any) => {
            const url = api.atlasImageURL(atlasId, t.image_index, heatmaps ? feature : undefined);
            return <img key={t.image_index} src={url} alt=""
              title={`img ${t.image_index} · act ${t.score.toFixed(2)}`} onClick={() => onZoom(url)} />;
          })}
        </div>
      ) : <p className="note">no activating images</p>}
    </div>
  );
}

// ---- notebook modal --------------------------------------------------------
function NotebookModal({ atlasId, entries, heatmaps, onZoom, onLocate, onClose }: {
  atlasId: string; entries: NotebookEntry[]; heatmaps: boolean;
  onZoom: (u: string) => void; onLocate: (f: number) => void; onClose: () => void;
}) {
  const exportJson = () => {
    const blob = new Blob([JSON.stringify({ atlas: atlasId, entries }, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `notebook_${atlasId.replace(/[\\/]/g, "_")}.json`;
    a.click(); URL.revokeObjectURL(a.href);
  };
  return (
    <div className="atlas-modal-backdrop" onClick={onClose}>
      <div className="atlas-modal notebook-modal" onClick={(e) => e.stopPropagation()}>
        <div className="atlas-modal-head">
          <b>Feature notebook · {entries.length}</b>
          <div style={{ display: "flex", gap: 10 }}>
            <button className="ghost" onClick={exportJson} disabled={!entries.length}>Export JSON</button>
            <button className="ghost" onClick={onClose}>Close</button>
          </div>
        </div>
        <div className="notebook-list">
          {entries.length === 0 && <p className="note" style={{ padding: 16 }}>
            No named features yet — click a point in the scatter and give it a name.</p>}
          {entries.map((e) => {
            const thumb = e.top_image_index != null
              ? api.atlasImageURL(atlasId, e.top_image_index, heatmaps ? e.feature : undefined) : null;
            return (
              <div className="notebook-entry" key={e.feature}>
                {thumb
                  ? <img className="notebook-thumb" src={thumb} alt="" onClick={() => onZoom(thumb)} />
                  : <div className="notebook-thumb placeholder" />}
                <div className="notebook-main">
                  <div className="feature-detail-head">
                    <b>F{e.feature}</b>
                    {e.dead && <span className="dead">dead</span>}
                    {e.firing_rate != null &&
                      <span className="note" style={{ marginLeft: "auto" }}>
                        fire {e.firing_rate.toFixed(3)} · str {(e.mean_act ?? 0).toFixed(2)}</span>}
                  </div>
                  <LabelEditor atlasId={atlasId} feature={e.feature} name={e.name} note={e.note} />
                  <button className="ghost small" onClick={() => onLocate(e.feature)}>Locate in plot</button>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
