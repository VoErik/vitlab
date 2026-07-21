// Feature Atlas: swappable scatter (UMAP / freq-vs-strength) -> select -> detail with
// top-k images and a heatmap toggle.
import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";
import { api } from "../lib/api";

export function AtlasView() {
  const { data: list } = useQuery({ queryKey: ["atlasList"], queryFn: api.atlasList });
  const [atlasId, setAtlasId] = useState("");
  const [scatter, setScatter] = useState<"umap" | "freq_strength">("umap");
  const [filter, setFilter] = useState("alive");
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<number[]>([]);
  const [heatmaps, setHeatmaps] = useState(true);
  const plot = useRef<HTMLDivElement>(null);

  const { data: feats } = useQuery({
    queryKey: ["atlasFeatures", atlasId, scatter, filter, search],
    queryFn: () => api.atlasFeatures(atlasId, {
      scatter, filter, limit: 8000, ...(search ? { search: Number(search) } : {}),
    }),
    enabled: !!atlasId,
  });

  useEffect(() => {
    if (!feats || !plot.current) return;
    const rows = feats.rows;
    Plotly.react(plot.current, [{
      x: rows.map((r: any) => r.x), y: rows.map((r: any) => r.y),
      customdata: rows.map((r: any) => r.feature),
      text: rows.map((r: any) => `F${r.feature} · fire ${r.firing_rate.toFixed(3)} · str ${r.mean_act.toFixed(2)}`),
      hoverinfo: "text", mode: "markers", type: "scattergl",
      marker: { size: 5, color: rows.map((r: any) => r.firing_rate), colorscale: "Viridis", showscale: true, colorbar: { title: "fire" } },
    }], {
      margin: { l: 36, r: 10, t: 10, b: 36 }, dragmode: "lasso",
      xaxis: { title: scatter === "umap" ? "UMAP-1" : "firing rate" },
      yaxis: { title: scatter === "umap" ? "UMAP-2" : "mean activation" },
    }, { responsive: true });
    const el = plot.current as any;
    el.removeAllListeners?.("plotly_selected");
    el.on?.("plotly_selected", (ev: any) => setSelected(ev ? ev.points.map((p: any) => p.customdata) : []));
    el.on?.("plotly_click", (ev: any) => setSelected([ev.points[0].customdata]));
  }, [feats, scatter]);

  return (
    <div className="cols">
      <div className="panel narrow">
        <h3>Atlas</h3>
        <select value={atlasId} onChange={(e) => { setAtlasId(e.target.value); setSelected([]); }}>
          <option value="">—</option>
          {list?.atlases.map((a) => <option key={a.id} value={a.id}>{a.id}</option>)}
        </select>
        <label>Scatter
          <select value={scatter} onChange={(e) => setScatter(e.target.value as any)}>
            <option value="umap">UMAP (dictionary)</option>
            <option value="freq_strength">Frequency vs strength</option>
          </select>
        </label>
        <label>Filter
          <select value={filter} onChange={(e) => setFilter(e.target.value)}>
            {["alive", "dead", "rare", "common", "high_strength"].map((f) => <option key={f} value={f}>{f}</option>)}
          </select>
        </label>
        <label>Search feature id
          <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="e.g. 1847" />
        </label>
        <label className="row"><input type="checkbox" checked={heatmaps}
          onChange={(e) => setHeatmaps(e.target.checked)} /> show heatmaps</label>
        <p className="note">{feats?.total ?? 0} features · {selected.length} selected</p>
      </div>

      <div className="panel grow"><div ref={plot} style={{ height: 580 }} /></div>

      <div className="panel">
        <h3>Selected</h3>
        {selected.slice(0, 12).map((f) => (
          <FeatureCard key={f} atlasId={atlasId} feature={f} heatmaps={heatmaps} />
        ))}
        {!selected.length && <p className="note">Lasso or click points to inspect features.</p>}
      </div>
    </div>
  );
}

function FeatureCard({ atlasId, feature, heatmaps }: { atlasId: string; feature: number; heatmaps: boolean }) {
  const { data } = useQuery({ queryKey: ["feat", atlasId, feature], queryFn: () => api.atlasFeature(atlasId, feature) });
  if (!data) return null;
  return (
    <div className="featcard">
      <b>F{feature}</b> fire {data.stats.firing_rate.toFixed(3)} · str {data.stats.mean_act.toFixed(2)}
      {data.stats.dead && <span className="dead"> dead</span>}
      <div className="thumbs">
        {data.top_images.slice(0, 8).map((t: any) => (
          <img key={t.image_index} width={54} height={54} alt=""
            src={api.atlasImageURL(atlasId, t.image_index, heatmaps ? feature : undefined)} />
        ))}
      </div>
    </div>
  );
}
