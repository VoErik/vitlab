# thanks claude :)

import json, base64, io, numpy as np
from PIL import Image as PILImage
from tqdm import tqdm
from typing import Dict, Tuple, List, Optional
from ..circuits.circuit import Circuit


def save_circuit_html(
    circuit:             Circuit,
    model,
    bank,
    dataloader,
    path:                str,
    top_n:               int   = 6,
    modal_size:          int   = 192,
    node_cell_size:      int   = 56,
    device:              str   = "cuda",
    model_key:           Optional[str] = None,
    title:               str   = "Feature Circuit",
    positive_edges_only: bool  = True,
    target_class:        Optional[int] = None,
    colormap:            str   = "viridis"
) -> None:
    """Self-contained interactive HTML circuit diagram. Hover a node to trace
    ancestors; click for top-activating images. Layout (prefix, patch grid) is read
    from model.spec, so it works on any backbone."""
    reader = model.reader
    prefix = model.spec.n_prefix_tokens
    patch_size = model.spec.patch_size
    image_size = model.spec.image_size
    """
    Self-contained interactive HTML circuit diagram.
      • Hover node   → ancestor path highlighted back to layer 0
      • Click node   → modal with top-n heatmapped images (Esc or backdrop to close)
    """
    GRID_R, GRID_C = 2, 3
    SEP = 2
    import matplotlib.pyplot as _plt
    cmap_hot  = _plt.get_cmap(colormap)
    _denorm_key = model_key or getattr(model.spec, "key", None)

    nodes_by_layer = {
        layer: [n.feature_idx for n in nodes]
        for layer, nodes in circuit.nodes.items() if nodes
    }

    top_hits: Dict[Tuple[int,int], list] = {}

    for batch in tqdm(dataloader, desc="Collecting thumbnails"):
      if isinstance(batch, dict):
          images, labels = batch["pixel_values"], batch.get("labels")
      else:
          images, labels = batch[0], (batch[1] if len(batch) > 1 else None)
      images = images.to(device)

      if target_class is not None and labels is not None:
        mask = (labels == target_class)
        if not mask.any():
            continue
        images = images[mask]          # (B_filtered, C, H, W)

      n_side = images.shape[-1] // patch_size

      for layer, feat_idxs in nodes_by_layer.items():
          site = f"blocks.{layer}.resid_post"
          if site not in bank.saes:
              continue
          acts = reader.read(images, site)[:, prefix:, :]   # (B, P, D)
          Bb, P, D = acts.shape
          z_all = bank[site].encode(acts.reshape(Bb * P, D)).reshape(Bb, P, -1)
          for b in range(Bb):
              z = z_all[b]                                   # (P, F)
              for feat_idx in feat_idxs:
                  acts    = z[:, feat_idx]
                  max_act = acts.max().item()
                  if max_act <= 0:
                      continue
                  key = (layer, feat_idx)
                  if key not in top_hits:
                      top_hits[key] = []
                  top_hits[key].append((
                      max_act,
                      images[b].cpu(),
                      acts.reshape(n_side, n_side).cpu().detach().numpy(),
                  ))
                  top_hits[key].sort(key=lambda x: -x[0])
                  top_hits[key] = top_hits[key][:top_n]

    def denorm(t):
        from ..datasets import denormalize
        if _denorm_key is not None:
            try:
                return denormalize(t, _denorm_key).permute(1, 2, 0).numpy().clip(0, 1)
            except Exception:
                pass
        arr = t.permute(1, 2, 0).numpy()
        return ((arr - arr.min()) / (np.ptp(arr) + 1e-8)).clip(0, 1)

    def blend(img_np, act_map):
        big = np.array(
            PILImage.fromarray((act_map / (act_map.max() + 1e-8) * 255)
                               .clip(0, 255).astype(np.uint8))
            .resize((image_size, image_size), PILImage.BILINEAR)
        ) / 255.0
        h   = cmap_hot(big)[:, :, :3]
        a   = np.clip(big * 0.75, 0, 0.65)[:, :, None]
        return (img_np * (1 - a) + h * a).clip(0, 1)

    def to_b64(arr, size=None, quality=85):
        img = PILImage.fromarray((arr * 255).clip(0, 255).astype(np.uint8)).convert("RGB")
        if size:
            img = img.resize((size, size), PILImage.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode()

    def make_grid_b64(hits):
        cells = []
        for _, img_t, act in hits[:top_n]:
            blended = blend(denorm(img_t), act)
            cell = np.array(
                PILImage.fromarray((blended * 255).astype(np.uint8))
                .resize((node_cell_size, node_cell_size), PILImage.BILINEAR)
            ) / 255.0
            cells.append(cell)
        while len(cells) < top_n:
            cells.append(np.full((node_cell_size, node_cell_size, 3), 0.87))

        gw = GRID_C * node_cell_size + (GRID_C - 1) * SEP
        gh = GRID_R * node_cell_size + (GRID_R - 1) * SEP
        canvas = np.full((gh, gw, 3), 0.35)
        for r in range(GRID_R):
            for c in range(GRID_C):
                y0 = r * (node_cell_size + SEP)
                x0 = c * (node_cell_size + SEP)
                canvas[y0:y0+node_cell_size, x0:x0+node_cell_size] = cells[r*GRID_C + c]
        img = PILImage.fromarray((canvas * 255).clip(0, 255).astype(np.uint8)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        return base64.b64encode(buf.getvalue()).decode(), gw, gh

    thumbs_b64: Dict[str, str] = {}
    modal_b64:  Dict[str, List[str]] = {}
    thumb_w, thumb_h = 0, 0

    for key, hits in top_hits.items():
        k = f"{key[0]}_{key[1]}"
        grid_b64, gw, gh = make_grid_b64(hits)
        thumbs_b64[k] = grid_b64
        thumb_w, thumb_h = gw, gh
        modal_b64[k] = [to_b64(blend(denorm(img_t), act), modal_size)
                        for _, img_t, act in hits]

    all_imp = [abs(n.node_importance)
               for nodes in circuit.nodes.values() for n in nodes]
    max_imp = max(all_imp) if all_imp else 1.0

    js_nodes = {
        str(l): [
            {"layer": n.layer, "feat": n.feature_idx,
             "imp": float(abs(n.node_importance)), "label": n.label}
            for n in sorted(nodes, key=lambda x: -abs(x.node_importance))
        ]
        for l, nodes in circuit.nodes.items() if nodes
    }
    js_edges = [
        {"ul": u.layer, "uf": u.feature_idx,
         "dl": d.layer, "df": d.feature_idx, "w": float(abs(w))}
        for u, d, w in circuit.edges
        if (not positive_edges_only or w > 0)
    ]
    circuit_json = json.dumps({"nodes": js_nodes, "edges": js_edges,
                               "max_imp": max_imp, "title": title})
    thumbs_json  = json.dumps(thumbs_b64)
    modal_json   = json.dumps(modal_b64)
    ratio        = round(thumb_h / max(thumb_w, 1), 4)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:#f4f4f0;color:#222}}
h1{{text-align:center;padding:20px 0 8px;font-size:17px;font-weight:600;color:#333;
    letter-spacing:.3px}}
#wrap{{overflow-x:auto;padding:12px 36px 36px}}
svg{{display:block}}
.node{{cursor:pointer;transition:opacity .12s}}
.edge{{transition:opacity .12s,stroke .12s}}
.lbl{{font-family:-apple-system,sans-serif;pointer-events:none}}

/* modal */
#modal{{display:none;position:fixed;inset:0;z-index:999;
       background:rgba(0,0,0,.65);align-items:center;justify-content:center}}
#modal.open{{display:flex}}
#mbox{{background:#fff;border-radius:12px;padding:24px;max-width:680px;
       width:90vw;box-shadow:0 20px 60px rgba(0,0,0,.35)}}
#mhdr{{display:flex;justify-content:space-between;align-items:flex-start;
       margin-bottom:14px}}
#mid{{font-size:14px;font-weight:700;font-family:monospace;color:#222}}
#mlabel{{font-size:12px;color:#777;font-style:italic;margin-top:3px}}
#mclose{{border:none;background:none;font-size:26px;cursor:pointer;
         color:#aaa;line-height:1;padding:0 2px}}
#mclose:hover{{color:#333}}
#mgrid{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}}
#mgrid img{{width:100%;border-radius:5px;
            box-shadow:0 2px 8px rgba(0,0,0,.14);display:block}}
#mhint{{font-size:11px;color:#bbb;text-align:center;margin-top:12px}}
</style>
</head>
<body>
<h1 id="ptitle"></h1>
<div id="wrap"><svg id="svg" xmlns="http://www.w3.org/2000/svg"></svg></div>

<div id="modal">
  <div id="mbox">
    <div id="mhdr">
      <div><div id="mid"></div><div id="mlabel"></div></div>
      <button id="mclose">&#x00D7;</button>
    </div>
    <div id="mgrid"></div>
    <p id="mhint">Press Esc or click outside to close</p>
  </div>
</div>

<script>
(function(){{

const C      = {circuit_json};
const THUMBS = {thumbs_json};
const MODS   = {modal_json};
const RATIO  = {ratio};

// layout
const NW=190, TPAD=10, TW=NW-TPAD*2, TH=Math.round(TW*RATIO);
const LBDH=30, NH=TH+LBDH+TPAD, CGAP=NW+55, RGAP=NH+28;
const PX=60, PY=50;

const layers=Object.keys(C.nodes).map(Number).sort((a,b)=>a-b);
const maxK=Math.max(...layers.map(l=>C.nodes[l].length));
const SVW=PX*2+(layers.length-1)*CGAP+NW;
const SVH=PY*2+maxK*RGAP+NH*0.5;

const lx={{}};
layers.forEach((l,i)=>{{ lx[l]=PX+i*CGAP; }});

const pos={{}};
layers.forEach(l=>{{
  const ns=C.nodes[l], n=ns.length;
  ns.forEach((nd,i)=>{{
    pos[`${{nd.layer}}_${{nd.feat}}`]={{
      cx: lx[l]+NW/2,
      cy: SVH/2 + (i-(n-1)/2)*RGAP
    }};
  }});
}});

// YlOrRd + power norm
function color(v){{
  const t=Math.pow(v/(C.max_imp+1e-9),.4);
  const s=[[0,[255,255,212]],[.2,[254,217,142]],[.4,[254,153,41]],
            [.6,[240,59,32]],[.8,[204,0,37]],[1,[128,0,38]]];
  for(let i=0;i<s.length-1;i++){{
    if(t>=s[i][0]&&t<=s[i+1][0]){{
      const r=(t-s[i][0])/(s[i+1][0]-s[i][0]);
      const lerp=(a,b)=>Math.round(a+r*(b-a));
      return `rgb(${{lerp(s[i][1][0],s[i+1][1][0])}},${{
                    lerp(s[i][1][1],s[i+1][1][1])}},${{
                    lerp(s[i][1][2],s[i+1][1][2])}})`;
    }}
  }}
  return 'rgb(128,0,38)';
}}

// ancestor tracing
function ancestors(key){{
  const an=new Set(), ae=new Set();
  function walk(k){{
    C.edges.forEach(e=>{{
      const dk=`${{e.dl}}_${{e.df}}`, uk=`${{e.ul}}_${{e.uf}}`;
      if(dk===k&&!an.has(uk)){{ an.add(uk); ae.add(uk+'|'+dk); walk(uk); }}
    }});
  }}
  walk(key);
  return {{an,ae}};
}}

// SVG helpers
const NS='http://www.w3.org/2000/svg';
const svg=document.getElementById('svg');
svg.setAttribute('width',SVW); svg.setAttribute('height',SVH);
svg.setAttribute('viewBox',`0 0 ${{SVW}} ${{SVH}}`);
document.getElementById('ptitle').textContent=C.title;

function el(tag,attrs,par){{
  const e=document.createElementNS(NS,tag);
  for(const[k,v]of Object.entries(attrs)) e.setAttribute(k,v);
  if(par)par.appendChild(e);
  return e;
}}

// arrow markers
const defs=el('defs',{{}},svg);
[['arr-dim','#ccc'],['arr-hi','#2166ac']].forEach(([id,fill])=>{{
  const m=el('marker',{{id,markerWidth:8,markerHeight:8,
                         refX:7,refY:4,orient:'auto'}},defs);
  el('polygon',{{points:'0 0,8 4,0 8',fill}},m);
}});

// edges
const eEls={{}};
const maxW=Math.max(...C.edges.map(e=>e.w),1e-9);
C.edges.forEach(e=>{{
  const uk=`${{e.ul}}_${{e.uf}}`, dk=`${{e.dl}}_${{e.df}}`;
  if(!pos[uk]||!pos[dk])return;
  const {{cx:x1,cy:y1}}=pos[uk], {{cx:x2,cy:y2}}=pos[dk];
  const rel=e.w/(maxW+1e-9);
  const line=el('line',{{
    class:'edge', id:'e_'+uk+'__'+dk,
    x1:x1+NW/2, y1, x2:x2-NW/2-5, y2,
    stroke:'#bbb', 'stroke-width':0.8+2.2*rel,
    'stroke-opacity':0.3+0.6*rel,
    'marker-end':'url(#arr-dim)',
  }},svg);
  eEls[uk+'|'+dk]={{line,bw:0.8+2.2*rel,ba:0.3+0.6*rel}};
}});

// nodes
const nEls={{}};
layers.forEach(l=>{{
  C.nodes[l].forEach(nd=>{{
    const key=`${{nd.layer}}_${{nd.feat}}`;
    const {{cx,cy}}=pos[key];
    const x=cx-NW/2, y=cy-NH/2;
    const fill=color(nd.imp);
    const dark=nd.imp/C.max_imp>.35;
    const tc=dark?'#fff':'#222';

    const g=el('g',{{class:'node',id:'n_'+key,'data-key':key}},svg);

    // background
    el('rect',{{x,y,width:NW,height:NH,rx:8,fill,
                stroke:'rgba(255,255,255,0.55)','stroke-width':1.5}},g);

    // thumbnail
    const tk=`${{nd.layer}}_${{nd.feat}}`;
    if(THUMBS[tk]){{
      const ci=`cp_${{key}}`;
      const cp=el('clipPath',{{id:ci}},defs);
      el('rect',{{x:x+TPAD,y:y+TPAD,width:TW,height:TH,rx:4}},cp);
      el('image',{{
        x:x+TPAD,y:y+TPAD,width:TW,height:TH,
        href:`data:image/jpeg;base64,${{THUMBS[tk]}}`,
        'clip-path':`url(#${{ci}})`,
        preserveAspectRatio:'xMidYMid slice',
      }},g);
    }}

    // feature id
    const ty=y+TH+TPAD+12;
    el('text',{{class:'lbl',x:cx,y:ty,'text-anchor':'middle',
               'font-size':10,'font-weight':700,fill:tc}},g)
      .textContent=`L${{nd.layer}}#${{nd.feat}}`;

    // semantic label
    if(nd.label){{
      const lbl=nd.label.length>22?nd.label.slice(0,21)+'…':nd.label;
      el('text',{{class:'lbl',x:cx,y:ty+13,'text-anchor':'middle',
                 'font-size':8.5,'font-style':'italic',
                 fill:dark?'rgba(255,255,255,.75)':'#555'}},g)
        .textContent=lbl;
    }}

    // invisible hit area
    el('rect',{{x,y,width:NW,height:NH,rx:8,fill:'transparent'}},g);

    nEls[key]=g;
  }});
}});

// layer labels
layers.forEach(l=>{{
  el('text',{{x:lx[l]+NW/2,y:SVH-10,'text-anchor':'middle',
             'font-size':12,fill:'#666',
             'font-family':'-apple-system,sans-serif'}},svg)
    .textContent=`Layer ${{l}}`;
}});

// ── interactions ──────────────────────────────────────────────────────
function highlight(key){{
  const {{an,ae}}=ancestors(key);
  Object.entries(nEls).forEach(([k,g])=>{{
    if(k===key||an.has(k)){{
      g.style.opacity=1;
      const r=g.querySelector('rect');
      r.setAttribute('stroke', k===key?'#1a5fa8':'rgba(100,160,220,0.8)');
      r.setAttribute('stroke-width', k===key?2.5:1.8);
    }}else{{
      g.style.opacity=0.1;
    }}
  }});
  Object.entries(eEls).forEach(([ek,{{line,bw,ba}}])=>{{
    if(ae.has(ek)){{
      line.setAttribute('stroke','#2166ac');
      line.setAttribute('stroke-width',Math.max(bw,1.5));
      line.setAttribute('stroke-opacity',.92);
      line.setAttribute('marker-end','url(#arr-hi)');
    }}else{{
      line.setAttribute('stroke','#ddd');
      line.setAttribute('stroke-width',bw*.5);
      line.setAttribute('stroke-opacity',.07);
      line.setAttribute('marker-end','url(#arr-dim)');
    }}
  }});
}}

function reset(){{
  Object.entries(nEls).forEach(([,g])=>{{
    g.style.opacity=1;
    const r=g.querySelector('rect');
    r.setAttribute('stroke','rgba(255,255,255,0.55)');
    r.setAttribute('stroke-width',1.5);
  }});
  Object.entries(eEls).forEach(([,{{line,bw,ba}}])=>{{
    line.setAttribute('stroke','#bbb');
    line.setAttribute('stroke-width',bw);
    line.setAttribute('stroke-opacity',ba);
    line.setAttribute('marker-end','url(#arr-dim)');
  }});
}}

svg.addEventListener('mouseover',e=>{{
  const g=e.target.closest('.node');
  if(g)highlight(g.dataset.key);
}});
svg.addEventListener('mouseleave',reset);

svg.addEventListener('click',e=>{{
  const g=e.target.closest('.node');
  if(g)openModal(g.dataset.key);
}});

// ── modal ─────────────────────────────────────────────────────────────
const modal=document.getElementById('modal');
const mgrid=document.getElementById('mgrid');

function openModal(key){{
  const imgs=MODS[key]; if(!imgs)return;
  const [l,f]=key.split('_').map(Number);
  const nd=(C.nodes[l]||[]).find(n=>n.feat===f);
  document.getElementById('mid').textContent=`L${{l}}#${{f}}`;
  const lbl=document.getElementById('mlabel');
  lbl.textContent=nd?.label||'';
  lbl.style.display=nd?.label?'block':'none';
  mgrid.innerHTML='';
  imgs.forEach(b64=>{{
    const img=document.createElement('img');
    img.src=`data:image/jpeg;base64,${{b64}}`;
    mgrid.appendChild(img);
  }});
  modal.classList.add('open');
}}

function closeModal(){{
  modal.classList.remove('open');
  mgrid.innerHTML='';
}}

document.getElementById('mclose').addEventListener('click',closeModal);
modal.addEventListener('click',e=>{{ if(!e.target.closest('#mbox'))closeModal(); }});
document.addEventListener('keydown',e=>{{ if(e.key==='Escape')closeModal(); }});

}})();
</script>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved interactive circuit to {path}")