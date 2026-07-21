// Renders a (h×w) grid_map as a viridis overlay on a canvas, sized to `size`.
import { useEffect, useRef } from "react";

// minimal viridis stops (t in [0,1] -> [r,g,b])
const VIR: [number, number, number][] = [
  [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
  [31, 158, 137], [53, 183, 121], [110, 206, 88], [181, 222, 43], [253, 231, 37],
];
function viridis(t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, t)) * (VIR.length - 1);
  const i = Math.floor(x), f = x - i, a = VIR[i], b = VIR[Math.min(i + 1, VIR.length - 1)];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

export function Heatmap({ grid, size = 448, alpha = 0.55 }:
  { grid: number[][]; size?: number; alpha?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current; if (!cv || !grid?.length) return;
    const h = grid.length, w = grid[0].length;
    let mn = Infinity, mx = -Infinity;
    for (const row of grid) for (const v of row) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
    const rng = mx - mn || 1;
    const off = document.createElement("canvas"); off.width = w; off.height = h;
    const octx = off.getContext("2d")!; const im = octx.createImageData(w, h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const [r, g, b] = viridis((grid[y][x] - mn) / rng);
      const k = (y * w + x) * 4;
      im.data[k] = r; im.data[k + 1] = g; im.data[k + 2] = b; im.data[k + 3] = Math.round(alpha * 255);
    }
    octx.putImageData(im, 0, 0);
    const ctx = cv.getContext("2d")!; ctx.clearRect(0, 0, size, size);
    ctx.imageSmoothingEnabled = true; ctx.drawImage(off, 0, 0, size, size);
  }, [grid, size, alpha]);
  return <canvas ref={ref} width={size} height={size}
    style={{ position: "absolute", top: 0, left: 0, pointerEvents: "none" }} />;
}
