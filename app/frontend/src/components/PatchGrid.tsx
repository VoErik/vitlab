// Clickable side×side grid over the model's exact image. Selection = flat patch indices.
import { useState } from "react";

export function PatchGrid({ src, side, selected, onChange, overlay, size = 448 }: {
  src: string; side: number; selected: Set<number>;
  onChange: (s: Set<number>) => void; overlay?: (idx: number) => string | undefined; size?: number;
}) {
  const [drag, setDrag] = useState(false);
  const cell = size / side;
  const apply = (i: number, add: boolean) => {
    const s = new Set(selected); add ? s.add(i) : s.delete(i); onChange(s);
  };
  return (
    <div className="pgrid" style={{ width: size, height: size, position: "relative" }}
      onMouseUp={() => setDrag(false)} onMouseLeave={() => setDrag(false)}>
      <img src={src} width={size} height={size} alt="input" />
      <svg width={size} height={size} style={{ position: "absolute", top: 0, left: 0 }}>
        {Array.from({ length: side * side }, (_, i) => {
          const r = Math.floor(i / side), c = i % side;
          const fill = overlay?.(i);
          return (
            <rect key={i} x={c * cell} y={r * cell} width={cell} height={cell}
              fill={fill ?? (selected.has(i) ? "rgba(94,79,162,0.5)" : "transparent")}
              stroke="rgba(255,255,255,0.12)"
              onMouseDown={() => { setDrag(true); apply(i, !selected.has(i)); }}
              onMouseEnter={() => drag && apply(i, true)}
              style={{ cursor: "pointer" }} />
          );
        })}
      </svg>
    </div>
  );
}
