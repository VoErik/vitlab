// Clean-vs-corrupted (or single) probability bars.
export function ProbBars({ probs, names, highlight, onPick, compare }:
  { probs: number[]; names?: string[]; highlight?: number; onPick?: (i: number) => void; compare?: number[] }) {
  return (
    <div>
      {probs.map((p, i) => (
        <div key={i} className={"barrow" + (i === highlight ? " sel" : "")}
          onClick={() => onPick?.(i)} style={{ cursor: onPick ? "pointer" : "default" }}>
          <span>{names?.[i] ?? i}</span>
          <div className="bar">
            <div style={{ width: `${p * 100}%` }} />
            {compare && <div className="cmp" style={{ width: `${compare[i] * 100}%` }} />}
          </div>
          <span>{(p * 100).toFixed(1)}%{compare ? ` → ${(compare[i] * 100).toFixed(1)}%` : ""}</span>
        </div>
      ))}
    </div>
  );
}
