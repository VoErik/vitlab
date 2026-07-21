// Reusable model/task/bank/upload picker driven by useSetup().
import { useSetup } from "../lib/useSetup";

export function SetupPanel({ s, showBank = true, showSite = true }:
  { s: ReturnType<typeof useSetup>; showBank?: boolean; showSite?: boolean }) {
  return (
    <div className="panel narrow">
      <h3>Setup</h3>
      <label>Model
        <select value={s.modelId} onChange={(e) => { s.setModelId(e.target.value); s.setTask(""); }}>
          <option value="">—</option>
          {s.models.map((m) => <option key={m.id} value={m.id}>{m.id}</option>)}
        </select>
      </label>
      {s.model && <label>Task
        <select value={s.task} onChange={(e) => s.setTask(e.target.value)}>
          <option value="">—</option>
          {s.model.tasks.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>
      </label>}
      {showBank && <label>SAE bank
        <select value={s.bankId} onChange={(e) => s.setBankId(e.target.value)}>
          <option value="">—</option>
          {s.banks.map((b) => <option key={b.id} value={b.id}>{b.id}</option>)}
        </select>
      </label>}
      {showSite && <label>Site
        <select value={s.site} onChange={(e) => s.setSite(e.target.value)}>
          {(s.banks.find((b) => b.id === s.bankId)?.sites ?? [s.site]).map((si) =>
            <option key={si} value={si}>{si}</option>)}
        </select>
      </label>}
      <label className="file">Upload image
        <input type="file" accept="image/*" disabled={!s.task}
          onChange={(e) => e.target.files?.[0] && s.upload(e.target.files[0])} />
      </label>
    </div>
  );
}
