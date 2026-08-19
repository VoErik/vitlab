import { NavLink, Route, Routes } from "react-router-dom";
import { SetupProvider } from "./lib/SetupContext";
import { AtlasView } from "./views/AtlasView";
import { ClassifyView } from "./views/ClassifyView";
import { TokenGroupsView } from "./views/TokenGroupsView";
import { CoverageView } from "./views/CoverageView";
import { InterveneView } from "./views/InterveneView";
import { LogitLensView } from "./views/LogitLensView";

const tabs = [
  { to: "/", label: "Classify + Attribute", el: <ClassifyView /> },
  { to: "/token-groups", label: "Token Groups", el: <TokenGroupsView /> },
  { to: "/coverage", label: "Coverage", el: <CoverageView /> },
  { to: "/atlas", label: "Feature Atlas", el: <AtlasView /> },
  { to: "/intervene", label: "Intervention", el: <InterveneView /> },
  { to: "/logit-lens", label: "Logit Lens", el: <LogitLensView /> },
];

export function App() {
  return (
    <SetupProvider>
      <div className="app">
        <nav className="nav">
          <span className="brand">vitlab Explorer</span>
          {tabs.map((t) => (
            <NavLink key={t.to} to={t.to} end className={({ isActive }) => (isActive ? "active" : "")}>
              {t.label}
            </NavLink>
          ))}
        </nav>
        <main className="main">
          <Routes>{tabs.map((t) => <Route key={t.to} path={t.to} element={t.el} />)}</Routes>
        </main>
      </div>
    </SetupProvider>
  );
}
